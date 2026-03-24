"""LLM-based skill.yaml generator.

When a skill directory has executable scripts but no typed parameter schema
(i.e. it follows the pure Anthropic SKILL.md format with no `parameters:` or
`scripts:` frontmatter), this module inspects the scripts and SKILL.md body,
calls the active model, and writes a `skill.yaml` sidecar that makes the skill
callable as a typed tool.

Typical trigger: the LLM tried to call a tool by name, but agentji loaded it
as a prompt-only skill because the SKILL.md had no `scripts.execute`.
"""

from __future__ import annotations

import re
import sys
import select
import threading
from pathlib import Path
from typing import Any

import yaml
import litellm


# ── Prompt ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a skill configuration expert for agentji, a multi-agent framework.
Your job is to inspect a skill directory and produce a skill.yaml sidecar file
that exposes the skill's scripts as a typed callable tool.

The skill.yaml format is:

```yaml
scripts:
  execute: scripts/run.py   # relative path to the main executable script

parameters:
  param_name:
    type: string            # string | integer | number | boolean | array | object
    description: What this parameter does.
    required: true
  optional_param:
    type: string
    description: Optional parameter.
    required: false
    default: "value"        # omit if no sensible default
```

Rules:
- Only include parameters the script actually reads from its stdin JSON.
- Infer types from how the script uses the values (int(), float(), bool(), etc.).
- Mark a parameter required: true if the script accesses it without a fallback.
- Include a default only if the script has an explicit one (e.g. params.get("x", "default")).
- Choose the simplest script in scripts/ as the execute entry point
  (prefer run.py, main.py, or the only .py file present).
- Output ONLY the raw YAML — no markdown fences, no explanation.
"""

_USER_TEMPLATE = """\
Skill name: {name}
Skill description: {description}

SKILL.md body (usage notes and examples):
{body}

Scripts found in scripts/:
{scripts_block}

Generate the skill.yaml for this skill.
"""


# ── Core converter ────────────────────────────────────────────────────────────

def convert_skill(skill_dir: Path, litellm_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Read a skill directory, call the LLM, and write skill.yaml.

    Args:
        skill_dir: Path to the skill directory (must contain SKILL.md).
        litellm_kwargs: The litellm completion kwargs for the active model
            (model, api_base, api_key, etc.) — reuses the agent's own config.

    Returns:
        {"success": True, "skill_yaml": "<yaml string>"}  on success
        {"success": False, "error": "<message>"}          on failure
    """
    skill_md_path = skill_dir / "SKILL.md"
    if not skill_md_path.exists():
        return {"success": False, "error": f"SKILL.md not found in {skill_dir}"}

    # ── Parse SKILL.md ────────────────────────────────────────────────────────
    text = skill_md_path.read_text(encoding="utf-8")
    fm_match = re.match(r"^\s*---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not fm_match:
        return {"success": False, "error": "SKILL.md has no frontmatter"}

    try:
        frontmatter: dict = yaml.safe_load(fm_match.group(1)) or {}
    except yaml.YAMLError as exc:
        return {"success": False, "error": f"Invalid SKILL.md frontmatter: {exc}"}

    body = text[fm_match.end():]
    name = frontmatter.get("slug") or frontmatter.get("name", skill_dir.name)
    description = frontmatter.get("description", "")

    # ── Collect scripts ───────────────────────────────────────────────────────
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.exists():
        return {"success": False, "error": f"No scripts/ directory in {skill_dir}"}

    script_files = sorted(scripts_dir.glob("*.py")) + sorted(scripts_dir.glob("*.sh"))
    if not script_files:
        return {"success": False, "error": "No .py or .sh scripts found"}

    scripts_block_parts = []
    for sf in script_files:
        try:
            src = sf.read_text(encoding="utf-8")
        except Exception:
            src = "(unreadable)"
        scripts_block_parts.append(f"--- {sf.name} ---\n{src}")
    scripts_block = "\n\n".join(scripts_block_parts)

    # ── Call LLM ──────────────────────────────────────────────────────────────
    user_msg = _USER_TEMPLATE.format(
        name=name,
        description=str(description).strip(),
        body=body.strip(),
        scripts_block=scripts_block,
    )

    # Strip streaming callback and other non-completion keys from kwargs
    safe_kwargs = {
        k: v for k, v in litellm_kwargs.items()
        if k in ("model", "api_base", "api_key", "base_url", "timeout",
                 "max_tokens", "temperature", "extra_headers")
    }
    safe_kwargs.setdefault("temperature", 0.1)
    safe_kwargs.setdefault("max_tokens", 1024)

    try:
        response = litellm.completion(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            **safe_kwargs,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        return {"success": False, "error": f"LLM call failed: {exc}"}

    # Strip markdown fences if the model wrapped output anyway
    raw = re.sub(r"^```(?:yaml)?\s*\n?", "", raw.strip(), flags=re.IGNORECASE)
    raw = re.sub(r"\n?```\s*$", "", raw.strip())
    raw = raw.strip()

    # ── Validate YAML ─────────────────────────────────────────────────────────
    try:
        parsed = yaml.safe_load(raw)
        if not isinstance(parsed, dict) or "scripts" not in parsed:
            return {"success": False, "error": "Generated YAML is missing 'scripts' key"}
    except yaml.YAMLError as exc:
        return {"success": False, "error": f"Generated YAML is invalid: {exc}"}

    # ── Write skill.yaml ──────────────────────────────────────────────────────
    out_path = skill_dir / "skill.yaml"
    out_path.write_text(raw + "\n", encoding="utf-8")

    return {"success": True, "skill_yaml": raw, "path": str(out_path)}


# ── User consent prompt ───────────────────────────────────────────────────────

def prompt_user_for_conversion(skill_name: str, timeout: int = 20) -> bool:
    """Print a conversion prompt to stderr and wait up to `timeout` seconds.

    Returns True if the user approves (or times out — auto-yes),
    False if the user explicitly declines ('n'/'no'/'skip').

    Works in both interactive terminals and non-interactive contexts
    (e.g. piped stdin): if stdin is not a TTY the function auto-approves.
    """
    msg = (
        f"\n[agentji] Skill '{skill_name}' has callable scripts but no skill.yaml.\n"
        f"          agentji can scan the scripts and generate one automatically.\n"
        f"          Proceeding in {timeout}s — type 'n' + Enter to skip: "
    )
    sys.stderr.write(msg)
    sys.stderr.flush()

    # Non-interactive stdin (piped / serve mode) → auto-approve
    if not sys.stdin.isatty():
        sys.stderr.write(f"\n[agentji] Non-interactive mode — auto-converting.\n")
        return True

    # Use select for a non-blocking read with timeout (Unix only)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], float(timeout))
        if ready:
            line = sys.stdin.readline().strip().lower()
            approved = line not in ("n", "no", "skip")
            if not approved:
                sys.stderr.write("[agentji] Skipping conversion.\n")
            return approved
        else:
            sys.stderr.write(f"\n[agentji] No response — proceeding with conversion.\n")
            return True
    except (OSError, ValueError):
        # select not available (Windows) or stdin closed
        sys.stderr.write(f"\n[agentji] Auto-converting '{skill_name}'.\n")
        return True
