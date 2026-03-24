"""SKILL.md parser and OpenAI tool schema translator.

Reads SKILL.md files and converts them into one of two forms:

  Tool skills (agentji format):
    Have ``parameters:`` and ``scripts.execute:`` in frontmatter.
    Translated to OpenAI-compatible tool JSON schemas — offered to the LLM
    as callable tools and executed as subprocesses.

  Prompt skills (Anthropic Claude format):
    Have only ``name:`` and ``description:`` — no ``parameters:``, no
    ``scripts.execute:``.  The Markdown body is a system-prompt extension
    injected into the agent's context.  Helper scripts in ``scripts/`` are
    made available for the agent to call via the ``bash`` built-in.

This dual-mode design means Anthropic's official skills (xlsx, docx, pdf, …)
work in agentji with zero modifications — just add ``builtins: [bash, ...]``
to the agent config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


# ── Types ─────────────────────────────────────────────────────────────────────

ToolSchema = dict[str, Any]
"""An OpenAI-compatible tool definition dict."""


# ── SKILL.md parser ───────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^\s*---\s*\n(.*?)\n---\s*\n", re.DOTALL)

SUPPORTED_PARAM_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


class SkillParseError(ValueError):
    """Raised when a SKILL.md file cannot be parsed or is missing required fields."""


def _parse_frontmatter(text: str, source: str) -> tuple[dict[str, Any], str]:
    """Extract YAML frontmatter and Markdown body from a SKILL.md string.

    Args:
        text: Full contents of the SKILL.md file.
        source: File path string used in error messages.

    Returns:
        A tuple of ``(frontmatter_dict, body_markdown)``.

    Raises:
        SkillParseError: If no frontmatter block is found or YAML is invalid.
    """
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise SkillParseError(
            f"'{source}' is missing a YAML frontmatter block. "
            f"SKILL.md files must start with '---' delimited YAML."
        )

    try:
        frontmatter: dict[str, Any] = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise SkillParseError(
            f"Invalid YAML frontmatter in '{source}': {exc}"
        ) from exc

    body = text[match.end():]
    return frontmatter, body


def _translate_parameters(
    raw_params: dict[str, Any], source: str
) -> tuple[dict[str, Any], list[str]]:
    """Convert SKILL.md parameter definitions to JSON Schema properties.

    Args:
        raw_params: The ``parameters`` dict from the frontmatter.
        source: File path string used in error messages.

    Returns:
        A tuple of ``(properties_dict, required_list)``.

    Raises:
        SkillParseError: If a parameter is missing the required ``type`` field or
            uses an unsupported type.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param_def in raw_params.items():
        if not isinstance(param_def, dict):
            raise SkillParseError(
                f"Parameter '{param_name}' in '{source}' must be a mapping, "
                f"got {type(param_def).__name__}."
            )

        param_type = param_def.get("type")
        if not param_type:
            raise SkillParseError(
                f"Parameter '{param_name}' in '{source}' is missing required field 'type'."
            )
        if param_type not in SUPPORTED_PARAM_TYPES:
            raise SkillParseError(
                f"Parameter '{param_name}' in '{source}' has unsupported type '{param_type}'. "
                f"Supported types: {sorted(SUPPORTED_PARAM_TYPES)}."
            )

        prop: dict[str, Any] = {"type": param_type}

        if "description" in param_def:
            prop["description"] = param_def["description"]

        if "default" in param_def:
            prop["default"] = param_def["default"]

        # Array items schema
        if param_type == "array" and "items" in param_def:
            prop["items"] = param_def["items"]

        # Enum constraint
        if "enum" in param_def:
            prop["enum"] = param_def["enum"]

        properties[param_name] = prop

        is_required = param_def.get("required", False)
        if is_required:
            required.append(param_name)

    return properties, required


# ── Public translator ─────────────────────────────────────────────────────────

def translate_skill(skill_path: str | Path) -> ToolSchema:
    """Parse a SKILL.md file and return a skill descriptor dict.

    Returns one of two forms depending on whether the skill has an executable
    script entry point:

    **Tool skill** (has ``scripts.execute`` in frontmatter):
        An OpenAI-compatible tool definition that the LLM can call as a
        function.  Internal keys ``_scripts`` and ``_skill_dir`` are consumed
        by ``executor.py`` and stripped before sending to the LLM.

    **Prompt skill** (no ``scripts.execute``; Anthropic Claude format):
        A descriptor with ``_prompt_only: True``, ``_body`` (the Markdown
        body to inject into the system prompt), and ``_skill_dir`` (path to
        the skill directory, so the agent can locate helper scripts).

    Args:
        skill_path: Path to the *directory* containing ``SKILL.md``, or
            directly to the ``SKILL.md`` file itself.

    Returns:
        A skill descriptor dict.

    Raises:
        FileNotFoundError: If the SKILL.md file does not exist.
        SkillParseError: If required frontmatter fields are missing or invalid.
    """
    path = Path(skill_path)
    if path.is_dir():
        skill_file = path / "SKILL.md"
    else:
        skill_file = path

    if not skill_file.exists():
        raise FileNotFoundError(
            f"SKILL.md not found at '{skill_file}'. "
            f"Each skill directory must contain a SKILL.md file."
        )

    source = str(skill_file)
    text = skill_file.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(text, source)

    # ── Required fields ────────────────────────────────────────────────────
    raw_name: str | None = frontmatter.get("name")
    if not raw_name:
        raise SkillParseError(
            f"'{source}' is missing required frontmatter field 'name'."
        )

    # Use `slug` as the canonical identifier if present (e.g. clawhub skills
    # ship with slug: data-analysis even when name: "Data Analysis").
    # This keeps agent skill references stable across versions/folder renames.
    name: str = frontmatter.get("slug") or raw_name

    description: str | None = frontmatter.get("description")
    if not description:
        raise SkillParseError(
            f"'{source}' (skill '{raw_name}') is missing required frontmatter field 'description'."
        )

    skill_dir = str(skill_file.parent)

    # ── Detect skill type ──────────────────────────────────────────────────
    # skill.yaml sidecar takes priority over inline frontmatter fields.
    # This supports the Anthropic-compatible split: SKILL.md stays clean,
    # skill.yaml carries the agentji tool config (scripts + parameters).
    sidecar_path = skill_file.parent / "skill.yaml"
    if sidecar_path.exists():
        try:
            sidecar: dict[str, Any] = yaml.safe_load(
                sidecar_path.read_text(encoding="utf-8")
            ) or {}
        except yaml.YAMLError as exc:
            raise SkillParseError(
                f"Invalid skill.yaml in '{skill_file.parent}': {exc}"
            ) from exc
        # Merge: sidecar scripts/parameters/timeout override frontmatter
        if "scripts" in sidecar:
            frontmatter["scripts"] = sidecar["scripts"]
        if "parameters" in sidecar:
            frontmatter["parameters"] = sidecar["parameters"]
        if "timeout" in sidecar:
            frontmatter["timeout"] = sidecar["timeout"]

    scripts: dict[str, str] = frontmatter.get("scripts") or {}
    execute_script = scripts.get("execute")

    if not execute_script:
        # Prompt-only skill (Anthropic Claude format).
        # The Markdown body is injected into the agent's system prompt.
        # Helper scripts in scripts/ are accessible via the bash built-in.
        return {
            "_prompt_only": True,
            "function": {
                "name": name,
                "description": description,
            },
            "_body": body,
            "_skill_dir": skill_dir,
        }

    # ── Tool skill: build OpenAI parameters schema ─────────────────────────
    raw_params: dict[str, Any] = frontmatter.get("parameters") or {}
    if raw_params:
        properties, required = _translate_parameters(raw_params, source)
    else:
        properties, required = {}, []

    parameters_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters_schema["required"] = required

    tool: ToolSchema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters_schema,
        },
        # Internal metadata consumed by executor.py — not sent to the LLM
        "_scripts": scripts,
        "_skill_dir": skill_dir,
        "_timeout": frontmatter.get("timeout"),  # None = use agent default
    }

    return tool


def translate_skills(skill_paths: list[str | Path]) -> list[ToolSchema]:
    """Translate a list of skill directories into OpenAI tool schemas.

    Args:
        skill_paths: List of paths to skill directories or SKILL.md files.

    Returns:
        List of OpenAI tool definition dicts.

    Raises:
        FileNotFoundError: If any SKILL.md is missing.
        SkillParseError: If any SKILL.md has invalid content.
    """
    return [translate_skill(p) for p in skill_paths]
