"""Post-session skill improvement extractor for agentji.

At the end of each session (browser tab close, explicit API call, or idle
timeout), this module sends the full conversation to the configured model and
extracts three types of learning signals:

- correction   — the user corrected a wrong or sub-optimal agent response
- affirmation  — the user confirmed something worked well (non-obvious choices)
- hint         — the user provided context mid-conversation that helped the agent

Extracted improvements are appended to ``improvements.jsonl`` alongside each
skill's ``SKILL.md``.  If a signal maps to no specific skill it is stored in
``improvements.jsonl`` in the working directory as ``skill="general"``.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import litellm


# ── Extraction prompt ─────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM = """\
You are an AI skill improvement analyst. Your task is to analyse a conversation
between a user and an AI assistant and extract actionable learning signals that
could improve the AI's performance on future tasks.

Identify three types of signals:
- "correction"   : the user corrected a wrong, incomplete, or sub-optimal agent response
- "affirmation"  : the user explicitly confirmed something worked well (only non-obvious cases)
- "hint"         : the user provided helpful context or constraints mid-conversation
                   that allowed the agent to proceed

For each signal produce a JSON object with:
  type      - one of: correction, affirmation, hint
  skill     - the most relevant skill name from the provided list, or "general"
  learning  - a concise, actionable sentence describing what was learned
  context   - one or two sentences from the conversation that evidence this signal

Return ONLY a valid JSON array. If no signals are found return [].

Example output:
[
  {
    "type": "correction",
    "skill": "sql-query",
    "learning": "Always alias columns in multi-table GROUP BY to avoid ambiguous references.",
    "context": "User: 'The query fails — column amount is ambiguous.' Agent fixed by adding table alias."
  },
  {
    "type": "affirmation",
    "skill": "data-analysis",
    "learning": "Present numeric trends as percentage change rather than raw delta.",
    "context": "User: 'Yes, exactly — this format is perfect, keep it like this.'"
  }
]\
"""


def _build_user_prompt(messages: list[dict], skill_names: list[str]) -> str:
    skills_str = ", ".join(skill_names) if skill_names else "none"
    parts = [f"Available skills: {skills_str}", "", "Conversation:"]
    for m in messages:
        role = str(m.get("role", "unknown")).upper()
        content = str(m.get("content", ""))
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


def _parse_json_array(text: str) -> list[dict]:
    """Extract a JSON array from an LLM response (handles markdown fences)."""
    text = text.strip()
    # Strip ```json ... ``` or ``` ... ``` fences
    fenced = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text)
    if fenced:
        text = fenced.group(1).strip()
    # Find the first [...] block as a fallback
    if not text.startswith("["):
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            text = m.group(0)
    result = json.loads(text)
    if not isinstance(result, list):
        return []
    return [item for item in result if isinstance(item, dict)]


# ── Public entry point ────────────────────────────────────────────────────────

def extract_and_save(
    messages: list[dict],
    session_id: str,
    skill_refs: list[dict],   # [{"name": str, "path": str}]
    model: str,               # litellm model string, e.g. "openai/gpt-4o-mini"
    litellm_kwargs: dict,     # api_key, api_base, etc.
    target_skills: list[str], # empty = all skills
    fallback_improvements_path: Path | None = None,
) -> list[dict]:
    """Extract improvement signals and save to per-skill improvements.jsonl files.

    Args:
        messages: Full conversation message list (role/content dicts).
        session_id: Session identifier stamped on every saved entry.
        skill_refs: List of ``{"name": ..., "path": ...}`` dicts from the pipeline.
        model: litellm model string for the extraction call.
        litellm_kwargs: Extra kwargs forwarded to ``litellm.completion`` (api_key, api_base).
        target_skills: If non-empty, only extract improvements for these skill names.
        fallback_improvements_path: Path for "general" (non-skill-specific) improvements.
            Defaults to ``Path.cwd() / "improvements.jsonl"``.

    Returns:
        List of extracted improvement dicts (useful for logging / tests).
    """
    if not messages:
        return []

    skill_by_name = {s["name"]: s["path"] for s in skill_refs}
    skill_names = [
        s["name"] for s in skill_refs
        if not target_skills or s["name"] in target_skills
    ]

    user_prompt = _build_user_prompt(messages, skill_names)

    try:
        call_kwargs = {k: v for k, v in litellm_kwargs.items() if k != "model"}
        resp = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            **call_kwargs,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception:
        return []

    try:
        improvements = _parse_json_array(raw)
    except (json.JSONDecodeError, ValueError):
        return []

    now = datetime.now(timezone.utc).isoformat()
    saved: list[dict] = []

    for item in improvements:
        skill_name = str(item.get("skill") or "general")
        entry = {
            "ts": now,
            "session_id": session_id,
            "skill": skill_name,
            "type": str(item.get("type") or "unknown"),
            "learning": str(item.get("learning") or ""),
            "context": str(item.get("context") or ""),
        }

        # Determine target file
        skill_path_str = skill_by_name.get(skill_name)
        if skill_path_str:
            target_file = Path(skill_path_str) / "improvements.jsonl"
        else:
            target_file = (
                fallback_improvements_path
                if fallback_improvements_path
                else Path.cwd() / "improvements.jsonl"
            )

        try:
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with target_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        saved.append(entry)

    return saved
