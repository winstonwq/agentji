"""In-session and cross-run memory backend for agentji.

Provides two complementary features when ``memory:`` is set in agentji.yaml:

Sliding window compression
    Before each LLM call, if the message list exceeds ``window_size``, the
    oldest half of the conversation is summarised into a single compressed-
    history block using the agent's own model. This keeps the active context
    bounded without dropping information entirely.

Long-term memory (LTM)
    At the start of each root run, relevant facts from previous runs are
    injected into the system prompt. At the end of each root run, the agent's
    final response is distilled into 3-7 concise facts and persisted to a
    per-user JSONL file under ``ltm_path``.

Both features are safe no-ops when ``memory:`` is absent from config.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import litellm

if TYPE_CHECKING:
    from agentji.config import MemoryConfig

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_SUMMARY_TAG = "[agentji:compressed-history]"

# Token-based thresholds (fraction of model context window).
# fallback_msgs used when litellm can't determine the context window.
_COMPRESSION_PRESETS: dict[str, dict] = {
    #              trigger  keep-tail  fallback-msgs
    "auto":       {"threshold": 0.75, "target": 0.40, "fallback_msgs": 40},
    "aggressive": {"threshold": 0.50, "target": 0.20, "fallback_msgs": 20},
}

_COMPRESS_SYSTEM = (
    "You are a conversation compressor for an AI agent. "
    "Produce a dense, factual summary of the following conversation exchanges. "
    "Preserve all task context, decisions made, key tool results, file paths, "
    "and any information the agent would need to continue the work coherently. "
    "Be thorough but concise. Output only the summary — no preamble, no headers."
)

_EXTRACT_SYSTEM = (
    "Extract 3-7 concise, factual key points from this AI agent run. "
    "Focus on outcomes, decisions made, and context useful in future sessions. "
    "Return a JSON array of short strings only — no markdown, no preamble."
)


# ── Public backend class ───────────────────────────────────────────────────────

class MemoryBackend:
    """In-session and cross-run memory backend.

    All public methods are safe no-ops when ``config`` is ``None`` or
    ``config.backend`` is not ``'local'``.

    Methods
    -------
    inject(system_prompt, user_message) -> str
        Called once before the first LLM call of a root run.
        Prepends LTM facts from previous sessions to the system prompt.

    maybe_compress(messages, litellm_kwargs) -> list[dict]
        Called before each LLM call.
        Returns a (possibly shorter) message list with older turns compressed.

    remember(run_id, summary, litellm_kwargs) -> None
        Called once after a root run completes.
        Extracts key facts from ``summary`` and appends them to the LTM store.
    """

    def __init__(self, config: "MemoryConfig | None" = None) -> None:
        self.config = config
        self.enabled = config is not None and config.backend == "local"
        self._ltm_file: Path | None = None

        if self.enabled:
            self._ltm_file = Path(config.ltm_path) / f"{config.user_id}.jsonl"

    # ── Public interface ───────────────────────────────────────────────────────

    def inject(self, system_prompt: str, user_message: str) -> str:
        """Return system_prompt with LTM facts block appended (if any exist)."""
        if not self.enabled or not self._ltm_file or not self._ltm_file.exists():
            return system_prompt

        facts = self._load_ltm_facts(self.config.inject_limit)
        if not facts:
            return system_prompt

        block = (
            "\n\n---\n## Memory from previous sessions\n"
            + "\n".join(f"- {f}" for f in facts)
        )
        return system_prompt + block

    def maybe_compress(
        self, messages: list[dict[str, Any]], litellm_kwargs: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Compress older messages into a summary when the context budget is exceeded.

        Uses token-based detection when the model's context window is known to
        litellm, otherwise falls back to a message-count heuristic. The system
        message at index 0 is always preserved. Prior compressed summaries are
        included in the next compression pass so context is never lost.

        Returns the (possibly modified) messages list.
        """
        if not self.enabled or self.config.compression == "off":
            return messages

        preset = _COMPRESSION_PRESETS[self.config.compression]
        model = litellm_kwargs.get("model", "")

        # ── Try token-based detection ─────────────────────────────────────────
        context_window = _get_context_window(model)
        if context_window:
            try:
                total_tokens = litellm.token_counter(model=model, messages=messages)
            except Exception:
                context_window = None  # fall through to message-count path

        if context_window:
            trigger_tokens = int(context_window * preset["threshold"])
            if total_tokens <= trigger_tokens:
                return messages  # within budget

            # Walk from the tail, keep messages up to the target budget.
            tail_budget = int(context_window * preset["target"])
            tail, tail_tokens = [], 0
            for msg in reversed(messages[1:]):
                try:
                    msg_tok = litellm.token_counter(model=model, messages=[msg])
                except Exception:
                    msg_tok = len(str(msg.get("content") or "")) // 4  # rough fallback
                if tail_tokens + msg_tok > tail_budget:
                    break
                tail.insert(0, msg)
                tail_tokens += msg_tok
        else:
            # ── Message-count fallback ────────────────────────────────────────
            fallback = preset["fallback_msgs"]
            non_system = messages[1:]
            if len(non_system) <= fallback:
                return messages
            keep_n = fallback // 2
            tail = non_system[-keep_n:]

        tail_start = len(messages) - len(tail)
        to_compress = messages[1:tail_start]

        if not to_compress:
            return messages  # nothing old enough to compress

        try:
            summary_text = _summarize(to_compress, litellm_kwargs)
        except Exception as exc:
            logger.warning("agentji.memory: compression failed, keeping full history: %s", exc)
            return messages

        summary_msg: dict[str, Any] = {
            "role": "system",
            "content": (
                f"{_SUMMARY_TAG}\n"
                "The following is a compressed summary of the earlier conversation:\n"
                f"{summary_text}"
            ),
        }
        return [messages[0], summary_msg] + tail

    def remember(
        self, run_id: str, summary: str, litellm_kwargs: dict[str, Any]
    ) -> None:
        """Extract key facts from ``summary`` and persist them to the LTM store."""
        if not self.enabled or not self.config.auto_remember or not self._ltm_file:
            return

        try:
            facts = _extract_facts(summary, litellm_kwargs)
        except Exception as exc:
            logger.warning("agentji.memory: LTM extraction failed: %s", exc)
            return

        if not facts:
            return

        self._ltm_file.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "facts": facts,
        }
        with self._ltm_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_ltm_facts(self, limit: int) -> list[str]:
        """Load the most recent ``limit`` entry-sets from the LTM JSONL, flattened."""
        try:
            lines = self._ltm_file.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

        entries: list[dict] = []
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= limit:
                break

        facts: list[str] = []
        for entry in reversed(entries):  # chronological order for readability
            date = entry.get("ts", "")[:10]  # YYYY-MM-DD
            for fact in entry.get("facts", []):
                facts.append(f"[{date}] {fact}" if date else str(fact))
        return facts


# ── LLM helpers (module-level, shared) ────────────────────────────────────────

def _get_context_window(model: str) -> int | None:
    """Return the model's input context window size in tokens, or None if unknown."""
    try:
        info = litellm.get_model_info(model)
        return info.get("max_input_tokens") or info.get("max_tokens") or None
    except Exception:
        return None


def _summarize(
    messages: list[dict[str, Any]], litellm_kwargs: dict[str, Any]
) -> str:
    """Call the LLM to produce a summary of ``messages``."""
    transcript = _format_as_transcript(messages)
    # Force non-streaming; strip tool bindings so any model can handle it.
    kwargs = {
        **litellm_kwargs,
        "stream": False,
        "tools": None,
        "tool_choice": None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    response = litellm.completion(
        messages=[
            {"role": "system", "content": _COMPRESS_SYSTEM},
            {"role": "user", "content": transcript},
        ],
        **kwargs,
    )
    return (response.choices[0].message.content or "").strip()


def _extract_facts(summary: str, litellm_kwargs: dict[str, Any]) -> list[str]:
    """Call the LLM to extract key facts from a run summary as a JSON list."""
    kwargs = {
        **litellm_kwargs,
        "stream": False,
        "tools": None,
        "tool_choice": None,
    }
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    response = litellm.completion(
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": summary},
        ],
        **kwargs,
    )
    raw = (response.choices[0].message.content or "").strip()
    # Strip markdown code fences if the model wraps its output.
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(f).strip() for f in parsed if str(f).strip()]
    except json.JSONDecodeError:
        pass
    # Fallback: treat each non-empty line as a fact.
    return [ln.lstrip("•-– ").strip() for ln in raw.splitlines() if ln.strip()]


def _format_as_transcript(messages: list[dict[str, Any]]) -> str:
    """Render ``messages`` as a plain-text transcript for the summariser LLM."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            # Pass prior compressed summaries through so context chains correctly.
            if _SUMMARY_TAG in content:
                lines.append(content)
            # Skip other system messages (they're usually injected boilerplate).
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                names = ", ".join(tc["function"]["name"] for tc in tool_calls)
                lines.append(f"[assistant called tools: {names}]")
            if content:
                lines.append(f"[assistant]: {content}")

        elif role == "tool":
            snippet = content[:600]
            if len(content) > 600:
                snippet += f"… ({len(content) - 600} chars truncated)"
            lines.append(f"[tool result]: {snippet}")

        elif role == "user":
            lines.append(f"[user]: {content}")

        else:
            lines.append(f"[{role}]: {content}")

    return "\n".join(lines)
