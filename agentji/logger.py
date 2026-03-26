"""Conversation logger for agentji.

Records every message, tool call, and tool result across the agentic loop
into a JSONL file. Each line is a self-contained JSON event.

One logger instance can span multiple ``run_agent()`` calls (a pipeline),
grouping them under a shared ``pipeline_id``.

Supports daily log rotation for long-running serve deployments.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


_TRUNCATE_AT = 2000  # chars — preview limit for long content in logs


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _preview(text: str | None, limit: int = _TRUNCATE_AT) -> str:
    """Return a truncated preview of a string."""
    if not text:
        return ""
    s = str(text)
    if len(s) > limit:
        return s[:limit] + f"  …[{len(s) - limit} chars truncated]"
    return s


class ConversationLogger:
    """Append-only JSONL logger for an agentji pipeline run.

    Each event is written as a single JSON line. Supports two modes:

    - **Fixed path** (default): always writes to the given ``log_path``.
    - **Daily rotation**: pass ``log_dir`` + ``rotation='daily'`` instead of
      ``log_path``. A new file is created each calendar day
      (``{prefix}_YYYY-MM-DD.jsonl``). The active path is re-evaluated on
      every write, so a long-running serve process rotates automatically at
      midnight without restarting.

    Args:
        log_path: Fixed path to the ``.jsonl`` file. Mutually exclusive with
            ``log_dir``/``rotation``.
        pipeline_id: Identifier grouping multiple agent runs. Defaults to a
            short random UUID.
        event_callback: Optional callback invoked with each event dict after
            it is written to disk (used by the SSE routing in server.py).
        session_id: Optional browser/client session identifier stamped on
            every event. Useful for grepping a single user's conversation
            from a shared daily log.
        log_dir: Directory for rotating log files. Required when
            ``rotation='daily'``.
        prefix: Filename prefix for rotating files. Defaults to ``"serve"``.
        rotation: ``"daily"`` or ``"none"`` (default ``"none"``).
    """

    def __init__(
        self,
        log_path: str | Path | None = None,
        pipeline_id: str | None = None,
        event_callback: Callable[[dict], None] | None = None,
        session_id: str | None = None,
        # rotation params
        log_dir: str | Path | None = None,
        prefix: str = "serve",
        rotation: str = "none",
    ) -> None:
        if log_path is None and log_dir is None:
            raise ValueError("Either log_path or log_dir must be provided.")
        self._fixed_path: Path | None = Path(log_path) if log_path else None
        self._log_dir: Path | None = Path(log_dir) if log_dir else None
        self._prefix = prefix
        self._rotation = rotation

        if self._fixed_path:
            self._fixed_path.parent.mkdir(parents=True, exist_ok=True)
        if self._log_dir:
            self._log_dir.mkdir(parents=True, exist_ok=True)

        self.pipeline_id: str = pipeline_id or uuid.uuid4().hex[:8]
        self.event_callback = event_callback
        self.session_id: str | None = session_id
        self._lock = threading.Lock()

    @property
    def log_path(self) -> Path:
        """Return the active log file path, rotating daily if configured."""
        if self._log_dir and self._rotation == "daily":
            return self._log_dir / f"{self._prefix}_{_today()}.jsonl"
        assert self._fixed_path is not None
        return self._fixed_path

    # ── Core write ────────────────────────────────────────────────────────────

    def _write(self, event: str, **fields: Any) -> None:
        """Append one JSON event line to the log file. Thread-safe."""
        entry: dict[str, Any] = {
            "ts": _now(),
            "pipeline": self.pipeline_id,
            "event": event,
            **fields,
        }
        if self.session_id:
            entry["session"] = self.session_id
        path = self.log_path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        if self.event_callback:
            self.event_callback(entry)

    # ── Event methods ─────────────────────────────────────────────────────────

    def run_start(self, agent: str, run_id: str, model: str, prompt: str) -> None:
        """Log the start of a single agent run."""
        self._write(
            "run_start",
            agent=agent,
            run_id=run_id,
            model=model,
            prompt=_preview(prompt),
        )

    def llm_call(
        self,
        agent: str,
        run_id: str,
        iteration: int,
        n_messages: int,
        n_tools: int,
    ) -> None:
        """Log an LLM API call being dispatched."""
        self._write(
            "llm_call",
            agent=agent,
            run_id=run_id,
            iteration=iteration,
            n_messages=n_messages,
            n_tools=n_tools,
        )

    def llm_response(
        self,
        agent: str,
        run_id: str,
        iteration: int,
        content: str | None,
        tool_calls: list[dict[str, Any]],
    ) -> None:
        """Log the LLM's response (text + any tool calls requested)."""
        simplified_calls = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "args_preview": _preview(tc.get("function", {}).get("arguments", ""), 300),
            }
            for tc in (tool_calls or [])
        ]
        self._write(
            "llm_response",
            agent=agent,
            run_id=run_id,
            iteration=iteration,
            content_preview=_preview(content),
            tool_calls=simplified_calls,
        )

    def tool_call(
        self,
        agent: str,
        run_id: str,
        tool_name: str,
        tool_type: str,
        args: dict[str, Any],
    ) -> None:
        """Log a tool being invoked (skill or MCP)."""
        # Truncate large argument values (e.g. long prompts) for the log
        args_preview = {
            k: _preview(str(v), 400) if isinstance(v, str) and len(str(v)) > 400 else v
            for k, v in args.items()
        }
        self._write(
            "tool_call",
            agent=agent,
            run_id=run_id,
            tool=tool_name,
            tool_type=tool_type,  # "skill" or "mcp"
            args=args_preview,
        )

    def tool_result(
        self,
        agent: str,
        run_id: str,
        tool_name: str,
        result: str,
        error: bool = False,
    ) -> None:
        """Log a tool result (success or error)."""
        self._write(
            "tool_result",
            agent=agent,
            run_id=run_id,
            tool=tool_name,
            error=error,
            result_preview=_preview(result),
            result_chars=len(result),
        )

    def run_end(
        self,
        agent: str,
        run_id: str,
        response: str,
        iterations: int,
    ) -> None:
        """Log the final response at the end of an agent run."""
        self._write(
            "run_end",
            agent=agent,
            run_id=run_id,
            response_preview=_preview(response),
            iterations=iterations,
        )

    def run_limit(
        self,
        agent: str,
        run_id: str,
        iterations: int,
        max_iterations: int,
        last_preview: str | None = None,
    ) -> None:
        """Log that an agent has hit its iteration limit without finishing."""
        self._write(
            "run_limit",
            agent=agent,
            run_id=run_id,
            iterations=iterations,
            max_iterations=max_iterations,
            last_preview=_preview(last_preview or "", 300),
        )

    def context_write(
        self,
        agent: str,
        key: str,
        size: int,
        offloaded: bool,
        path: str | None = None,
    ) -> None:
        """Log a value being written to the run context."""
        fields: dict[str, Any] = {
            "agent": agent,
            "key": key,
            "size": size,
            "offloaded": offloaded,
        }
        if path:
            fields["path"] = path
        self._write("context_write", **fields)

    def context_read(
        self,
        agent: str,
        key: str,
        offloaded: bool,
        path: str | None = None,
    ) -> None:
        """Log a value being resolved from the run context for a consuming agent."""
        fields: dict[str, Any] = {
            "agent": agent,
            "key": key,
            "offloaded": offloaded,
        }
        if path:
            fields["path"] = path
        self._write("context_read", **fields)
