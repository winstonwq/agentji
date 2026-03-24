"""Per-run scratch context for agentji pipelines.

Provides a shared key-value store for a single pipeline run. Values that
exceed the size threshold are automatically offloaded to the scratch directory
on disk and replaced with a file path reference. The consuming agent receives
the path and reads the full content via the read_file built-in.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agentji.logger import ConversationLogger


class RunContext:
    """Shared key-value store for a single pipeline run.

    Values above size_threshold are automatically offloaded to the scratch
    directory and replaced with a file path reference. The consuming agent
    receives the path and reads the file via read_file.

    Args:
        run_id: Pipeline identifier — used as the scratch directory name.
        scratch_dir: Directory for offloaded files. Created on instantiation.
        size_threshold: Character count above which values are written to disk.
        logger: Optional ConversationLogger for context_write events.
    """

    def __init__(
        self,
        run_id: str,
        scratch_dir: Path,
        size_threshold: int = 8000,
        logger: "ConversationLogger | None" = None,
    ) -> None:
        self.run_id = run_id
        self.scratch_dir = scratch_dir
        self.size_threshold = size_threshold
        self._logger = logger
        self._store: dict[str, dict[str, Any]] = {}
        scratch_dir.mkdir(parents=True, exist_ok=True)

    def set(self, key: str, value: str, agent: str) -> str:
        """Store a value, offloading to disk if it exceeds the size threshold.

        Args:
            key: Logical key for this output (e.g. "market_findings").
            value: The string content to store.
            agent: Name of the agent producing this output.

        Returns:
            The value itself if stored in memory, or the file path string if
            offloaded to disk.
        """
        size = len(value)
        offloaded = size > self.size_threshold

        if offloaded:
            path = self.scratch_dir / f"{key}.md"
            path.write_text(value, encoding="utf-8")
            stored = str(path)
        else:
            stored = value

        self._store[key] = {
            "agent": agent,
            "value": stored,
            "offloaded": offloaded,
            "size": size,
        }

        if self._logger:
            self._logger.context_write(
                agent=agent,
                key=key,
                size=size,
                offloaded=offloaded,
                path=stored if offloaded else None,
            )

        return stored

    def get(self, key: str) -> str | None:
        """Return the stored value or path string for a key.

        For offloaded keys this returns the file path, not the file contents —
        reading the file is the agent's responsibility via read_file.

        Returns:
            The stored string (content or path), or None if key is absent.
        """
        entry = self._store.get(key)
        if entry is None:
            return None
        return entry["value"]

    def summary(self) -> dict[str, dict[str, Any]]:
        """Return metadata for all stored keys.

        Returns:
            Mapping of key → {"agent": str, "size": int, "offloaded": bool}.
        """
        return {
            key: {
                "agent": entry["agent"],
                "size": entry["size"],
                "offloaded": entry["offloaded"],
            }
            for key, entry in self._store.items()
        }
