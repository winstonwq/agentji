"""Cross-run memory backend stub for agentji.

Stubbed until mem0 integration is implemented. Both methods are safe no-ops
when memory config is absent. The loop calls them unconditionally so that
adding a real backend later requires no loop changes.

To implement:
    pip install mem0ai
    See: https://docs.mem0.ai
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agentji.config import MemoryConfig


class MemoryBackend:
    """Cross-run memory backend. Stubbed until mem0 integration is implemented.

    Contract:
    - inject() is called before the orchestrator's first LLM call.
      It receives the assembled system prompt and the user's message,
      and returns a modified system prompt with relevant memories prepended.
    - remember() is called on run_end.
      It receives the run_id and a plain-text summary of the run
      (the orchestrator's final response) and stores it for future retrieval.

    Both methods remain callable when memory: is absent from config.
    The stub makes them safe no-ops.
    """

    def __init__(self, config: "MemoryConfig | None" = None) -> None:
        self.config = config
        self.enabled = config is not None

    def inject(self, system_prompt: str, user_message: str) -> str:
        """Return system_prompt unchanged. Replace with mem0 retrieval when implementing."""
        return system_prompt

    def remember(self, run_id: str, summary: str) -> None:
        """No-op. Replace with mem0 storage when implementing."""
        pass
