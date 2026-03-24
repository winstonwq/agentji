"""Skill script executor.

Runs the script associated with a skill tool call via subprocess, passes
parameters as JSON on stdin, and captures stdout as the tool result.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TIMEOUT_SECONDS = 30


class ExecutionError(RuntimeError):
    """Raised when a skill script exits with a non-zero status."""


def execute_skill(
    tool_schema: dict[str, Any],
    arguments: dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Execute the script for a skill tool call and return its stdout.

    The script is invoked with the current Python interpreter. Parameters are
    passed as a JSON string on stdin. Stdout is captured and returned as the
    tool result. Stderr is forwarded to the calling process for visibility.

    Args:
        tool_schema: The translated tool schema dict (from skill_translator).
            Must contain ``_scripts`` and ``_skill_dir`` metadata keys.
        arguments: The arguments dict from the LLM tool call.
        timeout: Maximum seconds to wait for the script to complete.

    Returns:
        The stdout output of the script (stripped of trailing whitespace).

    Raises:
        ExecutionError: If the script exits with a non-zero status or times out.
        FileNotFoundError: If the script file cannot be found.
    """
    scripts: dict[str, str] = tool_schema.get("_scripts", {})
    skill_dir = Path(tool_schema.get("_skill_dir", "."))

    script_rel = scripts.get("execute")
    if not script_rel:
        raise ExecutionError(
            f"Skill '{tool_schema['function']['name']}' has no 'execute' script defined. "
            f"Add a 'scripts.execute' entry to its SKILL.md."
        )

    script_path = skill_dir / script_rel
    if not script_path.exists():
        raise FileNotFoundError(
            f"Script '{script_path}' not found for skill "
            f"'{tool_schema['function']['name']}'. "
            f"Check the 'scripts.execute' path in SKILL.md."
        )

    stdin_payload = json.dumps(arguments, ensure_ascii=False)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise ExecutionError(
            f"Skill '{tool_schema['function']['name']}' timed out after {timeout}s."
        )

    if result.returncode != 0:
        stdout_snippet = result.stdout.strip()[:500] if result.stdout else ""
        stderr_snippet = result.stderr.strip()[:500] if result.stderr else ""
        detail = stdout_snippet or stderr_snippet or "(no output)"
        raise ExecutionError(
            f"Skill '{tool_schema['function']['name']}' exited with code "
            f"{result.returncode}. {detail}"
        )

    return result.stdout.rstrip()
