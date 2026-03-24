"""Built-in tools available to agentji agents.

Built-ins are primitive capabilities that don't require an external script or
MCP server — they're implemented directly in the runtime. They unlock
compatibility with Anthropic-format skills (SKILL.md body-only, no scripts.execute)
which rely on the agent having bash and file I/O access, the same way Claude Code
provides these natively.

Enable per-agent in agentji.yaml:

    agents:
      spreadsheet-agent:
        builtins: [bash, read_file, write_file]

Security: bash executes arbitrary shell commands. Only enable it for agents
where you trust the LLM's judgment and the execution environment is appropriate.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

# ── Runtime environment ───────────────────────────────────────────────────────

# Expose the active Python interpreter so bash tools can use it reliably.
# On many Linux systems only python3 is in PATH; AGENTJI_PYTHON is always set.
_PYTHON_EXE = sys.executable
_VENV_BIN = str(Path(_PYTHON_EXE).parent)


def _subprocess_env() -> dict[str, str]:
    """Build an environment for bash subprocesses.

    Prepends the venv's bin directory to PATH so 'python' and 'pip' resolve
    to the agentji virtual environment.  Also exports AGENTJI_PYTHON with the
    absolute path to the Python interpreter.
    """
    env = os.environ.copy()
    existing_path = env.get("PATH", "")
    env["PATH"] = _VENV_BIN + ":" + existing_path
    env["AGENTJI_PYTHON"] = _PYTHON_EXE
    return env


# ── Tool schemas (OpenAI format) ──────────────────────────────────────────────

BUILTIN_SCHEMAS: dict[str, dict[str, Any]] = {
    "bash": {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command and return its stdout, stderr, and exit code. "
                "Use this to run scripts, install packages, or perform system operations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds to wait (default: 30).",
                    },
                },
                "required": ["command"],
            },
        },
    },
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from disk and return its text content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": (
                "Write text content to a file on disk. "
                "Creates parent directories if they do not exist."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to write.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Text content to write to the file.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
}

VALID_BUILTINS: frozenset[str] = frozenset(BUILTIN_SCHEMAS)


# ── Executors ─────────────────────────────────────────────────────────────────

def execute_builtin(name: str, args: dict[str, Any], default_timeout: int = 60) -> str:
    """Dispatch a built-in tool call and return a JSON string result."""
    if name == "bash":
        # LLM-supplied timeout wins; fall back to the agent's configured default
        return _bash(args["command"], int(args.get("timeout", default_timeout)))
    if name == "read_file":
        return _read_file(args["path"])
    if name == "write_file":
        return _write_file(args["path"], args["content"])
    return json.dumps({"error": f"Unknown built-in tool: '{name}'"})


def _bash(command: str, timeout: int = 30) -> str:
    try:
        proc = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_subprocess_env(),
        )
        return json.dumps({
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "exit_code": proc.returncode,
        }, ensure_ascii=False)
    except subprocess.TimeoutExpired:
        return json.dumps({
            "error": f"Command timed out after {timeout}s",
            "exit_code": -1,
        })
    except Exception as exc:
        return json.dumps({"error": str(exc), "exit_code": -1})


def _read_file(path: str) -> str:
    try:
        content = Path(path).read_text(encoding="utf-8")
        return json.dumps({
            "content": content,
            "path": str(Path(path).resolve()),
            "size_bytes": len(content.encode()),
        }, ensure_ascii=False)
    except FileNotFoundError:
        return json.dumps({"error": f"File not found: {path}"})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


def _write_file(path: str, content: str) -> str:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({
            "status": "success",
            "path": str(p.resolve()),
            "size_bytes": len(content.encode()),
        })
    except Exception as exc:
        return json.dumps({"error": str(exc)})
