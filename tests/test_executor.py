"""Unit tests for agentji.executor."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentji.executor import ExecutionError, execute_skill


def make_tool_schema(script_path: Path, name: str = "test-skill") -> dict:
    """Build a minimal tool schema pointing at the given script."""
    return {
        "type": "function",
        "function": {"name": name, "description": "test", "parameters": {"type": "object", "properties": {}}},
        "_scripts": {"execute": script_path.name},
        "_skill_dir": str(script_path.parent),
    }


class TestExecuteSkill:
    def test_echo_skill(self, tmp_path: Path) -> None:
        script = tmp_path / "echo.py"
        script.write_text(
            "import json, sys\n"
            "params = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'greeting': f'Hello, {params[\"name\"]}!'}))\n"
        )
        tool_schema = make_tool_schema(script, name="echo")
        result = execute_skill(tool_schema, {"name": "Winston"})
        assert "Winston" in result

    def test_unicode_input(self, tmp_path: Path) -> None:
        script = tmp_path / "echo_unicode.py"
        script.write_text(
            "import json, sys\n"
            "params = json.loads(sys.stdin.read())\n"
            "print(json.dumps({'value': params['text']}, ensure_ascii=False))\n"
        )
        tool_schema = make_tool_schema(script, name="echo-unicode")
        result = execute_skill(tool_schema, {"text": "世界"})
        assert "世界" in result

    def test_script_exit_nonzero_raises(self, tmp_path: Path) -> None:
        script = tmp_path / "fail.py"
        script.write_text("import sys\nprint('err', file=sys.stderr)\nsys.exit(1)\n")
        tool_schema = make_tool_schema(script)
        with pytest.raises(ExecutionError, match="exit"):
            execute_skill(tool_schema, {})

    def test_missing_script_raises(self, tmp_path: Path) -> None:
        tool_schema = {
            "type": "function",
            "function": {"name": "test", "description": "", "parameters": {}},
            "_scripts": {"execute": "scripts/nonexistent.py"},
            "_skill_dir": str(tmp_path),
        }
        with pytest.raises(FileNotFoundError, match="not found"):
            execute_skill(tool_schema, {})

    def test_no_execute_script_raises(self, tmp_path: Path) -> None:
        tool_schema = {
            "type": "function",
            "function": {"name": "test", "description": "", "parameters": {}},
            "_scripts": {},
            "_skill_dir": str(tmp_path),
        }
        with pytest.raises(ExecutionError, match="execute"):
            execute_skill(tool_schema, {})

    def test_timeout_raises(self, tmp_path: Path) -> None:
        script = tmp_path / "slow.py"
        script.write_text("import time\ntime.sleep(10)\n")
        tool_schema = make_tool_schema(script)
        with pytest.raises(ExecutionError, match="timed out"):
            execute_skill(tool_schema, {}, timeout=1)
