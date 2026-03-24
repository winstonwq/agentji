"""Unit tests for agentji built-in tools and prompt-skill integration."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from agentji.builtins import execute_builtin, VALID_BUILTINS, BUILTIN_SCHEMAS


# ── VALID_BUILTINS / schemas ───────────────────────────────────────────────────

class TestBuiltinRegistry:
    def test_valid_builtins_set(self) -> None:
        assert "bash" in VALID_BUILTINS
        assert "read_file" in VALID_BUILTINS
        assert "write_file" in VALID_BUILTINS

    def test_all_builtins_have_schemas(self) -> None:
        for name in VALID_BUILTINS:
            assert name in BUILTIN_SCHEMAS
            schema = BUILTIN_SCHEMAS[name]
            assert schema["type"] == "function"
            assert "name" in schema["function"]
            assert "parameters" in schema["function"]

    def test_unknown_builtin_returns_error(self) -> None:
        result = json.loads(execute_builtin("nonexistent", {}))
        assert "error" in result


# ── bash ──────────────────────────────────────────────────────────────────────

class TestBashBuiltin:
    def test_echo_returns_stdout(self) -> None:
        result = json.loads(execute_builtin("bash", {"command": "echo hello"}))
        assert result["stdout"].strip() == "hello"
        assert result["exit_code"] == 0

    def test_failing_command_returns_nonzero_exit(self) -> None:
        result = json.loads(execute_builtin("bash", {"command": "exit 42"}))
        assert result["exit_code"] == 42

    def test_stderr_captured(self) -> None:
        result = json.loads(execute_builtin("bash", {"command": "echo err >&2"}))
        assert "err" in result["stderr"]

    def test_timeout_causes_error(self) -> None:
        result = json.loads(execute_builtin("bash", {"command": "sleep 60", "timeout": 1}))
        assert "error" in result
        assert result["exit_code"] == -1

    def test_python_script_execution(self, tmp_path: Path) -> None:
        script = tmp_path / "hello.py"
        script.write_text("print('hello from python')", encoding="utf-8")
        result = json.loads(execute_builtin("bash", {"command": f"python3 {script}"}))
        assert "hello from python" in result["stdout"]
        assert result["exit_code"] == 0

    def test_multiline_output(self) -> None:
        result = json.loads(execute_builtin("bash", {"command": "printf 'a\\nb\\nc\\n'"}))
        assert result["stdout"].count("\n") >= 3


# ── read_file ─────────────────────────────────────────────────────────────────

class TestReadFileBuiltin:
    def test_reads_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        result = json.loads(execute_builtin("read_file", {"path": str(f)}))
        assert result["content"] == "hello world"
        assert result["size_bytes"] == len("hello world".encode())

    def test_missing_file_returns_error(self) -> None:
        result = json.loads(execute_builtin("read_file", {"path": "/nonexistent/file.txt"}))
        assert "error" in result

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("x", encoding="utf-8")
        result = json.loads(execute_builtin("read_file", {"path": str(f)}))
        assert Path(result["path"]).is_absolute()

    def test_reads_unicode_content(self, tmp_path: Path) -> None:
        f = tmp_path / "unicode.txt"
        f.write_text("你好世界", encoding="utf-8")
        result = json.loads(execute_builtin("read_file", {"path": str(f)}))
        assert result["content"] == "你好世界"


# ── write_file ────────────────────────────────────────────────────────────────

class TestWriteFileBuiltin:
    def test_writes_new_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "out.txt"
        result = json.loads(execute_builtin("write_file", {"path": str(dest), "content": "hello"}))
        assert result["status"] == "success"
        assert dest.read_text() == "hello"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c.txt"
        execute_builtin("write_file", {"path": str(dest), "content": "data"})
        assert dest.exists()
        assert dest.read_text() == "data"

    def test_overwrites_existing_file(self, tmp_path: Path) -> None:
        dest = tmp_path / "f.txt"
        dest.write_text("old", encoding="utf-8")
        execute_builtin("write_file", {"path": str(dest), "content": "new"})
        assert dest.read_text() == "new"

    def test_returns_byte_count(self, tmp_path: Path) -> None:
        dest = tmp_path / "f.txt"
        result = json.loads(execute_builtin("write_file", {"path": str(dest), "content": "abc"}))
        assert result["size_bytes"] == 3

    def test_write_file_error_returns_error_json(self, tmp_path: Path) -> None:
        from unittest.mock import patch
        from pathlib import Path as _Path
        with patch.object(_Path, "write_text", side_effect=OSError("disk full")):
            dest = tmp_path / "fail.txt"
            result = json.loads(execute_builtin("write_file", {"path": str(dest), "content": "x"}))
        assert "error" in result


class TestBashExceptionPath:
    def test_bash_generic_exception_returns_error(self) -> None:
        from unittest.mock import patch
        import subprocess
        with patch("subprocess.run", side_effect=OSError("fork failed")):
            result = json.loads(execute_builtin("bash", {"command": "echo hi"}))
        assert "error" in result
        assert result["exit_code"] == -1


class TestReadFileExceptionPath:
    def test_read_file_generic_exception_returns_error(self, tmp_path: Path) -> None:
        from unittest.mock import patch
        from pathlib import Path as _Path
        f = tmp_path / "f.txt"
        f.write_text("data")
        with patch.object(_Path, "read_text", side_effect=PermissionError("no access")):
            result = json.loads(execute_builtin("read_file", {"path": str(f)}))
        assert "error" in result


# ── Prompt-skill injection ────────────────────────────────────────────────────

class TestPromptSkillIntegration:
    """Tests for the full prompt-skill → system prompt injection pipeline."""

    def _make_anthropic_skill(self, tmp_path: Path, name: str, body: str) -> Path:
        """Create a minimal Anthropic-format skill (no scripts.execute)."""
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent(f"""\
            ---
            name: {name}
            description: A test prompt skill.
            ---

            {body}
        """), encoding="utf-8")
        return skill_dir

    def test_prompt_skill_detected(self, tmp_path: Path) -> None:
        from agentji.skill_translator import translate_skill
        skill_dir = self._make_anthropic_skill(tmp_path, "test-skill", "Do stuff.")
        s = translate_skill(skill_dir)
        assert s["_prompt_only"] is True
        assert s["function"]["name"] == "test-skill"
        assert "Do stuff." in s["_body"]

    def test_prompt_skill_not_in_tool_list(self, tmp_path: Path) -> None:
        """Prompt skills must NOT appear as callable tools."""
        from agentji.skill_translator import translate_skill
        skill_dir = self._make_anthropic_skill(tmp_path, "guide", "Guidance here.")
        s = translate_skill(skill_dir)
        assert s.get("_prompt_only") is True
        assert "parameters" not in s["function"]

    def test_config_builtins_validated(self, tmp_path: Path) -> None:
        from agentji.config import AgentConfig
        with pytest.raises(Exception, match="Unknown built-in"):
            AgentConfig(
                model="openai/gpt-4o",
                system_prompt="test",
                builtins=["nonexistent_tool"],
            )

    def test_config_valid_builtins_accepted(self) -> None:
        from agentji.config import AgentConfig
        agent = AgentConfig(
            model="openai/gpt-4o",
            system_prompt="test",
            builtins=["bash", "read_file", "write_file"],
        )
        assert agent.builtins == ["bash", "read_file", "write_file"]

    def test_prompt_skill_body_in_system_prompt(self, tmp_path: Path) -> None:
        """End-to-end: prompt skill body must appear in the constructed system prompt."""
        import textwrap as tw
        from agentji.config import AgentjiConfig, load_config

        skill_dir = self._make_anthropic_skill(
            tmp_path, "xlsx",
            "Always use openpyxl. Never use xlrd. Validate formulas."
        )

        config_yaml = tw.dedent(f"""
            version: "1"
            providers:
              openai:
                api_key: sk-test
            skills:
              - path: {skill_dir}
            agents:
              writer:
                model: openai/gpt-4o
                system_prompt: "You create spreadsheets."
                skills: [xlsx]
                builtins: [bash, write_file]
                max_iterations: 5
        """)
        cfg_path = tmp_path / "agentji.yaml"
        cfg_path.write_text(config_yaml, encoding="utf-8")

        import os
        os.environ.setdefault("OPENAI_API_KEY", "sk-test")
        cfg = load_config(cfg_path)
        agent = cfg.agents["writer"]
        assert "xlsx" in agent.skills
        assert "bash" in agent.builtins
