"""Unit tests for the agentji CLI commands and _summarize_log helper."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from agentji.cli import app, _summarize_log
from agentji.logger import ConversationLogger


runner = CliRunner()


# ── _summarize_log edge cases ─────────────────────────────────────────────────

def _make_log(tmp_path: Path) -> Path:
    log = ConversationLogger(tmp_path / "run.jsonl", pipeline_id="p1")
    log.run_start("agent", "r1", "openai/gpt-4o", "hello")
    log.llm_call("agent", "r1", 1, 2, 0)
    log.llm_response("agent", "r1", 1, "Hi there!", [])
    log.run_end("agent", "r1", "Hi there!", 1)
    return tmp_path / "run.jsonl"


class TestSummarizeLogEdges:
    def test_empty_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        with pytest.raises(ValueError, match="empty"):
            _summarize_log(path, 200)

    def test_invalid_json_line_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text('{"ts": "x", "event": "run_start"}\nnot json\n')
        with pytest.raises(ValueError, match="Invalid JSON"):
            _summarize_log(path, 200)

    def test_missing_file_raises_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _summarize_log(tmp_path / "nonexistent.jsonl", 200)

    def test_preview_truncation(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "t.jsonl", pipeline_id="px")
        log.run_start("agent", "r1", "model", "short prompt")
        log.tool_call("agent", "r1", "my_tool", "skill", {"key": "A" * 500})
        log.tool_result("agent", "r1", "my_tool", "result")
        log.run_end("agent", "r1", "ok", 1)
        summary = _summarize_log(tmp_path / "t.jsonl", 10)
        assert "…[+" in summary  # truncation marker present

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "blanks.jsonl"
        path.write_text('\n{"ts":"x","pipeline":"p","event":"run_start","agent":"a","run_id":"r","model":"m","prompt":"q"}\n\n')
        summary = _summarize_log(path, 200)
        assert "p" in summary

    def test_error_tool_shows_error_count(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "e.jsonl", pipeline_id="ep")
        log.run_start("bot", "r1", "model", "task")
        log.tool_call("bot", "r1", "bad", "skill", {})
        log.tool_result("bot", "r1", "bad", "Error: boom", error=True)
        log.run_end("bot", "r1", "done", 1)
        summary = _summarize_log(tmp_path / "e.jsonl", 200)
        assert "Errors" in summary
        assert "❌" in summary


# ── `agentji logs` command ────────────────────────────────────────────────────

class TestLogsCommand:
    def test_prints_summary_for_valid_log(self, tmp_path: Path) -> None:
        log_path = _make_log(tmp_path)
        result = runner.invoke(app, ["logs", str(log_path)])
        assert result.exit_code == 0
        assert "p1" in result.output

    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["logs", str(tmp_path / "nope.jsonl")])
        assert result.exit_code != 0

    def test_invalid_log_exits_nonzero(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.jsonl"
        bad.write_text("not json\n")
        result = runner.invoke(app, ["logs", str(bad)])
        assert result.exit_code != 0

    def test_max_preview_option_accepted(self, tmp_path: Path) -> None:
        log_path = _make_log(tmp_path)
        result = runner.invoke(app, ["logs", str(log_path), "--max-preview", "50"])
        assert result.exit_code == 0


# ── `agentji init` command ────────────────────────────────────────────────────

class TestInitCommand:
    def test_creates_agentji_yaml(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert (tmp_path / "agentji.yaml").exists()

    def test_creates_env_file(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        assert (tmp_path / ".env").exists()

    def test_copies_agentji_skill(self, tmp_path: Path) -> None:
        runner.invoke(app, ["init", str(tmp_path)])
        assert (tmp_path / "skills" / "agentji").is_dir()
        assert (tmp_path / "skills" / "agentji" / "SKILL.md").exists()

    def test_skips_existing_agentji_yaml(self, tmp_path: Path) -> None:
        existing = tmp_path / "agentji.yaml"
        existing.write_text("# my config", encoding="utf-8")
        runner.invoke(app, ["init", str(tmp_path)])
        assert existing.read_text() == "# my config"

    def test_default_directory_is_cwd(self, tmp_path: Path) -> None:
        import os
        orig = os.getcwd()
        try:
            os.chdir(tmp_path)
            result = runner.invoke(app, ["init"])
            assert result.exit_code == 0
            assert (tmp_path / "agentji.yaml").exists()
        finally:
            os.chdir(orig)

    def test_skips_existing_agentji_skill(self, tmp_path: Path) -> None:
        # First init — copies the skill
        runner.invoke(app, ["init", str(tmp_path)])
        # Second init — skill dir already exists, should skip without error
        result = runner.invoke(app, ["init", str(tmp_path)])
        assert result.exit_code == 0
        assert "already exists" in result.output


# ── `agentji run` command ─────────────────────────────────────────────────────

class TestRunCommand:
    def _make_config(self, tmp_path: Path) -> Path:
        config_path = tmp_path / "agentji.yaml"
        config_path.write_text(textwrap.dedent("""\
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
                max_iterations: 3
        """), encoding="utf-8")
        return config_path

    def test_runs_agent_and_prints_response(self, tmp_path: Path) -> None:
        config_path = self._make_config(tmp_path)
        with patch("agentji.loop.litellm.completion") as mock_llm:
            msg = mock_llm.return_value.choices[0].message
            msg.content = "Test response"
            msg.tool_calls = None
            msg.model_dump.return_value = {"role": "assistant", "content": "Test response"}
            result = runner.invoke(app, [
                "run", "--config", str(config_path),
                "--agent", "qa", "--prompt", "hello"
            ])
        assert result.exit_code == 0
        assert "Test response" in result.output

    def test_unknown_agent_exits_nonzero(self, tmp_path: Path) -> None:
        config_path = self._make_config(tmp_path)
        result = runner.invoke(app, [
            "run", "--config", str(config_path),
            "--agent", "nonexistent", "--prompt", "hi"
        ])
        assert result.exit_code != 0

    def test_missing_config_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, [
            "run", "--config", str(tmp_path / "nope.yaml"),
            "--agent", "qa", "--prompt", "hi"
        ])
        assert result.exit_code != 0

    def test_agent_runtime_error_exits_nonzero(self, tmp_path: Path) -> None:
        config_path = self._make_config(tmp_path)
        with patch("agentji.loop.litellm.completion", side_effect=RuntimeError("boom")):
            result = runner.invoke(app, [
                "run", "--config", str(config_path),
                "--agent", "qa", "--prompt", "hi"
            ])
        assert result.exit_code != 0

    def test_log_dir_creates_jsonl_file(self, tmp_path: Path) -> None:
        config_path = self._make_config(tmp_path)
        log_dir = tmp_path / "logs"
        with patch("agentji.loop.litellm.completion") as mock_llm:
            msg = mock_llm.return_value.choices[0].message
            msg.content = "logged"
            msg.tool_calls = None
            msg.model_dump.return_value = {"role": "assistant", "content": "logged"}
            runner.invoke(app, [
                "run", "--config", str(config_path),
                "--agent", "qa", "--prompt", "log me",
                "--log-dir", str(log_dir),
            ])
        jsonl_files = list(log_dir.glob("*.jsonl"))
        assert len(jsonl_files) == 1
        events = [json.loads(l) for l in jsonl_files[0].read_text().splitlines()]
        assert any(e["event"] == "run_start" for e in events)
