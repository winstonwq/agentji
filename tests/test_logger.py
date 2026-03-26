"""Unit tests for ConversationLogger and the log-summary skill."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agentji.logger import ConversationLogger


# ── ConversationLogger ────────────────────────────────────────────────────────

class TestConversationLogger:
    def test_creates_log_file(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "test.jsonl")
        log.run_start("agent1", "run1", "qwen-plus", "hello")
        assert (tmp_path / "test.jsonl").exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        log_path = tmp_path / "deep" / "nested" / "run.jsonl"
        ConversationLogger(log_path)
        assert log_path.parent.exists()

    def test_each_event_is_valid_json_line(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "events.jsonl")
        log.run_start("a", "r1", "model", "prompt")
        log.llm_call("a", "r1", 1, 2, 3)
        log.llm_response("a", "r1", 1, "content", [])
        log.tool_call("a", "r1", "my_tool", "skill", {"x": 1})
        log.tool_result("a", "r1", "my_tool", "result text")
        log.run_end("a", "r1", "final answer", 2)

        lines = (tmp_path / "events.jsonl").read_text().splitlines()
        assert len(lines) == 6
        for line in lines:
            data = json.loads(line)
            assert "ts" in data
            assert "pipeline" in data
            assert "event" in data

    def test_event_types_are_correct(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "e.jsonl")
        log.run_start("a", "r1", "m", "p")
        log.llm_call("a", "r1", 1, 1, 0)
        log.llm_response("a", "r1", 1, None, [])
        log.tool_call("a", "r1", "t", "mcp", {})
        log.tool_result("a", "r1", "t", "ok")
        log.run_end("a", "r1", "done", 1)

        events = [json.loads(l)["event"] for l in (tmp_path / "e.jsonl").read_text().splitlines()]
        assert events == ["run_start", "llm_call", "llm_response", "tool_call", "tool_result", "run_end"]

    def test_pipeline_id_defaults_to_short_uuid(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "p.jsonl")
        log.run_start("a", "r", "m", "p")
        data = json.loads((tmp_path / "p.jsonl").read_text())
        assert len(data["pipeline"]) == 8

    def test_pipeline_id_can_be_set(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "p.jsonl", pipeline_id="my-pipeline")
        log.run_start("a", "r", "m", "p")
        data = json.loads((tmp_path / "p.jsonl").read_text())
        assert data["pipeline"] == "my-pipeline"

    def test_appends_across_multiple_writes(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "append.jsonl")
        log.run_start("a", "r1", "m", "p1")
        log.run_start("a", "r2", "m", "p2")
        lines = (tmp_path / "append.jsonl").read_text().splitlines()
        assert len(lines) == 2

    def test_long_content_is_truncated_in_run_start(self, tmp_path: Path) -> None:
        long_prompt = "x" * 3000
        log = ConversationLogger(tmp_path / "t.jsonl")
        log.run_start("a", "r", "m", long_prompt)
        data = json.loads((tmp_path / "t.jsonl").read_text())
        assert len(data["prompt"]) < 3000
        assert "truncated" in data["prompt"]

    def test_tool_result_error_flag(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "err.jsonl")
        log.tool_result("a", "r", "bad_tool", "Error: oops", error=True)
        data = json.loads((tmp_path / "err.jsonl").read_text())
        assert data["error"] is True
        assert data["tool"] == "bad_tool"

    def test_tool_call_with_long_string_args_truncated(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "args.jsonl")
        args = {"prompt": "A" * 500, "size": "1280*720"}
        log.tool_call("a", "r", "wan-image", "skill", args)
        data = json.loads((tmp_path / "args.jsonl").read_text())
        # The long "prompt" arg should be truncated
        stored_prompt = data["args"]["prompt"]
        assert len(stored_prompt) < 500

    def test_llm_response_with_tool_calls(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "tc.jsonl")
        tool_calls = [
            {"id": "tc1", "function": {"name": "my_tool", "arguments": '{"x": 1}'}},
        ]
        log.llm_response("a", "r", 1, "thinking...", tool_calls)
        data = json.loads((tmp_path / "tc.jsonl").read_text())
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "my_tool"

    def test_result_chars_recorded(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "chars.jsonl")
        log.tool_result("a", "r", "tool", "hello world")
        data = json.loads((tmp_path / "chars.jsonl").read_text())
        assert data["result_chars"] == len("hello world")

    def test_multiple_agents_same_log(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "multi.jsonl", pipeline_id="pipe1")
        log.run_start("agent-a", "r1", "model-a", "prompt a")
        log.run_end("agent-a", "r1", "response a", 1)
        log.run_start("agent-b", "r2", "model-b", "prompt b")
        log.run_end("agent-b", "r2", "response b", 2)

        lines = (tmp_path / "multi.jsonl").read_text().splitlines()
        events = [json.loads(l) for l in lines]
        agents = [e.get("agent") for e in events]
        assert agents == ["agent-a", "agent-a", "agent-b", "agent-b"]
        assert all(e["pipeline"] == "pipe1" for e in events)


# ── agentji logs CLI command ──────────────────────────────────────────────────

from agentji.cli import _summarize_log


def _make_log(tmp_path: Path) -> Path:
    """Create a realistic test log with two agent runs."""
    log = ConversationLogger(tmp_path / "test_pipeline.jsonl", pipeline_id="testpipe")
    # Run 1: data-fetcher with a tool call
    log.run_start("data-fetcher", "run1", "openai/qwen-plus", "Get metrics for AZN.L")
    log.llm_call("data-fetcher", "run1", 1, 2, 1)
    log.llm_response("data-fetcher", "run1", 1, None, [
        {"id": "tc1", "function": {"name": "get_key_metrics", "arguments": '{"ticker": "AZN.L"}'}}
    ])
    log.tool_call("data-fetcher", "run1", "get_key_metrics", "mcp", {"ticker": "AZN.L"})
    log.tool_result("data-fetcher", "run1", "get_key_metrics", '{"pe": 22, "revenue": "40B"}')
    log.llm_call("data-fetcher", "run1", 2, 3, 1)
    log.llm_response("data-fetcher", "run1", 2, "Here are the metrics: PE=22, Revenue=40B", [])
    log.run_end("data-fetcher", "run1", "Here are the metrics: PE=22, Revenue=40B", 2)

    # Run 2: analyst, no tool calls
    log.run_start("analyst", "run2", "openai/qwen-max", "Here are the metrics: PE=22, Revenue=40B")
    log.llm_call("analyst", "run2", 1, 2, 0)
    log.llm_response("analyst", "run2", 1, "**BUY** — Strong fundamentals.", [])
    log.run_end("analyst", "run2", "**BUY** — Strong fundamentals.", 1)

    return tmp_path / "test_pipeline.jsonl"


class TestLogSummaryCli:
    def test_summary_contains_pipeline_id(self, tmp_path: Path) -> None:
        summary = _summarize_log(_make_log(tmp_path), 200)
        assert "testpipe" in summary

    def test_summary_contains_agent_names(self, tmp_path: Path) -> None:
        summary = _summarize_log(_make_log(tmp_path), 200)
        assert "data-fetcher" in summary
        assert "analyst" in summary

    def test_summary_contains_tool_name(self, tmp_path: Path) -> None:
        summary = _summarize_log(_make_log(tmp_path), 200)
        assert "get_key_metrics" in summary

    def test_summary_contains_final_response(self, tmp_path: Path) -> None:
        summary = _summarize_log(_make_log(tmp_path), 200)
        assert "BUY" in summary

    def test_summary_shows_run_count(self, tmp_path: Path) -> None:
        summary = _summarize_log(_make_log(tmp_path), 200)
        assert "Agent runs**: 2" in summary

    def test_error_tool_result_highlighted(self, tmp_path: Path) -> None:
        log = ConversationLogger(tmp_path / "err.jsonl", pipeline_id="errpipe")
        log.run_start("bot", "r1", "model", "do stuff")
        log.tool_call("bot", "r1", "bad_tool", "skill", {})
        log.tool_result("bot", "r1", "bad_tool", "Error: failed", error=True)
        log.run_end("bot", "r1", "(no response)", 1)

        summary = _summarize_log(tmp_path / "err.jsonl", 200)
        assert "❌" in summary
        assert "Errors" in summary

    def test_missing_log_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            _summarize_log(tmp_path / "nonexistent.jsonl", 200)


# ── Thread safety ─────────────────────────────────────────────────────────────

def test_concurrent_writes_are_thread_safe(tmp_path: Path) -> None:
    """Multiple threads writing to the same logger must not corrupt the JSONL."""
    log = ConversationLogger(tmp_path / "concurrent.jsonl")
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            log.run_start(agent=f"agent-{i}", run_id=f"r{i}", model="openai/gpt-4o", prompt="hi")
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread errors: {errors}"
    lines = (tmp_path / "concurrent.jsonl").read_text().splitlines()
    assert len(lines) == 30
    # Every line must be valid JSON
    for line in lines:
        entry = json.loads(line)
        assert entry["event"] == "run_start"
