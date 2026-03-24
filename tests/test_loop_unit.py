"""Unit tests for agentji loop internals.

Tests for the internal graph nodes (_should_continue, _finalize, _execute_tools)
and run_agent paths that aren't covered by test_call_agent.py:
  - max_iterations reached
  - no assistant response (all-tool messages)
  - unknown tool → error injected
  - exception during skill execution → error injected
  - prompt-skill body injected into system_prompt
  - logger wired through run_agent
  - builtin executed through the full loop
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from agentji.loop import (
    _should_continue,
    _finalize,
    _execute_tools,
    run_agent,
    AgentState,
    END,
)
from agentji.config import load_config
from agentji.logger import ConversationLogger


# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> AgentState:
    """Minimal valid AgentState for unit tests."""
    state: AgentState = {
        "messages": [],
        "tools": [],
        "tool_map": {},
        "mcp_map": {},
        "builtin_set": [],
        "litellm_kwargs": {"model": "openai/gpt-4o", "api_key": "sk-test"},
        "max_iterations": 5,
        "iteration": 0,
        "final_response": "",
        "_agent_name": "test-agent",
        "_run_id": "run-test",
        "_logger": None,
        "_cfg": None,
        "_allowed_agents": [],
    }
    state.update(overrides)
    return state


def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "agentji.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


def _fake_done_response(content: str = "Done."):
    """Return a litellm-style response with no tool calls."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": content}
    resp = MagicMock()
    resp.choices = [MagicMock(message=msg)]
    return resp


# ── _should_continue ──────────────────────────────────────────────────────────

class TestShouldContinue:
    def test_no_tool_calls_stops(self) -> None:
        state = _base_state(messages=[{"role": "assistant", "content": "hi"}], iteration=1)
        assert _should_continue(state) == END

    def test_tool_calls_present_continues(self) -> None:
        state = _base_state(
            messages=[{"role": "assistant", "tool_calls": [{"id": "x"}]}],
            iteration=1,
        )
        assert _should_continue(state) == "tools"

    def test_max_iterations_stops(self) -> None:
        state = _base_state(
            messages=[{"role": "assistant", "tool_calls": [{"id": "x"}]}],
            iteration=5,
            max_iterations=5,
        )
        assert _should_continue(state) == END

    def test_exactly_at_limit_stops(self) -> None:
        state = _base_state(
            messages=[{"role": "assistant", "tool_calls": [{"id": "x"}]}],
            iteration=3,
            max_iterations=3,
        )
        assert _should_continue(state) == END

    def test_below_limit_with_tools_continues(self) -> None:
        state = _base_state(
            messages=[{"role": "assistant", "tool_calls": [{"id": "x"}]}],
            iteration=2,
            max_iterations=5,
        )
        assert _should_continue(state) == "tools"


# ── _finalize ─────────────────────────────────────────────────────────────────

class TestFinalize:
    def test_extracts_last_assistant_content(self) -> None:
        state = _base_state(messages=[
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ])
        result = _finalize(state)
        assert result["final_response"] == "world"

    def test_returns_no_response_when_no_assistant_message(self) -> None:
        state = _base_state(messages=[
            {"role": "user", "content": "hello"},
            {"role": "tool", "content": "tool result"},
        ])
        result = _finalize(state)
        assert result["final_response"] == "(no response)"

    def test_returns_no_response_when_assistant_has_no_content(self) -> None:
        state = _base_state(messages=[
            {"role": "assistant", "content": None},
        ])
        result = _finalize(state)
        assert result["final_response"] == "(no response)"

    def test_picks_last_assistant_message(self) -> None:
        state = _base_state(messages=[
            {"role": "assistant", "content": "first"},
            {"role": "tool", "content": "result"},
            {"role": "assistant", "content": "final"},
        ])
        result = _finalize(state)
        assert result["final_response"] == "final"

    def test_logger_run_end_called_on_success(self, tmp_path: Path) -> None:
        logger = ConversationLogger(tmp_path / "test.jsonl")
        state = _base_state(
            messages=[{"role": "assistant", "content": "done"}],
            _logger=logger,
            iteration=2,
        )
        _finalize(state)
        events = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()]
        assert events[-1]["event"] == "run_end"
        assert events[-1]["iterations"] == 2

    def test_logger_run_end_called_on_no_response(self, tmp_path: Path) -> None:
        logger = ConversationLogger(tmp_path / "test.jsonl")
        state = _base_state(
            messages=[{"role": "user", "content": "hi"}],
            _logger=logger,
        )
        _finalize(state)
        events = [json.loads(l) for l in (tmp_path / "test.jsonl").read_text().splitlines()]
        run_end = next(e for e in events if e["event"] == "run_end")
        assert "(no response)" in run_end["response_preview"]


# ── _execute_tools ─────────────────────────────────────────────────────────────

class TestExecuteTools:
    def _state_with_tool_call(self, name: str, args: dict, **overrides) -> AgentState:
        tool_call_dict = {
            "id": "tc_001",
            "function": {
                "name": name,
                "arguments": json.dumps(args),
            },
        }
        return _base_state(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "tool_calls": [tool_call_dict]},
            ],
            **overrides,
        )

    def test_unknown_tool_injects_error_message(self) -> None:
        state = self._state_with_tool_call("nonexistent_tool", {"x": 1})
        result_state = _execute_tools(state)
        tool_msg = result_state["messages"][-1]
        assert tool_msg["role"] == "tool"
        assert "unknown tool" in tool_msg["content"].lower()

    def test_exception_in_skill_execution_injects_error(self, tmp_path: Path) -> None:
        from agentji.executor import ExecutionError

        # Create a tool schema pointing at a script that always fails
        script = tmp_path / "fail.py"
        script.write_text("import sys\nprint('err', file=sys.stderr)\nsys.exit(1)\n")
        tool_schema = {
            "type": "function",
            "function": {"name": "bad-skill", "description": "fails", "parameters": {}},
            "_scripts": {"execute": "fail.py"},
            "_skill_dir": str(tmp_path),
        }
        state = self._state_with_tool_call(
            "bad-skill", {},
            tool_map={"bad-skill": tool_schema},
        )
        result_state = _execute_tools(state)
        tool_msg = result_state["messages"][-1]
        assert "Error" in tool_msg["content"]

    def test_builtin_executed_through_execute_tools(self) -> None:
        state = self._state_with_tool_call(
            "bash", {"command": "echo builtin_test"},
            builtin_set=["bash"],
        )
        result_state = _execute_tools(state)
        tool_msg = result_state["messages"][-1]
        result = json.loads(tool_msg["content"])
        assert "builtin_test" in result["stdout"]

    def test_logger_records_tool_call_and_result(self, tmp_path: Path) -> None:
        logger = ConversationLogger(tmp_path / "log.jsonl")
        state = self._state_with_tool_call(
            "bash", {"command": "echo logged"},
            builtin_set=["bash"],
            _logger=logger,
            _agent_name="agent",
            _run_id="r1",
        )
        _execute_tools(state)
        events = [json.loads(l) for l in (tmp_path / "log.jsonl").read_text().splitlines()]
        event_types = [e["event"] for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types

    def test_call_agent_no_cfg_returns_error(self) -> None:
        state = self._state_with_tool_call(
            "call_agent", {"agent": "worker", "prompt": "do it"},
            _cfg=None,
            _allowed_agents=["worker"],
        )
        result_state = _execute_tools(state)
        tool_msg = result_state["messages"][-1]
        assert "Error" in tool_msg["content"]

    def test_multiple_tool_calls_all_appended(self) -> None:
        tool_calls = [
            {"id": "t1", "function": {"name": "bash", "arguments": '{"command": "echo a"}'}},
            {"id": "t2", "function": {"name": "bash", "arguments": '{"command": "echo b"}'}},
        ]
        state = _base_state(
            messages=[
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": None, "tool_calls": tool_calls},
            ],
            builtin_set=["bash"],
        )
        result_state = _execute_tools(state)
        tool_msgs = [m for m in result_state["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2


# ── run_agent paths ───────────────────────────────────────────────────────────

class TestRunAgentPaths:
    def _simple_cfg(self, tmp_path: Path) -> object:
        return load_config(_write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              worker:
                model: openai/gpt-4o-mini
                system_prompt: You are a worker.
                max_iterations: 3
        """))

    def test_returns_final_response(self, tmp_path: Path) -> None:
        cfg = self._simple_cfg(tmp_path)
        with patch("litellm.completion", return_value=_fake_done_response("Hello!")):
            result = run_agent(cfg, "worker", "hi")
        assert result == "Hello!"

    def test_logger_run_start_called(self, tmp_path: Path) -> None:
        cfg = self._simple_cfg(tmp_path)
        logger = ConversationLogger(tmp_path / "log.jsonl")
        with patch("litellm.completion", return_value=_fake_done_response()):
            run_agent(cfg, "worker", "test prompt", logger=logger)
        lines = (tmp_path / "log.jsonl").read_text().splitlines()
        events = [json.loads(l) for l in lines]
        run_start = next(e for e in events if e["event"] == "run_start")
        assert run_start["agent"] == "worker"
        assert run_start["prompt"] == "test prompt"

    def test_prompt_skill_body_injected_into_system_prompt(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "guide"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(textwrap.dedent("""\
            ---
            name: guide
            description: A test prompt skill.
            ---

            Always use metric units.
        """), encoding="utf-8")

        cfg = load_config(_write_config(tmp_path, f"""
            version: "1"
            providers:
              openai:
                api_key: sk-test
            skills:
              - path: {skill_dir}
            agents:
              worker:
                model: openai/gpt-4o-mini
                system_prompt: "Base prompt."
                skills: [guide]
                max_iterations: 2
        """))

        captured_messages = []

        def fake_completion(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return _fake_done_response()

        with patch("litellm.completion", side_effect=fake_completion):
            run_agent(cfg, "worker", "convert 5 miles to km")

        system_msg = next(m for m in captured_messages if m["role"] == "system")
        assert "Always use metric units." in system_msg["content"]
        assert "Base prompt." in system_msg["content"]

    def test_max_iterations_reached_returns_no_response(self, tmp_path: Path) -> None:
        """Agent that always returns tool calls eventually hits max_iterations."""
        cfg = load_config(_write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              looper:
                model: openai/gpt-4o-mini
                system_prompt: You loop forever.
                max_iterations: 2
        """))

        def fake_completion(**kwargs):
            tool_call = {
                "id": "tc_loop",
                "function": {"name": "bash", "arguments": '{"command": "echo hi"}'},
            }
            msg = MagicMock()
            msg.content = None
            msg.tool_calls = [MagicMock()]
            msg.model_dump.return_value = {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call],
            }
            resp = MagicMock()
            resp.choices = [MagicMock(message=msg)]
            return resp

        with patch("litellm.completion", side_effect=fake_completion):
            result = run_agent(cfg, "looper", "loop")
        assert result == "(no response)"

    def test_run_id_propagated_to_logger(self, tmp_path: Path) -> None:
        cfg = self._simple_cfg(tmp_path)
        logger = ConversationLogger(tmp_path / "log.jsonl")
        with patch("litellm.completion", return_value=_fake_done_response()):
            run_agent(cfg, "worker", "test", logger=logger, run_id="my-run-id")
        lines = (tmp_path / "log.jsonl").read_text().splitlines()
        events = [json.loads(l) for l in lines]
        assert all(e["run_id"] == "my-run-id" for e in events)
