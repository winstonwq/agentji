"""Unit tests for the call_agent orchestration mechanism."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentji.loop import _build_call_agent_tool, run_agent
from agentji.config import load_config


# ── _build_call_agent_tool ─────────────────────────────────────────────────────

class TestBuildCallAgentTool:
    def test_function_name(self) -> None:
        tool = _build_call_agent_tool(["analyst", "reporter"])
        assert tool["function"]["name"] == "call_agent"

    def test_enum_contains_agents(self) -> None:
        tool = _build_call_agent_tool(["analyst", "reporter"])
        enum = tool["function"]["parameters"]["properties"]["agent"]["enum"]
        assert enum == ["analyst", "reporter"]

    def test_required_fields(self) -> None:
        tool = _build_call_agent_tool(["analyst"])
        required = tool["function"]["parameters"]["required"]
        assert "agent" in required
        assert "prompt" in required

    def test_internal_marker(self) -> None:
        tool = _build_call_agent_tool(["analyst"])
        assert tool.get("_call_agent") is True

    def test_internal_marker_stripped_from_llm_tools(self) -> None:
        """The _call_agent marker must be stripped before sending to LLM."""
        tool = _build_call_agent_tool(["analyst"])
        llm_tool = {k: v for k, v in tool.items() if not k.startswith("_")}
        assert "_call_agent" not in llm_tool
        assert "type" in llm_tool
        assert "function" in llm_tool

    def test_description_mentions_agents(self) -> None:
        tool = _build_call_agent_tool(["sql-runner", "analyst"])
        desc = tool["function"]["description"]
        assert "sql-runner" in desc
        assert "analyst" in desc

    def test_empty_list(self) -> None:
        """Empty agents list is valid — produces no enum constraint."""
        tool = _build_call_agent_tool([])
        enum = tool["function"]["parameters"]["properties"]["agent"]["enum"]
        assert enum == []


# ── Config wiring: agents field ────────────────────────────────────────────────

def _write_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "agentji.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


class TestOrchestratorConfig:
    def test_agents_field_loaded(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              orchestrator:
                model: openai/gpt-4o
                system_prompt: You are a coordinator.
                agents: [worker]
              worker:
                model: openai/gpt-4o-mini
                system_prompt: You are a worker.
        """)
        cfg = load_config(cfg_path)
        assert cfg.agents["orchestrator"].agents == ["worker"]

    def test_agents_field_defaults_empty(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              worker:
                model: openai/gpt-4o-mini
                system_prompt: You are a worker.
        """)
        cfg = load_config(cfg_path)
        assert cfg.agents["worker"].agents == []

    def test_call_agent_tool_added_for_orchestrator(self, tmp_path: Path) -> None:
        """run_agent should add call_agent to tools when agent.agents is set."""
        cfg_path = _write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              orchestrator:
                model: openai/gpt-4o
                system_prompt: You are a coordinator.
                agents: [worker]
              worker:
                model: openai/gpt-4o-mini
                system_prompt: You are a worker.
        """)
        cfg = load_config(cfg_path)

        captured_tools = []

        def fake_completion(**kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            msg = MagicMock()
            msg.content = "Done."
            msg.tool_calls = None
            msg.model_dump.return_value = {"role": "assistant", "content": "Done."}
            resp = MagicMock()
            resp.choices = [MagicMock(message=msg)]
            return resp

        with patch("litellm.completion", side_effect=fake_completion):
            run_agent(cfg, "orchestrator", "Do something")

        tool_names = [t["function"]["name"] for t in captured_tools]
        assert "call_agent" in tool_names

    def test_call_agent_tool_not_added_for_non_orchestrator(self, tmp_path: Path) -> None:
        cfg_path = _write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              worker:
                model: openai/gpt-4o-mini
                system_prompt: You are a worker.
        """)
        cfg = load_config(cfg_path)

        captured_tools = []

        def fake_completion(**kwargs):
            captured_tools.extend(kwargs.get("tools", []))
            msg = MagicMock()
            msg.content = "Done."
            msg.tool_calls = None
            msg.model_dump.return_value = {"role": "assistant", "content": "Done."}
            resp = MagicMock()
            resp.choices = [MagicMock(message=msg)]
            return resp

        with patch("litellm.completion", side_effect=fake_completion):
            run_agent(cfg, "worker", "Do something")

        tool_names = [t["function"]["name"] for t in captured_tools]
        assert "call_agent" not in tool_names


# ── call_agent dispatch ────────────────────────────────────────────────────────

class TestCallAgentDispatch:
    def _make_cfg(self, tmp_path: Path) -> object:
        cfg_path = _write_config(tmp_path, """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              orchestrator:
                model: openai/gpt-4o
                system_prompt: You are a coordinator.
                agents: [worker]
                max_iterations: 5
              worker:
                model: openai/gpt-4o-mini
                system_prompt: You are a worker. Reply with WORKER_OK.
                max_iterations: 3
        """)
        return load_config(cfg_path)

    def test_orchestrator_can_call_worker(self, tmp_path: Path) -> None:
        """Orchestrator's call_agent tool should invoke run_agent on the worker."""
        cfg = self._make_cfg(tmp_path)
        call_log = []

        def fake_completion(**kwargs):
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            tools = kwargs.get("tools", [])
            tool_names = [t["function"]["name"] for t in tools]

            if "gpt-4o" in model and "call_agent" in tool_names and not call_log:
                # Orchestrator first call: invoke call_agent
                call_log.append("orchestrator_called_agent")
                tool_call = MagicMock()
                tool_call.id = "tc_001"
                tool_call.function.name = "call_agent"
                tool_call.function.arguments = '{"agent": "worker", "prompt": "Do task X"}'
                msg = MagicMock()
                msg.content = None
                msg.tool_calls = [tool_call]
                msg.model_dump.return_value = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "tc_001",
                        "function": {"name": "call_agent",
                                     "arguments": '{"agent": "worker", "prompt": "Do task X"}'},
                    }],
                }
                resp = MagicMock()
                resp.choices = [MagicMock(message=msg)]
                return resp

            if "gpt-4o-mini" in model:
                # Worker response
                call_log.append("worker_responded")
                msg = MagicMock()
                msg.content = "WORKER_OK"
                msg.tool_calls = None
                msg.model_dump.return_value = {"role": "assistant", "content": "WORKER_OK"}
                resp = MagicMock()
                resp.choices = [MagicMock(message=msg)]
                return resp

            # Orchestrator final response after seeing worker result
            msg = MagicMock()
            msg.content = "Task complete."
            msg.tool_calls = None
            msg.model_dump.return_value = {"role": "assistant", "content": "Task complete."}
            resp = MagicMock()
            resp.choices = [MagicMock(message=msg)]
            return resp

        with patch("litellm.completion", side_effect=fake_completion):
            result = run_agent(cfg, "orchestrator", "Coordinate the task")

        assert "orchestrator_called_agent" in call_log
        assert "worker_responded" in call_log
        assert result == "Task complete."

    def test_call_agent_disallowed_agent_returns_error(self, tmp_path: Path) -> None:
        """Trying to call an agent not in the allowed list returns an error."""
        cfg = self._make_cfg(tmp_path)

        def fake_completion(**kwargs):
            model = kwargs.get("model", "")
            messages = kwargs.get("messages", [])
            # Check if the last message is a tool error response
            if len(messages) > 2 and messages[-1].get("role") == "tool":
                tool_content = messages[-1].get("content", "")
                if "not in the allowed" in tool_content:
                    msg = MagicMock()
                    msg.content = "Agent not allowed."
                    msg.tool_calls = None
                    msg.model_dump.return_value = {
                        "role": "assistant", "content": "Agent not allowed."
                    }
                    resp = MagicMock()
                    resp.choices = [MagicMock(message=msg)]
                    return resp

            # First call: try to call an unlisted agent
            tool_call = MagicMock()
            tool_call.id = "tc_002"
            tool_call.function.name = "call_agent"
            tool_call.function.arguments = '{"agent": "hacker", "prompt": "do evil"}'
            msg = MagicMock()
            msg.content = None
            msg.tool_calls = [tool_call]
            msg.model_dump.return_value = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "tc_002",
                    "function": {"name": "call_agent",
                                 "arguments": '{"agent": "hacker", "prompt": "do evil"}'},
                }],
            }
            resp = MagicMock()
            resp.choices = [MagicMock(message=msg)]
            return resp

        with patch("litellm.completion", side_effect=fake_completion):
            result = run_agent(cfg, "orchestrator", "Try something")

        # The error message should have been injected as a tool result
        assert result == "Agent not allowed."
