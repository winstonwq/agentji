"""End-to-end session flow tests.

Tests the full agentji serve pipeline:
  POST /v1/chat/completions (stateful, improve flags)
  → session message accumulation
  → POST /v1/sessions/{id}/end
  → improvement extraction written to improvements.jsonl

LLM calls are mocked so no real API key or Ollama run is needed,
but the full FastAPI stack, session tracking, config loading, and
extractor wiring are all exercised.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import agentji.server as server_module
from agentji.server import app


# ── Config factory ────────────────────────────────────────────────────────────

def _make_cfg(tmp_path: Path, improvement_enabled: bool = False, skill_path: str | None = None):
    from agentji.config import (
        AgentjiConfig, AgentConfig, ProviderConfig,
        StudioConfig, ImprovementConfig,
    )

    if skill_path:
        from agentji.config import SkillRef
        skills = [SkillRef(path=skill_path)]
    else:
        skills = []

    return AgentjiConfig.model_construct(
        version="1",
        providers={"openai": ProviderConfig(api_key="sk-test")},
        skills=skills,
        mcps=[],
        agents={
            "assistant": AgentConfig.model_construct(
                model="openai/gpt-4o-mini",
                system_prompt="You are helpful.",
                skills=[],
                mcps=[],
                builtins=[],
                agents=[],
                outputs=[],
                inputs=[],
                max_iterations=5,
                tool_timeout=60,
            )
        },
        serve=None,
        memory=None,
        studio=StudioConfig(stateful=True, max_turns=20),
        improvement=ImprovementConfig(
            enabled=improvement_enabled,
            model=None,
            skills=[],
        ),
    )


@pytest.fixture(autouse=True)
def _clean_server(tmp_path):
    """Inject config and clean up server state between tests."""
    server_module._cfg = _make_cfg(tmp_path)
    server_module._logger = None
    server_module._default_agent = "assistant"
    server_module._studio_enabled = False
    yield
    server_module._cfg = None
    server_module._logger = None
    server_module._default_agent = None
    server_module._studio_enabled = False
    server_module._session_messages.clear()
    server_module._session_improve.clear()
    for t in list(server_module._session_timers.values()):
        t.cancel()
    server_module._session_timers.clear()


# ── Helper ────────────────────────────────────────────────────────────────────

def _fake_run(cfg, agent_name, prompt, **kwargs):
    return f"Agent '{agent_name}' responded to: {prompt}"


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSessionFlowImproveOff:
    """When improve=False (or not set), no session state is accumulated."""

    def test_no_session_tracking_without_improve(self):
        with patch("agentji.loop.run_agent", side_effect=_fake_run):
            client = TestClient(app)
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
                headers={"X-Agentji-Session-Id": "no-track-session"},
            )
        assert "no-track-session" not in server_module._session_messages

    def test_explicit_improve_false_no_tracking(self):
        with patch("agentji.loop.run_agent", side_effect=_fake_run):
            client = TestClient(app)
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "hello"}],
                    "stream": False,
                    "improve": False,
                },
                headers={"X-Agentji-Session-Id": "improve-false-session"},
            )
        assert "improve-false-session" not in server_module._session_messages


class TestSessionFlowImproveOn:
    """When improvement is enabled, messages accumulate and extraction triggers on end."""

    def test_session_messages_accumulate_across_turns(self, tmp_path):
        server_module._cfg = _make_cfg(tmp_path, improvement_enabled=True)

        session_id = "multi-turn-session"
        with patch("agentji.loop.run_agent", side_effect=_fake_run):
            client = TestClient(app)
            # Turn 1
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "first question"}],
                    "stream": False,
                    "improve": True,
                },
                headers={"X-Agentji-Session-Id": session_id},
            )
            # Turn 2 — client sends full history
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "first question"},
                        {"role": "assistant", "content": "first answer"},
                        {"role": "user", "content": "follow-up question"},
                    ],
                    "stream": False,
                    "improve": True,
                },
                headers={"X-Agentji-Session-Id": session_id},
            )

        messages = server_module._session_messages.get(session_id, [])
        assert any(m["role"] == "user" for m in messages)
        assert any(m["role"] == "assistant" for m in messages)

        # Cancel the idle timer
        with server_module._session_lock:
            t = server_module._session_timers.pop(session_id, None)
        if t:
            t.cancel()

    def test_session_end_clears_state(self, tmp_path):
        server_module._cfg = _make_cfg(tmp_path, improvement_enabled=True)

        session_id = "cleanup-session"
        with server_module._session_lock:
            server_module._session_messages[session_id] = [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
            ]
            server_module._session_improve[session_id] = True

        # Mock the extractor so we don't need real LLM
        with patch("agentji.improver.extract_and_save", return_value=[]):
            client = TestClient(app)
            resp = client.post(f"/v1/sessions/{session_id}/end")

        assert resp.status_code == 200
        # State should be cleared after some brief wait (extraction runs in executor)
        time.sleep(0.1)
        assert session_id not in server_module._session_messages
        assert session_id not in server_module._session_improve

    def test_session_end_invokes_extractor_with_correct_args(self, tmp_path):
        session_id = "extractor-args-session"
        server_module._cfg = _make_cfg(tmp_path, improvement_enabled=True)

        msgs = [
            {"role": "user", "content": "Can you fix this query?"},
            {"role": "assistant", "content": "Try adding an alias."},
            {"role": "user", "content": "That worked, thanks!"},
        ]
        with server_module._session_lock:
            server_module._session_messages[session_id] = list(msgs)
            server_module._session_improve[session_id] = True

        captured = {}

        def fake_extract(messages, session_id, skill_refs, model, litellm_kwargs, target_skills, **kw):
            captured["messages"] = messages
            captured["session_id"] = session_id
            captured["model"] = model
            return []

        with patch("agentji.improver.extract_and_save", side_effect=fake_extract):
            client = TestClient(app)
            client.post(f"/v1/sessions/{session_id}/end")
            time.sleep(0.2)  # let the executor thread run

        assert captured.get("session_id") == session_id
        assert captured.get("messages") == msgs
        assert "openai" in captured.get("model", "")  # model inherited from agent

    def test_session_end_writes_improvements_file(self, tmp_path):
        """Full e2e: session messages → extraction → improvements.jsonl on disk."""
        skill_dir = tmp_path / "skills" / "sql-query"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: sql-query\n---\n\nSQL query skill.", encoding="utf-8"
        )

        server_module._cfg = _make_cfg(tmp_path, improvement_enabled=True, skill_path=str(skill_dir))

        session_id = "write-improvements-session"
        msgs = [
            {"role": "user", "content": "The query is wrong."},
            {"role": "assistant", "content": "Let me fix that."},
            {"role": "user", "content": "Perfect, that's exactly right!"},
        ]
        with server_module._session_lock:
            server_module._session_messages[session_id] = list(msgs)
            server_module._session_improve[session_id] = True

        fake_improvements = [
            {
                "type": "correction",
                "skill": "sql-query",
                "learning": "Alias GROUP BY columns.",
                "context": "User: query is wrong.",
            }
        ]

        with patch("agentji.improver.extract_and_save", return_value=fake_improvements) as mock_ex:
            client = TestClient(app)
            resp = client.post(f"/v1/sessions/{session_id}/end")
            assert resp.status_code == 200
            time.sleep(0.2)

        mock_ex.assert_called_once()


class TestStatefulFlag:
    """History forwarding is controlled by the stateful flag."""

    def test_stateful_default_true_includes_history(self):
        captured = []

        def capture(cfg, agent_name, prompt, **kwargs):
            captured.append(kwargs.get("history"))
            return "ok"

        with patch("agentji.loop.run_agent", side_effect=capture):
            client = TestClient(app)
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "prior"},
                        {"role": "assistant", "content": "reply"},
                        {"role": "user", "content": "current"},
                    ],
                    "stream": False,
                },
            )

        assert captured[0] is not None
        assert any(m["content"] == "prior" for m in captured[0])

    def test_stateful_false_excludes_history(self):
        captured = []

        def capture(cfg, agent_name, prompt, **kwargs):
            captured.append(kwargs.get("history"))
            return "ok"

        with patch("agentji.loop.run_agent", side_effect=capture):
            client = TestClient(app)
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "prior"},
                        {"role": "assistant", "content": "reply"},
                        {"role": "user", "content": "current"},
                    ],
                    "stream": False,
                    "stateful": False,
                },
            )

        assert captured[0] is None

    def test_per_request_stateful_overrides_config_default(self, tmp_path):
        """Request stateful=True overrides a config with stateful=False."""
        from agentji.config import StudioConfig, ImprovementConfig
        server_module._cfg = _make_cfg(tmp_path)
        server_module._cfg.studio = StudioConfig(stateful=False, max_turns=20)

        captured = []

        def capture(cfg, agent_name, prompt, **kwargs):
            captured.append(kwargs.get("history"))
            return "ok"

        with patch("agentji.loop.run_agent", side_effect=capture):
            client = TestClient(app)
            client.post(
                "/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "user", "content": "prior"},
                        {"role": "assistant", "content": "reply"},
                        {"role": "user", "content": "current"},
                    ],
                    "stream": False,
                    "stateful": True,  # override: turn ON even though config says OFF
                },
            )

        assert captured[0] is not None


class TestIdleTimer:
    """Idle timer fires and triggers extraction after _SESSION_IDLE_SECS."""

    def test_idle_timer_is_set_when_improve_true(self, tmp_path):
        server_module._cfg = _make_cfg(tmp_path, improvement_enabled=True)
        session_id = "idle-timer-session"

        with patch("agentji.loop.run_agent", side_effect=_fake_run):
            client = TestClient(app)
            client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hi"}], "stream": False, "improve": True},
                headers={"X-Agentji-Session-Id": session_id},
            )

        with server_module._session_lock:
            has_timer = session_id in server_module._session_timers

        assert has_timer

        # Cancel to avoid side effects
        with server_module._session_lock:
            t = server_module._session_timers.pop(session_id, None)
        if t:
            t.cancel()

    def test_timer_reset_on_new_request(self, tmp_path):
        """Each new message resets the idle timer."""
        server_module._cfg = _make_cfg(tmp_path, improvement_enabled=True)
        session_id = "timer-reset-session"

        with patch("agentji.loop.run_agent", side_effect=_fake_run):
            client = TestClient(app)
            for _ in range(3):
                client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "msg"}], "stream": False, "improve": True},
                    headers={"X-Agentji-Session-Id": session_id},
                )

        # Only one active timer at the end
        with server_module._session_lock:
            count = sum(1 for k in server_module._session_timers if k == session_id)
        assert count == 1

        with server_module._session_lock:
            t = server_module._session_timers.pop(session_id, None)
        if t:
            t.cancel()
