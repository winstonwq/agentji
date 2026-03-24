"""Tests for agentji.server (FastAPI serve endpoints).

All tests mock run_agent / run_agent_streaming — no real LLM calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import agentji.server as server_module
from agentji.server import app


# ── Minimal config fixture ─────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _inject_cfg(tmp_path):
    """Inject a minimal AgentjiConfig into the server module for every test."""
    from agentji.config import (
        AgentjiConfig,
        AgentConfig,
        ProviderConfig,
        AgentOutput,
        AgentInput,
    )

    cfg = AgentjiConfig.model_construct(
        version="1",
        providers={"openai": ProviderConfig(api_key="sk-test")},
        skills=[],
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
            )
        },
        serve=None,
        memory=None,
    )

    server_module._cfg = cfg
    server_module._logger = None
    server_module._default_agent = "assistant"
    server_module._studio_enabled = False
    yield
    # Cleanup
    server_module._cfg = None
    server_module._logger = None
    server_module._default_agent = None
    server_module._studio_enabled = False
    server_module._session_messages.clear()
    server_module._session_improve.clear()
    for t in list(server_module._session_timers.values()):
        t.cancel()
    server_module._session_timers.clear()


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_pipeline_endpoint():
    """GET /v1/pipeline returns valid JSON with agents and default_agent keys."""
    client = TestClient(app)
    resp = client.get("/v1/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    assert "agents" in data
    assert "default_agent" in data
    assert "assistant" in data["agents"]
    assert data["default_agent"] == "assistant"
    agent_info = data["agents"]["assistant"]
    assert "model" in agent_info
    assert "skills" in agent_info
    assert "sub_agents" in agent_info


def test_chat_completion_non_streaming():
    """POST /v1/chat/completions with stream=false returns valid OpenAI chat completion."""
    with patch("agentji.loop.run_agent", return_value="Mock response") as mock_run:
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": False},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["message"]["content"] == "Mock response"
    assert data["choices"][0]["finish_reason"] == "stop"
    assert data["id"].startswith("chatcmpl-")


def test_chat_completion_streaming():
    """POST /v1/chat/completions with stream=true returns text/event-stream ending with [DONE]."""

    def fake_streaming(cfg, agent_name, prompt, on_token, **kwargs):
        on_token("Hello")
        on_token(", world!")
        return "Hello, world!"

    with patch("agentji.loop.run_agent_streaming", side_effect=fake_streaming):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "data:" in body
    assert "data: [DONE]" in body


def test_run_id_in_response_header():
    """Both streaming and non-streaming responses include X-Agentji-Run-Id header."""
    with patch("agentji.loop.run_agent", return_value="ok"):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "ping"}], "stream": False},
        )
    assert "x-agentji-run-id" in resp.headers
    run_id = resp.headers["x-agentji-run-id"]
    assert len(run_id) == 8  # short UUID hex


def test_unknown_agent_returns_400():
    """POST with agent='nonexistent' returns HTTP 400."""
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "agent": "nonexistent",
        },
    )
    assert resp.status_code == 400
    assert "nonexistent" in resp.json()["detail"]


def test_empty_messages_returns_400():
    """POST with empty messages list returns HTTP 400."""
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": []},
    )
    assert resp.status_code == 400


# ── Studio gate tests ──────────────────────────────────────────────────────────

def test_studio_disabled_by_default():
    """GET / returns JSON message when studio is disabled."""
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "message" in data
    assert "Studio" in data["message"] or "studio" in data["message"].lower()


def test_studio_enabled_serves_html():
    """GET / returns HTML when studio is enabled and index.html exists."""
    from pathlib import Path
    import agentji.server as sm
    sm._studio_enabled = True
    client = TestClient(app)
    resp = client.get("/")
    sm._studio_enabled = False
    # HTML file exists in the package, so should return 200 text/html
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


# ── Pipeline endpoint new fields ───────────────────────────────────────────────

def test_pipeline_includes_improvement_and_stateful_fields():
    """GET /v1/pipeline includes improvement_enabled and stateful fields."""
    client = TestClient(app)
    resp = client.get("/v1/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    assert "improvement_enabled" in data
    assert "stateful" in data
    assert data["improvement_enabled"] is False   # default
    assert data["stateful"] is True               # default


# ── Session tracking tests ─────────────────────────────────────────────────────

def test_session_end_unknown_session_returns_ended():
    """POST /v1/sessions/{id}/end for unknown session returns 'ended' status."""
    client = TestClient(app)
    resp = client.post("/v1/sessions/unknown-session-xyz/end")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ended"
    assert data["session_id"] == "unknown-session-xyz"


def test_improve_flag_tracks_session_messages(tmp_path):
    """When improve=True, messages are accumulated in _session_messages."""
    from agentji.config import ImprovementConfig, StudioConfig
    import agentji.server as sm

    # Enable improvement in config
    sm._cfg.improvement = ImprovementConfig(enabled=True)
    sm._cfg.studio = StudioConfig(stateful=True, max_turns=20)

    with patch("agentji.loop.run_agent", return_value="test response"):
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}], "stream": False, "improve": True},
            headers={"X-Agentji-Session-Id": "test-session-improve"},
        )

    assert resp.status_code == 200
    assert "test-session-improve" in sm._session_messages
    msgs = sm._session_messages["test-session-improve"]
    assert any(m["role"] == "user" for m in msgs)
    assert any(m["role"] == "assistant" for m in msgs)

    # Cancel the idle timer to avoid side effects
    with sm._session_lock:
        t = sm._session_timers.pop("test-session-improve", None)
    if t:
        t.cancel()


def test_session_end_triggers_extraction_scheduled(tmp_path):
    """POST /v1/sessions/{id}/end when session has messages returns extraction_scheduled."""
    import agentji.server as sm
    from agentji.config import ImprovementConfig

    sm._cfg.improvement = ImprovementConfig(enabled=True)

    session_id = "test-end-session"
    with sm._session_lock:
        sm._session_messages[session_id] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        sm._session_improve[session_id] = True

    with patch("agentji.improver.extract_and_save", return_value=[]) as mock_extract:
        client = TestClient(app)
        resp = client.post(f"/v1/sessions/{session_id}/end")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "extraction_scheduled"
    assert data["session_id"] == session_id


def test_stateful_false_sends_no_history():
    """When stateful=False is sent, prior messages are not forwarded to run_agent."""
    captured_history = []

    def capture_run(cfg, agent_name, prompt, **kwargs):
        captured_history.append(kwargs.get("history"))
        return "ok"

    with patch("agentji.loop.run_agent", side_effect=capture_run):
        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "second"},
                ],
                "stream": False,
                "stateful": False,
            },
        )

    assert captured_history[0] is None


def test_stateful_true_forwards_history():
    """When stateful=True (or default), prior messages are forwarded to run_agent."""
    captured_history = []

    def capture_run(cfg, agent_name, prompt, **kwargs):
        captured_history.append(kwargs.get("history"))
        return "ok"

    with patch("agentji.loop.run_agent", side_effect=capture_run):
        client = TestClient(app)
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "first"},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "second"},
                ],
                "stream": False,
                "stateful": True,
            },
        )

    assert captured_history[0] is not None
    assert any(m["role"] == "user" and m["content"] == "first" for m in captured_history[0])
