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


# ── Custom Studio UI ───────────────────────────────────────────────────────────

def test_custom_ui_served_when_configured(tmp_path):
    """GET / serves the custom HTML file when studio.custom_ui is set."""
    import agentji.server as sm
    from agentji.config import StudioConfig

    custom_html = tmp_path / "my-ui.html"
    custom_html.write_text("<html><body>custom</body></html>", encoding="utf-8")

    original_custom_ui = sm._cfg.studio.custom_ui if hasattr(sm._cfg, "studio") else None
    sm._cfg.studio = StudioConfig(custom_ui=str(custom_html))
    sm._studio_enabled = True

    try:
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert b"custom" in resp.content
    finally:
        sm._cfg.studio = StudioConfig(custom_ui=original_custom_ui)
        sm._studio_enabled = False


def test_missing_custom_ui_falls_back_to_builtin(tmp_path):
    """GET / falls back to built-in Studio when custom_ui path does not exist."""
    import agentji.server as sm
    from agentji.config import StudioConfig

    sm._cfg.studio = StudioConfig(custom_ui="/nonexistent/path/ui.html")
    sm._studio_enabled = True

    try:
        client = TestClient(app)
        resp = client.get("/")
        assert resp.status_code == 200
        # Falls back to built-in Studio (HTML) or JSON if studio HTML missing
        assert resp.headers.get("content-type", "").startswith("text/html") or \
               "application/json" in resp.headers.get("content-type", "")
    finally:
        sm._cfg.studio = StudioConfig()
        sm._studio_enabled = False


# ── output_format in pipeline endpoint ────────────────────────────────────────

def test_pipeline_agent_includes_output_format():
    """GET /v1/pipeline agent entries include output_format field (default 'text')."""
    client = TestClient(app)
    resp = client.get("/v1/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    agent_info = data["agents"]["assistant"]
    assert "output_format" in agent_info
    assert agent_info["output_format"] == "text"


def test_pipeline_agent_output_format_image():
    """Pipeline returns output_format='image' when agent config specifies it."""
    import agentji.server as sm
    from agentji.config import AgentConfig

    original = sm._cfg.agents["assistant"]
    sm._cfg.agents["assistant"] = AgentConfig.model_construct(
        model="openai/gpt-4o",
        system_prompt="You render images.",
        skills=[], mcps=[], builtins=[], agents=[],
        outputs=[], inputs=[],
        max_iterations=5,
        output_format="image",
    )
    try:
        client = TestClient(app)
        resp = client.get("/v1/pipeline")
        assert resp.status_code == 200
        assert resp.json()["agents"]["assistant"]["output_format"] == "image"
    finally:
        sm._cfg.agents["assistant"] = original


# ── /v1/files/ endpoint ───────────────────────────────────────────────────────

def test_files_endpoint_serves_existing_file(tmp_path):
    """GET /v1/files/{path} serves a file relative to CWD."""
    import os
    import agentji.server as sm

    # Create a test file in a temp location accessible relative to CWD
    # We'll monkeypatch Path.cwd in the test
    test_file = tmp_path / "output.txt"
    test_file.write_text("hello file content")

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        client = TestClient(app)
        resp = client.get("/v1/files/output.txt")
        assert resp.status_code == 200
        assert b"hello file content" in resp.content
    finally:
        os.chdir(original_cwd)


def test_files_endpoint_returns_404_for_missing(tmp_path):
    """GET /v1/files/{path} returns 404 when file doesn't exist."""
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        client = TestClient(app)
        resp = client.get("/v1/files/nonexistent_file_xyz.txt")
        assert resp.status_code == 404
    finally:
        os.chdir(original_cwd)


def test_files_endpoint_blocks_path_traversal(tmp_path):
    """GET /v1/files with ../ path traversal returns 403."""
    import os

    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        client = TestClient(app)
        resp = client.get("/v1/files/../../etc/passwd")
        assert resp.status_code in (403, 404)
    finally:
        os.chdir(original_cwd)


# ── accepted_inputs in pipeline topology ──────────────────────────────────────

def test_pipeline_includes_accepted_inputs():
    """GET /v1/pipeline returns accepted_inputs for each agent."""
    client = TestClient(app)
    resp = client.get("/v1/pipeline")
    assert resp.status_code == 200
    data = resp.json()
    agent_info = data["agents"]["assistant"]
    assert "accepted_inputs" in agent_info
    assert "text" in agent_info["accepted_inputs"]


# ── multimodal ChatMessage content ────────────────────────────────────────────

def test_chat_completion_accepts_multimodal_content():
    """POST /v1/chat/completions accepts list content (multimodal) in messages."""
    with patch("agentji.loop.run_agent", return_value="I see an image") as mock_run:
        client = TestClient(app)
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is in this image?"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                    ],
                }],
                "stream": False,
            },
        )
    assert resp.status_code == 200
    # run_agent should have been called with the multimodal list
    called_prompt = mock_run.call_args.args[2] if mock_run.call_args.args else mock_run.call_args.kwargs.get("prompt")
    assert isinstance(called_prompt, list)
    assert called_prompt[0]["type"] == "text"
    assert called_prompt[1]["type"] == "image_url"


# ── POST /v1/files/upload ─────────────────────────────────────────────────────

def test_upload_file_returns_path_and_filename(tmp_path, monkeypatch):
    """POST /v1/files/upload saves the file and returns path + filename."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/v1/files/upload",
        files={"file": ("test.png", b"\x89PNG\r\n\x1a\n", "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "path" in data
    assert "filename" in data
    assert data["filename"].endswith(".png")
    # File should actually exist
    from pathlib import Path as _Path
    assert _Path(data["path"]).exists()


def test_upload_file_saves_content(tmp_path, monkeypatch):
    """Uploaded file content is written to disk correctly."""
    monkeypatch.chdir(tmp_path)
    content = b"\x89PNG\r\n\x1a\nFAKE_PNG_DATA"
    client = TestClient(app)
    resp = client.post(
        "/v1/files/upload",
        files={"file": ("image.png", content, "image/png")},
    )
    assert resp.status_code == 200
    from pathlib import Path as _Path
    saved = _Path(resp.json()["path"]).read_bytes()
    assert saved == content


# ── GET /v1/media/{filepath} ──────────────────────────────────────────────────

def test_media_endpoint_serves_file_inline(tmp_path, monkeypatch):
    """GET /v1/media/{path} serves the file inline (no attachment disposition)."""
    monkeypatch.chdir(tmp_path)
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n")
    client = TestClient(app)
    resp = client.get(f"/v1/media/photo.png")
    assert resp.status_code == 200
    assert "attachment" not in resp.headers.get("content-disposition", "")


def test_media_endpoint_404_on_missing_file(tmp_path, monkeypatch):
    """GET /v1/media/nonexistent returns 404."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    resp = client.get("/v1/media/does_not_exist.png")
    assert resp.status_code == 404


def test_media_endpoint_403_on_path_traversal(tmp_path, monkeypatch):
    """GET /v1/media/../../etc/passwd returns 403."""
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)
    resp = client.get("/v1/media/../../etc/passwd")
    assert resp.status_code in (403, 404)
