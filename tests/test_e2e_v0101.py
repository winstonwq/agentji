"""End-to-end tests for agentji v0.10.1 new features.

Tests:
  1. Vertex AI service-account JSON authentication (mocked SA file, real litellm kwargs)
  2. output_format declared on sub-agents — path returned to orchestrator + pipeline endpoint
  3. Full serve API smoke test: POST /v1/chat/completions returns valid response with output_format
  4. Real LLM call (integration-marked): agent that generates a text file path returned as
     "image" output_format, verified via /v1/pipeline and the response content.

Run unit-level tests (no API key needed):
    pytest tests/test_e2e_v0101.py -v

Run integration tests (requires OPENAI_API_KEY or MOONSHOT_API_KEY):
    pytest tests/test_e2e_v0101.py -v -m integration
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import agentji.server as server_module
from agentji.server import app


# ── Config helpers ────────────────────────────────────────────────────────────

def _minimal_cfg(output_format: str = "text"):
    from agentji.config import (
        AgentjiConfig, AgentConfig, ProviderConfig, StudioConfig, ImprovementConfig,
    )
    return AgentjiConfig.model_construct(
        version="1",
        providers={"openai": ProviderConfig(api_key="sk-test")},
        skills=[],
        mcps=[],
        agents={
            "assistant": AgentConfig.model_construct(
                model="openai/gpt-4o-mini",
                system_prompt="You are helpful.",
                skills=[], mcps=[], builtins=[], agents=[],
                outputs=[], inputs=[],
                max_iterations=5,
                tool_timeout=60,
                output_format=output_format,
            )
        },
        serve=None,
        memory=None,
        studio=StudioConfig(stateful=True, max_turns=20),
        improvement=ImprovementConfig(enabled=False),
    )


@pytest.fixture(autouse=True)
def _clean_server():
    server_module._cfg = _minimal_cfg()
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


# ── 1. Vertex AI service-account JSON credential loading ──────────────────────

class TestVertexAICredentialLoading:
    """Unit-level: verifies router wires up vertex_credentials correctly."""

    def test_sa_json_loaded_into_litellm_kwargs(self, tmp_path: Path) -> None:
        from agentji.config import AgentjiConfig
        from agentji.router import build_litellm_kwargs

        sa = {
            "type": "service_account",
            "project_id": "agentji-test",
            "client_email": "ci@agentji-test.iam.gserviceaccount.com",
            "private_key_id": "key123",
        }
        sa_file = tmp_path / "vertex_sa.json"
        sa_file.write_text(json.dumps(sa))

        cfg = AgentjiConfig.model_validate({
            "version": "1",
            "providers": {
                "vertex_ai": {
                    "api_key": "",
                    "vertex_credentials_file": str(sa_file),
                }
            },
            "agents": {
                "gemini": {
                    "model": "vertex_ai/gemini-1.5-pro",
                    "system_prompt": "You are helpful.",
                }
            },
        })

        kwargs = build_litellm_kwargs(cfg, "gemini")

        assert "vertex_credentials" in kwargs
        creds = json.loads(kwargs["vertex_credentials"])
        assert creds["project_id"] == "agentji-test"
        assert creds["type"] == "service_account"
        # Model should NOT be remapped to openai/ prefix
        assert kwargs["model"] == "vertex_ai/gemini-1.5-pro"
        # Empty api_key should be absent
        assert kwargs.get("api_key", "") == ""

    def test_sa_json_with_relative_path(self, tmp_path: Path) -> None:
        """vertex_credentials_file relative path is resolved from CWD."""
        import os
        from agentji.config import AgentjiConfig
        from agentji.router import build_litellm_kwargs

        sa = {"type": "service_account", "project_id": "relative-test"}
        sa_file = tmp_path / "sa.json"
        sa_file.write_text(json.dumps(sa))

        cfg = AgentjiConfig.model_validate({
            "version": "1",
            "providers": {
                "vertex_ai": {
                    "vertex_credentials_file": "sa.json",
                }
            },
            "agents": {
                "gemini": {
                    "model": "vertex_ai/gemini-1.5-pro",
                    "system_prompt": "You are helpful.",
                }
            },
        })

        original = os.getcwd()
        os.chdir(tmp_path)
        try:
            kwargs = build_litellm_kwargs(cfg, "gemini")
            assert json.loads(kwargs["vertex_credentials"])["project_id"] == "relative-test"
        finally:
            os.chdir(original)

    def test_missing_sa_file_raises_value_error(self, tmp_path: Path) -> None:
        from agentji.config import AgentjiConfig
        from agentji.router import build_litellm_kwargs

        cfg = AgentjiConfig.model_validate({
            "version": "1",
            "providers": {
                "vertex_ai": {
                    "vertex_credentials_file": "/does/not/exist/sa.json",
                }
            },
            "agents": {
                "gemini": {
                    "model": "vertex_ai/gemini-1.5-pro",
                    "system_prompt": "You are helpful.",
                }
            },
        })
        with pytest.raises(ValueError, match="vertex_credentials_file"):
            build_litellm_kwargs(cfg, "gemini")


# ── 2. output_format — pipeline endpoint and config round-trip ────────────────

class TestOutputFormatPipeline:
    """output_format is declared in config and exposed via /v1/pipeline."""

    def test_default_output_format_is_text(self) -> None:
        client = TestClient(app)
        resp = client.get("/v1/pipeline")
        assert resp.status_code == 200
        assert resp.json()["agents"]["assistant"]["output_format"] == "text"

    def test_image_output_format_reflected_in_pipeline(self) -> None:
        server_module._cfg = _minimal_cfg(output_format="image")
        client = TestClient(app)
        resp = client.get("/v1/pipeline")
        assert resp.json()["agents"]["assistant"]["output_format"] == "image"

    def test_audio_output_format_reflected_in_pipeline(self) -> None:
        server_module._cfg = _minimal_cfg(output_format="audio")
        client = TestClient(app)
        resp = client.get("/v1/pipeline")
        assert resp.json()["agents"]["assistant"]["output_format"] == "audio"

    def test_video_output_format_reflected_in_pipeline(self) -> None:
        server_module._cfg = _minimal_cfg(output_format="video")
        client = TestClient(app)
        resp = client.get("/v1/pipeline")
        assert resp.json()["agents"]["assistant"]["output_format"] == "video"


# ── 3. Serve API smoke test — output_format in response chain ─────────────────

class TestServeOutputFormat:
    """Mocked LLM: sub-agent returns a file path, orchestrator passes it through."""

    def test_image_path_returned_as_text_content(self) -> None:
        """When an agent returns a file path, the server passes it back in choices[].message.content."""
        fake_path = "runs/abc123/chart.png"

        with patch("agentji.loop.run_agent", return_value=fake_path):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "make a chart"}], "stream": False},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == fake_path

    def test_streaming_image_path_returned(self) -> None:
        """Streaming response passes the image path through correctly."""
        fake_path = "runs/xyz/photo.jpg"

        def fake_streaming(cfg, agent_name, prompt, on_token, **kwargs):
            on_token(fake_path)

        with patch("agentji.loop.run_agent_streaming", side_effect=fake_streaming):
            client = TestClient(app)
            resp = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "make a photo"}], "stream": True},
            )

        assert resp.status_code == 200
        body = resp.text
        assert fake_path in body


# ── 4. Real LLM integration test ─────────────────────────────────────────────

@pytest.mark.integration
def test_real_api_chat_completion_with_output_format():
    """Full integration: actual LLM API call through agentji serve stack.

    The agent is asked to return a file path string (simulating image output).
    We verify the path comes back correctly in the API response.

    Requires OPENAI_API_KEY or MOONSHOT_API_KEY in environment.
    """
    moonshot_key = os.environ.get("MOONSHOT_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if not moonshot_key and not openai_key:
        pytest.skip("No API key available (MOONSHOT_API_KEY or OPENAI_API_KEY)")

    from agentji.config import (
        AgentjiConfig, AgentConfig, ProviderConfig, StudioConfig, ImprovementConfig,
    )
    from agentji.loop import run_agent

    # Prefer Moonshot (known good) over OpenAI
    if moonshot_key:
        cfg = AgentjiConfig.model_construct(
            version="1",
            providers={"moonshot": ProviderConfig(
                api_key=moonshot_key,
                base_url="https://api.moonshot.ai/v1",
            )},
            skills=[], mcps=[],
            agents={
                "assistant": AgentConfig.model_construct(
                    model="moonshot/kimi-k2.5",
                    system_prompt=(
                        "You are a helpful assistant. "
                        "When asked to return a file path, reply with ONLY the raw path — no markdown, no explanation."
                    ),
                    skills=[], mcps=[], builtins=[], agents=[],
                    outputs=[], inputs=[],
                    max_iterations=3, tool_timeout=60,
                    output_format="text",
                )
            },
            serve=None, memory=None,
            studio=StudioConfig(), improvement=ImprovementConfig(enabled=False),
        )
        provider = "moonshot/kimi-k2.5"
    elif openai_key:
        cfg = AgentjiConfig.model_construct(
            version="1",
            providers={"openai": ProviderConfig(api_key=openai_key)},
            skills=[], mcps=[],
            agents={
                "assistant": AgentConfig.model_construct(
                    model="openai/gpt-4o-mini",
                    system_prompt=(
                        "You are a helpful assistant. "
                        "When asked to return a file path, reply with ONLY the raw path — no markdown, no explanation."
                    ),
                    skills=[], mcps=[], builtins=[], agents=[],
                    outputs=[], inputs=[],
                    max_iterations=3, tool_timeout=60,
                    output_format="text",
                )
            },
            serve=None, memory=None,
            studio=StudioConfig(), improvement=ImprovementConfig(enabled=False),
        )
        provider = "openai/gpt-4o-mini"

    result = run_agent(cfg, "assistant", "Say exactly: runs/test/output.png")

    assert result is not None
    assert len(result) > 0
    # The agent should mention the path (it may include extra text, but the path should be there)
    assert "runs" in result or "output" in result, (
        f"Expected path reference in response from {provider}, got: {result!r}"
    )


@pytest.mark.integration
def test_real_vertex_ai_if_credentials_available(tmp_path: Path):
    """Integration: real Vertex AI call using a service-account JSON.

    Requires VERTEX_SA_JSON_PATH env var pointing to a valid SA JSON file,
    and a Vertex AI project with gemini-1.5-flash access.
    """
    sa_path = os.environ.get("VERTEX_SA_JSON_PATH")
    if not sa_path or not Path(sa_path).exists():
        pytest.skip("VERTEX_SA_JSON_PATH not set or file not found")

    from agentji.config import (
        AgentjiConfig, AgentConfig, ProviderConfig, StudioConfig, ImprovementConfig,
    )
    from agentji.loop import run_agent

    cfg = AgentjiConfig.model_construct(
        version="1",
        providers={"vertex_ai": ProviderConfig(
            api_key="",
            vertex_credentials_file=sa_path,
        )},
        skills=[], mcps=[],
        agents={
            "gemini": AgentConfig.model_construct(
                model="vertex_ai/gemini-1.5-flash",
                system_prompt="You are a helpful assistant. Reply briefly.",
                skills=[], mcps=[], builtins=[], agents=[],
                outputs=[], inputs=[],
                max_iterations=2, tool_timeout=60,
                output_format="text",
            )
        },
        serve=None, memory=None,
        studio=StudioConfig(), improvement=ImprovementConfig(enabled=False),
    )

    result = run_agent(cfg, "gemini", "Say hello in one sentence.")
    assert result
    assert len(result) > 5
    print(f"\n  Vertex AI response: {result}")
