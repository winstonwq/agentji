"""Tests for --root-path / reverse-proxy support."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
from starlette.testclient import TestClient

import agentji.server as server_module
from agentji.server import app, _RootPathStripperMiddleware


def _make_config(tmp_path: Path) -> "AgentjiConfig":
    from agentji.config import load_config

    cfg_path = tmp_path / "agentji.yaml"
    cfg_path.write_text(
        textwrap.dedent("""\
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
                max_iterations: 3
        """),
        encoding="utf-8",
    )
    return load_config(cfg_path)


@pytest.fixture()
def _server(tmp_path, monkeypatch):
    """Inject server globals and return a TestClient wrapping the stripped-prefix app."""
    cfg = _make_config(tmp_path)
    monkeypatch.setattr(server_module, "_cfg", cfg)
    monkeypatch.setattr(server_module, "_default_agent", "qa")
    monkeypatch.setattr(server_module, "_studio_enabled", True)
    monkeypatch.setattr(server_module, "_root_path", "/foo")

    wrapped = _RootPathStripperMiddleware(app, root_path="/foo")
    return TestClient(wrapped, raise_server_exceptions=True)


class TestRootPath:
    def test_studio_html_returned_at_prefixed_root(self, _server: TestClient) -> None:
        with patch.object(server_module, "_root_path", "/foo"):
            r = _server.get("/foo/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "__AGENTJI_ROOT_PATH__" in r.text
        assert '"/foo"' in r.text

    def test_pipeline_reachable_with_prefix(self, _server: TestClient) -> None:
        r = _server.get("/foo/v1/pipeline")
        assert r.status_code == 200
        data = r.json()
        assert "agents" in data

    def test_chat_completions_with_prefix(self, _server: TestClient, tmp_path) -> None:
        with patch("agentji.loop.run_agent", return_value="mock response"):
            r = _server.post(
                "/foo/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )
        assert r.status_code == 200
        data = r.json()
        assert data["choices"][0]["message"]["content"] == "mock response"

    def test_unprefixed_path_returns_404(self, _server: TestClient) -> None:
        r = _server.get("/v1/pipeline", follow_redirects=False)
        assert r.status_code == 404

    def test_unprefixed_chat_returns_404(self, _server: TestClient) -> None:
        r = _server.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
        )
        assert r.status_code == 404


class TestRootPathStripping:
    """Unit tests for the middleware itself."""

    @pytest.mark.asyncio
    async def test_exact_prefix_maps_to_root(self) -> None:
        calls = []

        async def dummy(scope, receive, send):
            calls.append(scope["path"])

        mw = _RootPathStripperMiddleware(dummy, root_path="/foo")
        await mw({"type": "http", "path": "/foo"}, None, None)
        assert calls == ["/"]

    @pytest.mark.asyncio
    async def test_prefixed_subpath_stripped(self) -> None:
        calls = []

        async def dummy(scope, receive, send):
            calls.append(scope["path"])

        mw = _RootPathStripperMiddleware(dummy, root_path="/foo")
        await mw({"type": "http", "path": "/foo/v1/bar"}, None, None)
        assert calls == ["/v1/bar"]

    @pytest.mark.asyncio
    async def test_empty_root_path_is_passthrough(self) -> None:
        calls = []

        async def dummy(scope, receive, send):
            calls.append(scope["path"])

        mw = _RootPathStripperMiddleware(dummy, root_path="")
        await mw({"type": "http", "path": "/v1/chat"}, None, None)
        assert calls == ["/v1/chat"]
