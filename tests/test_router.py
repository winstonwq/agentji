"""Unit tests for agentji.router — endpoint probe and fallback logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentji.config import AgentjiConfig, ProviderConfig
from agentji.router import (
    _cache_key,
    _probe,
    _load_cache,
    _save_cache,
    resolve_base_url,
    build_litellm_kwargs,
)


# ── _cache_key ─────────────────────────────────────────────────────────────────

class TestCacheKey:
    def test_deterministic(self) -> None:
        k1 = _cache_key("key", "https://a.com", "https://b.com")
        k2 = _cache_key("key", "https://a.com", "https://b.com")
        assert k1 == k2

    def test_different_inputs_differ(self) -> None:
        assert _cache_key("key1", "https://a.com", "https://b.com") != \
               _cache_key("key2", "https://a.com", "https://b.com")

    def test_length_16(self) -> None:
        k = _cache_key("x", "y", "z")
        assert len(k) == 16


# ── _probe ─────────────────────────────────────────────────────────────────────

class TestProbe:
    def test_returns_true_on_200(self) -> None:
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_response):
            assert _probe("https://api.example.com/v1", "sk-abc") is True

    def test_returns_false_on_401(self) -> None:
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            url=None, code=401, msg="Unauthorized", hdrs=None, fp=None
        )):
            assert _probe("https://api.example.com/v1", "sk-bad") is False

    def test_returns_false_on_403(self) -> None:
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            url=None, code=403, msg="Forbidden", hdrs=None, fp=None
        )):
            assert _probe("https://api.example.com/v1", "sk-bad") is False

    def test_returns_true_on_other_http_error(self) -> None:
        """A 404 or 500 means the endpoint exists (just wrong path/server error)."""
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.HTTPError(
            url=None, code=404, msg="Not Found", hdrs=None, fp=None
        )):
            assert _probe("https://api.example.com/v1", "sk-abc") is True

    def test_returns_false_on_connection_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("refused")):
            assert _probe("https://unreachable.example.com/v1", "sk-abc") is False

    def test_url_appends_models(self) -> None:
        captured = []
        def fake_open(req, timeout=None):
            captured.append(req.full_url)
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        with patch("urllib.request.urlopen", side_effect=fake_open):
            _probe("https://api.moonshot.ai/v1", "sk-abc")
        assert captured[0] == "https://api.moonshot.ai/v1/models"

    def test_url_trailing_slash_stripped(self) -> None:
        captured = []
        def fake_open(req, timeout=None):
            captured.append(req.full_url)
            m = MagicMock()
            m.__enter__ = lambda s: s
            m.__exit__ = MagicMock(return_value=False)
            return m
        with patch("urllib.request.urlopen", side_effect=fake_open):
            _probe("https://api.moonshot.ai/v1/", "sk-abc")
        assert captured[0] == "https://api.moonshot.ai/v1/models"


# ── resolve_base_url ───────────────────────────────────────────────────────────

class TestResolveBaseUrl:
    def _provider(self, base_url=None, fallback_base_url=None):
        return ProviderConfig(
            api_key="sk-test",
            base_url=base_url,
            fallback_base_url=fallback_base_url,
        )

    def test_no_fallback_returns_base_url_unchanged(self) -> None:
        p = self._provider(base_url="https://primary.example.com/v1")
        assert resolve_base_url(p) == "https://primary.example.com/v1"

    def test_no_base_url_returns_none(self) -> None:
        p = self._provider()
        assert resolve_base_url(p) is None

    def test_uses_primary_when_it_probes_ok(self, tmp_path: Path) -> None:
        p = self._provider(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        cache_file = tmp_path / "cache.json"
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe", return_value=True) as mock_probe:
            result = resolve_base_url(p)
        assert result == "https://api.moonshot.ai/v1"
        mock_probe.assert_called_once()  # only primary probed

    def test_falls_back_when_primary_fails(self, tmp_path: Path) -> None:
        p = self._provider(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        cache_file = tmp_path / "cache.json"
        probe_results = [False, True]  # primary fails, fallback works
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe", side_effect=probe_results):
            result = resolve_base_url(p)
        assert result == "https://api.moonshot.cn/v1"

    def test_returns_primary_when_both_fail(self, tmp_path: Path) -> None:
        p = self._provider(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        cache_file = tmp_path / "cache.json"
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe", return_value=False):
            result = resolve_base_url(p)
        assert result == "https://api.moonshot.ai/v1"

    def test_caches_result_to_disk(self, tmp_path: Path) -> None:
        p = self._provider(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        cache_file = tmp_path / "cache.json"
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe", return_value=True):
            resolve_base_url(p)
        saved = json.loads(cache_file.read_text())
        assert "https://api.moonshot.ai/v1" in saved.values()

    def test_cache_hit_skips_probe(self, tmp_path: Path) -> None:
        p = self._provider(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        key = _cache_key(p.api_key, p.base_url, p.fallback_base_url)
        cache_file = tmp_path / "cache.json"
        cache_file.write_text(json.dumps({key: "https://api.moonshot.cn/v1"}))
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe") as mock_probe:
            result = resolve_base_url(p)
        assert result == "https://api.moonshot.cn/v1"
        mock_probe.assert_not_called()


# ── build_litellm_kwargs with fallback ─────────────────────────────────────────

class TestBuildLitellmKwargsWithFallback:
    def _make_cfg(self, base_url, fallback_base_url=None):
        from agentji.config import AgentjiConfig
        import textwrap
        extra = f"\n                fallback_base_url: {fallback_base_url}" if fallback_base_url else ""
        yaml_text = textwrap.dedent(f"""
            version: "1"
            providers:
              moonshot:
                api_key: sk-test
                base_url: {base_url}{extra}
            agents:
              kimi:
                model: moonshot/kimi-k2.5
                system_prompt: Be helpful.
        """)
        import yaml
        raw = yaml.safe_load(yaml_text)
        return AgentjiConfig.model_validate(raw)

    def test_uses_resolved_url_as_api_base(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        cache_file = tmp_path / "cache.json"
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe", return_value=True):
            kwargs = build_litellm_kwargs(cfg, "kimi")
        assert kwargs["api_base"] == "https://api.moonshot.ai/v1"
        assert kwargs["model"] == "openai/kimi-k2.5"

    def test_fallback_url_used_as_api_base(self, tmp_path: Path) -> None:
        cfg = self._make_cfg(
            base_url="https://api.moonshot.ai/v1",
            fallback_base_url="https://api.moonshot.cn/v1",
        )
        cache_file = tmp_path / "cache.json"
        with patch("agentji.router._CACHE_PATH", cache_file), \
             patch("agentji.router._probe", side_effect=[False, True]):
            kwargs = build_litellm_kwargs(cfg, "kimi")
        assert kwargs["api_base"] == "https://api.moonshot.cn/v1"


# ── Vertex AI credentials ───────────────────────────────────────────────────────

class TestVertexAICredentials:
    """Tests for Google Cloud service-account JSON authentication."""

    def _make_vertex_cfg(self, sa_path: str) -> "AgentjiConfig":
        from agentji.config import AgentjiConfig
        return AgentjiConfig.model_validate({
            "version": "1",
            "providers": {
                "vertex_ai": {
                    "api_key": "",
                    "vertex_credentials_file": sa_path,
                }
            },
            "agents": {
                "gemini": {
                    "model": "vertex_ai/gemini-1.5-pro",
                    "system_prompt": "You are helpful.",
                }
            },
        })

    def test_vertex_credentials_loaded_into_kwargs(self, tmp_path: Path) -> None:
        """vertex_credentials_file is read and passed as vertex_credentials JSON string."""
        sa = {"type": "service_account", "project_id": "my-project", "client_email": "sa@my-project.iam.gserviceaccount.com"}
        sa_file = tmp_path / "vertex_sa.json"
        sa_file.write_text(json.dumps(sa))

        cfg = self._make_vertex_cfg(str(sa_file))
        kwargs = build_litellm_kwargs(cfg, "gemini")

        assert "vertex_credentials" in kwargs
        parsed = json.loads(kwargs["vertex_credentials"])
        assert parsed["project_id"] == "my-project"
        assert parsed["type"] == "service_account"

    def test_vertex_model_not_remapped_to_openai_prefix(self, tmp_path: Path) -> None:
        """Vertex AI providers have no base_url, so model stays as vertex_ai/..."""
        sa_file = tmp_path / "sa.json"
        sa_file.write_text(json.dumps({"type": "service_account"}))

        cfg = self._make_vertex_cfg(str(sa_file))
        kwargs = build_litellm_kwargs(cfg, "gemini")

        assert kwargs["model"] == "vertex_ai/gemini-1.5-pro"
        assert not kwargs["model"].startswith("openai/")

    def test_empty_api_key_not_included_in_kwargs(self, tmp_path: Path) -> None:
        """Empty api_key should be omitted from litellm kwargs."""
        sa_file = tmp_path / "sa.json"
        sa_file.write_text(json.dumps({"type": "service_account"}))

        cfg = self._make_vertex_cfg(str(sa_file))
        kwargs = build_litellm_kwargs(cfg, "gemini")

        # api_key should not be present (or be empty) when provider.api_key == ""
        assert "api_key" not in kwargs or kwargs.get("api_key") == ""

    def test_missing_credentials_file_raises(self, tmp_path: Path) -> None:
        """A vertex_credentials_file that doesn't exist should raise ValueError."""
        cfg = self._make_vertex_cfg("/nonexistent/path/sa.json")
        with pytest.raises(ValueError, match="vertex_credentials_file"):
            build_litellm_kwargs(cfg, "gemini")

    def test_non_vertex_provider_with_api_key_works(self, tmp_path: Path) -> None:
        """Standard providers (OpenAI, etc.) still pass api_key normally."""
        from agentji.config import AgentjiConfig
        cfg = AgentjiConfig.model_validate({
            "version": "1",
            "providers": {"openai": {"api_key": "sk-live"}},
            "agents": {
                "assistant": {
                    "model": "openai/gpt-4o",
                    "system_prompt": "Be helpful.",
                }
            },
        })
        kwargs = build_litellm_kwargs(cfg, "assistant")
        assert kwargs["api_key"] == "sk-live"
        assert "vertex_credentials" not in kwargs


# ── model_params merged into litellm kwargs ────────────────────────────────────

class TestModelParamsInKwargs:
    def _make_cfg(self, model_params: dict) -> "AgentjiConfig":
        from agentji.config import AgentjiConfig
        return AgentjiConfig.model_validate({
            "version": "1",
            "providers": {"openai": {"api_key": "sk-test"}},
            "agents": {
                "assistant": {
                    "model": "openai/gpt-4o",
                    "system_prompt": "Be helpful.",
                    "model_params": model_params,
                }
            },
        })

    def test_temperature_included_in_kwargs(self) -> None:
        cfg = self._make_cfg({"temperature": 0.7})
        kwargs = build_litellm_kwargs(cfg, "assistant")
        assert kwargs["temperature"] == 0.7

    def test_multiple_params_all_included(self) -> None:
        cfg = self._make_cfg({"temperature": 0.5, "top_p": 0.9, "max_tokens": 1000})
        kwargs = build_litellm_kwargs(cfg, "assistant")
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9
        assert kwargs["max_tokens"] == 1000

    def test_drop_params_set_when_model_params_present(self) -> None:
        cfg = self._make_cfg({"temperature": 0.3})
        kwargs = build_litellm_kwargs(cfg, "assistant")
        assert kwargs.get("drop_params") is True

    def test_drop_params_absent_when_no_model_params(self) -> None:
        cfg = self._make_cfg({})
        kwargs = build_litellm_kwargs(cfg, "assistant")
        assert "drop_params" not in kwargs

    def test_model_params_warning_logged(self, caplog) -> None:
        import logging
        cfg = self._make_cfg({"temperature": 0.8, "seed": 42})
        with caplog.at_level(logging.WARNING, logger="agentji.router"):
            build_litellm_kwargs(cfg, "assistant")
        assert any("model_params" in r.message for r in caplog.records)
        assert any("temperature" in r.message or "seed" in r.message for r in caplog.records)
