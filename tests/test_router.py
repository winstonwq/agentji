"""Unit tests for agentji.router — endpoint probe and fallback logic."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agentji.config import ProviderConfig
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
