"""Pytest configuration: load .env, register custom markers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


# ── Load .env from project root ───────────────────────────────────────────────

def pytest_configure(config: pytest.Config) -> None:
    """Load .env before any tests run and register custom markers."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)  # don't override already-set vars
        except ImportError:
            pass  # python-dotenv not available — env vars must be set manually

    config.addinivalue_line(
        "markers",
        "integration: real API calls — requires API keys in .env (e.g. DASHSCOPE_API_KEY)",
    )
    config.addinivalue_line(
        "markers",
        "local: requires a running Ollama instance with a model pulled",
    )


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture()
def dashscope_api_key() -> str:
    """Return the DashScope API key, skipping the test if not set."""
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    if not key or key.startswith("sk-...") or len(key) < 10:
        pytest.skip("DASHSCOPE_API_KEY not set — skipping integration test")
    return key


@pytest.fixture()
def ollama_base_url() -> str:
    """Return the Ollama base URL, skipping if Ollama is not reachable."""
    import urllib.request

    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # Normalise: strip /v1 when pinging the native /api/tags endpoint
    ping_base = base_url.rstrip("/v1").rstrip("/")
    try:
        urllib.request.urlopen(f"{ping_base}/api/tags", timeout=2)
    except Exception:
        pytest.skip(
            f"Ollama not reachable at {ping_base} — "
            "install Ollama and pull a model: ollama pull llama3.2:3b"
        )
    return base_url


@pytest.fixture()
def ollama_model() -> str:
    """Return the first available Ollama model, skipping if none are pulled."""
    import json
    import urllib.request

    preferred = os.environ.get("OLLAMA_MODEL", "")
    base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ping_base = base_url.rstrip("/v1").rstrip("/")

    try:
        resp = urllib.request.urlopen(f"{ping_base}/api/tags", timeout=2)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
    except Exception:
        pytest.skip("Ollama not reachable")

    if not models:
        pytest.skip("No Ollama models pulled — run: ollama pull llama3.2:3b")

    if preferred and preferred in models:
        return preferred
    return models[0]
