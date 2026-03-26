"""LiteLLM model routing configuration.

Translates agentji provider configs into litellm-compatible parameters and
provides a thin wrapper for model calls used by the agentic loop.

Endpoint probing
----------------
When a provider defines ``fallback_base_url``, agentji probes both URLs on
first use (lightweight GET /models with the API key) and caches the working
one to ``~/.agentji/endpoint_cache.json``.  This is mainly useful for
providers that ship separate regional endpoints with incompatible API keys
(e.g. Moonshot global vs. Moonshot China).
"""

from __future__ import annotations

import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from agentji.config import AgentjiConfig, AgentConfig, ProviderConfig


# ── Persistent endpoint cache ─────────────────────────────────────────────────

_CACHE_PATH = Path.home() / ".agentji" / "endpoint_cache.json"


def _load_cache() -> dict[str, str]:
    try:
        return json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(cache: dict[str, str]) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(api_key: str, primary: str, fallback: str) -> str:
    """Stable key for the (api_key, primary_url, fallback_url) triple."""
    raw = f"{api_key}|{primary}|{fallback}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Endpoint probe ─────────────────────────────────────────────────────────────

def _probe(base_url: str, api_key: str, timeout: int = 5) -> bool:
    """Return True if base_url/models responds successfully with this api_key.

    A 200 response means the key is valid for this endpoint.
    A 401 means the endpoint exists but the key is wrong for it.
    Any connection error means the endpoint is unreachable.
    """
    url = base_url.rstrip("/") + "/models"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError as exc:
        # 401 = key rejected; anything else (403, 404…) = endpoint reachable but wrong
        return exc.code not in (401, 403)
    except Exception:
        return False


def resolve_base_url(provider: ProviderConfig) -> str | None:
    """Return the best base_url for a provider, probing if a fallback is set.

    If ``provider.fallback_base_url`` is None the primary ``base_url`` is
    returned unchanged (no probe needed).

    On the first call the function probes both URLs, saves the winner to the
    on-disk cache, and returns it.  Subsequent calls hit the cache directly.
    """
    primary = provider.base_url
    fallback = provider.fallback_base_url

    if not primary or not fallback:
        return primary  # nothing to probe

    cache = _load_cache()
    key = _cache_key(provider.api_key, primary, fallback)

    if key in cache:
        return cache[key]

    # Probe primary first; fall back only if primary fails
    if _probe(primary, provider.api_key):
        chosen = primary
    elif _probe(fallback, provider.api_key):
        chosen = fallback
    else:
        # Both failed — return primary and let litellm surface the real error
        chosen = primary

    cache[key] = chosen
    _save_cache(cache)
    return chosen


# ── Main builder ───────────────────────────────────────────────────────────────

def build_litellm_kwargs(
    cfg: AgentjiConfig, agent_name: str
) -> dict[str, Any]:
    """Build keyword arguments for a litellm.completion() call.

    Args:
        cfg: The loaded AgentjiConfig.
        agent_name: The name of the agent whose model to configure.

    Returns:
        A dict with ``model``, ``api_key``, and optionally ``api_base``
        that can be unpacked directly into litellm.completion().

    Raises:
        KeyError: If the agent or its provider is not found in the config.
    """
    agent: AgentConfig = cfg.agents[agent_name]
    provider_name = agent.model.split("/")[0]
    provider = cfg.providers[provider_name]

    model_string = agent.model
    resolved_base_url = resolve_base_url(provider)

    if resolved_base_url:
        # Custom base_url means the provider exposes an OpenAI-compatible endpoint.
        # litellm requires the "openai/" prefix to route through its OpenAI client.
        # Strip whatever provider prefix the user wrote (e.g. "qwen/") and replace
        # with "openai/" so litellm uses the right transport.
        model_name = model_string.split("/", 1)[1]
        model_string = f"openai/{model_name}"

    kwargs: dict[str, Any] = {"model": model_string}

    # Only pass api_key when non-empty (e.g. service-account auth has no key)
    if provider.api_key:
        kwargs["api_key"] = provider.api_key

    if resolved_base_url:
        kwargs["api_base"] = resolved_base_url

    # ── Vertex AI service-account credentials ────────────────────────────────
    if provider.vertex_credentials_file:
        import json as _json
        from pathlib import Path as _Path
        creds_path = _Path(provider.vertex_credentials_file)
        if not creds_path.is_absolute():
            creds_path = _Path.cwd() / creds_path
        try:
            kwargs["vertex_credentials"] = creds_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"Cannot read vertex_credentials_file '{creds_path}': {exc}"
            ) from exc

    return kwargs
