"""Integration tests against a locally running Ollama instance.

These tests require Ollama installed and at least one model pulled.
Run with: pytest -m local

Setup:
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # Pull a small model that fits in 4 GB VRAM / RAM
    ollama pull llama3.2:3b      # ~2 GB
    ollama pull qwen2.5:3b       # ~2 GB
    ollama pull phi3.5            # ~2.2 GB

    # Override which model is used (optional):
    export OLLAMA_MODEL=llama3.2:3b

    # Override the Ollama endpoint (optional, defaults to localhost:11434):
    export OLLAMA_BASE_URL=http://localhost:11434/v1
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agentji.config import load_config
from agentji.loop import run_agent


def _write_test_config(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "agentji.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


# ── Basic local model call ────────────────────────────────────────────────────

@pytest.mark.local
class TestOllamaBasic:
    def test_simple_prompt(self, tmp_path: Path, ollama_base_url: str, ollama_model: str) -> None:
        """A simple prompt should return a non-empty response from Ollama."""
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              ollama:
                api_key: ollama
                base_url: {ollama_base_url}
            agents:
              local-qa:
                model: ollama/{ollama_model}
                system_prompt: "You are a concise assistant."
                max_iterations: 3
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "local-qa", "Reply with exactly: LOCAL_OK")
        assert len(result) > 0

    def test_hello_world_skill_local(
        self, tmp_path: Path, ollama_base_url: str, ollama_model: str
    ) -> None:
        """Local model should be able to call the hello-world skill."""
        skills_root = Path(__file__).parent.parent / "skills"
        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              ollama:
                api_key: ollama
                base_url: {ollama_base_url}
            skills:
              - path: {skills_root / "hello-world"}
            agents:
              local-greeter:
                model: ollama/{ollama_model}
                system_prompt: "Use the hello-world tool to greet the user."
                skills: [hello-world]
                max_iterations: 5
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "local-greeter", "Greet Winston using the tool")
        assert len(result) > 0


# ── Local model with Yahoo Finance MCP ───────────────────────────────────────

@pytest.mark.local
class TestOllamaWithYahooFinanceMcp:
    def test_local_analyst(self, tmp_path: Path, ollama_base_url: str, ollama_model: str) -> None:
        """Local model should produce a brief financial summary from raw data."""
        import json
        import importlib.util

        # Get real data directly from the MCP server module
        spec = importlib.util.spec_from_file_location(
            "yf_mcp",
            Path(__file__).parent.parent / "examples" / "ftse-analysis" / "yahoo_finance_mcp.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        raw_metrics = mod.get_key_metrics("AZN.L")

        config_path = _write_test_config(tmp_path, f"""
            version: "1"
            providers:
              ollama:
                api_key: ollama
                base_url: {ollama_base_url}
            agents:
              local-analyst:
                model: ollama/{ollama_model}
                system_prompt: >
                  You are a financial analyst.
                  Given financial metrics, write a one-paragraph summary
                  covering valuation and profitability.
                max_iterations: 3
        """)
        cfg = load_config(config_path)
        result = run_agent(cfg, "local-analyst", f"Analyse this data:\n{raw_metrics}")
        assert len(result) > 30
