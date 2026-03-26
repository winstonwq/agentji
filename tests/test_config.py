"""Unit tests for agentji.config."""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
import yaml

from agentji.config import AgentjiConfig, load_config, _interpolate


# ── Helpers ────────────────────────────────────────────────────────────────────

def write_yaml(tmp_path: Path, content: str) -> Path:
    """Write a YAML string to a temporary agentji.yaml file."""
    p = tmp_path / "agentji.yaml"
    p.write_text(textwrap.dedent(content), encoding="utf-8")
    return p


MINIMAL_CONFIG = """
    version: "1"
    providers:
      openai:
        api_key: sk-test
    agents:
      qa:
        model: openai/gpt-4o-mini
        system_prompt: You are a helpful assistant.
"""


# ── Interpolation tests ────────────────────────────────────────────────────────

class TestInterpolate:
    def test_scalar_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "abc123")
        assert _interpolate("${MY_KEY}") == "abc123"

    def test_partial_string_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOST", "localhost")
        assert _interpolate("http://${HOST}:8080") == "http://localhost:8080"

    def test_nested_dict_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TOKEN", "tok")
        result = _interpolate({"providers": {"x": {"api_key": "${TOKEN}"}}})
        assert result["providers"]["x"]["api_key"] == "tok"

    def test_list_replaced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert _interpolate(["${A}", "${B}"]) == ["1", "2"]

    def test_non_string_passthrough(self) -> None:
        assert _interpolate(42) == 42
        assert _interpolate(True) is True
        assert _interpolate(None) is None

    def test_missing_env_var_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_VAR_XYZ", raising=False)
        with pytest.raises(ValueError, match="NONEXISTENT_VAR_XYZ"):
            _interpolate("${NONEXISTENT_VAR_XYZ}")


# ── Valid config loading ───────────────────────────────────────────────────────

class TestLoadConfigValid:
    def test_minimal_config(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.version == "1"
        assert "openai" in cfg.providers
        assert cfg.providers["openai"].api_key == "sk-test"
        assert "qa" in cfg.agents
        assert cfg.agents["qa"].model == "openai/gpt-4o-mini"

    def test_default_max_iterations(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.agents["qa"].max_iterations == 10

    def test_config_with_skills_and_mcps(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              qwen:
                api_key: sk-q
                base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
            skills:
              - path: ./skills/hello-world
            mcps:
              - name: filesystem
                command: npx
                args: ["-y", "@modelcontextprotocol/server-filesystem", "./data"]
            agents:
              analyst:
                model: qwen/qwen-max
                system_prompt: "You are an analyst."
                skills: [hello-world]
                mcps: [filesystem]
                max_iterations: 5
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.providers["qwen"].base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert cfg.skills[0].path == "./skills/hello-world"
        assert cfg.mcps[0].name == "filesystem"
        assert cfg.agents["analyst"].max_iterations == 5
        assert "hello-world" in cfg.agents["analyst"].skills

    def test_env_var_interpolation_in_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("TEST_API_KEY", "sk-live-123")
        content = """
            version: "1"
            providers:
              openai:
                api_key: ${TEST_API_KEY}
            agents:
              qa:
                model: openai/gpt-4o
                system_prompt: "hello"
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.providers["openai"].api_key == "sk-live-123"

    def test_serve_config_parsed(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
            serve:
              port: 9000
              host: 127.0.0.1
              openai_compatible: true
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.serve is not None
        assert cfg.serve.port == 9000
        assert cfg.serve.host == "127.0.0.1"


# ── Schema validation errors ───────────────────────────────────────────────────

class TestLoadConfigInvalid:
    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="agentji init"):
            load_config(tmp_path / "nonexistent.yaml")

    def test_wrong_version_raises(self, tmp_path: Path) -> None:
        content = MINIMAL_CONFIG.replace('version: "1"', 'version: "2"')
        p = write_yaml(tmp_path, content)
        with pytest.raises(Exception, match="version"):
            load_config(p)

    def test_model_without_slash_raises(self, tmp_path: Path) -> None:
        content = MINIMAL_CONFIG.replace("openai/gpt-4o-mini", "gpt-4o-mini")
        p = write_yaml(tmp_path, content)
        with pytest.raises(Exception, match="provider/model"):
            load_config(p)

    def test_agent_references_unknown_provider_raises(self, tmp_path: Path) -> None:
        content = MINIMAL_CONFIG.replace("openai/gpt-4o-mini", "anthropic/claude-3")
        p = write_yaml(tmp_path, content)
        with pytest.raises(Exception, match="anthropic"):
            load_config(p)

    def test_agent_references_unknown_skill_raises(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o
                system_prompt: "hello"
                skills: [nonexistent-skill]
        """
        p = write_yaml(tmp_path, content)
        with pytest.raises(Exception, match="nonexistent-skill"):
            load_config(p)

    def test_agent_references_unknown_mcp_raises(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o
                system_prompt: "hello"
                mcps: [unknown-mcp]
        """
        p = write_yaml(tmp_path, content)
        with pytest.raises(Exception, match="unknown-mcp"):
            load_config(p)

    def test_not_a_mapping_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "agentji.yaml"
        p.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_config(p)


# ── StudioConfig (renamed fields) ─────────────────────────────────────────────

class TestStudioConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.studio.stateful is True
        assert cfg.studio.max_turns == 20

    def test_custom_studio_values(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            studio:
              stateful: false
              max_turns: 5
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.studio.stateful is False
        assert cfg.studio.max_turns == 5


# ── ImprovementConfig ──────────────────────────────────────────────────────────

class TestImprovementConfig:
    def test_defaults_disabled(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.improvement.enabled is False
        assert cfg.improvement.model is None
        assert cfg.improvement.skills == []

    def test_enabled_with_model(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            improvement:
              enabled: true
              model: openai/gpt-4o-mini
              skills:
                - sql-query
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.improvement.enabled is True
        assert cfg.improvement.model == "openai/gpt-4o-mini"
        assert cfg.improvement.skills == ["sql-query"]

    def test_enabled_with_no_model_inherits_none(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            improvement:
              enabled: true
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.improvement.enabled is True
        assert cfg.improvement.model is None


# ── New fields: parallel_agents, memory, studio.custom_ui ─────────────────────

class TestNewV0101Fields:
    def test_parallel_agents_default_true(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.agents["qa"].parallel_agents is True

    def test_parallel_agents_can_be_disabled(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
                parallel_agents: false
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.agents["qa"].parallel_agents is False

    def test_memory_config_local_defaults(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            memory:
              backend: local
              user_id: winston
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.memory is not None
        assert cfg.memory.backend == "local"
        assert cfg.memory.compression == "auto"
        assert cfg.memory.ltm_path == ".agentji/memory"
        assert cfg.memory.inject_limit == 5
        assert cfg.memory.auto_remember is True

    def test_memory_compression_options(self, tmp_path: Path) -> None:
        for compression in ["off", "auto", "aggressive"]:
            content = f"""
                version: "1"
                providers:
                  openai:
                    api_key: sk-test
                memory:
                  backend: local
                  user_id: testuser
                  compression: {compression}
                agents:
                  qa:
                    model: openai/gpt-4o-mini
                    system_prompt: test
            """
            p = write_yaml(tmp_path, content)
            cfg = load_config(p)
            assert cfg.memory.compression == compression

    def test_memory_none_by_default(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.memory is None

    def test_studio_custom_ui_field(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            studio:
              custom_ui: ./my-ui/dist/index.html
            agents:
              qa:
                model: openai/gpt-4o-mini
                system_prompt: You are a helpful assistant.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.studio.custom_ui == "./my-ui/dist/index.html"

    def test_studio_custom_ui_default_none(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.studio.custom_ui is None


# ── Vertex AI / GCP service-account auth ──────────────────────────────────────

class TestVertexAIConfig:
    def test_vertex_credentials_file_parsed(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              vertex_ai:
                api_key: ""
                vertex_credentials_file: ./vertex_sa.json
            agents:
              gemini:
                model: vertex_ai/gemini-1.5-pro
                system_prompt: You are helpful.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.providers["vertex_ai"].vertex_credentials_file == "./vertex_sa.json"

    def test_vertex_api_key_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """api_key now defaults to '' so Vertex AI configs can omit it."""
        content = """
            version: "1"
            providers:
              vertex_ai:
                vertex_credentials_file: ./vertex_sa.json
            agents:
              gemini:
                model: vertex_ai/gemini-1.5-pro
                system_prompt: You are helpful.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.providers["vertex_ai"].api_key == ""

    def test_vertex_no_credentials_file_is_none(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.providers["openai"].vertex_credentials_file is None

    def test_vertex_credentials_file_with_env_interpolation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SA_PATH", "/secrets/sa.json")
        content = """
            version: "1"
            providers:
              vertex_ai:
                vertex_credentials_file: ${SA_PATH}
            agents:
              gemini:
                model: vertex_ai/gemini-1.5-pro
                system_prompt: You are helpful.
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.providers["vertex_ai"].vertex_credentials_file == "/secrets/sa.json"


# ── output_format on AgentConfig ──────────────────────────────────────────────

class TestOutputFormat:
    def test_output_format_defaults_to_text(self, tmp_path: Path) -> None:
        p = write_yaml(tmp_path, MINIMAL_CONFIG)
        cfg = load_config(p)
        assert cfg.agents["qa"].output_format == "text"

    def test_output_format_image(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              painter:
                model: openai/gpt-4o
                system_prompt: You generate images.
                output_format: image
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.agents["painter"].output_format == "image"

    def test_output_format_audio(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              narrator:
                model: openai/gpt-4o
                system_prompt: You generate audio.
                output_format: audio
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.agents["narrator"].output_format == "audio"

    def test_output_format_video(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              director:
                model: openai/gpt-4o
                system_prompt: You generate video.
                output_format: video
        """
        p = write_yaml(tmp_path, content)
        cfg = load_config(p)
        assert cfg.agents["director"].output_format == "video"

    def test_invalid_output_format_raises(self, tmp_path: Path) -> None:
        content = """
            version: "1"
            providers:
              openai:
                api_key: sk-test
            agents:
              qa:
                model: openai/gpt-4o
                system_prompt: You are helpful.
                output_format: binary
        """
        p = write_yaml(tmp_path, content)
        with pytest.raises(Exception):
            load_config(p)
