"""YAML configuration loader and Pydantic schema validator for agentji.

Loads agentji.yaml, resolves ${ENV_VAR} interpolations, and validates the
result against the V1 schema.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

_VALID_BUILTINS = frozenset({"bash", "read_file", "write_file"})


# ── Regex for ${VAR_NAME} interpolation ──────────────────────────────────────
_ENV_VAR_RE = re.compile(r"\$\{([^}]+)\}")


def _interpolate(value: Any) -> Any:
    """Recursively resolve ``${VAR_NAME}`` placeholders from the environment.

    Args:
        value: A scalar, list, or dict from the parsed YAML tree.

    Returns:
        The same structure with all placeholders replaced by their environment
        variable values.

    Raises:
        ValueError: If a referenced environment variable is not set.
    """
    if isinstance(value, str):
        def _replace(match: re.Match) -> str:
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                raise ValueError(
                    f"Environment variable '{var_name}' is not set. "
                    f"Add it to your shell or to a .env file and try again."
                )
            return env_value

        return _ENV_VAR_RE.sub(_replace, value)
    if isinstance(value, dict):
        return {k: _interpolate(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(item) for item in value]
    return value


# ── Sub-models ────────────────────────────────────────────────────────────────

class ProviderConfig(BaseModel):
    """LLM provider credentials and endpoint."""

    api_key: str = Field(
        "",
        description=(
            "API key for this provider. "
            "May be left empty when authenticating via a service-account file "
            "(e.g. Google Cloud Vertex AI with vertex_credentials_file)."
        ),
    )
    base_url: str | None = Field(
        None,
        description="Optional custom base URL (e.g. for DashScope or a local proxy).",
    )
    fallback_base_url: str | None = Field(
        None,
        description=(
            "Fallback base URL tried when base_url is unreachable or returns 401. "
            "Useful for providers with separate regional endpoints "
            "(e.g. api.moonshot.ai vs api.moonshot.cn). "
            "agentji probes both on first use and caches the working one."
        ),
    )
    vertex_credentials_file: str | None = Field(
        None,
        description=(
            "Path to a Google Cloud service-account JSON file for Vertex AI authentication. "
            "Relative paths are resolved from the working directory where agentji is launched. "
            "When set, the JSON is read and passed to litellm as vertex_credentials, "
            "so api_key can be left empty. "
            "Only relevant for providers using vertex_ai/* model strings."
        ),
    )


class SkillRef(BaseModel):
    """Reference to a skill directory containing a SKILL.md file."""

    path: str = Field(..., description="Path to the skill directory (relative or absolute).")


class MCPConfig(BaseModel):
    """MCP server definition (command-based launch)."""

    name: str = Field(..., description="Unique name for this MCP server.")
    command: str = Field(..., description="Executable to launch the MCP server.")
    args: list[str] = Field(default_factory=list, description="Arguments for the command.")
    env: dict[str, str] | None = Field(
        None, description="Optional environment variables for the server process."
    )


class AgentOutput(BaseModel):
    """A named output that an agent writes to the run context at completion."""

    key: str = Field(..., description="Logical key for this output (e.g. 'market_findings').")
    description: str = Field(..., description="Human-readable description of what this output contains.")


class AgentInput(BaseModel):
    """A named input that an agent reads from the run context before starting."""

    key: str = Field(..., description="Logical key to read from the run context.")
    description: str = Field(..., description="Human-readable description of what this input contains.")


class MemoryConfig(BaseModel):
    """In-session and cross-run memory configuration."""

    backend: Literal["local", "mem0"] = Field(
        "local",
        description=(
            "'local' — built-in file-based LTM + sliding window compression (no extra dependencies). "
            "'mem0' — reserved for future mem0 integration."
        ),
    )
    user_id: str = Field(..., description="User identifier for memory scoping.")
    auto_remember: bool = Field(True, description="Automatically extract and store key facts after each run.")
    inject_limit: int = Field(5, ge=1, description="Maximum number of past-session facts to inject into the system prompt.")
    # Sliding window compression
    compression: Literal["off", "auto", "aggressive"] = Field(
        "auto",
        description=(
            "'off'        — no compression, full history always sent. "
            "'auto'       — compress when context is ~75%% full, keep most recent ~40%% verbatim. "
            "               Uses token counts when the model's context window is known to litellm; "
            "               falls back to a 40-message heuristic for unknown models. "
            "'aggressive' — compress at ~50%% full, keep ~20%% verbatim (good for small-context models). "
            "               Falls back to 20 messages."
        ),
    )
    # LTM storage (local backend)
    ltm_path: str = Field(
        ".agentji/memory",
        description="Directory for local LTM storage. A JSONL file per user_id is written here.",
    )

    @field_validator("compression", mode="before")
    @classmethod
    def _coerce_compression(cls, v: object) -> object:
        # YAML parses bare 'off' as boolean False; map it back to the string literal.
        if v is False:
            return "off"
        return v


class AgentConfig(BaseModel):
    """Configuration for a single agent."""

    model: str = Field(
        ...,
        description=(
            "Model string in 'provider/model-name' format, e.g. 'qwen/qwen-max' or "
            "'openai/gpt-4o'. The provider name must match a key under 'providers'."
        ),
    )
    system_prompt: str = Field(..., description="System prompt for this agent.")
    skills: list[str] = Field(
        default_factory=list,
        description="List of skill names (must match a loaded skill's name).",
    )
    mcps: list[str] = Field(
        default_factory=list,
        description="List of MCP server names to connect (must match a key under 'mcps').",
    )
    max_iterations: int = Field(
        10,
        ge=1,
        description="Maximum number of agentic loop iterations before stopping.",
    )
    tool_timeout: int = Field(
        60,
        ge=1,
        description=(
            "Default timeout in seconds for tool execution (skills and bash). "
            "Skills may declare their own timeout in skill.yaml which takes precedence. "
            "For agents that run heavy scripts (charts, docx, pip installs), "
            "set this higher, e.g. 120 or 300."
        ),
    )
    parallel_agents: bool = Field(
        True,
        description=(
            "When True (default), multiple call_agent tool calls emitted in a single "
            "LLM response are dispatched concurrently rather than sequentially. "
            "Only applies when the LLM chooses to emit ≥2 call_agent calls at once. "
            "Set to False to force sequential sub-agent execution."
        ),
    )
    builtins: list[str] = Field(
        default_factory=list,
        description=(
            "Built-in tools to enable for this agent. "
            "Valid values: bash, read_file, write_file. "
            "Required for running Anthropic-format prompt skills."
        ),
    )
    agents: list[str] = Field(
        default_factory=list,
        description="Sub-agent names this agent can delegate to (orchestrator use).",
    )
    output_format: Literal["text", "image", "audio", "video"] = Field(
        "text",
        description=(
            "Declared output format for this agent's final response. "
            "'text' (default) — plain text or markdown. "
            "'image' — the final response is a file path to an image the agent produced. "
            "'audio' — the final response is a file path to an audio file. "
            "'video' — the final response is a file path to a video file. "
            "Non-text formats are rendered inline in the Studio UI and served via /v1/files/. "
            "Only meaningful for non-main (sub-)agents called via call_agent; "
            "the orchestrator receives the path as a string."
        ),
    )
    outputs: list[AgentOutput] = Field(
        default_factory=list,
        description="Named outputs this agent writes to the run context at completion.",
    )
    inputs: list[AgentInput] = Field(
        default_factory=list,
        description="Named inputs this agent reads from the run context before starting.",
    )

    @field_validator("builtins")
    @classmethod
    def builtins_must_be_valid(cls, v: list[str]) -> list[str]:
        invalid = [b for b in v if b not in _VALID_BUILTINS]
        if invalid:
            raise ValueError(
                f"Unknown built-in(s): {invalid}. "
                f"Valid built-ins: {sorted(_VALID_BUILTINS)}."
            )
        return v

    @field_validator("model")
    @classmethod
    def model_format(cls, v: str) -> str:
        """Ensure model string contains exactly one slash (provider/model)."""
        if "/" not in v:
            raise ValueError(
                f"Model string '{v}' must be in 'provider/model-name' format, "
                f"e.g. 'openai/gpt-4o' or 'qwen/qwen-max'."
            )
        return v


class ServeConfig(BaseModel):
    """HTTP serving configuration (V1.5 — parsed but not activated in V1)."""

    port: int = Field(8000, ge=1, le=65535, description="Port to bind the HTTP server.")
    host: str = Field("0.0.0.0", description="Host to bind the HTTP server.")
    openai_compatible: bool = Field(
        True,
        description="Expose an OpenAI-compatible /v1/chat/completions endpoint.",
    )


class LogConfig(BaseModel):
    """Log rotation and retention configuration for agentji serve."""

    rotation: Literal["daily", "none"] = Field(
        "daily",
        description=(
            "Log rotation strategy. "
            "'daily' creates a new file each day (serve_2026-03-22.jsonl). "
            "'none' appends to a single serve.jsonl forever."
        ),
    )
    keep_days: int | None = Field(
        30,
        ge=1,
        description=(
            "Delete log files older than this many days on server startup. "
            "Only applies when rotation='daily'. Set to null to keep all files."
        ),
    )


class StudioConfig(BaseModel):
    """Studio UI configuration — controls session behaviour in the browser."""

    stateful: bool = Field(
        True,
        description=(
            "Maintain conversation history across turns within a session. "
            "When True, prior messages are included in each request so the agent "
            "remembers the conversation. Each browser tab is an independent session; "
            "refreshing clears history. Can be overridden per-request via the "
            "'stateful' field in POST /v1/chat/completions."
        ),
    )
    max_turns: int = Field(
        20,
        ge=1,
        description="Maximum number of user+assistant exchange pairs to keep in session history.",
    )
    custom_ui: str | None = Field(
        None,
        description=(
            "Path to a custom single-file HTML UI to serve at GET / instead of the built-in Studio. "
            "Relative paths are resolved from the working directory where agentji serve is launched. "
            "The entire /v1/ API surface remains unchanged — your UI talks to the same endpoints. "
            "Only active when --studio flag is passed."
        ),
    )


class ImprovementConfig(BaseModel):
    """Skill improvement extraction configuration.

    When enabled, agentji tracks conversation sessions and — at the end of each
    session — uses the configured model to extract learning signals (corrections,
    affirmations, and user-provided hints) and appends them to each skill's
    ``improvements.jsonl`` file alongside the SKILL.md.
    """

    enabled: bool = Field(
        False,
        description=(
            "Enable post-session skill improvement extraction. "
            "When True, session messages are tracked and analysed after the session ends. "
            "Can be overridden per-request via the 'improve' field in "
            "POST /v1/chat/completions."
        ),
    )
    model: str | None = Field(
        None,
        description=(
            "Model to use for extraction. None (default) inherits the default agent's "
            "model string. Specify a cheaper model (e.g. 'openai/gpt-4o-mini') to "
            "reduce extraction cost."
        ),
    )
    skills: list[str] = Field(
        default_factory=list,
        description=(
            "Skill names to collect improvements for. "
            "Empty list (default) includes all loaded skills."
        ),
    )


# ── Root config model ─────────────────────────────────────────────────────────

class AgentjiConfig(BaseModel):
    """Root configuration model for agentji.yaml."""

    version: str = Field(..., description="Schema version. Must be '1' for V1.")
    providers: dict[str, ProviderConfig] = Field(
        ..., description="Named LLM provider configurations."
    )
    skills: list[SkillRef] = Field(
        default_factory=list, description="Skill directories to load."
    )
    mcps: list[MCPConfig] = Field(
        default_factory=list, description="MCP server definitions."
    )
    agents: dict[str, AgentConfig] = Field(
        ..., description="Named agent definitions."
    )
    serve: ServeConfig | None = Field(
        None, description="Optional HTTP serving config (activated in V1.5)."
    )
    studio: StudioConfig = Field(
        default_factory=StudioConfig,
        description="Optional Studio UI configuration (session stateful mode, etc.).",
    )
    improvement: ImprovementConfig = Field(
        default_factory=ImprovementConfig,
        description="Optional skill improvement extraction configuration.",
    )
    logs: LogConfig = Field(
        default_factory=LogConfig,
        description="Optional log rotation and retention configuration.",
    )
    memory: MemoryConfig | None = Field(
        None,
        description=(
            "Optional memory configuration. "
            "When set, enables sliding window compression (within each run) and "
            "LTM fact injection/extraction (across runs). "
            "backend: 'local' requires no extra dependencies."
        ),
    )

    @field_validator("version")
    @classmethod
    def version_must_be_one(cls, v: str) -> str:
        """Enforce schema version '1' for V1 compatibility."""
        if v != "1":
            raise ValueError(
                f"Unsupported config version '{v}'. agentji V1 requires version: \"1\"."
            )
        return v

    @model_validator(mode="after")
    def agents_reference_valid_providers(self) -> "AgentjiConfig":
        """Ensure every agent's provider prefix exists in the providers map."""
        for agent_name, agent in self.agents.items():
            provider = agent.model.split("/")[0]
            if provider not in self.providers:
                raise ValueError(
                    f"Agent '{agent_name}' references provider '{provider}' "
                    f"(from model '{agent.model}'), but '{provider}' is not defined "
                    f"under 'providers'. Add it or fix the model string."
                )
        return self

    @model_validator(mode="after")
    def skills_references_valid(self) -> "AgentjiConfig":
        """Ensure every agent's skill references match a loaded skill name."""
        loaded_skill_names: set[str] = set()
        for s in self.skills:
            skill_file = Path(s.path) / "SKILL.md"
            if skill_file.exists():
                text = skill_file.read_text(encoding="utf-8")
                # Extract slug or name from frontmatter (same logic as skill_translator)
                slug_match = re.search(r"^slug:\s*(.+)$", text, re.MULTILINE)
                name_match = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
                if slug_match:
                    loaded_skill_names.add(slug_match.group(1).strip())
                elif name_match:
                    loaded_skill_names.add(name_match.group(1).strip())
            # Also accept the folder name as a fallback
            loaded_skill_names.add(Path(s.path).name)

        for agent_name, agent in self.agents.items():
            for skill_ref in agent.skills:
                if skill_ref not in loaded_skill_names:
                    raise ValueError(
                        f"Agent '{agent_name}' references skill '{skill_ref}', "
                        f"but no skill with that name is listed under 'skills'. "
                        f"Add the skill path or remove the reference."
                    )
        return self

    @model_validator(mode="after")
    def mcps_references_valid(self) -> "AgentjiConfig":
        """Ensure every agent's MCP references match a defined MCP server."""
        defined_mcps = {m.name for m in self.mcps}
        for agent_name, agent in self.agents.items():
            for mcp_ref in agent.mcps:
                if mcp_ref not in defined_mcps:
                    raise ValueError(
                        f"Agent '{agent_name}' references MCP server '{mcp_ref}', "
                        f"but '{mcp_ref}' is not defined under 'mcps'. "
                        f"Add the MCP definition or remove the reference."
                    )
        return self


# ── Public loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str | Path) -> AgentjiConfig:
    """Load, interpolate, and validate an agentji.yaml config file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        A validated :class:`AgentjiConfig` instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If environment variables are missing or schema validation fails.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: '{path}'. "
            f"Run 'agentji init' to create a starter config."
        )

    with path.open("r", encoding="utf-8") as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Config file '{path}' is not a valid YAML mapping. "
            f"Expected a dictionary at the top level."
        )

    interpolated = _interpolate(raw)
    return AgentjiConfig.model_validate(interpolated)
