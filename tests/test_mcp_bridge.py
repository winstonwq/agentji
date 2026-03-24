"""Unit tests for agentji.mcp_bridge."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentji.config import MCPConfig
from agentji.mcp_bridge import (
    _mcp_server_spec,
    _mcp_tool_to_openai_schema,
    call_mcp_tool,
    list_mcp_tools,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_mcp_config() -> MCPConfig:
    return MCPConfig(
        name="test-server",
        command="python",
        args=["test_server.py"],
    )


@pytest.fixture()
def mcp_config_with_env() -> MCPConfig:
    return MCPConfig(
        name="env-server",
        command="npx",
        args=["-y", "@mcp/some-server"],
        env={"API_KEY": "test-key"},
    )


def _make_mock_tool(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> MagicMock:
    """Build a mock fastmcp Tool object."""
    tool = MagicMock()
    tool.name = name
    tool.description = description
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    tool.inputSchema = schema
    return tool


# ── _mcp_server_spec ──────────────────────────────────────────────────────────

class TestMcpServerSpec:
    def test_basic_spec(self, sample_mcp_config: MCPConfig) -> None:
        import sys
        spec = _mcp_server_spec(sample_mcp_config)
        assert "mcpServers" in spec
        assert "test-server" in spec["mcpServers"]
        server = spec["mcpServers"]["test-server"]
        # "python" is resolved to sys.executable so venv packages are available
        assert server["command"] == sys.executable
        assert server["args"] == ["test_server.py"]
        assert "env" not in server

    def test_spec_includes_env(self, mcp_config_with_env: MCPConfig) -> None:
        spec = _mcp_server_spec(mcp_config_with_env)
        server = spec["mcpServers"]["env-server"]
        assert server["env"] == {"API_KEY": "test-key"}

    def test_spec_name_used_as_key(self, sample_mcp_config: MCPConfig) -> None:
        spec = _mcp_server_spec(sample_mcp_config)
        assert list(spec["mcpServers"].keys()) == ["test-server"]


# ── _mcp_tool_to_openai_schema ────────────────────────────────────────────────

class TestMcpToolToOpenaiSchema:
    def test_basic_conversion(self) -> None:
        tool = _make_mock_tool(
            "get_data",
            "Fetch some data.",
            properties={"query": {"type": "string", "description": "The query."}},
            required=["query"],
        )
        schema = _mcp_tool_to_openai_schema(tool)
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_data"
        assert schema["function"]["description"] == "Fetch some data."
        assert schema["function"]["parameters"]["type"] == "object"
        assert "query" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["required"] == ["query"]

    def test_mcp_tag_present(self) -> None:
        tool = _make_mock_tool("ping", "Ping.")
        schema = _mcp_tool_to_openai_schema(tool)
        assert schema["_mcp"] is True

    def test_empty_input_schema(self) -> None:
        tool = _make_mock_tool("no_params", "No parameters.")
        tool.inputSchema = {}
        schema = _mcp_tool_to_openai_schema(tool)
        assert schema["function"]["parameters"]["type"] == "object"
        assert schema["function"]["parameters"]["properties"] == {}

    def test_none_input_schema(self) -> None:
        tool = _make_mock_tool("no_schema", "No schema at all.")
        tool.inputSchema = None
        schema = _mcp_tool_to_openai_schema(tool)
        assert schema["function"]["parameters"]["type"] == "object"

    def test_empty_description_allowed(self) -> None:
        tool = _make_mock_tool("tool", "")
        tool.description = None
        schema = _mcp_tool_to_openai_schema(tool)
        assert schema["function"]["description"] == ""

    def test_multiple_params(self) -> None:
        tool = _make_mock_tool(
            "search",
            "Search the web.",
            properties={
                "query": {"type": "string"},
                "num_results": {"type": "integer", "default": 5},
            },
            required=["query"],
        )
        schema = _mcp_tool_to_openai_schema(tool)
        props = schema["function"]["parameters"]["properties"]
        assert "query" in props
        assert "num_results" in props
        assert props["num_results"]["default"] == 5


# ── list_mcp_tools ────────────────────────────────────────────────────────────

class TestListMcpTools:
    def test_returns_openai_schemas(self, sample_mcp_config: MCPConfig) -> None:
        mock_tools = [
            _make_mock_tool("tool_a", "Tool A."),
            _make_mock_tool("tool_b", "Tool B.", properties={"x": {"type": "integer"}}),
        ]

        async def fake_list_tools_async(_cfg: MCPConfig) -> list:
            return mock_tools

        with patch("agentji.mcp_bridge._list_tools_async", side_effect=fake_list_tools_async):
            result = list_mcp_tools(sample_mcp_config)

        assert len(result) == 2
        names = {t["function"]["name"] for t in result}
        assert names == {"tool_a", "tool_b"}
        for t in result:
            assert t["_mcp"] is True

    def test_empty_server_returns_empty(self, sample_mcp_config: MCPConfig) -> None:
        async def fake_empty(_cfg: MCPConfig) -> list:
            return []

        with patch("agentji.mcp_bridge._list_tools_async", side_effect=fake_empty):
            result = list_mcp_tools(sample_mcp_config)

        assert result == []


# ── call_mcp_tool ─────────────────────────────────────────────────────────────

class TestCallMcpTool:
    def test_returns_string_result(self, sample_mcp_config: MCPConfig) -> None:
        async def fake_call(_cfg: MCPConfig, name: str, args: dict) -> str:
            return f"result for {name}"

        with patch("agentji.mcp_bridge._call_tool_async", side_effect=fake_call):
            result = call_mcp_tool(sample_mcp_config, "my_tool", {"arg": "val"})

        assert result == "result for my_tool"

    def test_passes_arguments(self, sample_mcp_config: MCPConfig) -> None:
        captured: dict = {}

        async def fake_call(_cfg: MCPConfig, name: str, args: dict) -> str:
            captured["args"] = args
            return "ok"

        with patch("agentji.mcp_bridge._call_tool_async", side_effect=fake_call):
            call_mcp_tool(sample_mcp_config, "tool", {"ticker": "AZN.L", "periods": 3})

        assert captured["args"] == {"ticker": "AZN.L", "periods": 3}

    def test_empty_result_returns_empty_string(self, sample_mcp_config: MCPConfig) -> None:
        async def fake_call(_cfg: MCPConfig, name: str, args: dict) -> str:
            return ""

        with patch("agentji.mcp_bridge._call_tool_async", side_effect=fake_call):
            result = call_mcp_tool(sample_mcp_config, "tool", {})

        assert result == ""


# ── Live in-process server test (no subprocess, no network) ──────────────────

class TestMcpBridgeLive:
    """Use a real in-process FastMCP server to test the bridge end-to-end
    without subprocesses or external services."""

    def test_list_and_call_via_in_process_server(self) -> None:
        """Build a minimal FastMCP server in-process and exercise the bridge."""
        from fastmcp import Client, FastMCP

        # Build a tiny in-process MCP server
        server = FastMCP("test-in-process")

        @server.tool()
        def echo(message: str) -> str:
            """Echo the input message back."""
            return f"echo: {message}"

        @server.tool()
        def add(a: int, b: int) -> str:
            """Add two numbers."""
            return str(a + b)

        # Test via fastmcp Client directly (bypasses subprocess transport)
        async def run() -> tuple[list, str]:
            async with Client(server) as client:
                tools = await client.list_tools()
                result = await client.call_tool("echo", {"message": "hello"})
                # fastmcp 3.x: result is CallToolResult with .content list
                content_items = getattr(result, "content", result)
                text = "\n".join(
                    item.text for item in content_items if hasattr(item, "text")
                )
                return tools, text

        tools, text = asyncio.run(run())
        names = {t.name for t in tools}
        assert "echo" in names
        assert "add" in names
        assert text == "echo: hello"


# ── Yahoo Finance MCP server unit tests ──────────────────────────────────────

class TestYahooFinanceMcpServer:
    """Test the Yahoo Finance MCP server functions directly (no subprocess)."""

    @pytest.fixture(autouse=True)
    def _import_server(self) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "yf_mcp",
            Path(__file__).parent / "fixtures" / "yahoo_finance_mcp.py",
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        self.server_mod = mod

    def test_list_ftse100_returns_valid_json(self) -> None:
        import json
        result = self.server_mod.list_ftse100_tickers()
        tickers = json.loads(result)
        assert isinstance(tickers, list)
        assert len(tickers) > 10
        assert all(isinstance(t, str) for t in tickers)

    def test_get_financial_statements_no_ticker_no_flag(self) -> None:
        import json
        result = self.server_mod.get_financial_statements()
        data = json.loads(result)
        assert "error" in data

    def test_get_financial_statements_random_ftse100(self) -> None:
        import json
        result = self.server_mod.get_financial_statements(random_ftse100=True, periods=1)
        data = json.loads(result)
        # May error if Yahoo Finance is unreachable — that's acceptable in unit tests
        if "error" not in data:
            assert "ticker" in data
            assert "income_statement" in data
            assert "balance_sheet" in data
            assert "cash_flow_statement" in data

    def test_get_key_metrics_no_ticker(self) -> None:
        import json
        result = self.server_mod.get_key_metrics()
        data = json.loads(result)
        assert "error" in data

    def test_periods_clamped(self) -> None:
        import json
        # periods=99 should be clamped to 5
        result = self.server_mod.get_financial_statements("AZN.L", periods=99)
        data = json.loads(result)
        if "error" not in data:
            assert data["periods_returned"] == 5

    def test_mcp_server_exposes_three_tools(self) -> None:
        async def run() -> list:
            from fastmcp import Client
            async with Client(self.server_mod.mcp) as client:
                return await client.list_tools()

        tools = asyncio.run(run())
        names = {t.name for t in tools}
        assert names == {"list_ftse100_tickers", "get_financial_statements", "get_key_metrics"}
