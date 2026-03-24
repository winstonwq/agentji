"""FastMCP client bridge for agentji.

Connects to MCP servers defined in agentji.yaml, discovers their tools,
converts them to OpenAI-compatible schemas, and executes tool calls.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Any

# Suppress verbose INFO logs emitted by the mcp / fastmcp libraries
# (e.g. "INFO:mcp.client.stdio:..." connection lifecycle messages).
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("fastmcp").setLevel(logging.WARNING)

from agentji.config import MCPConfig


# ── Internal async helpers ────────────────────────────────────────────────────

def _mcp_server_spec(mcp_config: MCPConfig) -> dict[str, Any]:
    """Build the MCPConfig dict expected by fastmcp Client.

    Uses the standard MCP server config format with a single named server.
    When there is exactly one server, fastmcp connects directly without
    prefixing tool names.

    Args:
        mcp_config: agentji MCP server definition.

    Returns:
        A dict in ``{"mcpServers": {name: {command, args, [env]}}}`` format.
    """
    # Resolve 'python' / 'python3' to the active interpreter so that
    # MCP server scripts can import packages from the current venv.
    command = mcp_config.command
    if command in ("python", "python3"):
        command = sys.executable

    server: dict[str, Any] = {
        "command": command,
        "args": mcp_config.args,
    }
    if mcp_config.env:
        server["env"] = mcp_config.env
    return {"mcpServers": {mcp_config.name: server}}


async def _list_tools_async(mcp_config: MCPConfig) -> list[Any]:
    """Async: connect to MCP server and return its tool list."""
    from fastmcp import Client

    async with Client(_mcp_server_spec(mcp_config)) as client:
        return await client.list_tools()


async def _call_tool_async(
    mcp_config: MCPConfig, tool_name: str, arguments: dict[str, Any]
) -> str:
    """Async: connect to MCP server and execute a tool call."""
    from fastmcp import Client

    async with Client(_mcp_server_spec(mcp_config)) as client:
        result = await client.call_tool(tool_name, arguments)
        # fastmcp 3.x returns a CallToolResult with a .content list
        content_items = getattr(result, "content", result)
        parts = []
        for item in content_items:
            if hasattr(item, "text"):
                parts.append(item.text)
            else:
                parts.append(str(item))
        return "\n".join(parts) if parts else ""


def _run_async(coro: Any) -> Any:
    """Run a coroutine synchronously, compatible with already-running event loops.

    Args:
        coro: An awaitable coroutine.

    Returns:
        The coroutine's return value.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # We're inside a running event loop (e.g. pytest-asyncio, Jupyter).
        # Use a new thread with its own event loop to avoid nesting.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


# ── Schema conversion ─────────────────────────────────────────────────────────

def _mcp_tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Convert a fastmcp Tool object to an OpenAI-compatible tool definition.

    Args:
        tool: A fastmcp Tool object with ``name``, ``description``,
              and ``inputSchema`` attributes.

    Returns:
        An OpenAI tool definition dict.
    """
    input_schema: dict[str, Any] = {}
    if hasattr(tool, "inputSchema") and tool.inputSchema:
        raw = tool.inputSchema
        # inputSchema is already a JSON Schema object dict
        if isinstance(raw, dict):
            input_schema = raw
        else:
            # Pydantic model or similar — call model_dump if available
            input_schema = raw.model_dump() if hasattr(raw, "model_dump") else dict(raw)

    # Ensure it has the right top-level shape
    if "type" not in input_schema:
        input_schema["type"] = "object"
    if "properties" not in input_schema:
        input_schema["properties"] = {}

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": input_schema,
        },
        # Internal tag so the loop knows this is an MCP tool, not a skill
        "_mcp": True,
    }


# ── Public sync API ───────────────────────────────────────────────────────────

def list_mcp_tools(mcp_config: MCPConfig) -> list[dict[str, Any]]:
    """Discover tools from an MCP server and return OpenAI-compatible schemas.

    Connects to the MCP server, lists available tools, and converts each
    to an OpenAI tool definition dict.

    Args:
        mcp_config: The MCP server to connect to.

    Returns:
        List of OpenAI tool definition dicts, each with ``_mcp: True`` tag.

    Raises:
        RuntimeError: If the MCP server cannot be started or connected to.
    """
    tools = _run_async(_list_tools_async(mcp_config))
    return [_mcp_tool_to_openai_schema(t) for t in tools]


def call_mcp_tool(
    mcp_config: MCPConfig, tool_name: str, arguments: dict[str, Any]
) -> str:
    """Execute a tool on an MCP server and return the text result.

    Args:
        mcp_config: The MCP server that owns the tool.
        tool_name: Name of the tool to call.
        arguments: Arguments dict to pass to the tool.

    Returns:
        The tool's text output as a string.

    Raises:
        RuntimeError: If the tool call fails.
    """
    return _run_async(_call_tool_async(mcp_config, tool_name, arguments))
