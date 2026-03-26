"""LangGraph-based agentic execution loop.

Implements the core prompt → tool call → execute → result injection cycle.
Routes tool calls to either skill scripts (executor.py) or MCP servers
(mcp_bridge.py) based on how each tool was registered.

Logging is opt-in: pass a ConversationLogger to run_agent() to record
every LLM call, tool invocation, and result to a JSONL file.
"""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, TypedDict

import litellm
from langgraph.graph import StateGraph, END

from agentji.builtins import BUILTIN_SCHEMAS, VALID_BUILTINS
from agentji.config import AgentjiConfig, MCPConfig
from agentji.executor import execute_skill
from agentji.logger import ConversationLogger
from agentji.memory import MemoryBackend
from agentji.router import build_litellm_kwargs
from agentji.run_context import RunContext
from agentji.skill_translator import translate_skills


# ── Graph state ───────────────────────────────────────────────────────────────

_STUCK_THRESHOLD = 3   # consecutive error-iterations before injecting intervention


class AgentState(TypedDict):
    """Mutable state threaded through the LangGraph nodes."""

    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    tool_map: dict[str, dict[str, Any]]    # skill tool name → full tool schema
    mcp_map: dict[str, MCPConfig]          # MCP tool name → MCPConfig
    builtin_set: list[str]                 # enabled built-in tool names
    litellm_kwargs: dict[str, Any]
    max_iterations: int
    iteration: int
    final_response: str
    # Logging context (None when logging is disabled)
    _agent_name: str
    _run_id: str
    _logger: Any  # ConversationLogger | None
    # Orchestration context (None when not an orchestrator)
    _cfg: Any             # AgentjiConfig | None — for call_agent dispatch
    _allowed_agents: list[str]  # sub-agent names this agent may call
    # Run context for file-based handoff between agents
    _run_context: Any     # RunContext | None
    # Optional streaming callback — set only for root orchestrator, never for sub-agents
    _stream_callback: Any  # Callable[[str], None] | None
    # All loaded skill descriptors (tool + prompt-only) — used by skill-converter
    _all_skills: list[dict[str, Any]]
    # Default timeout (seconds) for skill and bash tool execution
    _tool_timeout: int
    # Optional cancel signal — callable returns True when user requests stop
    _cancel_fn: Any  # Callable[[], bool] | None
    # Consecutive error tracking — reset on any successful tool call iteration
    _consecutive_errors: int
    # Memory backend — handles sliding window compression each iteration
    _memory: Any  # MemoryBackend | None
    # Whether to fan out multiple call_agent calls concurrently
    _parallel_agents: bool


# ── call_agent tool ───────────────────────────────────────────────────────────

def _build_call_agent_tool(agent_names: list[str]) -> dict[str, Any]:
    """Build the call_agent tool schema for an orchestrator agent.

    The ``enum`` constraint is populated from the parent agent's ``agents:``
    list so the LLM can only delegate to explicitly allowed sub-agents.
    """
    return {
        "type": "function",
        "function": {
            "name": "call_agent",
            "description": (
                "Delegate a task to a specialized sub-agent and receive its full response. "
                "Use this to break complex tasks into focused steps handled by the right expert. "
                f"Available agents: {', '.join(agent_names)}. "
                "When tasks are independent — they don't need each other's outputs — "
                "include multiple call_agent calls in a single response to dispatch them in parallel."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent": {
                        "type": "string",
                        "enum": agent_names,
                        "description": "Name of the sub-agent to call.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": (
                            "The task or question to send to the agent. "
                            "Be specific — include all context the agent needs since it has no memory of prior calls."
                        ),
                    },
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional list of local file paths to include as image inputs for the sub-agent. "
                            "Only valid when the target agent has 'image' in its accepted_inputs. "
                            "Paths should be in the run scratch directory."
                        ),
                    },
                },
                "required": ["agent", "prompt"],
            },
        },
        "_call_agent": True,  # internal marker — stripped before sending to LLM
    }


# ── Graph nodes ───────────────────────────────────────────────────────────────

def _call_llm(state: AgentState) -> AgentState:
    """Call the LLM with the current message history and bound tools.

    Supports two modes:
    - Non-streaming (default): calls litellm.completion() and buffers the full response.
    - Streaming: calls litellm.completion(stream=True), calls _stream_callback(chunk)
      for each content delta, then assembles the full message dict from accumulated chunks.
      Tool call chunks are accumulated silently; only content chunks trigger the callback.
    """
    logger: ConversationLogger | None = state.get("_logger")
    agent_name: str = state.get("_agent_name", "")
    run_id: str = state.get("_run_id", "")
    stream_callback: Callable[[str], None] | None = state.get("_stream_callback")

    # ── Cancellation check ────────────────────────────────────────────────────
    cancel_fn = state.get("_cancel_fn")
    if cancel_fn and cancel_fn():
        state["messages"].append({"role": "assistant", "content": "Run stopped by user."})
        state["_cancelled"] = True
        return state

    # ── Stuck detection: inject intervention after N consecutive error iterations ──
    consecutive = state.get("_consecutive_errors", 0)
    if consecutive >= _STUCK_THRESHOLD:
        state["messages"].append({
            "role": "user",
            "content": (
                f"[Intervention] You have encountered {consecutive} consecutive tool failures "
                f"without making progress. Do NOT retry the same approach again. "
                f"Step back and choose a fundamentally different strategy:\n"
                f"- If a script keeps failing, simplify it drastically or skip that step entirely\n"
                f"- If a path or import is wrong, verify it before retrying\n"
                f"- If the task is taking too long on one sub-task, move on and come back\n"
                f"Acknowledge this, state your new approach in one sentence, then proceed."
            ),
        })
        state["_consecutive_errors"] = 0
        if logger:
            logger.tool_result(
                agent=agent_name,
                run_id=run_id,
                tool_name="[intervention]",
                result=f"Stuck after {consecutive} consecutive errors — intervention injected",
                error=False,
            )

    # ── Sliding window compression ─────────────────────────────────────────
    memory = state.get("_memory")
    if memory is not None:
        state["messages"] = memory.maybe_compress(
            state["messages"], state["litellm_kwargs"]
        )

    kwargs: dict[str, Any] = {
        **state["litellm_kwargs"],
        "messages": state["messages"],
    }

    llm_tools: list[dict[str, Any]] = []
    if state["tools"]:
        # Strip internal metadata keys before sending to the LLM
        llm_tools = [
            {k: v for k, v in t.items() if not k.startswith("_")}
            for t in state["tools"]
        ]
        kwargs["tools"] = llm_tools
        kwargs["tool_choice"] = "auto"

    if logger:
        logger.llm_call(
            agent=agent_name,
            run_id=run_id,
            iteration=state["iteration"] + 1,
            n_messages=len(state["messages"]),
            n_tools=len(llm_tools),
        )

    if stream_callback is not None:
        # ── Streaming mode ────────────────────────────────────────────────────
        response = litellm.completion(stream=True, **kwargs)
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_call_acc: dict[int, dict[str, Any]] = {}

        for chunk in response:
            choice = chunk.choices[0]
            delta = choice.delta

            # Accumulate reasoning/thinking tokens (e.g. Kimi K2.5, o-series)
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                reasoning_parts.append(rc)

            if delta.content:
                content_parts.append(delta.content)
                stream_callback(delta.content)

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_call_acc:
                        tool_call_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    acc = tool_call_acc[idx]
                    if tc.id:
                        acc["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            acc["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            acc["function"]["arguments"] += tc.function.arguments

        content: str | None = "".join(content_parts) or None
        tool_calls_list = [tool_call_acc[i] for i in sorted(tool_call_acc)]

        message_dict: dict[str, Any] = {"role": "assistant"}
        if reasoning_parts:
            message_dict["reasoning_content"] = "".join(reasoning_parts)
        if content is not None:
            message_dict["content"] = content
        if tool_calls_list:
            message_dict["tool_calls"] = tool_calls_list
    else:
        # ── Non-streaming mode (unchanged existing behavior) ──────────────────
        response = litellm.completion(**kwargs)
        message = response.choices[0].message
        message_dict = message.model_dump(exclude_none=True)

    if logger:
        tool_calls_for_log = message_dict.get("tool_calls") or []
        logger.llm_response(
            agent=agent_name,
            run_id=run_id,
            iteration=state["iteration"] + 1,
            content=message_dict.get("content"),
            tool_calls=tool_calls_for_log,
        )

    state["messages"].append(message_dict)
    state["iteration"] += 1
    return state


def _try_skill_conversion(
    state: AgentState, tool_name: str, args: dict[str, Any]
) -> tuple[str, bool] | None:
    """Attempt to convert a prompt-only skill with scripts into a callable tool.

    Returns (result_str, is_error) if the tool was found and handled
    (conversion succeeded → execute result, or declined → error message).
    Returns None if no matching skill was found (caller should report unknown tool).
    """
    from pathlib import Path as _Path

    all_skills: list[dict[str, Any]] = state.get("_all_skills") or []
    matching = next(
        (s for s in all_skills
         if s.get("_prompt_only") and s["function"]["name"] == tool_name),
        None,
    )
    if matching is None:
        return None  # not a known skill at all

    skill_dir = _Path(matching["_skill_dir"])
    if not (skill_dir / "scripts").exists():
        return None  # prompt-only with no scripts — not convertible

    # ── Ask the user ──────────────────────────────────────────────────────────
    from agentji.skill_converter import prompt_user_for_conversion, convert_skill
    approved = prompt_user_for_conversion(tool_name)
    if not approved:
        return (
            f"Skill '{tool_name}' was not converted. "
            f"Add a skill.yaml to {skill_dir} to make it callable.",
            True,
        )

    # ── Run conversion ────────────────────────────────────────────────────────
    import sys
    sys.stderr.write(f"[agentji] Converting skill '{tool_name}'…\n")
    sys.stderr.flush()

    conv = convert_skill(skill_dir, state["litellm_kwargs"])
    if not conv["success"]:
        return (
            f"Skill conversion failed for '{tool_name}': {conv.get('error')}",
            True,
        )

    sys.stderr.write(
        f"[agentji] skill.yaml written to {conv['path']} — retrying tool call.\n"
    )
    sys.stderr.flush()

    # ── Reload skill and execute ───────────────────────────────────────────────
    from agentji.skill_translator import translate_skill
    try:
        new_tool = translate_skill(skill_dir)
    except Exception as exc:
        return (f"Skill reload failed after conversion: {exc}", True)

    if new_tool.get("_prompt_only"):
        return (
            f"Converted skill.yaml for '{tool_name}' did not produce a callable tool. "
            f"Check {skill_dir / 'skill.yaml'}.",
            True,
        )

    # Register into live state so future calls also work
    state["tools"].append(new_tool)
    state["tool_map"][new_tool["function"]["name"]] = new_tool

    # Execute immediately
    try:
        result = execute_skill(new_tool, args)
        return (result, False)
    except Exception as exc:
        return (f"Error executing converted skill '{tool_name}': {exc}", True)


def _build_image_content_block(path: str) -> dict[str, Any]:
    """Convert a local image file path to a litellm-compatible image_url content block."""
    import base64
    import mimetypes
    data = Path(path).read_bytes()
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    b64 = base64.b64encode(data).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}


def _dispatch_call_agent(
    state: AgentState,
    args: dict[str, Any],
    agent_name: str,
    run_id: str,
    logger: "ConversationLogger | None",
) -> tuple[str, bool]:
    """Execute a single call_agent invocation. Returns (result, is_error).

    Safe to call from multiple threads simultaneously — all shared state
    (RunContext, ConversationLogger) is protected by locks.
    """
    target = args.get("agent", "")
    sub_prompt = args.get("prompt", "")
    attachments: list[str] = args.get("attachments") or []
    cfg = state.get("_cfg")
    allowed_agents: list[str] = state.get("_allowed_agents") or []
    run_context = state.get("_run_context")

    if not cfg:
        return "Error: call_agent not available (no config in state)", True
    if target not in allowed_agents:
        return (
            f"Error: agent '{target}' is not in the allowed sub-agents list. "
            f"Allowed: {allowed_agents}",
            True,
        )

    target_cfg = cfg.agents.get(target)
    if run_context and target_cfg and target_cfg.inputs:
        found_lines = []
        for inp in target_cfg.inputs:
            value = run_context.get(inp.key)
            if value is None:
                continue
            found_lines.append(f"- {inp.key}: {value}")
            if logger:
                summary_entry = run_context.summary().get(inp.key, {})
                logger.context_read(
                    agent=target,
                    key=inp.key,
                    offloaded=summary_entry.get("offloaded", False),
                    path=value if summary_entry.get("offloaded") else None,
                )
        if found_lines:
            input_lines = (
                ["The following inputs are available for this task:"]
                + found_lines
                + ["\nRead each file using read_file before beginning your work."]
            )
            sub_prompt = "\n".join(input_lines) + "\n\n" + sub_prompt

    # ── Build multimodal user content if attachments present ──────────────
    sub_content: str | list[dict[str, Any]] = sub_prompt
    if attachments:
        parts: list[dict[str, Any]] = [{"type": "text", "text": sub_prompt}]
        for att in attachments:
            try:
                parts.append(_build_image_content_block(att))
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "call_agent: could not encode attachment '%s': %s", att, exc
                )
        sub_content = parts

    result = run_agent(
        cfg,
        target,
        sub_content,
        logger=logger,
        run_id=f"{run_id}:{target}",
        run_context=run_context,
    )

    if run_context and target_cfg and target_cfg.outputs:
        for output in target_cfg.outputs:
            run_context.set(output.key, result, target)

    return result, False


def _execute_tools(state: AgentState) -> AgentState:
    """Execute all tool calls in the last assistant message.

    Non-call_agent tools (skills, MCP, builtins) run sequentially.
    call_agent invocations run concurrently when parallel_agents=True and
    the LLM emitted ≥2 call_agent calls in the same response.
    Results are always appended to state["messages"] in original tool-call order.
    """
    from agentji.mcp_bridge import call_mcp_tool
    from agentji.builtins import execute_builtin, VALID_BUILTINS

    logger: ConversationLogger | None = state.get("_logger")
    agent_name: str = state.get("_agent_name", "")
    run_id: str = state.get("_run_id", "")
    allow_parallel: bool = state.get("_parallel_agents", True)
    allowed_agents: list[str] = state.get("_allowed_agents") or []
    builtin_set = set(state.get("builtin_set") or [])

    last_message = state["messages"][-1]
    tool_calls: list[dict[str, Any]] = last_message.get("tool_calls") or []

    # results[i] = (tool_call_id, content, is_error)
    results: dict[int, tuple[str, str, bool]] = {}

    # ── Partition into call_agent vs everything else ───────────────────────────
    ca_indices = [i for i, tc in enumerate(tool_calls) if tc["function"]["name"] == "call_agent"]
    other_indices = [i for i, tc in enumerate(tool_calls) if tc["function"]["name"] != "call_agent"]

    # ── Execute non-call_agent tools sequentially (unchanged behavior) ─────────
    for i in other_indices:
        tool_call = tool_calls[i]
        fn = tool_call["function"]
        name: str = fn["name"]
        tool_id = tool_call.get("id", name)

        try:
            args: dict[str, Any] = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError as exc:
            err = f"Error: tool arguments contained invalid JSON ({exc}). Rewrite the tool call with properly escaped arguments."
            if logger:
                logger.tool_result(agent=agent_name, run_id=run_id, tool_name=name, result=err, error=True)
            results[i] = (tool_id, err, True)
            continue

        if name in state["mcp_map"]:
            tool_type = "mcp"
        elif name in builtin_set:
            tool_type = "builtin"
        else:
            tool_type = "skill"

        if logger:
            logger.tool_call(agent=agent_name, run_id=run_id, tool_name=name, tool_type=tool_type, args=args)

        error = False
        try:
            if name in state["mcp_map"]:
                mcp_config = state["mcp_map"][name]
                result = call_mcp_tool(mcp_config, name, args)
            elif name in state["tool_map"]:
                tool_schema = state["tool_map"][name]
                skill_timeout = tool_schema.get("_timeout")
                agent_timeout = state.get("_tool_timeout", 60)
                result = execute_skill(tool_schema, args, timeout=skill_timeout or agent_timeout)
            elif name in builtin_set and name in VALID_BUILTINS:
                result = execute_builtin(name, args, default_timeout=state.get("_tool_timeout", 60))
            else:
                conv_result = _try_skill_conversion(state, name, args)
                if conv_result is not None:
                    result, error = conv_result
                else:
                    available = list(state["tool_map"]) + list(state["mcp_map"]) + list(builtin_set)
                    result = f"Error: unknown tool '{name}'. Available: {available}"
                    error = True
        except Exception as exc:
            result = f"Error executing '{name}': {exc}"
            error = True

        if logger:
            logger.tool_result(agent=agent_name, run_id=run_id, tool_name=name, result=result, error=error)
        results[i] = (tool_id, result, error)

    # ── Execute call_agent tools — parallel when ≥2 and allowed ───────────────
    use_parallel = allow_parallel and len(ca_indices) > 1

    def _run_call_agent(i: int) -> tuple[int, str, str, bool]:
        tool_call = tool_calls[i]
        fn = tool_call["function"]
        tool_id = tool_call.get("id", "call_agent")
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError as exc:
            err = f"Error: invalid JSON arguments ({exc})."
            if logger:
                logger.tool_result(agent=agent_name, run_id=run_id, tool_name="call_agent", result=err, error=True)
            return i, tool_id, err, True

        if logger:
            logger.tool_call(agent=agent_name, run_id=run_id, tool_name="call_agent", tool_type="agent", args=args)
        try:
            result, error = _dispatch_call_agent(state, args, agent_name, run_id, logger)
        except Exception as exc:
            result, error = f"Error executing call_agent: {exc}", True

        if logger:
            logger.tool_result(agent=agent_name, run_id=run_id, tool_name="call_agent", result=result, error=error)
        return i, tool_id, result, error

    if use_parallel:
        with ThreadPoolExecutor(max_workers=len(ca_indices)) as ex:
            for i, tool_id, result, error in ex.map(_run_call_agent, ca_indices):
                results[i] = (tool_id, result, error)
    else:
        for i in ca_indices:
            i, tool_id, result, error = _run_call_agent(i)
            results[i] = (tool_id, result, error)

    # ── Append results in original order; track consecutive errors ────────────
    for i in range(len(tool_calls)):
        if i not in results:
            continue
        tool_id, content, error = results[i]
        state["messages"].append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": content,
        })
        if error:
            state["_consecutive_errors"] = state.get("_consecutive_errors", 0) + 1
        else:
            state["_consecutive_errors"] = 0

    return state


def _should_continue(state: AgentState) -> str:
    """Decide whether to loop (more tool calls) or stop."""
    if state.get("_cancelled") or state["iteration"] >= state["max_iterations"]:
        return END
    last_message = state["messages"][-1]
    if last_message.get("tool_calls"):
        return "tools"
    return END


def _finalize(state: AgentState) -> AgentState:
    """Extract the final text response and log run completion."""
    logger: ConversationLogger | None = state.get("_logger")
    agent_name: str = state.get("_agent_name", "")
    run_id: str = state.get("_run_id", "")
    hit_limit = state["iteration"] >= state["max_iterations"]

    for msg in reversed(state["messages"]):
        if msg.get("role") == "assistant":
            content = msg.get("content")
            if content:
                state["final_response"] = content
                if logger:
                    if hit_limit:
                        logger.run_limit(
                            agent=agent_name,
                            run_id=run_id,
                            iterations=state["iteration"],
                            max_iterations=state["max_iterations"],
                            last_preview=content,
                        )
                    logger.run_end(
                        agent=agent_name,
                        run_id=run_id,
                        response=content,
                        iterations=state["iteration"],
                    )
                return state

    state["final_response"] = "(no response)"
    if logger:
        if hit_limit:
            logger.run_limit(
                agent=agent_name,
                run_id=run_id,
                iterations=state["iteration"],
                max_iterations=state["max_iterations"],
                last_preview=None,
            )
        logger.run_end(
            agent=agent_name,
            run_id=run_id,
            response="(no response)",
            iterations=state["iteration"],
        )
    return state


# ── Graph builder ─────────────────────────────────────────────────────────────

def _build_graph() -> Any:
    """Construct and compile the LangGraph agentic loop."""
    graph = StateGraph(AgentState)
    graph.add_node("llm", _call_llm)
    graph.add_node("tools", _execute_tools)
    graph.add_node("finalize", _finalize)
    graph.set_entry_point("llm")
    graph.add_conditional_edges(
        "llm", _should_continue, {"tools": "tools", END: "finalize"}
    )
    graph.add_edge("tools", "llm")
    graph.add_edge("finalize", END)
    return graph.compile()


_COMPILED_GRAPH: Any = None


def _get_graph() -> Any:
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = _build_graph()
    return _COMPILED_GRAPH


# ── Public interface ──────────────────────────────────────────────────────────

def run_agent(
    cfg: AgentjiConfig,
    agent_name: str,
    prompt: str | list[dict[str, Any]],
    logger: ConversationLogger | None = None,
    run_id: str | None = None,
    run_context: RunContext | None = None,
    history: list[dict[str, Any]] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> str:
    """Run an agent from config against a user prompt and return the response.

    Args:
        cfg: The loaded AgentjiConfig.
        agent_name: Name of the agent to run (must be in cfg.agents).
        prompt: The user's input prompt.
        logger: Optional ConversationLogger. When provided, all LLM calls,
            tool invocations, and results are appended to the log file.
        run_id: Optional identifier for this specific run within a pipeline.
            Defaults to a short UUID.
        run_context: Shared per-pipeline scratch context. Created automatically
            on the first (root) call and passed to all sub-agents.

    Returns:
        The agent's final text response.
    """
    from agentji.mcp_bridge import list_mcp_tools

    agent = cfg.agents[agent_name]
    run_id = run_id or uuid.uuid4().hex[:8]

    # ── RunContext: create on root call, inherit on sub-agent calls ────────
    is_root = run_context is None
    if run_context is None:
        pipeline_id = logger.pipeline_id if logger else uuid.uuid4().hex[:8]
        scratch_dir = Path("./runs") / pipeline_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        run_context = RunContext(pipeline_id, scratch_dir, logger=logger)

    # ── Memory backend (stub — no-op until mem0 integration) ───────────────
    memory = MemoryBackend(cfg.memory)

    # Determine model string (after router translation)
    litellm_kwargs = build_litellm_kwargs(cfg, agent_name)
    model_str = litellm_kwargs.get("model", agent.model)

    if logger:
        logger.run_start(
            agent=agent_name,
            run_id=run_id,
            model=model_str,
            prompt=prompt if isinstance(prompt, str) else f"[multimodal: {len(prompt)} parts]",
        )

    # ── Load skills (tool + prompt) ────────────────────────────────────────
    skill_paths = [s.path for s in cfg.skills]
    all_skill_descriptors = translate_skills(skill_paths) if skill_paths else []
    agent_skill_names = set(agent.skills)

    # Separate tool skills (have scripts.execute) from prompt skills (body-only)
    tool_skills: list[dict[str, Any]] = []
    prompt_skills: list[dict[str, Any]] = []
    for s in all_skill_descriptors:
        if s.get("_prompt_only"):
            if s["function"]["name"] in agent_skill_names:
                prompt_skills.append(s)
        else:
            if s["function"]["name"] in agent_skill_names:
                tool_skills.append(s)

    skill_tool_map = {t["function"]["name"]: t for t in tool_skills}

    # ── Build system prompt (base + prompt-skill bodies) ───────────────────
    system_prompt = agent.system_prompt
    for ps in prompt_skills:
        skill_name = ps["function"]["name"]
        skill_dir = ps["_skill_dir"]
        scripts_dir = Path(skill_dir) / "scripts"
        body = ps.get("_body", "")
        header = f"\n\n---\n## Skill context: {skill_name}\n"
        if scripts_dir.exists():
            header += f"Scripts directory: {scripts_dir.resolve()}\n\n"
        system_prompt = system_prompt + header + body

    # ── Inject scratch directory paragraph ─────────────────────────────────
    scratch_note = (
        f"\n\n---\nRun scratch directory: ./runs/{run_context.run_id}/\n"
        "When your output exceeds what fits comfortably in a single prompt "
        "(roughly 6,000 characters), write it to the scratch directory using "
        "write_file and pass the recipient the file path — not the raw content. "
        "When you receive a file path from another agent, read it using read_file "
        "before processing. The scratch directory persists for the lifetime of this run."
    )
    system_prompt = system_prompt + scratch_note

    # ── Inject outputs instructions ────────────────────────────────────────
    if agent.outputs:
        out_lines = ["\n\nAt the end of your work, save your final output to the run context:"]
        for out in agent.outputs:
            out_lines.append(f"- Key: {out.key} — {out.description}")
        has_write_file = "write_file" in agent.builtins
        if has_write_file:
            paths = ", ".join(
                f"./runs/{run_context.run_id}/{out.key}.md" for out in agent.outputs
            )
            out_lines.append(
                f"\nWrite the full content to {paths} using write_file, "
                "then confirm the path in your response."
            )
        else:
            out_lines.append(
                "\nEnsure your final response contains your complete output — "
                "it will be saved to the run context automatically."
            )
        system_prompt = system_prompt + "\n".join(out_lines)

    # ── Memory inject (root/orchestrator only) ─────────────────────────────
    if is_root:
        system_prompt = memory.inject(system_prompt, prompt)

    # ── Load MCP tools ─────────────────────────────────────────────────────
    mcp_tools: list[dict[str, Any]] = []
    mcp_map: dict[str, MCPConfig] = {}
    mcp_config_by_name = {m.name: m for m in cfg.mcps}
    for mcp_name in agent.mcps:
        mcp_config = mcp_config_by_name[mcp_name]
        discovered = list_mcp_tools(mcp_config)
        for tool in discovered:
            mcp_map[tool["function"]["name"]] = mcp_config
        mcp_tools.extend(discovered)

    # ── Load built-in tools ────────────────────────────────────────────────
    enabled_builtins = [b for b in agent.builtins if b in VALID_BUILTINS]
    builtin_tools = [BUILTIN_SCHEMAS[b] for b in enabled_builtins]

    # ── call_agent tool (orchestrators only) ──────────────────────────────
    allowed_agents = agent.agents  # empty list for non-orchestrators
    call_agent_tools = (
        [_build_call_agent_tool(allowed_agents)] if allowed_agents else []
    )

    all_tools = tool_skills + mcp_tools + builtin_tools + call_agent_tools

    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": prompt},
        ],
        "tools": all_tools,
        "tool_map": skill_tool_map,
        "mcp_map": mcp_map,
        "builtin_set": enabled_builtins,
        "litellm_kwargs": litellm_kwargs,
        "max_iterations": agent.max_iterations,
        "iteration": 0,
        "final_response": "",
        "_agent_name": agent_name,
        "_run_id": run_id,
        "_logger": logger,
        "_cfg": cfg,
        "_allowed_agents": allowed_agents,
        "_run_context": run_context,
        "_stream_callback": None,
        "_all_skills": all_skill_descriptors,
        "_consecutive_errors": 0,
        "_tool_timeout": agent.tool_timeout,
        "_cancel_fn": cancel_fn,
        "_memory": memory,
        "_parallel_agents": agent.parallel_agents,
    }

    graph = _get_graph()
    final_state: AgentState = graph.invoke(initial_state)
    final_response = final_state["final_response"]

    # ── Memory remember (root/orchestrator only) ───────────────────────────
    if is_root:
        memory.remember(run_id, final_response, litellm_kwargs)

    return final_response


def run_agent_streaming(
    cfg: AgentjiConfig,
    agent_name: str,
    prompt: str | list[dict[str, Any]],
    on_token: Callable[[str], None],
    logger: ConversationLogger | None = None,
    run_id: str | None = None,
    run_context: RunContext | None = None,
    history: list[dict[str, Any]] | None = None,
    cancel_fn: Callable[[], bool] | None = None,
) -> str:
    """Run an agent with streaming token delivery via a callback.

    Identical to :func:`run_agent` except that LLM content chunks from the root
    agent are delivered to ``on_token`` as they arrive. Sub-agents called via
    ``call_agent`` never stream — only the root orchestrator streams its final
    synthesis to the caller.

    Args:
        cfg: The loaded AgentjiConfig.
        agent_name: Name of the agent to run.
        prompt: The user's input prompt.
        on_token: Called once per content chunk (text delta) as it streams.
        logger: Optional ConversationLogger.
        run_id: Optional run identifier. Defaults to a short UUID.
        run_context: Shared per-pipeline scratch context. Created automatically
            on the first (root) call and passed to all sub-agents.

    Returns:
        The agent's complete final text response (same as run_agent).
    """
    from agentji.mcp_bridge import list_mcp_tools

    agent = cfg.agents[agent_name]
    run_id = run_id or uuid.uuid4().hex[:8]

    is_root = run_context is None
    if run_context is None:
        pipeline_id = logger.pipeline_id if logger else uuid.uuid4().hex[:8]
        scratch_dir = Path("./runs") / pipeline_id
        scratch_dir.mkdir(parents=True, exist_ok=True)
        run_context = RunContext(pipeline_id, scratch_dir, logger=logger)

    memory = MemoryBackend(cfg.memory)

    litellm_kwargs = build_litellm_kwargs(cfg, agent_name)
    model_str = litellm_kwargs.get("model", agent.model)

    if logger:
        logger.run_start(
            agent=agent_name,
            run_id=run_id,
            model=model_str,
            prompt=prompt if isinstance(prompt, str) else f"[multimodal: {len(prompt)} parts]",
        )

    skill_paths = [s.path for s in cfg.skills]
    all_skill_descriptors = translate_skills(skill_paths) if skill_paths else []
    agent_skill_names = set(agent.skills)

    tool_skills: list[dict[str, Any]] = []
    prompt_skills: list[dict[str, Any]] = []
    for s in all_skill_descriptors:
        if s.get("_prompt_only"):
            if s["function"]["name"] in agent_skill_names:
                prompt_skills.append(s)
        else:
            if s["function"]["name"] in agent_skill_names:
                tool_skills.append(s)

    skill_tool_map = {t["function"]["name"]: t for t in tool_skills}

    system_prompt = agent.system_prompt
    for ps in prompt_skills:
        skill_name = ps["function"]["name"]
        skill_dir = ps["_skill_dir"]
        scripts_dir = Path(skill_dir) / "scripts"
        body = ps.get("_body", "")
        header = f"\n\n---\n## Skill context: {skill_name}\n"
        if scripts_dir.exists():
            header += f"Scripts directory: {scripts_dir.resolve()}\n\n"
        system_prompt = system_prompt + header + body

    scratch_note = (
        f"\n\n---\nRun scratch directory: ./runs/{run_context.run_id}/\n"
        "When your output exceeds what fits comfortably in a single prompt "
        "(roughly 6,000 characters), write it to the scratch directory using "
        "write_file and pass the recipient the file path — not the raw content. "
        "When you receive a file path from another agent, read it using read_file "
        "before processing. The scratch directory persists for the lifetime of this run."
    )
    system_prompt = system_prompt + scratch_note

    if agent.outputs:
        out_lines = ["\n\nAt the end of your work, save your final output to the run context:"]
        for out in agent.outputs:
            out_lines.append(f"- Key: {out.key} — {out.description}")
        has_write_file = "write_file" in agent.builtins
        if has_write_file:
            paths = ", ".join(
                f"./runs/{run_context.run_id}/{out.key}.md" for out in agent.outputs
            )
            out_lines.append(
                f"\nWrite the full content to {paths} using write_file, "
                "then confirm the path in your response."
            )
        else:
            out_lines.append(
                "\nEnsure your final response contains your complete output — "
                "it will be saved to the run context automatically."
            )
        system_prompt = system_prompt + "\n".join(out_lines)

    if is_root:
        system_prompt = memory.inject(system_prompt, prompt)

    mcp_tools: list[dict[str, Any]] = []
    mcp_map: dict[str, MCPConfig] = {}
    mcp_config_by_name = {m.name: m for m in cfg.mcps}
    for mcp_name in agent.mcps:
        mcp_config = mcp_config_by_name[mcp_name]
        discovered = list_mcp_tools(mcp_config)
        for tool in discovered:
            mcp_map[tool["function"]["name"]] = mcp_config
        mcp_tools.extend(discovered)

    enabled_builtins = [b for b in agent.builtins if b in VALID_BUILTINS]
    builtin_tools = [BUILTIN_SCHEMAS[b] for b in enabled_builtins]

    allowed_agents = agent.agents
    call_agent_tools = (
        [_build_call_agent_tool(allowed_agents)] if allowed_agents else []
    )

    all_tools = tool_skills + mcp_tools + builtin_tools + call_agent_tools

    initial_state: AgentState = {
        "messages": [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": prompt},
        ],
        "tools": all_tools,
        "tool_map": skill_tool_map,
        "mcp_map": mcp_map,
        "builtin_set": enabled_builtins,
        "litellm_kwargs": litellm_kwargs,
        "max_iterations": agent.max_iterations,
        "iteration": 0,
        "final_response": "",
        "_agent_name": agent_name,
        "_run_id": run_id,
        "_logger": logger,
        "_cfg": cfg,
        "_allowed_agents": allowed_agents,
        "_run_context": run_context,
        "_stream_callback": on_token,  # only difference from run_agent()
        "_all_skills": all_skill_descriptors,
        "_consecutive_errors": 0,
        "_tool_timeout": agent.tool_timeout,
        "_cancel_fn": cancel_fn,
        "_memory": memory,
        "_parallel_agents": agent.parallel_agents,
    }

    graph = _get_graph()
    final_state: AgentState = graph.invoke(initial_state)
    final_response = final_state["final_response"]

    if is_root:
        memory.remember(run_id, final_response, litellm_kwargs)

    return final_response
