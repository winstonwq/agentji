"""agentji FastAPI server — OpenAI-compatible agent serving endpoint.

Exposes three endpoints:
  POST /v1/chat/completions  — run an agent (streaming or non-streaming)
  GET  /v1/events/{run_id}   — SSE stream of pipeline log events for a run
  GET  /v1/pipeline          — current pipeline topology as JSON

Start via: agentji serve --config agentji.yaml
"""

from __future__ import annotations

import asyncio
import json
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any

_SESSION_IDLE_SECS = 30  # seconds of inactivity before auto-triggering extraction

try:
    from fastapi import FastAPI, File, HTTPException, Request, UploadFile
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "agentji serve requires FastAPI. Install with: pip install agentji[serve]"
    )

from pydantic import BaseModel

import pathlib

from agentji.config import AgentjiConfig
from agentji.logger import ConversationLogger

STUDIO_HTML = pathlib.Path(__file__).parent / "studio" / "index.html"


# ── Module-level state injected by the CLI at startup ─────────────────────────

_cfg: AgentjiConfig | None = None
_logger: ConversationLogger | None = None   # startup logger — provides log_path
_default_agent: str | None = None
_studio_enabled: bool = False               # True only when --studio flag is passed

# SSE event routing: pipeline_id → asyncio.Queue of log event dicts
_event_subscribers: dict[str, asyncio.Queue] = {}
_cancel_events: dict[str, threading.Event] = {}
_event_loop: asyncio.AbstractEventLoop | None = None

_executor = ThreadPoolExecutor(max_workers=8)

# ── Session-level state for improvement extraction ─────────────────────────────
# Maps session_id → accumulated full message list (all turns, user + assistant).
_session_messages: dict[str, list[dict]] = {}
# Maps session_id → whether improvement is enabled for that session.
_session_improve: dict[str, bool] = {}
# Maps session_id → active idle Timer (reset on each new request).
_session_timers: dict[str, threading.Timer] = {}
_session_lock = threading.Lock()


# ── FastAPI application ────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(application: FastAPI):
    global _event_loop
    _event_loop = asyncio.get_event_loop()
    yield


app = FastAPI(
    title="agentji",
    description="OpenAI-compatible agent serving endpoint",
    version="0.1.0",
    lifespan=_lifespan,
)


# ── Request / response models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatCompletionRequest(BaseModel):
    model: str = ""          # ignored — agentji uses agent config
    messages: list[ChatMessage]
    stream: bool = False
    agent: str | None = None         # agentji extension — override default agent
    stateful: bool | None = None     # override config studio.stateful per-request
    improve: bool | None = None      # override config improvement.enabled per-request


# ── Improvement extraction helpers ────────────────────────────────────────────

def _trigger_improvement(session_id: str) -> None:
    """Run improvement extraction for a session, then clean up session state."""
    with _session_lock:
        # Cancel any pending timer
        timer = _session_timers.pop(session_id, None)
        if timer is not None:
            timer.cancel()
        messages = _session_messages.pop(session_id, [])
        _session_improve.pop(session_id, None)

    if not messages or _cfg is None:
        return

    cfg = _cfg
    default_agent = _default_agent or (next(iter(cfg.agents)) if cfg.agents else None)
    if default_agent is None:
        return

    imp_cfg = cfg.improvement
    from agentji.router import build_litellm_kwargs
    litellm_kwargs = build_litellm_kwargs(cfg, default_agent)
    model = imp_cfg.model or litellm_kwargs.get("model", "")
    if not model:
        return

    # Collect skill_refs from config
    import re as _re
    from pathlib import Path as _Path
    skill_refs: list[dict] = []
    for s in cfg.skills:
        path = _Path(s.path)
        skill_md = path / "SKILL.md"
        resolved_name = path.name
        if skill_md.exists():
            text = skill_md.read_text(encoding="utf-8")
            slug_m = _re.search(r"^slug:\s*(.+)$", text, _re.MULTILINE)
            name_m = _re.search(r"^name:\s*(.+)$", text, _re.MULTILINE)
            if slug_m:
                resolved_name = slug_m.group(1).strip()
            elif name_m:
                resolved_name = name_m.group(1).strip()
        skill_refs.append({"name": resolved_name, "path": s.path})

    from agentji.improver import extract_and_save
    _executor.submit(
        extract_and_save,
        messages,
        session_id,
        skill_refs,
        model,
        litellm_kwargs,
        list(imp_cfg.skills),
    )


def _schedule_idle_timer(session_id: str) -> None:
    """(Re)start a _SESSION_IDLE_SECS idle timer for a session."""
    with _session_lock:
        old = _session_timers.pop(session_id, None)
        if old is not None:
            old.cancel()
        timer = threading.Timer(_SESSION_IDLE_SECS, _trigger_improvement, args=(session_id,))
        timer.daemon = True
        timer.start()
        _session_timers[session_id] = timer


# ── Event routing callback ─────────────────────────────────────────────────────

def _dispatch_event(event: dict) -> None:
    """Route a logger event to the matching SSE subscriber queue (thread-safe)."""
    pipeline_id = event.get("pipeline", "")
    q = _event_subscribers.get(pipeline_id)
    if q is not None and _event_loop is not None and not _event_loop.is_closed():
        try:
            _event_loop.call_soon_threadsafe(q.put_nowait, event)
        except Exception:
            pass


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_agent(request_agent: str | None) -> str:
    """Return the agent name to use, with fallback to default/first agent."""
    cfg = _cfg
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server not initialised")
    if request_agent:
        if request_agent not in cfg.agents:
            raise HTTPException(
                status_code=400,
                detail=f"Agent '{request_agent}' not found. Available: {list(cfg.agents)}",
            )
        return request_agent
    if _default_agent and _default_agent in cfg.agents:
        return _default_agent
    return next(iter(cfg.agents))


def _make_request_logger(run_id: str, session_id: str | None = None) -> ConversationLogger | None:
    """Create a per-request logger sharing the serve log file but with a unique pipeline_id."""
    if _logger is None:
        return None
    # Re-use the startup logger's rotation settings; log_path is a property
    # that returns the correct current-day file for rotating loggers.
    if _logger._log_dir and _logger._rotation == "daily":
        return ConversationLogger(
            log_dir=_logger._log_dir,
            prefix=_logger._prefix,
            rotation="daily",
            pipeline_id=run_id,
            event_callback=_dispatch_event,
            session_id=session_id,
        )
    return ConversationLogger(
        _logger.log_path,
        pipeline_id=run_id,
        event_callback=_dispatch_event,
        session_id=session_id,
    )


def _completion_object(run_id: str, agent_name: str, content: str) -> dict:
    return {
        "id": f"chatcmpl-{run_id}",
        "object": "chat.completion",
        "model": agent_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


def _chunk_object(run_id: str, agent_name: str, token: str, finish: bool = False) -> str:
    chunk = {
        "id": f"chatcmpl-{run_id}",
        "object": "chat.completion.chunk",
        "model": agent_name,
        "choices": [{
            "index": 0,
            "delta": {} if finish else {"content": token},
            "finish_reason": "stop" if finish else None,
        }],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ── POST /v1/chat/completions ─────────────────────────────────────────────────

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, raw: Request) -> Any:
    """OpenAI-compatible chat completions endpoint."""
    from agentji.loop import run_agent, run_agent_streaming

    cfg = _cfg
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server not initialised")

    # Extract prompt from last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found in messages list")
    prompt = user_messages[-1].content

    agent_name = _resolve_agent(request.agent)
    run_id = uuid.uuid4().hex[:8]
    session_id = raw.headers.get("X-Agentji-Session-Id") or None
    request_logger = _make_request_logger(run_id, session_id=session_id)

    response_headers = {"X-Agentji-Run-Id": run_id}

    # ── Session history ────────────────────────────────────────────────────
    # Effective stateful: request override → config default
    effective_stateful = request.stateful if request.stateful is not None else cfg.studio.stateful
    history: list[dict] | None = None
    if effective_stateful and len(request.messages) > 1:
        prior = [{"role": m.role, "content": m.content} for m in request.messages[:-1]]
        # keep last max_turns pairs (user + assistant = 2 messages per turn)
        prior = prior[-(cfg.studio.max_turns * 2):]
        history = prior or None

    # ── Improvement session tracking ───────────────────────────────────────
    effective_improve = (
        request.improve if request.improve is not None else cfg.improvement.enabled
    )
    if effective_improve and session_id:
        with _session_lock:
            # Register improve intent for this session
            _session_improve[session_id] = True
            # Accumulate all messages sent so far (history already contains prior turns)
            if session_id not in _session_messages:
                _session_messages[session_id] = []
            # Add all prior messages not yet tracked
            current_tracked = len(_session_messages[session_id])
            all_messages = [{"role": m.role, "content": m.content} for m in request.messages]
            if current_tracked < len(all_messages):
                _session_messages[session_id] = all_messages
        _schedule_idle_timer(session_id)

    cancel_evt = threading.Event()
    _cancel_events[run_id] = cancel_evt

    if not request.stream:
        # ── Non-streaming ─────────────────────────────────────────────────
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _executor,
                lambda: run_agent(cfg, agent_name, prompt, logger=request_logger, run_id=run_id, history=history, cancel_fn=cancel_evt.is_set),
            )
        finally:
            _cancel_events.pop(run_id, None)
        # Append assistant reply to session messages
        if effective_improve and session_id and result:
            with _session_lock:
                if session_id in _session_messages:
                    _session_messages[session_id].append(
                        {"role": "assistant", "content": result}
                    )
        return JSONResponse(
            content=_completion_object(run_id, agent_name, result),
            headers=response_headers,
        )

    # ── Streaming ─────────────────────────────────────────────────────────
    token_queue: queue.Queue = queue.Queue()
    _streamed_tokens: list[str] = []

    def _on_token(token: str) -> None:
        token_queue.put(token)
        if effective_improve and session_id:
            _streamed_tokens.append(token)

    def _run_in_thread() -> None:
        try:
            run_agent_streaming(
                cfg,
                agent_name,
                prompt,
                on_token=_on_token,
                logger=request_logger,
                run_id=run_id,
                history=history,
                cancel_fn=cancel_evt.is_set,
            )
        except Exception as exc:
            token_queue.put(exc)
        finally:
            _cancel_events.pop(run_id, None)
            # Append completed assistant reply to session messages
            if effective_improve and session_id and _streamed_tokens:
                full_reply = "".join(_streamed_tokens)
                with _session_lock:
                    if session_id in _session_messages:
                        _session_messages[session_id].append(
                            {"role": "assistant", "content": full_reply}
                        )
            token_queue.put(None)  # sentinel: done

    _executor.submit(_run_in_thread)

    def _generate():
        while True:
            item = token_queue.get()
            if item is None:
                yield _chunk_object(run_id, agent_name, "", finish=True)
                yield "data: [DONE]\n\n"
                break
            elif isinstance(item, Exception):
                err_chunk = {
                    "id": f"chatcmpl-{run_id}",
                    "object": "chat.completion.chunk",
                    "model": agent_name,
                    "choices": [{"index": 0, "delta": {"content": f"[Error: {item}]"}, "finish_reason": "stop"}],
                }
                yield f"data: {json.dumps(err_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                break
            else:
                yield _chunk_object(run_id, agent_name, item)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers=response_headers,
    )


# ── GET /v1/events/{run_id} ───────────────────────────────────────────────────

@app.get("/v1/events/{run_id}")
async def events_stream(run_id: str) -> StreamingResponse:
    """SSE stream of pipeline log events for a specific run.

    Open this before or immediately after POST /v1/chat/completions.
    Streams JSONL log events as they arrive. Closes on root run_end or 5-min timeout.
    """
    q: asyncio.Queue = asyncio.Queue()
    _event_subscribers[run_id] = q

    async def _generate():
        loop = asyncio.get_event_loop()
        deadline = loop.time() + 7200  # 2-hour hard deadline
        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    yield "data: [DONE]\n\n"
                    return
                try:
                    event = await asyncio.wait_for(q.get(), timeout=min(30.0, remaining))
                except asyncio.TimeoutError:
                    if loop.time() >= deadline:
                        yield "data: [DONE]\n\n"
                        return
                    # Send a keepalive comment so long LLM calls don't drop the connection
                    yield ": heartbeat\n\n"
                    continue

                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                # Close on root run_end (sub-agent run_ids contain ":")
                if (
                    event.get("event") == "run_end"
                    and ":" not in event.get("run_id", "?")
                ):
                    yield "data: [DONE]\n\n"
                    return
        finally:
            _event_subscribers.pop(run_id, None)

    return StreamingResponse(_generate(), media_type="text/event-stream")


# ── POST /v1/cancel/{run_id} ──────────────────────────────────────────────────

@app.post("/v1/cancel/{run_id}")
async def cancel_run(run_id: str) -> dict:
    """Signal the running pipeline to stop at the next iteration boundary."""
    evt = _cancel_events.get(run_id)
    if evt is None:
        raise HTTPException(status_code=404, detail=f"No active run '{run_id}'")
    evt.set()
    return {"cancelled": run_id}


# ── GET / — serve studio ─────────────────────────────────────────────────

@app.get("/")
async def serve_studio():
    if _studio_enabled:
        # Custom UI overrides the built-in Studio when configured.
        if _cfg and _cfg.studio.custom_ui:
            custom_path = pathlib.Path(_cfg.studio.custom_ui)
            if not custom_path.is_absolute():
                custom_path = pathlib.Path.cwd() / custom_path
            if custom_path.exists():
                return FileResponse(custom_path)
        if STUDIO_HTML.exists():
            return FileResponse(STUDIO_HTML)
    return JSONResponse(
        content={
            "message": (
                "agentji serve running. "
                "Studio UI is disabled. Start with --studio to enable the browser interface."
                if not _studio_enabled
                else "agentji serve running."
            )
        }
    )


# ── GET /v1/pipeline ──────────────────────────────────────────────────────────

@app.get("/v1/pipeline")
async def pipeline_topology() -> dict:
    """Return the current pipeline topology as JSON."""
    import re

    cfg = _cfg
    if cfg is None:
        raise HTTPException(status_code=503, detail="Server not initialised")

    agents_info: dict[str, Any] = {}
    for name, agent in cfg.agents.items():
        agents_info[name] = {
            "model": agent.model,
            "skills": list(agent.skills),
            "mcps": list(agent.mcps),
            "builtins": list(agent.builtins),
            "sub_agents": list(agent.agents),
            "output_format": getattr(agent, "output_format", "text"),
            "accepted_inputs": getattr(agent, "accepted_inputs", ["text"]),
            "inputs": [{"key": i.key, "description": i.description} for i in agent.inputs],
            "outputs": [{"key": o.key, "description": o.description} for o in agent.outputs],
        }

    # Resolve skill names (slug > name > folder name) and include paths for UI
    skill_refs: list[dict] = []
    for s in cfg.skills:
        path = Path(s.path)
        skill_md = path / "SKILL.md"
        resolved_name = path.name
        if skill_md.exists():
            text = skill_md.read_text(encoding="utf-8")
            slug_m = re.search(r"^slug:\s*(.+)$", text, re.MULTILINE)
            name_m = re.search(r"^name:\s*(.+)$", text, re.MULTILINE)
            if slug_m:
                resolved_name = slug_m.group(1).strip()
            elif name_m:
                resolved_name = name_m.group(1).strip()
        skill_refs.append({"name": resolved_name, "path": s.path})

    default = _default_agent or (next(iter(cfg.agents)) if cfg.agents else None)

    return {
        "agents": agents_info,
        "default_agent": default,
        "mcps": [m.name for m in cfg.mcps],
        "skills": [r["name"] for r in skill_refs],   # backward compat
        "skill_refs": skill_refs,                      # includes paths for source inference
        "stateful": cfg.studio.stateful,
        "improvement_enabled": cfg.improvement.enabled,
    }


# ── POST /v1/sessions/{session_id}/end ───────────────────────────────────────

@app.post("/v1/sessions/{session_id}/end")
async def end_session(session_id: str) -> dict:
    """Signal that a session has ended and trigger improvement extraction.

    Called automatically by the Studio UI on tab close (beforeunload).
    API callers can invoke this explicitly after a conversation is complete.
    Returns immediately; extraction runs in a background thread.
    """
    with _session_lock:
        should_extract = _session_improve.get(session_id, False)
        has_messages = bool(_session_messages.get(session_id))

    if should_extract and has_messages:
        # Run extraction in the thread pool; don't block the response
        _executor.submit(_trigger_improvement, session_id)
        return {"session_id": session_id, "status": "extraction_scheduled"}

    # Clean up even if no extraction needed
    with _session_lock:
        _session_messages.pop(session_id, None)
        _session_improve.pop(session_id, None)
        timer = _session_timers.pop(session_id, None)
    if timer is not None:
        timer.cancel()
    return {"session_id": session_id, "status": "ended"}


# ── POST /v1/files/upload — accept file uploads for multimodal input ─────────

@app.post("/v1/files/upload")
async def upload_file(file: UploadFile) -> dict:
    """Upload a file (image, audio, etc.) and receive a local path reference.

    The returned path can be included in a multimodal ChatMessage content list
    as an image_url content block, or passed as a call_agent attachment.
    """
    upload_dir = Path(".agentji/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload").suffix or ".bin"
    fname = f"{uuid.uuid4().hex[:8]}{suffix}"
    dest = upload_dir / fname
    content = await file.read()
    dest.write_bytes(content)
    return {"path": str(dest), "filename": fname}


# ── GET /v1/files/{filepath} — serve local files as downloads ────────────────

@app.get("/v1/files/{filepath:path}")
async def download_file(filepath: str):
    """Serve a local file as a download. Path is relative to the server working directory."""
    import os
    cwd = Path.cwd()
    target = (cwd / filepath).resolve()
    # Prevent path traversal
    try:
        target.relative_to(cwd)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    return FileResponse(
        path=str(target),
        filename=target.name,
        headers={"Content-Disposition": f'attachment; filename="{target.name}"'},
    )


# ── GET /v1/media/{filepath} — serve files inline for Studio rendering ────────

@app.get("/v1/media/{filepath:path}")
async def serve_media(filepath: str):
    """Serve a local file inline (for image rendering in Studio). Path is relative to CWD."""
    import mimetypes as _mimetypes
    cwd = Path.cwd()
    target = (cwd / filepath).resolve()
    try:
        target.relative_to(cwd)
    except ValueError:
        raise HTTPException(status_code=403, detail="Access denied")
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="Not a file")
    media_type, _ = _mimetypes.guess_type(str(target))
    return FileResponse(path=str(target), media_type=media_type or "application/octet-stream")


# ── Server launcher (used by CLI) ─────────────────────────────────────────────

def start(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    """Start the uvicorn server. Called by the CLI after injecting module globals."""
    uvicorn.run(
        "agentji.server:app",
        host=host,
        port=port,
        reload=reload,
    )
