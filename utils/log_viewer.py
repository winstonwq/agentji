#!/usr/bin/env python3
"""agentji log viewer — renders a JSONL session log as a Slack-like web UI.

Usage:
    python utils/log_viewer.py logs/run.jsonl
    python utils/log_viewer.py logs/run.jsonl --port 8765
    python utils/log_viewer.py logs/run.jsonl --out viewer.html   # write file, don't serve
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import sys
import tempfile
import threading
import webbrowser
from datetime import datetime, timezone
from pathlib import Path


# ── Colour palette (one per agent, cycles) ────────────────────────────────────

_AGENT_COLOURS = [
    ("#4f8ef7", "#1a2d52"),  # blue
    ("#a78bfa", "#2d1f52"),  # violet
    ("#34d399", "#1a3d30"),  # green
    ("#fb923c", "#3d2410"),  # orange
    ("#f472b6", "#3d1428"),  # pink
    ("#38bdf8", "#0f2d3d"),  # sky
    ("#facc15", "#3d3010"),  # yellow
]


def _agent_colour(name: str, registry: dict[str, tuple[str, str]]) -> tuple[str, str]:
    if name not in registry:
        idx = len(registry) % len(_AGENT_COLOURS)
        registry[name] = _AGENT_COLOURS[idx]
    return registry[name]


# ── Log parsing ────────────────────────────────────────────────────────────────

def _parse_log(path: Path) -> list[dict]:
    events = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {i}: {exc}") from exc
    if not events:
        raise ValueError("Log file is empty.")
    return events


def _fmt_ts(ts_str: str) -> str:
    try:
        dt = datetime.fromisoformat(ts_str)
        local = dt.astimezone()
        return local.strftime("%H:%M:%S")
    except Exception:
        return ts_str


def _fmt_args(args: dict | None) -> str:
    if not args:
        return ""
    try:
        return json.dumps(args, ensure_ascii=False, indent=2)
    except Exception:
        return str(args)


def _escape(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )


# ── HTML rendering ─────────────────────────────────────────────────────────────

_CSS = """
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #1a1d21;
    color: #d1d2d3;
    display: flex;
    height: 100vh;
    overflow: hidden;
}

/* Sidebar */
#sidebar {
    width: 240px;
    min-width: 200px;
    background: #19171d;
    border-right: 1px solid #2a2a2e;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    flex-shrink: 0;
}
#sidebar h1 {
    font-size: 15px;
    font-weight: 700;
    padding: 18px 16px 8px;
    color: #fff;
    letter-spacing: 0.01em;
}
.sidebar-section {
    padding: 8px 0 4px;
}
.sidebar-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a7f8a;
    padding: 0 16px 6px;
}
.sidebar-item {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 16px;
    cursor: pointer;
    border-radius: 4px;
    margin: 0 6px;
    font-size: 13px;
    color: #c9ccd0;
    transition: background 0.1s;
}
.sidebar-item:hover, .sidebar-item.active {
    background: #2a2d34;
    color: #fff;
}
.agent-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sidebar-meta {
    padding: 12px 16px;
    border-top: 1px solid #2a2a2e;
    margin-top: auto;
    font-size: 11px;
    color: #7a7f8a;
    line-height: 1.6;
}

/* Main */
#main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
}
#header {
    padding: 14px 20px;
    border-bottom: 1px solid #2a2a2e;
    background: #1a1d21;
    display: flex;
    align-items: center;
    gap: 10px;
}
#header h2 { font-size: 15px; font-weight: 700; color: #fff; }
#header .subtitle { font-size: 12px; color: #7a7f8a; margin-left: 4px; }

#feed {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px 32px;
}

/* Messages */
.msg {
    display: flex;
    gap: 10px;
    margin-bottom: 4px;
    align-items: flex-start;
}
.msg.compact { margin-bottom: 1px; }
.msg.with-gap { margin-top: 14px; }

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 2px;
    color: #fff;
}
.avatar.hidden { visibility: hidden; }

.bubble {
    flex: 1;
    max-width: 820px;
}
.bubble-header {
    display: flex;
    align-items: baseline;
    gap: 8px;
    margin-bottom: 3px;
}
.bubble-header .agent-name {
    font-size: 13px;
    font-weight: 700;
    color: #fff;
}
.bubble-header .ts {
    font-size: 11px;
    color: #7a7f8a;
}
.bubble-header .model-badge {
    font-size: 10px;
    color: #7a7f8a;
    background: #2a2d34;
    padding: 1px 6px;
    border-radius: 10px;
}

.text-content {
    font-size: 13.5px;
    line-height: 1.55;
    color: #d1d2d3;
    white-space: pre-wrap;
    word-break: break-word;
}

/* Tool call card */
.tool-card {
    border: 1px solid #2e3138;
    border-radius: 8px;
    overflow: hidden;
    margin: 4px 0;
    max-width: 700px;
    background: #222529;
}
.tool-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    cursor: pointer;
    user-select: none;
    background: #24272d;
    border-bottom: 1px solid #2e3138;
    transition: background 0.1s;
}
.tool-header:hover { background: #2a2d34; }
.tool-icon { font-size: 13px; }
.tool-name { font-size: 12px; font-weight: 600; color: #a0a8b8; }
.tool-type-badge {
    font-size: 10px;
    padding: 1px 6px;
    border-radius: 10px;
    background: #2e3138;
    color: #7a7f8a;
}
.tool-status {
    margin-left: auto;
    font-size: 11px;
    color: #7a7f8a;
}
.tool-status.ok { color: #34d399; }
.tool-status.err { color: #f87171; }
.chevron {
    font-size: 10px;
    color: #7a7f8a;
    transition: transform 0.15s;
    margin-left: 4px;
}
.tool-body {
    padding: 10px 12px;
}
.tool-body.hidden { display: none; }
.tool-section-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #7a7f8a;
    margin-bottom: 4px;
}
.tool-code {
    font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
    font-size: 11.5px;
    color: #c9ccd0;
    background: #1a1d21;
    border-radius: 4px;
    padding: 8px 10px;
    white-space: pre-wrap;
    word-break: break-all;
    max-height: 200px;
    overflow-y: auto;
    line-height: 1.5;
    border: 1px solid #2e3138;
}
.tool-result {
    margin-top: 8px;
}
.tool-result .tool-code.error-result {
    color: #f87171;
    background: #2d1414;
    border-color: #5c2020;
}

/* Events */
.event-row {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 3px 0 3px 42px;
    font-size: 11.5px;
    color: #7a7f8a;
}
.event-row .ev-icon { font-size: 12px; }
.event-pill {
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.ev-start { background: #1e3a2a; color: #34d399; }
.ev-end   { background: #1e2a3a; color: #60a5fa; }
.ev-llm   { background: #2a2030; color: #a78bfa; }

/* Agent call nesting */
.nested-agent {
    margin: 6px 0 6px 42px;
    border-left: 2px solid #2e3138;
    padding-left: 12px;
}

/* Separator */
.day-sep {
    text-align: center;
    font-size: 11px;
    color: #7a7f8a;
    margin: 16px 0;
    position: relative;
}
.day-sep::before {
    content: "";
    position: absolute;
    left: 0; right: 0; top: 50%;
    border-top: 1px solid #2e3138;
}
.day-sep span {
    position: relative;
    background: #1a1d21;
    padding: 0 12px;
}

/* Filter bar */
#filter-bar {
    padding: 8px 20px;
    border-bottom: 1px solid #2a2a2e;
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
}
.filter-btn {
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    border: 1px solid #2e3138;
    background: transparent;
    color: #7a7f8a;
    cursor: pointer;
    transition: all 0.1s;
}
.filter-btn:hover { background: #2a2d34; color: #fff; }
.filter-btn.active { background: #2a2d34; color: #fff; border-color: #4f5566; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #3a3d44; border-radius: 3px; }
"""

_JS = """
function toggleTool(id) {
    const body = document.getElementById('body-' + id);
    const chev = document.getElementById('chev-' + id);
    body.classList.toggle('hidden');
    chev.textContent = body.classList.contains('hidden') ? '▶' : '▼';
}

function filterAgent(name) {
    const btns = document.querySelectorAll('.filter-btn');
    btns.forEach(b => b.classList.remove('active'));
    const clicked = document.getElementById('fb-' + name);
    if (clicked) clicked.classList.add('active');

    const rows = document.querySelectorAll('.agent-section');
    rows.forEach(r => {
        if (name === '__all__' || r.dataset.agent === name) {
            r.style.display = '';
        } else {
            r.style.display = 'none';
        }
    });
}

document.querySelectorAll('.sidebar-item').forEach(item => {
    item.addEventListener('click', () => {
        document.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
        item.classList.add('active');
        filterAgent(item.dataset.agent);
    });
});
"""


def _build_html(events: list[dict], log_path: Path) -> str:
    colour_registry: dict[str, tuple[str, str]] = {}
    agents: list[str] = []
    for e in events:
        a = e.get("agent", "unknown")
        if a not in agents:
            agents.append(a)
        _agent_colour(a, colour_registry)

    pipeline_id = events[0].get("pipeline", "?") if events else "?"
    start_ts = events[0].get("ts", "") if events else ""
    n_events = len(events)
    n_tool_calls = sum(1 for e in events if e.get("event") == "tool_call")
    n_errors = sum(1 for e in events if e.get("event") == "tool_result" and e.get("error"))

    # ── Sidebar ────────────────────────────────────────────────────────────────
    sidebar_items = []
    sidebar_items.append(
        '<div class="sidebar-item active" data-agent="__all__" id="sb-__all__">'
        '<span style="font-size:14px">🗂</span> All agents</div>'
    )
    for a in agents:
        fg, bg = colour_registry[a]
        initials = a[:2].upper()
        sidebar_items.append(
            f'<div class="sidebar-item" data-agent="{_escape(a)}" id="sb-{_escape(a)}">'
            f'<span class="agent-dot" style="background:{fg}"></span>'
            f'{_escape(a)}</div>'
        )

    # ── Filter buttons ─────────────────────────────────────────────────────────
    filter_btns = ['<button class="filter-btn active" id="fb-__all__" onclick="filterAgent(\'__all__\')">All</button>']
    for a in agents:
        filter_btns.append(
            f'<button class="filter-btn" id="fb-{_escape(a)}" onclick="filterAgent(\'{_escape(a)}\')">'
            f'{_escape(a)}</button>'
        )

    # ── Feed ───────────────────────────────────────────────────────────────────
    feed_html = _render_feed(events, colour_registry)

    meta_lines = [
        f"Pipeline: {pipeline_id}",
        f"Started: {_fmt_ts(start_ts)}",
        f"Events: {n_events}",
        f"Tool calls: {n_tool_calls}",
        f"Errors: {n_errors}",
    ]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>agentji log — {_escape(log_path.name)}</title>
<style>{_CSS}</style>
</head>
<body>

<div id="sidebar">
  <h1>agentji</h1>
  <div class="sidebar-section">
    <div class="sidebar-label">Agents</div>
    {''.join(sidebar_items)}
  </div>
  <div class="sidebar-meta">
    {'<br>'.join(_escape(l) for l in meta_lines)}
  </div>
</div>

<div id="main">
  <div id="header">
    <h2>{_escape(log_path.name)}</h2>
    <span class="subtitle">— pipeline {_escape(pipeline_id)}</span>
  </div>
  <div id="filter-bar">{''.join(filter_btns)}</div>
  <div id="feed">{feed_html}</div>
</div>

<script>{_JS}</script>
</body>
</html>"""


def _render_feed(events: list[dict], colour_registry: dict) -> str:
    parts = []
    tool_counter = [0]
    prev_agent = None
    prev_event = None

    def next_tool_id() -> str:
        tool_counter[0] += 1
        return f"t{tool_counter[0]}"

    # Group tool_call + tool_result pairs
    result_map: dict[str, dict] = {}
    pending_call_idx: dict[str, int] = {}
    for i, e in enumerate(events):
        ev = e.get("event")
        if ev == "tool_call":
            key = f"{e.get('agent')}:{e.get('run_id')}:{e.get('tool')}"
            pending_call_idx[key] = i
        elif ev == "tool_result":
            key = f"{e.get('agent')}:{e.get('run_id')}:{e.get('tool')}"
            if key in pending_call_idx:
                result_map[pending_call_idx[key]] = e
                del pending_call_idx[key]

    skip_indices: set[int] = set(
        i for e in result_map.values()
        for i, ev in enumerate(events) if ev is e
    )

    depth: dict[str, int] = {}

    for idx, event in enumerate(events):
        if idx in skip_indices:
            continue

        ev = event.get("event")
        agent = event.get("agent", "unknown")
        ts = _fmt_ts(event.get("ts", ""))
        fg, bg = colour_registry.get(agent, ("#888", "#222"))
        initials = agent[:2].upper()

        gap_class = "with-gap" if agent != prev_agent else ""
        show_header = agent != prev_agent or prev_event in ("run_start", "run_end", None)

        section_agent = _escape(agent)

        if ev == "run_start":
            model = event.get("model", "")
            prompt = event.get("prompt", "")
            nested = ":" in event.get("run_id", "")

            parts.append(f'<div class="agent-section" data-agent="{section_agent}">')
            if nested:
                parts.append('<div class="nested-agent">')

            parts.append(
                f'<div class="event-row">'
                f'<span class="event-pill ev-start">run start</span>'
                f'<span style="color:{fg};font-weight:600">{_escape(agent)}</span>'
                f'<span class="model-badge" style="background:{bg};color:{fg}">{_escape(model)}</span>'
                f'<span style="color:#7a7f8a;font-size:11px">{ts}</span>'
                f'</div>'
            )
            # Show the prompt as a user bubble
            parts.append(
                f'<div class="msg with-gap">'
                f'<div class="avatar" style="background:#3a3d44;color:#c9ccd0">U</div>'
                f'<div class="bubble">'
                f'<div class="bubble-header"><span class="agent-name" style="color:#c9ccd0">user</span>'
                f'<span class="ts">{ts}</span></div>'
                f'<div class="text-content">{_escape(prompt)}</div>'
                f'</div></div>'
            )

        elif ev == "llm_response":
            content = event.get("content_preview", "")
            tool_calls = event.get("tool_calls", []) or []
            iteration = event.get("iteration", "")

            if show_header:
                parts.append(
                    f'<div class="msg {gap_class}">'
                    f'<div class="avatar" style="background:{bg};color:{fg}">{_escape(initials)}</div>'
                    f'<div class="bubble">'
                    f'<div class="bubble-header">'
                    f'<span class="agent-name" style="color:{fg}">{_escape(agent)}</span>'
                    f'<span class="ts">{ts}</span>'
                    f'<span class="event-pill ev-llm" style="font-size:10px">iter {iteration}</span>'
                    f'</div>'
                )
            else:
                parts.append(
                    f'<div class="msg compact">'
                    f'<div class="avatar hidden" style="background:{bg};color:{fg}">{_escape(initials)}</div>'
                    f'<div class="bubble">'
                )

            if content:
                parts.append(f'<div class="text-content">{_escape(content)}</div>')

            for tc in tool_calls:
                tc_name = tc.get("name", "?")
                tc_preview = tc.get("args_preview", "")
                tid = next_tool_id()
                parts.append(
                    f'<div class="tool-card">'
                    f'<div class="tool-header" onclick="toggleTool(\'{tid}\')">'
                    f'<span class="tool-icon">⚙</span>'
                    f'<span class="tool-name">{_escape(tc_name)}</span>'
                    f'<span class="tool-status" id="status-{tid}">queued</span>'
                    f'<span class="chevron" id="chev-{tid}">▼</span>'
                    f'</div>'
                    f'<div class="tool-body" id="body-{tid}">'
                    f'<div class="tool-section-label">Arguments preview</div>'
                    f'<div class="tool-code">{_escape(tc_preview)}</div>'
                    f'</div></div>'
                )

            parts.append('</div></div>')

        elif ev == "tool_call":
            result = result_map.get(idx)
            args = event.get("args", {})
            tool_name = event.get("tool", "?")
            tool_type = event.get("tool_type", "")
            tid = next_tool_id()

            is_err = result and result.get("error")
            result_preview = result.get("result_preview", "") if result else ""
            result_chars = result.get("result_chars", 0) if result else 0
            result_ts = _fmt_ts(result.get("ts", "")) if result else ""

            status_class = "err" if is_err else "ok"
            status_text = "error" if is_err else f"ok · {result_chars} chars"

            parts.append(
                f'<div class="agent-section" data-agent="{section_agent}">'
                f'<div style="padding-left:42px;margin:3px 0">'
                f'<div class="tool-card">'
                f'<div class="tool-header" onclick="toggleTool(\'{tid}\')">'
                f'<span class="tool-icon">{"🔴" if is_err else "🔧"}</span>'
                f'<span class="tool-name">{_escape(tool_name)}</span>'
                f'<span class="tool-type-badge">{_escape(tool_type)}</span>'
                f'<span class="tool-status {status_class}">{_escape(status_text)}</span>'
                f'<span class="chevron" id="chev-{tid}">▶</span>'
                f'</div>'
                f'<div class="tool-body hidden" id="body-{tid}">'
            )

            if args:
                parts.append(
                    f'<div class="tool-section-label">Arguments</div>'
                    f'<div class="tool-code">{_escape(_fmt_args(args))}</div>'
                )

            if result_preview:
                err_class = " error-result" if is_err else ""
                parts.append(
                    f'<div class="tool-result">'
                    f'<div class="tool-section-label">Result {result_ts}</div>'
                    f'<div class="tool-code{err_class}">{_escape(result_preview)}</div>'
                    f'</div>'
                )

            parts.append('</div></div></div></div>')

        elif ev == "run_end":
            response = event.get("response_preview", "")
            iters = event.get("iterations", "?")
            nested = ":" in event.get("run_id", "")

            if response:
                parts.append(
                    f'<div class="agent-section" data-agent="{section_agent}">'
                    f'<div class="msg with-gap">'
                    f'<div class="avatar" style="background:{bg};color:{fg}">{_escape(initials)}</div>'
                    f'<div class="bubble">'
                    f'<div class="bubble-header">'
                    f'<span class="agent-name" style="color:{fg}">{_escape(agent)}</span>'
                    f'<span class="ts">{ts}</span>'
                    f'</div>'
                    f'<div class="text-content">{_escape(response)}</div>'
                    f'</div></div></div>'
                )

            parts.append(
                f'<div class="agent-section" data-agent="{section_agent}">'
                f'<div class="event-row">'
                f'<span class="event-pill ev-end">run end</span>'
                f'<span style="color:{fg};font-weight:600">{_escape(agent)}</span>'
                f'<span style="color:#7a7f8a">{iters} iterations · {ts}</span>'
                f'</div></div>'
            )

            if nested:
                parts.append('</div>')  # close nested-agent

        elif ev == "llm_call":
            pass  # shown implicitly via llm_response

        prev_agent = agent
        prev_event = ev

    return "\n".join(parts)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="agentji log viewer")
    parser.add_argument("log", help="Path to a .jsonl log file")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port (default 8765)")
    parser.add_argument("--out", help="Write HTML to this file instead of serving")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    events = _parse_log(log_path)
    html = _build_html(events, log_path)

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(html, encoding="utf-8")
        print(f"Written to {out_path}")
        return

    # Serve from a temp file
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", delete=False, encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()
    tmp_path = Path(tmp.name)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(tmp_path.read_bytes())

        def log_message(self, fmt, *a):
            pass  # silence request logs

    url = f"http://localhost:{args.port}"
    print(f"Serving log viewer at {url}  (Ctrl+C to stop)")

    def open_browser():
        import time
        time.sleep(0.4)
        webbrowser.open(url)

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        with http.server.HTTPServer(("", args.port), Handler) as srv:
            srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
