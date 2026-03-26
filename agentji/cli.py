"""agentji CLI — entry point for the `agentji` command.

Commands:
  agentji init       Scaffold a new agentji project in the current directory.
  agentji run        Run an agent from a config file with a prompt.
  agentji logs       Summarize an agentji conversation log (JSONL).
  agentji --help     Show help.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

app = typer.Typer(
    name="agentji",
    help="Universal configuration and execution layer for AI agents.",
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)


# ── Starter config template ───────────────────────────────────────────────────

_STARTER_CONFIG = """\
# agentji.yaml — starter configuration
# Run:   agentji run --agent assistant --prompt "hello"
# Serve: agentji serve --config agentji.yaml --studio
version: "1"

providers:
  openai:
    api_key: ${OPENAI_API_KEY}
  # qwen:
  #   api_key: ${DASHSCOPE_API_KEY}
  #   base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

agents:
  assistant:
    model: openai/gpt-4o-mini
    system_prompt: "You are a helpful assistant."
    max_iterations: 10

# Session options (all optional — shown here with defaults)
# studio:
#   stateful: true      # maintain conversation history across turns
#   max_turns: 20       # max prior turns to include

# Skill improvement extraction (optional — disabled by default)
# improvement:
#   enabled: true
#   model: null         # null = use the default agent's model
#   skills: []          # empty = all loaded skills
"""

_STARTER_ENV = """\
# Copy from .env.example — fill in your API keys
OPENAI_API_KEY=sk-...
# DASHSCOPE_API_KEY=sk-...
# MOONSHOT_API_KEY=sk-...
"""

_BUNDLED_SKILLS_SRC = Path(__file__).parent.parent / "skills"


# ── init command ──────────────────────────────────────────────────────────────

@app.command()
def init(
    directory: Optional[str] = typer.Argument(
        None,
        help="Directory to initialize. Defaults to the current directory.",
    ),
) -> None:
    """Scaffold a new agentji project with a starter config and example skills."""
    target = Path(directory) if directory else Path.cwd()
    target.mkdir(parents=True, exist_ok=True)

    config_path = target / "agentji.yaml"
    env_path = target / ".env"
    skills_target = target / "skills"

    # ── Write starter agentji.yaml ─────────────────────────────────────────
    if config_path.exists():
        console.print(f"[yellow]agentji.yaml already exists — skipping.[/yellow]")
    else:
        config_path.write_text(_STARTER_CONFIG, encoding="utf-8")
        console.print(f"[green]Created[/green] {config_path}")

    # ── Write .env ─────────────────────────────────────────────────────────
    if env_path.exists():
        console.print(f"[yellow].env already exists — skipping.[/yellow]")
    else:
        env_path.write_text(_STARTER_ENV, encoding="utf-8")
        console.print(f"[green]Created[/green] {env_path}")

    # ── Copy agentji meta-skill ────────────────────────────────────────────
    agentji_skill_src = _BUNDLED_SKILLS_SRC / "agentji"
    if agentji_skill_src.exists():
        skills_target.mkdir(exist_ok=True)
        dest = skills_target / "agentji"
        if dest.exists():
            console.print(f"[yellow]skills/agentji already exists — skipping.[/yellow]")
        else:
            shutil.copytree(agentji_skill_src, dest)
            console.print(f"[green]Copied[/green] skills/agentji")

    console.print()
    console.print(Panel(
        "[bold]Next steps:[/bold]\n"
        "1. Edit [cyan]agentji.yaml[/cyan] and set your model\n"
        "2. Add your API key to [cyan].env[/cyan] (or export it)\n"
        "3. Run: [bold cyan]agentji run --agent assistant --prompt \"hello\"[/bold cyan]",
        title="agentji init complete",
        border_style="green",
    ))


# ── run command ───────────────────────────────────────────────────────────────

@app.command()
def run(
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Name of the agent to run (must be defined in the config file).",
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        "-p",
        help="User prompt to send to the agent.",
    ),
    config: str = typer.Option(
        "agentji.yaml",
        "--config",
        "-c",
        help="Path to the agentji.yaml config file.",
    ),
    log_dir: Optional[str] = typer.Option(
        None,
        "--log-dir",
        "-l",
        help="Directory to write conversation logs (JSONL). Enables logging when set.",
    ),
    keep_runs: bool = typer.Option(
        True,
        "--keep-runs/--no-keep-runs",
        help="Keep the ./runs/ scratch directory after the run (default: keep). No-op in V1.",
    ),
) -> None:
    """Run an agent from a config file with a prompt."""
    import datetime
    from agentji.config import load_config
    from agentji.loop import run_agent
    from agentji.logger import ConversationLogger

    try:
        cfg = load_config(config)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        err_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    if agent not in cfg.agents:
        available = ", ".join(cfg.agents.keys())
        err_console.print(
            f"[red]Error:[/red] Agent '{agent}' not found in config. "
            f"Available agents: {available}"
        )
        raise typer.Exit(1)

    logger: ConversationLogger | None = None
    log_path: Path | None = None
    if log_dir:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(log_dir) / f"{agent}_{ts}.jsonl"
        logger = ConversationLogger(log_path)
        console.print(f"[dim]Logging to:[/dim] {log_path}")

    console.print(f"[dim]Running agent:[/dim] [bold]{agent}[/bold]")
    console.print(f"[dim]Prompt:[/dim] {prompt}")
    console.print()

    try:
        result = run_agent(cfg, agent_name=agent, prompt=prompt, logger=logger)
        console.print(result)
        if log_path:
            console.print()
            console.print(f"[dim]Log written to:[/dim] {log_path}")
    except Exception as exc:
        err_console.print(f"[red]Agent error:[/red] {exc}")
        raise typer.Exit(1)


# ── logs command ──────────────────────────────────────────────────────────────

def _preview(text: str | None, limit: int) -> str:
    if not text:
        return ""
    s = str(text)
    return s if len(s) <= limit else s[:limit] + f"…[+{len(s) - limit}]"


def _format_args(args: dict, limit: int) -> str:
    if not args:
        return "(no args)"
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > limit:
            v_str = v_str[:limit] + f"…[+{len(v_str) - limit}]"
        parts.append(f"{k}={v_str!r}")
    return ", ".join(parts)


def _summarize_log(log_path: Path, max_tool_preview: int, session_filter: str | None = None) -> str:
    """Parse an agentji JSONL log and return a Markdown timeline string."""
    import json

    events: list[dict] = []
    with log_path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {i}: {exc}")

    if not events:
        raise ValueError("Log file is empty.")

    if session_filter:
        events = [e for e in events if e.get("session") == session_filter]
        if not events:
            raise ValueError(f"No events found for session '{session_filter}'.")

    # Collect unique pipelines (daily logs have many)
    pipelines_seen: list[str] = []
    for e in events:
        pid = e.get("pipeline", "?")
        if pid not in pipelines_seen:
            pipelines_seen.append(pid)

    run_starts = [e for e in events if e.get("event") == "run_start"]
    errors = [e for e in events if e.get("event") == "tool_result" and e.get("error")]

    lines: list[str] = []
    lines.append("# agentji Pipeline Log")
    lines.append("")
    if session_filter:
        lines.append(f"**Session**: `{session_filter}`  ")
    pipeline_ids = ", ".join(f"`{p}`" for p in pipelines_seen)
    lines.append(f"**Pipelines**: {len(pipelines_seen)} ({pipeline_ids})  ")
    lines.append(f"**Agent runs**: {len(run_starts)}  ")
    if errors:
        lines.append(f"**Errors**: {len(errors)}")
    lines.append("")

    run_id_order: list[str] = []
    by_run: dict[str, list[dict]] = {}
    for e in events:
        rid = e.get("run_id", "")
        if rid not in by_run:
            by_run[rid] = []
            run_id_order.append(rid)
        by_run[rid].append(e)

    for run_num, rid in enumerate(run_id_order, 1):
        run_events = by_run[rid]
        start = next((e for e in run_events if e.get("event") == "run_start"), None)
        end = next((e for e in run_events if e.get("event") == "run_end"), None)

        agent = start.get("agent", "?") if start else "?"
        model = start.get("model", "?") if start else "?"
        prompt_preview = _preview(start.get("prompt", ""), 120) if start else ""
        iterations = end.get("iterations", "?") if end else "?"
        response_preview = _preview(end.get("response_preview", ""), 300) if end else ""

        lines.append(f"## Run {run_num}: `{agent}` (`{rid}`)")
        lines.append("")
        lines.append(f"- **Model**: `{model}`")
        lines.append(f"- **Iterations**: {iterations}")
        lines.append(f"- **Prompt**: {prompt_preview}")
        lines.append("")

        llm_calls = [e for e in run_events if e.get("event") == "llm_call"]
        tool_calls = [e for e in run_events if e.get("event") == "tool_call"]
        tool_results = {
            e.get("tool"): e
            for e in run_events
            if e.get("event") == "tool_result"
        }

        if llm_calls:
            lines.append(f"**LLM calls**: {len(llm_calls)}")
            lines.append("")

        if tool_calls:
            lines.append("**Tool calls**:")
            lines.append("")
            for tc in tool_calls:
                tool_name = tc.get("tool", "?")
                tool_type = tc.get("tool_type", "?")
                args_str = _format_args(tc.get("args", {}), max_tool_preview)
                result_event = tool_results.get(tool_name)
                is_error = result_event.get("error", False) if result_event else False
                result_str = _preview(
                    result_event.get("result_preview", "") if result_event else "",
                    max_tool_preview,
                )
                status = "❌" if is_error else "✅"
                lines.append(f"- {status} **{tool_name}** ({tool_type})")
                lines.append(f"  - Args: `{args_str}`")
                if result_str:
                    lines.append(f"  - Result: {result_str}")
            lines.append("")

        if response_preview:
            lines.append("**Final response**:")
            lines.append(f"> {response_preview.replace(chr(10), '  ')}")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


@app.command()
def logs(
    log_path: str = typer.Argument(..., help="Path to the .jsonl conversation log file."),
    max_preview: int = typer.Option(
        200,
        "--max-preview",
        help="Maximum characters shown for tool args and results.",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="Filter output to a single session ID (X-Agentji-Session-Id).",
    ),
) -> None:
    """Summarize an agentji conversation log (JSONL) as a Markdown timeline."""
    from rich.markdown import Markdown

    path = Path(log_path)
    if not path.exists():
        err_console.print(f"[red]Error:[/red] File not found: {path}")
        raise typer.Exit(1)

    try:
        summary = _summarize_log(path, max_preview, session_filter=session)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    console.print(Markdown(summary))


# ── serve command ─────────────────────────────────────────────────────────────

@app.command()
def serve(
    config: Path = typer.Option(
        Path("agentji.yaml"),
        "--config",
        "-c",
        help="Path to the agentji.yaml config file.",
    ),
    agent: str = typer.Option(
        "",
        "--agent",
        "-a",
        help="Default agent name. Defaults to the first agent defined in config.",
    ),
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind."),
    port: int = typer.Option(8000, "--port", help="Port to bind."),
    log_dir: Path = typer.Option(
        Path("./logs"),
        "--log-dir",
        "-l",
        help="Directory for the serve.jsonl log file.",
    ),
    studio: bool = typer.Option(
        False,
        "--studio",
        help="Enable the Studio browser UI at / (disabled by default).",
    ),
    reload: bool = typer.Option(
        False,
        "--reload",
        help="Auto-reload on code changes (development only).",
    ),
) -> None:
    """Start the agentji OpenAI-compatible HTTP server."""
    from agentji.config import load_config
    from agentji.logger import ConversationLogger
    import agentji.server as server_module

    try:
        cfg = load_config(config)
    except FileNotFoundError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        err_console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(1)

    # Resolve default agent
    default_agent = agent if agent and agent in cfg.agents else next(iter(cfg.agents))
    if agent and agent not in cfg.agents:
        err_console.print(
            f"[yellow]Warning:[/yellow] Agent '{agent}' not found — "
            f"falling back to '{default_agent}'."
        )

    # Set up logging
    import datetime as _dt
    log_dir.mkdir(parents=True, exist_ok=True)
    log_cfg = cfg.logs

    if log_cfg.rotation == "daily":
        startup_logger = ConversationLogger(
            log_dir=log_dir,
            prefix="serve",
            rotation="daily",
        )
        log_display = str(log_dir / "serve_YYYY-MM-DD.jsonl")
        # Prune old daily log files
        if log_cfg.keep_days is not None:
            cutoff = _dt.datetime.now() - _dt.timedelta(days=log_cfg.keep_days)
            for old in log_dir.glob("serve_*.jsonl"):
                try:
                    date_str = old.stem[len("serve_"):]
                    file_date = _dt.datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff:
                        old.unlink()
                        console.print(f"[dim]Pruned old log:[/dim] {old.name}")
                except (ValueError, IndexError):
                    pass
    else:
        log_path = log_dir / "serve.jsonl"
        startup_logger = ConversationLogger(log_path)
        log_display = str(log_path)

    # Inject globals into server module
    server_module._cfg = cfg
    server_module._logger = startup_logger
    server_module._default_agent = default_agent
    server_module._studio_enabled = studio

    # Print startup info
    default_model = cfg.agents[default_agent].model
    console.print()
    console.print("[bold]agentji serve[/bold]")
    console.print(f"  config:  {config}")
    console.print(f"  agent:   {default_agent} ({default_model})")
    console.print(f"  listen:  http://{host}:{port}")
    if studio:
        console.print(f"  studio:  http://localhost:{port}")
    else:
        console.print(f"  studio:  disabled (use --studio to enable)")
    console.print(f"  log:     {log_display}")
    if log_cfg.rotation == "daily":
        keep_str = f"{log_cfg.keep_days}d" if log_cfg.keep_days else "forever"
        console.print(f"  rotate:  daily  keep={keep_str}")
    if cfg.improvement.enabled:
        imp_model = cfg.improvement.model or f"{default_model} (default agent)"
        console.print(f"  improve: enabled  model={imp_model}")
    console.print()
    console.print("[bold]Endpoints:[/bold]")
    console.print(f"  POST http://{host}:{port}/v1/chat/completions")
    console.print(f"  GET  http://{host}:{port}/v1/events/{{run_id}}")
    console.print(f"  GET  http://{host}:{port}/v1/pipeline")
    console.print(f"  POST http://{host}:{port}/v1/sessions/{{session_id}}/end")
    console.print()

    from agentji.server import start
    start(host=host, port=port, reload=reload)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry point."""
    app()


if __name__ == "__main__":
    main()
