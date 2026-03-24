---
name: agentji
description: >
  Build, configure, run, and debug agentji multi-agent pipelines. Use when a
  developer asks to create an agentji config, define a skill, run an agent,
  start the serve API, open the studio UI, validate a config, scaffold a new
  skill, or convert an Anthropic-format skill to a callable tool. Provides the
  full developer reference and six executable actions: run, validate,
  list-agents, new-skill, convert-skill, serve-info.
---

# agentji Developer Reference

agentji is a Python runtime for multi-agent AI pipelines. Define every agent —
model, tools, MCP servers, sub-agents — in one YAML file and run it on any LLM.
Skills written for Anthropic Claude Code work without modification.

> **Tip:** Use the `skill-converter` to automatically generate a `skill.yaml`
> for any Anthropic-format skill that has executable scripts. See the
> Skill format section below.

---

## Core concepts

**Providers** — API credentials and optional custom base URLs. agentji supports
any provider that exposes an OpenAI-compatible endpoint. The `provider/model`
string in each agent disambiguates which credentials to use.

**Skills** — capabilities an agent can use. Two types:
- **Tool skills**: have a `skill.yaml` (or legacy `scripts.execute` in frontmatter)
  → translated to an OpenAI tool schema, executed as a subprocess with JSON stdin/stdout.
- **Prompt skills**: Anthropic Claude Code format — no `skill.yaml`, no `scripts.execute`
  → markdown body injected into the agent's system prompt. Works with zero modification.

**skill.yaml sidecar** — the recommended way to make a skill callable as a tool
while keeping `SKILL.md` pure Anthropic format. Place `skill.yaml` alongside
`SKILL.md`. agentji merges them at load time.

**Builtins** — primitive tools built into the runtime: `bash`, `read_file`,
`write_file`. Enable per-agent in the config. Required for prompt skills that
rely on executing helper scripts or writing files.

**Agents** — each agent has a model, system prompt, skills, optional MCP
servers, optional builtins, and an optional list of sub-agents it can call.
Agents can declare `inputs` (consumed from RunContext) and `outputs` (saved
to RunContext) for file-based handoff between agents.

**Orchestration** — set `agents: [sub-agent-name]` to give an agent a
`call_agent(agent, prompt)` tool. Sub-agents run as nested calls and share
the same log file. Large outputs are automatically offloaded to disk via
RunContext so orchestrators don't run out of context.

**serve** — `agentji serve` starts an OpenAI-compatible HTTP server with three
endpoints, plus a built-in browser Studio for interactive testing.

---

## agentji.yaml — full schema

```yaml
version: "1"

providers:
  <provider-name>:
    api_key: ${ENV_VAR}                        # required
    base_url: https://...                      # optional — OpenAI-compat endpoint
    fallback_base_url: https://...             # optional — probed if base_url fails

skills:
  - path: ./skills/my-skill                   # one entry per skill directory

mcps:
  - name: my-server                           # unique name referenced by agents
    command: python
    args: [./path/to/server.py]
    env:                                       # optional env vars for the process
      MY_VAR: value

agents:
  <agent-name>:
    model: <provider>/<model-name>            # required
    system_prompt: "..."                       # required
    skills: [skill-name, ...]                 # optional — names/slugs from skills above
    mcps: [server-name, ...]                  # optional — names from mcps above
    builtins: [bash, read_file, write_file]   # optional — built-in tools
    agents: [sub-agent-name, ...]             # optional — enables call_agent tool
    max_iterations: 10                         # optional — default 10
    tool_timeout: 60                           # optional — default 60s; per-skill timeout in skill.yaml wins
    inputs:                                    # optional — consumed from RunContext
      - key: findings
        description: Full analyst findings document.
    outputs:                                   # optional — saved to RunContext
      - key: findings
        description: Complete findings including all tables and recommendations.
```

**Provider examples:**

```yaml
providers:
  qwen:
    api_key: ${DASHSCOPE_API_KEY}
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

  moonshot:
    api_key: ${MOONSHOT_API_KEY}
    base_url: https://api.moonshot.ai/v1
    fallback_base_url: https://api.moonshot.cn/v1   # auto-probed on first use

  anthropic:
    api_key: ${ANTHROPIC_API_KEY}                   # no base_url — native routing

  openai:
    api_key: ${OPENAI_API_KEY}

  ollama:
    api_key: ollama
    base_url: http://localhost:11434/v1
```

**Model strings** (always `platform/model-name`):

| Platform | Example model string |
|---|---|
| DashScope (Qwen, MiniMax, …) | `qwen/qwen-max`, `qwen/MiniMax/MiniMax-M2.7` |
| Moonshot | `moonshot/kimi-k2.5` |
| Anthropic | `anthropic/claude-haiku-4-5` |
| OpenAI | `openai/gpt-4o` |
| Ollama (local) | `ollama/llama3.2` |

---

## Skill format

Every skill is a directory containing a `SKILL.md` file. Two types:

### Prompt skill (Anthropic Claude Code format — zero changes needed)

```markdown
---
name: brand-guidelines
description: >
  Applies brand colors and typography to visual artifacts.
---

# Brand Guidelines

Primary color: #141413. Accent: #d97757. ...
```

No `skill.yaml` needed. The markdown body is injected into the agent's system
prompt when the skill is referenced. Helper scripts in `scripts/` are accessible
via the `bash` builtin.

### Tool skill (callable via structured tool call)

Two ways to define a tool skill:

**Option A — `skill.yaml` sidecar (recommended, keeps SKILL.md Anthropic-compatible):**

```
my-skill/
  SKILL.md        ← pure Anthropic format (name + description + body)
  skill.yaml      ← agentji tool config
  scripts/
    run.py
```

`skill.yaml`:
```yaml
scripts:
  execute: scripts/run.py

parameters:
  query:
    type: string
    description: The SQL query to execute.
    required: true
  database:
    type: string
    description: Path to the SQLite database file.
    required: false
    default: "./data/chinook.db"
```

**Option B — inline frontmatter (legacy, still supported):**

```markdown
---
name: my-skill
description: What this skill does and when to use it.
parameters:
  query:
    type: string
    description: The query to run.
    required: true
scripts:
  execute: scripts/run.py
---
```

**Script interface** — receives JSON on stdin, writes JSON to stdout:

```python
import json, sys

def main():
    params = json.load(sys.stdin)
    query = params["query"]
    db = params.get("database", "./data/chinook.db")
    # ... do work ...
    print(json.dumps({"row_count": 5, "rows": [...]}))

if __name__ == "__main__":
    main()
```

### skill-converter — auto-generate skill.yaml

If the LLM tries to call a skill that was loaded as prompt-only but has
executable scripts, agentji automatically offers to convert it:

```
[agentji] Skill 'my-tool' has callable scripts but no skill.yaml.
          agentji can scan the scripts and generate one automatically.
          Proceeding in 20s — type 'n' + Enter to skip:
```

The runtime uses the active agent's own model to inspect the scripts and
generate a `skill.yaml`. You can also trigger this manually:

```bash
# Via the agentji meta-skill (if available to the agent):
# action: convert-skill, skill_dir: ./skills/my-tool

# Or directly via Python:
from agentji.skill_converter import convert_skill
from pathlib import Path
result = convert_skill(Path("./skills/my-tool"), litellm_kwargs)
```

**Slug for stable references** — add `slug:` to SKILL.md frontmatter to keep
agent config stable across version upgrades:

```markdown
---
name: Data Analysis
slug: data-analysis   # agents reference this: skills: [data-analysis]
description: ...
---
```

---

## Multi-agent orchestration

```yaml
agents:
  orchestrator:
    model: moonshot/kimi-k2.5
    system_prompt: |
      Coordinate analysis tasks. Delegate data work to the analyst,
      then pass findings to the reporter to produce a Word document.
    agents: [analyst, reporter]    # enables call_agent tool
    max_iterations: 8

  analyst:
    model: qwen/MiniMax/MiniMax-M2.7
    skills: [sql-query, data-analysis]
    builtins: [write_file, read_file]
    system_prompt: "..."
    max_iterations: 15
    outputs:
      - key: analyst_output
        description: Complete findings document including all tables.

  reporter:
    model: qwen/qwen-max
    skills: [docx, brand-guidelines]
    builtins: [bash, read_file, write_file]
    system_prompt: "..."
    max_iterations: 12
    inputs:
      - key: analyst_output
        description: Full analyst findings. Read with read_file before building report.
```

- Sub-agent outputs are automatically saved to a shared RunContext (per pipeline run).
- Values over ~8000 chars are offloaded to `./runs/<pipeline_id>/<key>.md` on disk.
- The reporter receives the file path automatically and should `read_file` it first.
- All events across the entire pipeline appear in one JSONL log with a shared `pipeline_id`.

---

## Running agents

**CLI:**
```bash
# Basic run
agentji run --config agentji.yaml --agent analyst --prompt "summarise sales"

# With logging
agentji run --config agentji.yaml --agent orchestrator \
  --prompt "growth strategy?" --log-dir ./logs

# Start the HTTP server + Studio UI
agentji serve --config agentji.yaml --agent orchestrator --port 8000
```

**Python API:**
```python
from dotenv import load_dotenv
load_dotenv()

from agentji.config import load_config
from agentji.loop import run_agent, run_agent_streaming
from agentji.logger import ConversationLogger

cfg = load_config("agentji.yaml")

# Simple call
result = run_agent(cfg, "analyst", "What are the top markets?")

# With logging
logger = ConversationLogger("logs/run.jsonl", pipeline_id="my-pipeline")
result = run_agent(cfg, "orchestrator", "Full analysis", logger=logger)

# Streaming (tokens delivered to callback as they arrive)
def on_token(token: str):
    print(token, end="", flush=True)

run_agent_streaming(cfg, "orchestrator", "Full analysis", on_token=on_token)
```

---

## agentji serve — HTTP server + Studio

`agentji serve` starts a FastAPI server with three endpoints:

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible chat completions (streaming supported) |
| `GET /v1/events/{run_id}` | SSE stream of pipeline log events for a run |
| `GET /v1/pipeline` | Current pipeline topology as JSON |
| `GET /` | agentji Studio — browser UI for interactive testing |

**Studio UI** — open `http://localhost:8000` in a browser after `agentji serve`:
- Left sidebar: agent pipeline tree with live status dots, skills list with source badges
- Center: streaming chat interface — each LLM turn appears as a separate message bubble
- Right panel: live event log (LLM calls, tool calls, sub-agent calls, results)
- Drag handles on all panel borders to resize sidebar widths and input area height

**Response headers:**
- `X-Agentji-Run-Id` — short pipeline ID on every `/v1/chat/completions` response;
  use it to open the events SSE stream for that run.

---

## Logging

agentji writes JSONL logs with one event per line. Read them with:

```bash
agentji logs ./logs/orchestrator_20260322_143140.jsonl
```

Each event has: `event`, `pipeline`, `run_id`, `agent`, `ts`, and event-specific fields.

| Event | Key fields |
|---|---|
| `run_start` | `model`, `prompt` |
| `run_end` | `iterations`, `response_preview` |
| `llm_call` | `iteration`, `n_messages`, `n_tools` |
| `llm_response` | `content_preview`, `tool_calls` |
| `tool_call` | `tool`, `tool_type`, `args` |
| `tool_result` | `tool`, `error`, `result_preview` |
| `context_write` | `key`, `path`, `size` |
| `context_read` | `key`, `offloaded` |

---

## Environment variables

agentji reads `${VAR_NAME}` placeholders in config at load time. It does **not**
load `.env` automatically — do that in your shell or via `python-dotenv`:

```bash
export DASHSCOPE_API_KEY=sk-...
agentji run ...
```

Or in Python:
```python
from dotenv import load_dotenv
load_dotenv()  # reads .env before load_config()
```

---

## Bundled skills

These ship with agentji and are copied on `agentji init`:

| Skill | Type | What it does |
|---|---|---|
| `agentji` | tool + prompt | This skill — build, run, and debug agentji pipelines |

Community skills from `anthropics/skills` (drop-in, no modification needed):

| Skill | Source | What it does |
|---|---|---|
| `brand-guidelines` | anthropics/skills | Anthropic brand colors and typography |
| `docx` | anthropics/skills | Create styled Word documents |
| `skill-creator` | anthropics/skills | Build and iterate on new skills interactively |

---

## Project layout

```
agentji.yaml              ← config
.env                      ← API keys (gitignored)
skills/
  my-skill/
    SKILL.md              ← Anthropic-format: name, description, body
    skill.yaml            ← agentji tool config: scripts + parameters (auto-generated if missing)
    scripts/
      run.py
examples/
  data-analyst/           ← multi-agent Chinook example
    agentji.yaml
    data/chinook.db
    skills/
    logs/
logs/                     ← JSONL logs (gitignored)
runs/                     ← per-run scratch dirs for agent handoff (gitignored)
```
