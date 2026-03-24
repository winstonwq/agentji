# agentji

Run any agent skill on any model. One YAML file.

Anthropic's official skills — `docx`, `brand-guidelines`, `data-analysis` — work here unchanged, on Qwen, Kimi, MiniMax, or a local Ollama model. Swap the model with one config line. No code changes.

```yaml
agents:
  orchestrator:
    model: moonshot/kimi-k2.5      # change this line to switch providers
    agents: [analyst, reporter]

  analyst:
    model: qwen/MiniMax/MiniMax-M2.7
    skills: [sql-query, data-analysis]

  reporter:
    model: qwen/glm-5
    skills: [docx-template]
    builtins: [bash, write_file]
    max_iterations: 20
```

*Orchestrated by Kimi K2.5 · Analysed by MiniMax M2.7 · Reported by GLM-5 · Zero Claude.*

---

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-beta-orange)

```bash
pip install agentji
```

---

## Quickstart

Three paths. Pick the one that fits.

**Path A — free, offline, no API keys**
Uses a local Ollama model. You get a working weather agent in a browser UI.

```bash
pip install "agentji[serve]" mcp-weather-server
ollama pull qwen3:4b
cd examples/weather-reporter
agentji serve --studio
```

Open [http://localhost:8000](http://localhost:8000) → ask: *"Weather in Seoul, Tokyo, London?"*

---

**Path B — cloud models, multi-agent pipeline**
Three providers, one pipeline. You get a Word document with a full market analysis.

```bash
pip install "agentji[serve]" python-docx matplotlib
export MOONSHOT_API_KEY=your_key
export DASHSCOPE_API_KEY=your_key
cd examples/data-analyst && python data/download_chinook.py
agentji serve --studio
```

Open [http://localhost:8000](http://localhost:8000) → ask: *"Which markets should we prioritise for growth? Full report."*
→ `output/growth_strategy.docx` is written to disk when the run completes.

---

**Path C — CLI, no server**
No browser, no server. Pipe it into a script or run it headless.

```bash
agentji run --config examples/data-analyst/agentji.yaml \
  --agent orchestrator \
  --prompt "Which genres are high-margin but low-volume?"
```

---

## Skills

A skill is a directory with a `SKILL.md`. Skills from any registry work without modification:

| Skill | Source | Type |
|---|---|---|
| `sql-query` | Bundled (agentji) | Tool skill |
| `data-analysis` | [ClawHub — ivangdavila](https://clawhub.ai/ivangdavila/data-analysis) | Prompt skill |
| Any Claude Code skill | [Anthropic official](https://github.com/anthropics/skills) | Prompt skill |

Claude Code's Anthropic-format skills work here unchanged. The model is a config line.

### Two skill types

**Prompt skills** — the SKILL.md body is injected into the agent's system prompt. Anthropic's official skills (`brand-guidelines`, `docx`, `data-analysis`) are all prompt skills. They work on any model because they're instructions, not code.

**Tool skills** — a `skill.yaml` sidecar alongside SKILL.md adds the tool config: script path, parameters, timeout. SKILL.md stays in pure Anthropic format; `skill.yaml` is the agentji extension.

```
skills/sql-query/
├── SKILL.md       ← pure Anthropic format: name + description + body
├── skill.yaml     ← agentji tool config: scripts.execute + parameters
└── scripts/
    └── run_query.py
```

### Skill converter

If a skill has callable scripts but no `skill.yaml`, agentji detects it and offers to auto-generate one using the active agent's model. No separate setup.

---

## Multi-agent orchestration

Set `agents:` on any agent to make it an orchestrator. agentji injects a `call_agent(agent, prompt)` tool whose `enum` constraint limits delegation to declared sub-agents — no hallucinated agent names.

```yaml
agents:
  orchestrator:
    model: moonshot/kimi-k2.5
    agents: [analyst, reporter]   # call_agent tool added automatically

  analyst:
    model: qwen/MiniMax/MiniMax-M2.7
    skills: [sql-query, data-analysis]

  reporter:
    model: qwen/glm-5
    skills: [docx-template]
    builtins: [bash, write_file]
```

Sub-agent calls appear in the same log file — the entire pipeline in one JSONL, linked by a shared `pipeline_id`.

---

## MCP servers

Declare an MCP server in YAML; agentji connects via FastMCP and exposes its tools to the agent automatically.

```yaml
mcps:
  - name: weather
    command: python
    args: [-m, mcp_weather_server]   # launched as subprocess, stdio transport

agents:
  weather-reporter:
    model: ollama/qwen3:4b
    mcps: [weather]                  # tools discovered at runtime
```

---

## agentji serve

```bash
pip install "agentji[serve]"

# API only (default) — suitable for production, CI, headless deployments
agentji serve --config agentji.yaml --port 8000

# API + Studio browser UI
agentji serve --config agentji.yaml --port 8000 --studio
```

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming, returns `X-Agentji-Run-Id` header |
| `GET /v1/events/{run_id}` | SSE stream of all agent events (tool calls, sub-agent delegations) |
| `GET /v1/pipeline` | Pipeline topology JSON |
| `POST /v1/sessions/{id}/end` | End a session and trigger skill improvement extraction |
| `GET /` | agentji Studio (only when `--studio` flag is set) |

### Sessions

Pass `X-Agentji-Session-Id` to track a conversation across turns. Control history per request:

```json
{ "messages": [...], "stateful": true, "improve": true }
```

Or configure defaults in YAML:

```yaml
studio:
  stateful: true    # carry conversation history across turns
  max_turns: 20
```

### agentji Studio

```
┌──────────────┬────────────────────────┬─────────────┐
│ agent graph  │  chat + thinking cards │  live log   │
│ skill badges │  streaming response    │  SSE events │
│ status dots  │  file download links   │  stats bar  │
└──────────────┴────────────────────────┴─────────────┘
```

- Parallel tool calls grouped with a left border
- `context_write` / `context_read` events in amber — file handoffs between agents
- Orchestrator step tracker — live phase list with pending → running → done status
- Iteration limit banner with **Continue** button — never lose work at `max_iterations`
- **■ Stop** button — cancel a run at the next iteration boundary
- File download links — `.docx`, `.csv`, `.md` paths become clickable
- **Stateful toggle** — switch between stateful and stateless sessions in the header
- **Skill improvement checkbox** — opt individual sessions in/out of improvement extraction

---

## Skill improvement

At session end, agentji uses the configured model to review the conversation and extract three types of learning signals — corrections, affirmations, and hints — then appends them to each skill's `improvements.jsonl`:

```yaml
improvement:
  enabled: true
  model: null     # null = inherit default agent model
  skills: []      # empty = all loaded skills
```

Signal types written to `skills/sql-query/improvements.jsonl`:
```json
{"type": "correction", "skill": "sql-query", "learning": "Use InvoiceLine.UnitPrice * Quantity for revenue, not Invoice.Total.", "context": "User corrected a query that used Invoice.Total which includes tax adjustments."}
{"type": "hint",       "skill": "sql-query", "learning": "The Chinook database covers 2009–2013 only; scope date filters to this range.", "context": "User noted this mid-conversation."}
```

Session end is triggered automatically on tab close, via `POST /v1/sessions/{id}/end`, or after 30 seconds of inactivity. The Studio checkbox lets users opt sessions in/out individually.

---

## Built-in tools

| Builtin | What it does |
|---|---|
| `bash` | Execute shell commands |
| `read_file` | Read a file from disk |
| `write_file` | Write a file to disk |

These replicate the native tools Claude Code provides, enabling prompt skills that rely on file I/O to run on any model.

---

## Provider support

| Provider | Model string | Notes |
|---|---|---|
| Qwen (DashScope) | `qwen/qwen-max` | |
| MiniMax (DashScope) | `qwen/MiniMax/MiniMax-M2.7` | Via DashScope routing |
| GLM (DashScope) | `qwen/glm-5` | |
| Kimi (Moonshot) | `moonshot/kimi-k2.5` | `fallback_base_url` for China/global auto-detect |
| Anthropic | `anthropic/claude-haiku-4-5` | No `base_url` needed |
| OpenAI | `openai/gpt-4o` | |
| Ollama (local) | `ollama/qwen3:4b` | Free, runs offline, no API key |
| Any litellm provider | — | [full list →](https://litellm.ai) |

**Dual-endpoint auto-detection** — set `fallback_base_url` for providers with regional endpoints (e.g. Moonshot global vs China). agentji probes both on first use and caches the result.

```yaml
providers:
  moonshot:
    api_key: ${MOONSHOT_API_KEY}
    base_url: https://api.moonshot.ai/v1
    fallback_base_url: https://api.moonshot.cn/v1   # auto-probed on first use
```

---

## Roadmap

**Shipped**
- [x] Skill translation (SKILL.md → OpenAI tool schema)
- [x] skill.yaml sidecar (tool config separate from Anthropic format)
- [x] Skill converter (auto-generate skill.yaml from scripts via LLM)
- [x] Prompt skills (Anthropic format, body injected into system prompt)
- [x] Multi-provider routing via litellm
- [x] Agentic loop via LangGraph
- [x] MCP server integration via FastMCP
- [x] Built-in tools (bash, read_file, write_file)
- [x] Multi-agent orchestration (call_agent with enum constraint)
- [x] Per-run RunContext (file-based context handoff between agents)
- [x] Conversation logging (JSONL, pipeline_id, session_id, daily rotation)
- [x] Provider endpoint auto-detection + caching
- [x] agentji serve (OpenAI-compatible HTTP endpoint)
- [x] agentji Studio (chat UI, pipeline tree, event log)
- [x] Studio flag (--studio; API-only by default)
- [x] Stateful / stateless session toggle (per-config and per-request)
- [x] Skill improvement extraction (post-session, per-skill improvements.jsonl)
- [x] Consecutive error intervention (stuck detection)
- [x] Iteration limit banner with Continue / Stop
- [x] Per-agent tool timeout (tool_timeout in agentji.yaml)
- [x] Run cancellation (POST /v1/cancel/{run_id})

**Coming**
- [ ] Parallel sub-agent dispatch
- [ ] Persistent memory (mem0 / Zep)
- [ ] Plugin system for community skill registries

---

## Why agentji

Built for developers working across the global AI ecosystem — for teams where Qwen, Kimi, and local models are first-class requirements, not afterthoughts. If you're locked to one provider because your skills won't port, agentji is the unlock.

**机** (jī) — machine, engine. The runtime.
**集** (jí) — assemble. Skills, models, tools, agents.
**极** (jí) — ultimate. Any skill on any model.

---

## Contributing

Issues and PRs welcome. Adding a skill or a provider integration is the best first PR.

```bash
pytest                   # unit tests
pytest -m integration    # requires API keys in .env
pytest -m local          # requires Ollama running locally
```

---

## License

MIT