# agentji

Skills written for Claude Code run on any model. Define agents in YAML — model, tools, MCP servers, sub-agents. `pip install agentji`. Qwen, Kimi, Ollama, GPT-4, Claude — same config.

![Python](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-early_dev-orange)

---

## Example
```yaml
providers:
  moonshot:
    api_key: ${MOONSHOT_API_KEY}
    base_url: https://api.moonshot.ai/v1
  qwen:
    api_key: ${DASHSCOPE_API_KEY}
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1

agents:
  orchestrator:
    model: moonshot/kimi-k2.5
    system_prompt: "Coordinate the analysis and report pipeline."
    agents: [analyst, reporter]

  analyst:
    model: qwen/MiniMax/MiniMax-M2.7
    skills: [sql-query, data-analysis]
    max_iterations: 15

  reporter:
    model: qwen/glm-5
    skills: [docx, brand-guidelines]
    builtins: [bash, write_file]
    max_iterations: 20
```

*Orchestrated by Kimi K2.5 (Moonshot) · Analysed by MiniMax M2.7 (DashScope) · Reported by glm5 (DashScope) · Zero Claude.*

---

## Quickstart

**Path A — zero cost, zero API keys (local Ollama)**

```bash
pip install agentji mcp-weather-server
ollama pull qwen3:4b   # ollama must be running: ollama serve
agentji run --config examples/weather-reporter/agentji.yaml \
  --agent weather-reporter \
  --prompt "Weather in Seoul, Tokyo, London, Paris, New York?"
```

**Path B — cloud model**

```bash
pip install agentji
export MOONSHOT_API_KEY=your_key
export DASHSCOPE_API_KEY=your_key
cd examples/data-analyst && python data/download_chinook.py
agentji run --config agentji.yaml --agent orchestrator \
  --prompt "Which markets should we prioritise for growth?"
```

---

## Skills

A skill is a directory with a `SKILL.md`. Skills from any registry work without modification:

| Skill | Source | Works on |
|---|---|---|
| `sql-query` | Bundled (agentji) | Any model |
| `data-analysis` | [ClawHub — ivangdavila](https://clawhub.ai/ivangdavila/data-analysis) | Any model |
| `self-improving` | [ClawHub — ivangdavila](https://clawhub.ai/ivangdavila/self-improving) | Any model |
| `brand-guidelines` | [Anthropic official](https://github.com/anthropics/skills) | Any model |
| `docx` | [Anthropic official](https://github.com/anthropics/skills) | Any model |

*Skills written for Claude Code work here unchanged. Skills from ClawHub work here unchanged. The model is a config line.*

```yaml
model: qwen/qwen-max        # change this
model: moonshot/kimi-k2.5   # to this
model: ollama/qwen3:4b      # or this — free, local
```

### Two skill types

**Prompt skills** — the SKILL.md body is injected into the agent's system prompt. Anthropic's official skills (`brand-guidelines`, `docx`, `data-analysis`) are all prompt skills. They work on any model because they're instructions, not code.

**Tool skills** — have a `skill.yaml` sidecar alongside SKILL.md. SKILL.md stays in pure Anthropic format; `skill.yaml` carries the tool config (script path, parameters, timeout). Both formats coexist in the same directory.

```
skills/sql-query/
├── SKILL.md       ← pure Anthropic format: name + description + body
├── skill.yaml     ← agentji tool config: scripts.execute + parameters + timeout
└── scripts/
    └── run_query.py
```

### Skill converter

If a skill has callable scripts but no `skill.yaml`, agentji detects it when the LLM tries to call it and offers to auto-generate one:

```
[agentji] Skill 'my-tool' has callable scripts but no skill.yaml.
          Proceeding in 20s — type 'n' + Enter to skip.
```

The converter uses the active agent's model to inspect the scripts and produce `skill.yaml`. No separate setup.

---

## Multi-agent orchestration

Set `agents:` on any agent to make it an orchestrator. agentji exposes a `call_agent(agent, prompt)` tool whose `enum` constraint limits delegation to the declared sub-agents — no hallucinated agent names.

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

Sub-agent calls appear in the same log file — the entire pipeline in one JSONL.

---

## agentji serve

```bash
pip install "agentji[serve]"

# API only (default)
agentji serve --config agentji.yaml --port 8000

# API + Studio browser UI
agentji serve --config agentji.yaml --port 8000 --studio
```

Starts an OpenAI-compatible server. Add `--studio` to also serve the agentji Studio browser UI at `http://localhost:8000`.

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI-compatible, streaming, returns `X-Agentji-Run-Id` header |
| `GET /v1/events/{run_id}` | SSE stream of all agent events (tool calls, sub-agent delegations, context handoffs) |
| `GET /v1/pipeline` | Pipeline topology JSON |
| `POST /v1/sessions/{id}/end` | End a session and trigger skill improvement extraction |
| `GET /` | agentji Studio (only when `--studio` flag is set) |

### Session modes

Each request can be stateless (context-free) or stateful (conversation history carried forward):

```yaml
studio:
  stateful: true    # default — history sent with each turn
  max_turns: 20     # how many prior turns to include
```

Per-request override via the `stateful` field in the request body:
```json
{ "messages": [...], "stateful": false }
```

### agentji Studio

```
┌──────────────┬────────────────────────┬─────────────┐
│ agent graph  │  chat + thinking cards │  live log   │
│ skill badges │  streaming response    │  SSE events │
│ status dots  │  file download links   │  stats bar  │
└──────────────┴────────────────────────┴─────────────┘
```

- Parallel tool calls grouped with a left border — you see the agent batching work
- `context_write` / `context_read` events in amber — you see file handoffs between agents
- Orchestrator step tracker — live phase list with pending → running → done status dots
- Iteration limit banner with **Continue** button — never lose work at `max_iterations`
- **■ Stop** button — cancel a run at the next iteration boundary
- File download links — `.docx`, `.csv`, `.md` paths in responses become clickable
- **Stateful toggle** — switch between stateful and stateless sessions in the header
- **Skill improvement checkbox** — opt this session in/out of improvement extraction

---

## Built-in tools

| Builtin | What it does |
|---|---|
| `bash` | Execute shell commands |
| `read_file` | Read a file from disk |
| `write_file` | Write a file to disk |

These replicate the native tools Claude Code provides, enabling prompt skills that rely on file I/O to run on any model.

---

## What shipped recently

**Consecutive error intervention** — after 3 consecutive failed tool calls, agentji injects a strategy-change message before the next LLM call. Shown as an amber `stuck` badge in the studio log. The counter resets on any success.

**Studio flag** — the Studio UI is now opt-in. `agentji serve` alone exposes API endpoints only; add `--studio` to enable the browser interface. This avoids accidentally exposing the UI in headless / CI deployments.

**Stateful sessions** — conversation history per browser tab (renamed from `session_memory`):
```yaml
studio:
  stateful: true
  max_turns: 20
```

**Skill improvement** — at the end of a session, agentji uses the configured model to extract corrections, affirmations, and user hints, then appends them to each skill's `improvements.jsonl`:
```yaml
improvement:
  enabled: true
  model: null           # null = inherit default agent model
  skills: []            # empty = all loaded skills
```
The Studio checkbox lets users opt individual sessions in/out. API callers pass `"improve": true` in the request body. Session end is triggered automatically on tab close (`beforeunload`) or via `POST /v1/sessions/{id}/end`. A 30-second idle timer fires automatically if neither signal arrives.

**Daily log rotation**:
```yaml
logs:
  rotation: daily   # creates serve_2026-03-22.jsonl
  keep_days: 30
```

**Per-agent tool timeout** — set `tool_timeout` in seconds on any agent (default 60s). Skills can also declare their own timeout in `skill.yaml`. Prevents hung SQL queries or runaway chart scripts from burning iterations.

**`agentji logs --session <id>`** — filter a shared daily log to one browser session.

**Reasoning content fix** — Kimi K2.5's `reasoning_content` in streaming is handled correctly. Without this fix, Kimi responses arrive empty.

**Run cancellation** — `POST /v1/cancel/{run_id}` stops a pipeline at the next iteration boundary. Wired to the Studio's Stop button and the limit banner.

---

## Provider support

| Provider | Model string | Notes |
|---|---|---|
| Qwen (DashScope) | `qwen/qwen-max` | |
| MiniMax (DashScope) | `qwen/MiniMax/MiniMax-M2.7` | Via DashScope routing |
| Kimi (Moonshot) | `moonshot/kimi-k2.5` | `fallback_base_url` for China/global auto-detect |
| Anthropic | `anthropic/claude-haiku-4-5` | No `base_url` needed |
| OpenAI | `openai/gpt-4o` | |
| Ollama (local) | `ollama/qwen3:4b` | Free, runs offline, no API key |
| Any litellm provider | — | [full list →](https://litellm.ai) |

**Dual-endpoint auto-detection** — set `fallback_base_url` for providers with regional endpoints (e.g. Moonshot global vs China). agentji probes both on first use and caches the result.

---

## Roadmap

```
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
- [x] agentji Studio (chat UI, pipeline tree, event log, stateful sessions)
- [x] Studio flag (--studio; API-only by default)
- [x] Skill improvement extraction (post-session, per-skill improvements.jsonl)
- [x] Stateful / stateless session toggle (per-config and per-request)
- [x] Consecutive error intervention (stuck detection)
- [x] Iteration limit banner with Continue / Stop
- [x] Per-agent tool timeout (tool_timeout in agentji.yaml)
- [x] Run cancellation (POST /v1/cancel/{run_id})
- [ ] Parallel sub-agent dispatch
- [ ] Persistent memory (mem0 / Zep)
- [ ] Plugin system for community skill registries
```

---

## Why agentji

**机** (jī) — machine, engine. The runtime.
**集** (jí) — assemble. Skills, models, tools, agents.
**极** (jí) — ultimate. Any skill should run on any model.

Built for developers working across the global AI ecosystem — especially those who need Qwen, Kimi, and local models to be first-class citizens, not afterthoughts.

---

## Contributing

Issues and PRs welcome. Adding a skill or a provider integration is the best first PR. Run the test suite with:

```bash
pytest                   # unit tests
pytest -m integration    # requires API keys in .env
pytest -m local          # requires Ollama running locally
```

---

## License

MIT
