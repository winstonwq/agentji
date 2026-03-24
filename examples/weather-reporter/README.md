# agentji example: weather-reporter

Live weather for any cities using a free local model and a free no-auth weather API. Zero API keys, zero cloud accounts, zero cost.

```
weather-reporter  (ollama/qwen3:4b)
└── weather MCP   (mcp-weather-server → Open-Meteo API, no key required)
```

---

## What this demonstrates

- A local Ollama model making parallel MCP tool calls in a single iteration
- `mcp-weather-server` connecting to Open-Meteo — completely free, no auth
- agentji's MCP bridge via FastMCP — declare the server in YAML, tools appear automatically
- Parallel tool call grouping in the Studio log

This is the smallest possible agentji config: one provider, one MCP server, one agent, ~20 lines of YAML.

---

## Setup

```bash
pip install "agentji[serve]" mcp-weather-server
ollama pull qwen3:4b
```

Ollama must be running. On macOS it starts automatically after install. On Linux: `ollama serve &`

You only need to `ollama pull qwen3:4b` once — the model is cached locally after the first pull.

---

## Run with Studio

```bash
cd examples/weather-reporter
agentji serve --studio
```

Open [http://localhost:8000](http://localhost:8000) and ask: *"Weather in Seoul, Tokyo, London, Paris, New York?"*

## Run from CLI

```bash
agentji run --config examples/weather-reporter/agentji.yaml \
  --agent weather-reporter \
  --prompt "Get current weather for Seoul, Tokyo, London, Paris, New York."
```

Expected output:

| City | Condition | Temp (°C) | Humidity (%) | Wind (km/h) |
|---|---|---|---|---|
| Seoul | Partly cloudy | 7.5 | 89 | 1.5 |
| Tokyo | Overcast | 9.5 | 70 | 2.0 |
| London | Overcast | 13.6 | 68 | 23.0 |
| Paris | Overcast | 18.5 | 36 | 15.3 |
| New York | Clear sky | 2.5 | 66 | 17.0 |

Data is live — your results will differ.

---

## Performance

The 4B local model takes 2–3 minutes on typical hardware. Switch to a faster model in one line — the MCP config and everything else stays the same:

```yaml
model: ollama/qwen3:14b              # faster local model
model: qwen/qwen-max                 # cloud, ~10× faster, needs DASHSCOPE_API_KEY
model: anthropic/claude-haiku-4-5   # fastest, needs ANTHROPIC_API_KEY
```

---

## agentji.yaml

```yaml
version: "1"

providers:
  ollama:
    api_key: ollama               # required by litellm; not checked by Ollama
    base_url: http://localhost:11434/v1

mcps:
  - name: weather
    command: python
    args: [-m, mcp_weather_server]  # launched as subprocess, connected via FastMCP

agents:
  weather-reporter:
    model: ollama/qwen3:4b
    system_prompt: |
      You are a weather reporter. When the user asks for weather data:
      1. Identify every city the user mentioned.
      2. Call get_current_weather exactly once per city.
      3. Wait for ALL results before writing anything.
      4. Present results in a single Markdown table:
         City | Condition | Temp (°C) | Humidity (%) | Wind (km/h)
      5. Add one sentence noting the most notable weather difference.
    mcps: [weather]
    max_iterations: 20
```

---

## Available tools

`mcp-weather-server` exposes three tools agentji discovers automatically:

| Tool | What it does |
|---|---|
| `get_current_weather` | Current conditions for a city |
| `get_weather_byDateTimeRange` | Forecast for a date/time range |
| `get_weather_details` | Detailed breakdown (hourly, UV, pressure) |
