# agentji example: weather-reporter

Run a live weather agent for 7 cities using a free local model and a free no-auth weather API. Zero API keys, zero cloud accounts, zero cost.

```
weather-reporter  (ollama/qwen3:4b)
└── weather MCP   (mcp-weather-server → Open-Meteo API, no key required)
```

---

## What this demonstrates

- A local Ollama model making 7 parallel MCP tool calls in a single iteration
- `mcp-weather-server` connecting to Open-Meteo — completely free, no auth
- agentji's MCP bridge via FastMCP — declare the server in YAML, tools appear automatically
- Parallel tool call grouping in the studio log

This is the smallest possible agentji config: one provider, one MCP server, one agent, ~20 lines of YAML.

---

## Setup

```bash
pip install agentji mcp-weather-server
ollama pull qwen3:4b
```

Ollama must be running before you execute the next command. On macOS it starts automatically after install. On Linux: `ollama serve &`

---

## Run

```bash
agentji run --config examples/weather-reporter/agentji.yaml \
  --agent weather-reporter \
  --prompt "Get current weather for New York, San Francisco, London, Paris, Beijing, Seoul and Tokyo. Show as a table."
```

Expected output:

| City | Condition | Temp (°C) | Humidity (%) | Wind (km/h) |
|---|---|---|---|---|
| New York | Mainly clear | 11.1 | 75 | 19.9 |
| San Francisco | Clear sky | 16.5 | 71 | 1.3 |
| London | Clear sky | 12.6 | 51 | 10.1 |
| Paris | Partly cloudy | 17.3 | 39 | 13.4 |
| Beijing | Overcast | 10.1 | 56 | 4.5 |
| Seoul | Partly cloudy | 3.6 | 59 | 4.8 |
| Tokyo | Clear sky | 5.8 | 46 | 3.7 |

Data is live — your results will differ. The agent made 7 parallel tool calls in one iteration.

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
      You are a weather reporter. Call get_current_weather ONCE PER CITY.
      Never invent or estimate weather data — only use what the tool returns.
      Present results in a Markdown table.
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
