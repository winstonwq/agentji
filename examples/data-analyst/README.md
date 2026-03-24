# agentji example: data-analyst

A three-agent pipeline that answers any question about the Chinook database —
from quick lookups to full strategic analyses with branded Word reports.
Orchestrated by Kimi K2.5 across three cloud providers.

```
orchestrator  (moonshot/kimi-k2.5)          ← plans, decides workflow, writes briefs
├── analyst   (qwen/MiniMax/MiniMax-M2.7)   ← SQL queries + analytical reasoning
└── reporter  (qwen/glm-5)                   ← charts + branded Word document
```

The orchestrator classifies the request and routes it:

| Request | Workflow |
|---|---|
| "Which market has the most revenue?" | analyst → answer |
| "Analyse genre performance" | analyst → summary |
| "Full analysis + Word doc" | analyst → reporter → confirm paths |
| "Format this text into a docx" | reporter only — no analyst |

The analyst **always** writes findings to the run scratch directory (`./runs/<run_id>/analyst_output.md`). The orchestrator passes that file path to the reporter only if the user asked for a document.

---

## Skills used

| Skill | Source | Role |
|---|---|---|
| `sql-query` | Bundled agentji (tool skill) | Executes SQL against Chinook SQLite |
| `data-analysis` | [ClawHub — ivangdavila](https://clawhub.ai/ivangdavila/data-analysis) | Analytical methodology |
| `docx-template` | Local (examples/data-analyst/skills) | python-docx helpers, brand colors, chart patterns |

---

## Setup

**1. Download the Chinook database (one-time, ~1 MB)**

```bash
cd examples/data-analyst
python data/download_chinook.py
```

**2. Install document and chart dependencies**

```bash
pip install python-docx matplotlib
```

**3. Set API keys**

```bash
export MOONSHOT_API_KEY=your_key      # moonshot.ai or moonshot.cn — auto-detected
export DASHSCOPE_API_KEY=your_key     # Qwen + MiniMax via DashScope
```

**4. Run**

```bash
agentji run --config agentji.yaml --agent orchestrator \
  --prompt "Which markets and genres should we prioritise for growth? \
Produce a full strategic report."
```

**5. Open the outputs**

Outputs are written to `./runs/<run_id>/` — the run ID is shown in the agent log. For example:

```bash
cat runs/a1b2c3d4/analyst_output.md        # structured analysis with all tables
open runs/a1b2c3d4/growth_strategy.docx    # branded Word document
```

---

## What happens when you run this

1. **Orchestrator** (Kimi K2.5) reads the prompt, classifies it as full-pipeline, and writes a detailed brief for the analyst — specifying the dimensions to analyse, the output sections, and the file path to write to.

2. **Analyst** (MiniMax M2.7) runs 10–15 SQL queries across multiple iterations, self-correcting on failures. Writes complete findings to `./runs/<run_id>/analyst_output.md` via `write_file`. Confirms the path in its final response.

3. **Orchestrator** receives the confirmation, decides a Word document was requested, and writes a second brief for the reporter — specifying the findings file, the output path, which sections to include, and what charts to generate.

4. **Reporter** (GLM-5) reads the findings file, generates matplotlib charts, and builds the Word document with python-docx. Verifies the file exists and reports path and size back to the orchestrator.

5. **Orchestrator** returns an executive summary to the user and confirms both output paths.

The file handoff (step 2→3) prevents the context overflow that occurs when large analyst output is passed as text — the orchestrator passes a path, not content.

---

## What the analyst actually does

The analyst runs SQL, not `SELECT *`. Three of the queries it runs on every full analysis:

```sql
-- Revenue and customers by market
SELECT BillingCountry,
       ROUND(SUM(Total), 2)            AS Revenue,
       COUNT(DISTINCT CustomerId)      AS Customers
FROM Invoice
GROUP BY BillingCountry
ORDER BY Revenue DESC;

-- Revenue, track count, and unit economics by genre
SELECT g.Name                                        AS Genre,
       ROUND(SUM(il.UnitPrice * il.Quantity), 2)    AS Revenue,
       COUNT(il.TrackId)                             AS TracksSold,
       ROUND(AVG(il.UnitPrice), 2)                  AS AvgUnitPrice
FROM InvoiceLine il
JOIN Track t  ON il.TrackId  = t.TrackId
JOIN Genre g  ON t.GenreId   = g.GenreId
GROUP BY g.Name
ORDER BY Revenue DESC;

-- Market × genre cross-analysis
SELECT i.BillingCountry,
       g.Name                                        AS Genre,
       ROUND(SUM(il.UnitPrice * il.Quantity), 2)    AS Revenue
FROM Invoice i
JOIN InvoiceLine il ON i.InvoiceId  = il.InvoiceId
JOIN Track t        ON il.TrackId   = t.TrackId
JOIN Genre g        ON t.GenreId    = g.GenreId
GROUP BY i.BillingCountry, g.Name
ORDER BY i.BillingCountry, Revenue DESC;
```

Every figure in the output is backed by a query. The analyst never fills gaps with estimates.

---

## What the pipeline produces

After a successful run, `./runs/<run_id>/analyst_output.md` contains a full strategic analysis. Here is the opening finding from a real run:

> "Rock dominates the catalogue with $826.65 in revenue (35.5% of total),
> but Latin delivers superior market penetration in three of the top five
> markets (USA: 17.2%, Canada: 19.5%, UK: 27.2%), representing a
> high-growth opportunity with only 340 tracks versus Rock's 745."

The reporter then converts this into `./runs/<run_id>/growth_strategy.docx` — a multi-page branded Word document with embedded charts, data tables, and a two-axis genre positioning matrix.

---

## More prompts to try

```bash
# Quick data question — no report
"Which country generates the most revenue? Show the top 5."

# Trend analysis
"Is total revenue growing or declining year-over-year? Flag any markets at risk."

# Genre deep-dive
"Which genres are high-margin but low-volume? Where should we invest in catalogue depth?"

# Cross-analysis
"Which markets are under-indexed on Latin music relative to their revenue size?"

# Full pipeline
"Full growth strategy report."

# Format existing content — reporter only, no analyst
"Format this text into a branded Word doc: ..."
```

---

## Skill sources

| Skill | Type | Source | Reference name |
|---|---|---|---|
| `sql-query` | tool | Bundled | `sql-query` |
| `data-analysis` | prompt | [clawhub: data-analysis v1.0.2](https://clawhub.ai/skills/data-analysis) | `data-analysis` (slug) |
| `docx-template` | prompt | Local | `docx-template` |

Agents reference skills by the `slug:` field in SKILL.md (if present), otherwise by `name:`. This means renaming a folder for versioning (`data-analysis-1.0.3/`) doesn't break the agent config — the slug stays stable.

---

## Switch any model — zero other changes

```yaml
# In agentji.yaml — swap one line per agent, nothing else changes
orchestrator:
  model: openai/gpt-4o            # instead of moonshot/kimi-k2.5

analyst:
  model: ollama/qwen3:4b          # free, local — needs Ollama running

reporter:
  model: anthropic/claude-haiku-4-5
```
