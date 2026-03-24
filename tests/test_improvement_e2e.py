"""
Real end-to-end test for the skill improvement extractor.

Calls extract_and_save() directly with a crafted conversation containing
clear corrections and hints, using the actual Kimi K2.5 API (no server needed).

Requires: MOONSHOT_API_KEY set in environment or .env

Run:
    cd /path/to/agentji
    pytest tests/test_improvement_e2e.py -v -s -m integration
"""

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── Crafted conversation with clear learning signals ──────────────────────────
#
# Contains three explicit signals:
#   1. correction  — user corrects a formatting choice
#   2. hint        — user gives a domain rule to follow
#   3. affirmation — user confirms a good approach
#

FAKE_CONVERSATION = [
    {
        "role": "user",
        "content": "Write a SQL query to get the top 5 countries by total revenue.",
    },
    {
        "role": "assistant",
        "content": (
            "SELECT c.Country, SUM(i.Total) AS Revenue\n"
            "FROM Invoice i JOIN Customer c ON i.CustomerId = c.CustomerId\n"
            "GROUP BY c.Country ORDER BY Revenue DESC LIMIT 5;"
        ),
    },
    {
        "role": "user",
        "content": (
            "Correction: do not use Invoice.Total — it may include tax adjustments. "
            "Always compute revenue as SUM(il.UnitPrice * il.Quantity) "
            "from InvoiceLine. Fix the query."
        ),
    },
    {
        "role": "assistant",
        "content": (
            "Understood. Here is the corrected query:\n\n"
            "SELECT c.Country, SUM(il.UnitPrice * il.Quantity) AS Revenue\n"
            "FROM InvoiceLine il\n"
            "JOIN Invoice i ON il.InvoiceId = i.InvoiceId\n"
            "JOIN Customer c ON i.CustomerId = c.CustomerId\n"
            "GROUP BY c.Country ORDER BY Revenue DESC LIMIT 5;"
        ),
    },
    {
        "role": "user",
        "content": (
            "Good. Also, hint for future SQL queries: the Chinook database has sales "
            "only from 2009–2013, so never filter by dates outside that range."
        ),
    },
    {
        "role": "assistant",
        "content": "Noted — I will scope all date filters to 2009–2013.",
    },
    {
        "role": "user",
        "content": "Perfect, that SQL pattern with InvoiceLine is exactly right. Keep doing it that way.",
    },
    {
        "role": "assistant",
        "content": "Will do — I will consistently use InvoiceLine for revenue calculations.",
    },
]

SKILL_REFS = [
    {"name": "sql-query",      "path": "skills/sql-query"},
    {"name": "data-analysis",  "path": "skills/data-analysis"},
]


@pytest.mark.integration
def test_extract_and_save_real_api():
    """extract_and_save writes ≥1 improvements using the real Kimi K2.5 API."""
    api_key = os.environ.get("MOONSHOT_API_KEY")
    if not api_key:
        pytest.skip("MOONSHOT_API_KEY not set")

    from agentji.improver import extract_and_save

    session_id = str(uuid.uuid4())

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Create fake skill dirs with SKILL.md so paths resolve
        (tmp / "skills" / "sql-query").mkdir(parents=True)
        (tmp / "skills" / "sql-query" / "SKILL.md").write_text("# sql-query skill\n")
        (tmp / "skills" / "data-analysis").mkdir(parents=True)
        (tmp / "skills" / "data-analysis" / "SKILL.md").write_text("# data-analysis skill\n")

        # Rewrite paths to point at tmp
        skill_refs = [
            {"name": "sql-query",     "path": str(tmp / "skills" / "sql-query")},
            {"name": "data-analysis", "path": str(tmp / "skills" / "data-analysis")},
        ]

        litellm_kwargs = {
            "model": "moonshot/kimi-k2.5",
            "api_key": api_key,
            "api_base": "https://api.moonshot.ai/v1",
        }

        improvements = extract_and_save(
            messages=FAKE_CONVERSATION,
            session_id=session_id,
            skill_refs=skill_refs,
            model="moonshot/kimi-k2.5",
            litellm_kwargs=litellm_kwargs,
            target_skills=[],
            fallback_improvements_path=tmp / "improvements.jsonl",
        )

        # ── Assertions ────────────────────────────────────────────────────────

        assert len(improvements) >= 1, (
            f"Expected at least 1 improvement signal, got {len(improvements)}.\n"
            f"The conversation contains an explicit correction and a hint — "
            "the model should have found at least one."
        )

        types_found = {e["type"] for e in improvements}
        assert "correction" in types_found or "hint" in types_found, (
            f"Expected at least one 'correction' or 'hint', got types: {types_found}"
        )

        # Every entry must have required fields and the right session_id
        for e in improvements:
            assert e["session_id"] == session_id
            assert e["type"] in ("correction", "affirmation", "hint", "unknown")
            assert e["skill"]
            assert e["learning"]
            assert e["ts"]

        # At least one improvements.jsonl file must exist on disk
        written_files = list(tmp.rglob("improvements.jsonl"))
        assert written_files, "No improvements.jsonl files were written to disk"

        # Verify disk contents parse correctly
        all_disk_entries: list[dict] = []
        for f in written_files:
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    all_disk_entries.append(json.loads(line))

        assert len(all_disk_entries) == len(improvements), (
            f"Disk entries ({len(all_disk_entries)}) != returned entries ({len(improvements)})"
        )

        # ── Print for visual inspection ───────────────────────────────────────
        print(f"\n  Extracted {len(improvements)} signal(s):")
        for e in improvements:
            print(f"    [{e['type']:12s}] skill={e['skill']}")
            print(f"      learning : {e['learning']}")
            print(f"      context  : {e['context'][:120]}")
        print(f"\n  Written files: {[str(f.relative_to(tmp)) for f in written_files]}")
