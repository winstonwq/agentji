"""
End-to-end test for the skill improvement pipeline.

Constructs a 2-turn conversation that contains clear learning signals
(a correction and a hint), triggers session end, then verifies that
improvements.jsonl was written alongside the relevant skill's SKILL.md.

Usage:
    cd examples/data-analyst
    agentji serve --studio --port 8000 &   # or use the running server
    python test_improvement.py
"""

import json
import sys
import time
import uuid
from pathlib import Path

import requests

BASE = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())
HEADERS = {
    "Content-Type": "application/json",
    "X-Agentji-Session-Id": SESSION_ID,
}

SKILL_DIRS = [
    Path("skills/sql-query"),
    Path("skills/data-analysis-1.0.2"),
    Path("skills/docx-template"),
]


def _delete_improvements():
    """Remove any leftover improvements.jsonl from previous runs."""
    for d in SKILL_DIRS:
        f = d / "improvements.jsonl"
        if f.exists():
            f.unlink()
    cwd_file = Path("improvements.jsonl")
    if cwd_file.exists():
        cwd_file.unlink()


def _ask(messages: list[dict], timeout: int = 90) -> str:
    payload = {
        "messages": messages,
        "stateful": False,   # we're managing history manually
        "improve": True,     # override: enable improvement for this session
    }
    r = requests.post(
        f"{BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=timeout
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _end_session() -> dict:
    r = requests.post(f"{BASE}/v1/sessions/{SESSION_ID}/end", timeout=10)
    r.raise_for_status()
    return r.json()


def _wait_for_improvements(timeout_secs: int = 60) -> list[Path]:
    """Poll until at least one improvements.jsonl appears (or timeout)."""
    deadline = time.time() + timeout_secs
    while time.time() < deadline:
        found = [d / "improvements.jsonl" for d in SKILL_DIRS
                 if (d / "improvements.jsonl").exists()]
        fallback = Path("improvements.jsonl")
        if fallback.exists():
            found.append(fallback)
        if found:
            return found
        time.sleep(2)
    return []


def main():
    print(f"Session: {SESSION_ID}\n")

    # ── Clean slate ───────────────────────────────────────────────────────────
    _delete_improvements()

    # ── Turn 1: get a real agent answer ───────────────────────────────────────
    print("Turn 1: asking for total revenue…")
    t1_messages = [
        {"role": "user", "content": "What is the total revenue across all countries?"}
    ]
    reply1 = _ask(t1_messages)
    print(f"  Agent: {reply1[:200]}…\n")

    # ── Turn 2: user issues a correction + a hint ─────────────────────────────
    # We include the full history so the server tracks all turns.
    # The user message contains a clear correction and a forward-looking hint.
    print("Turn 2: user provides correction and hint…")
    t2_messages = t1_messages + [
        {"role": "assistant", "content": reply1},
        {
            "role": "user",
            "content": (
                "Correction: you presented the revenue without showing the per-country breakdown. "
                "Always include a per-country table when the user asks about 'all countries'. "
                "Hint for sql-query: when aggregating revenue always join "
                "Invoice → InvoiceLine and use SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) — "
                "never rely on Invoice.Total alone as it can include tax. "
                "Now please give me revenue by country as a table."
            ),
        },
    ]
    reply2 = _ask(t2_messages)
    print(f"  Agent: {reply2[:200]}…\n")

    # ── End session → trigger extraction ─────────────────────────────────────
    print("Ending session…")
    status = _end_session()
    print(f"  Server: {status}\n")

    if status.get("status") != "extraction_scheduled":
        print("WARN: server did not schedule extraction "
              f"(status={status.get('status')}). "
              "Check that improve=True was accepted.")

    # ── Wait for improvements.jsonl ───────────────────────────────────────────
    print("Waiting for extraction to complete (up to 60s)…")
    found_files = _wait_for_improvements(timeout_secs=60)

    if not found_files:
        print("\nFAIL — no improvements.jsonl files were written within 60 seconds.")
        sys.exit(1)

    # ── Display results ───────────────────────────────────────────────────────
    print(f"\nPASS — improvements written to {len(found_files)} file(s):\n")
    all_entries: list[dict] = []
    for path in found_files:
        print(f"  {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                all_entries.append(entry)
                print(f"    [{entry['type']:12s}] skill={entry['skill']}")
                print(f"      learning : {entry['learning']}")
                print(f"      context  : {entry['context'][:120]}")
                print()

    print(f"Total signals extracted: {len(all_entries)}")

    # Validate structure
    for e in all_entries:
        assert e.get("ts"), "missing ts"
        assert e.get("session_id") == SESSION_ID, "session_id mismatch"
        assert e.get("type") in ("correction", "affirmation", "hint", "unknown"), \
            f"unexpected type: {e.get('type')}"
        assert e.get("skill"), "missing skill"
        assert e.get("learning"), "missing learning"

    print("Structure validation passed.")
    sys.exit(0)


if __name__ == "__main__":
    main()
