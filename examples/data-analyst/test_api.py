"""
Quick smoke test for the data-analyst agentji API.

Usage:
    cd examples/data-analyst
    python test_api.py
"""
import json
import sys
import time
import uuid

import requests

BASE = "http://localhost:8000"
SESSION_ID = str(uuid.uuid4())
HEADERS = {
    "Content-Type": "application/json",
    "X-Agentji-Session-Id": SESSION_ID,
}


def ask(question: str, timeout: int = 120) -> str:
    payload = {
        "messages": [{"role": "user", "content": question}],
        "stateful": True,
    }
    print(f"\n>>> {question}")
    t0 = time.time()
    r = requests.post(f"{BASE}/v1/chat/completions", headers=HEADERS, json=payload, timeout=timeout)
    elapsed = time.time() - t0
    if r.status_code != 200:
        print(f"    ERROR {r.status_code}: {r.text[:300]}")
        return ""
    reply = r.json()["choices"][0]["message"]["content"]
    print(f"    ({elapsed:.1f}s) {reply[:600]}{'...' if len(reply) > 600 else ''}")
    return reply


def check_pipeline():
    r = requests.get(f"{BASE}/v1/pipeline")
    assert r.status_code == 200, f"pipeline check failed: {r.status_code}"
    p = r.json()
    print(f"Pipeline OK — agents: {list(p['agents'].keys())}, stateful: {p['stateful']}")


def end_session():
    r = requests.post(f"{BASE}/v1/sessions/{SESSION_ID}/end")
    print(f"\nSession end: {r.status_code} {r.json()}")


if __name__ == "__main__":
    print(f"Session: {SESSION_ID}")

    # 1. Verify server is up
    check_pipeline()

    # 2. Simple data question — should NOT trigger reporter
    reply1 = ask("What is the total revenue across all countries?", timeout=90)

    # 3. Another quick question
    reply2 = ask("Which genre has the highest number of tracks?", timeout=90)

    # 4. End session cleanly
    end_session()

    ok = bool(reply1 and reply2)
    print(f"\n{'PASS' if ok else 'FAIL'} — got responses: q1={bool(reply1)}, q2={bool(reply2)}")
    sys.exit(0 if ok else 1)
