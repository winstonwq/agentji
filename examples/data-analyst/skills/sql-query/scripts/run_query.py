#!/usr/bin/env python3
"""
SQL query executor for the sql-query skill.
Reads parameters from stdin as JSON, executes a read-only SQLite query,
writes results to stdout as JSON.
"""
import json
import pathlib
import sqlite3
import sys


_ALLOWED_FIRST_WORDS = frozenset({"SELECT", "WITH", "EXPLAIN", "PRAGMA"})


def main() -> None:
    try:
        params = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        print(json.dumps({"error": f"Invalid JSON input: {exc}"}))
        sys.exit(1)

    query: str = params.get("query", "").strip()
    database: str = params.get("database", "./data/chinook.db")

    if not query:
        print(json.dumps({"error": "No query provided"}))
        sys.exit(1)

    # Strip leading SQL line comments (-- ...) before checking the first keyword
    stripped = "\n".join(
        line for line in query.splitlines() if not line.strip().startswith("--")
    ).strip()
    first_word = stripped.split()[0].upper() if stripped.split() else ""
    if first_word not in _ALLOWED_FIRST_WORDS:
        print(json.dumps({
            "error": (
                f"Only read-only queries are allowed (SELECT, WITH, EXPLAIN). "
                f"Got: {first_word}"
            )
        }))
        sys.exit(1)

    db_path = pathlib.Path(database)
    if not db_path.exists():
        print(json.dumps({
            "error": (
                f"Database not found: {database}. "
                f"Run: python data/download_chinook.py"
            )
        }))
        sys.exit(1)

    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query)
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()

        print(json.dumps({
            "row_count": len(rows),
            "columns": list(rows[0].keys()) if rows else [],
            "rows": rows,
        }, ensure_ascii=False))

    except sqlite3.Error as exc:
        print(json.dumps({"error": f"SQL error: {exc}", "query": query}))
        sys.exit(1)


if __name__ == "__main__":
    main()
