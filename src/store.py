"""SQLite persistence for Question/Answer rows."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    question        TEXT NOT NULL,
    content         TEXT NOT NULL,
    retries         INTEGER DEFAULT 0,
    cost_usd        REAL DEFAULT 0.0,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_EXTRA_COLUMNS = [
    ("model", "TEXT"),
    ("confidence", "REAL"),
    ("sources_json", "TEXT"),
    ("schema_version", "TEXT DEFAULT 'v1'"),
]


def ensure_schema(db_path: str | Path) -> None:
    """Create the answers table and add any missing columns. Safe to rerun."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_BASE_SCHEMA)
        existing = {
            row[1] for row in conn.execute("PRAGMA table_info(answers)").fetchall()
        }
        for col_name, col_def in _EXTRA_COLUMNS:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE answers ADD COLUMN {col_name} {col_def}")
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect(db_path: str | Path) -> Generator[sqlite3.Connection, None, None]:
    ensure_schema(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def save_answer(
    conn: sqlite3.Connection,
    *,
    question: str,
    content: str,
    retries: int,
    cost_usd: float,
    model: str,
    confidence: float,
    sources: list[str],
    schema_version: str = "v1",
) -> int:
    cur = conn.execute(
        """
        INSERT INTO answers (
            question, content, retries, cost_usd, model,
            confidence, sources_json, schema_version
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            question,
            content,
            retries,
            cost_usd,
            model,
            confidence,
            json.dumps(sources),
            schema_version,
        ),
    )
    return int(cur.lastrowid or 0)


def query_results(
    conn: sqlite3.Connection, model: str | None = None
) -> list[dict[str, object]]:
    if model is None:
        cur = conn.execute("SELECT * FROM answers ORDER BY id")
    else:
        cur = conn.execute(
            "SELECT * FROM answers WHERE model = ? ORDER BY id", (model,)
        )
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row, strict=True)) for row in cur.fetchall()]
