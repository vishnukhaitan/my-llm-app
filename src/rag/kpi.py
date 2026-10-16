"""KPI math for the naive RAG golden set (A7). No network calls."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_GOLDEN_PATH = Path("data/golden_questions.jsonl")


def load_golden(path: Path | None = None) -> list[dict[str, str]]:
    """Load question rows: id, question, expected_source_id."""
    src = path or DEFAULT_GOLDEN_PATH
    rows: list[dict[str, str]] = []
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "id": str(row["id"]),
                "question": str(row["question"]),
                "expected_source_id": str(row["expected_source_id"]),
            }
        )
    return rows


def retrieval_hit(chunk_ids: list[str], expected_source_id: str) -> bool:
    """True when any retrieved chunk belongs to the expected document."""
    prefix = expected_source_id + "#"
    return any(chunk_id == expected_source_id or chunk_id.startswith(prefix) for chunk_id in chunk_ids)


def p50(values: list[float]) -> float:
    """Median. Even counts use the mean of the two middle values."""
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return float(ordered[mid - 1] + ordered[mid]) / 2.0


def summarize(rows: list[dict]) -> dict[str, float | int]:
    """Aggregate hit rate, mean chat cost, and p50 latency in milliseconds.

    Each row needs hit (bool), cost_usd (float), latency_ms (float).
    cost_usd is the chat completion only; the question embedding is extra.
    """
    n = len(rows)
    if n == 0:
        return {"n": 0, "hits": 0, "hit_rate": 0.0, "mean_cost_usd": 0.0, "p50_latency_ms": 0.0}
    hits = sum(1 for row in rows if row["hit"])
    return {
        "n": n,
        "hits": hits,
        "hit_rate": hits / n,
        "mean_cost_usd": sum(float(row["cost_usd"]) for row in rows) / n,
        "p50_latency_ms": p50([float(row["latency_ms"]) for row in rows]),
    }
