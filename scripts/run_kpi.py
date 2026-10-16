"""Run the golden questions through ask_rag and write a KPI snapshot.

Usage (repo root, venv active, real 1536-d index):

    python scripts/run_kpi.py
    python scripts/run_kpi.py --out docs/kpi/naive-rag-snapshot.md

The index must include every expected source. A --source leave_policy
index will miss the other 11 questions.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ask_rag import ask_rag
from src.rag.debug_log import log_step
from src.rag.embeddings import load_index
from src.rag.kpi import load_golden, retrieval_hit, summarize
from src.settings import Settings

logger = logging.getLogger(__name__)
DEFAULT_OUT = Path("docs/kpi/naive-rag-snapshot.md")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive RAG KPI snapshot")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--k", type=int, default=3)
    return parser.parse_args()


def _render(rows: list[dict], summary: dict, index_sources: list[str]) -> str:
    lines = [
        "# Naive RAG KPI snapshot (A7)",
        "",
        "Ran `data/golden_questions.jsonl` through `ask_rag` (k=3) on `data/embeddings.json`.",
        "",
        f"Index source ids: {', '.join(index_sources) if index_sources else '(empty)'}",
        "",
        "## Headline numbers",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Questions | {summary['n']} |",
        f"| Retrieval hits | {summary['hits']} / {summary['n']} |",
        f"| Retrieval hit rate | {summary['hit_rate']:.0%} |",
        f"| Mean chat cost / query | ${summary['mean_cost_usd']:.6f} |",
        f"| Latency p50 | {summary['p50_latency_ms']:.0f} ms |",
        "",
        "Hit means the expected document id appears in the retrieved chunk ids (`leave_policy#0` counts for `leave_policy`).",
        "Cost is the chat completion only. Embedding the question is a separate call and is not included.",
        "",
        "## Per question",
        "",
        "| id | expected | hit | top sources | ms | cost_usd |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        sources = ", ".join(row["sources"])
        lines.append(
            f"| {row['id']} | {row['expected_source_id']} | {'yes' if row['hit'] else 'no'} | {sources} | {row['latency_ms']:.0f} | {row['cost_usd']:.6f} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(name)s: %(message)s",
    )
    if settings.use_fake:
        raise SystemExit("USE_FAKE=true cannot score a 1536-d index. Leave USE_FAKE=false.")
    golden = load_golden()
    index = load_index()
    index_sources = sorted({str(row.get("source_id")) for row in index})
    log_step(logger, f"STEP kpi START n={len(golden)} index_sources={len(index_sources)}")
    rows: list[dict] = []
    for item in golden:
        started = time.perf_counter()
        answer = ask_rag(item["question"], index, settings=settings, k=args.k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        hit = retrieval_hit(answer.sources, item["expected_source_id"])
        row = {
            "id": item["id"],
            "expected_source_id": item["expected_source_id"],
            "sources": answer.sources,
            "hit": hit,
            "latency_ms": elapsed_ms,
            "cost_usd": answer.cost_usd,
        }
        rows.append(row)
        logger.debug(
            "kpi %s expected=%s hit=%s sources=%s ms=%.0f",
            item["id"],
            item["expected_source_id"],
            hit,
            answer.sources,
            elapsed_ms,
        )
    summary = summarize(rows)
    text = _render(rows, summary, index_sources)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    log_step(
        logger,
        f"STEP kpi DONE hits={summary['hits']}/{summary['n']} hit_rate={summary['hit_rate']:.0%}",
    )
    print(text)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
