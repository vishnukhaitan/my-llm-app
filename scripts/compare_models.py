"""Run the same questions through one or more models and persist results.

Usage:
    python scripts/compare_models.py
    python scripts/compare_models.py --models gpt-4o-mini,gpt-4o --db data/answers.db

Fake (no API spend):
    USE_FAKE=true python scripts/compare_models.py
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.main import ask_llm, persist_answer
from src.models import Question
from src.settings import Settings


def load_questions(path: Path) -> list[str]:
    with path.open() as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if rows and rows[0] and rows[0][0].lower().startswith("question"):
        rows = rows[1:]
    return [row[0].strip() for row in rows if row and row[0].strip()]


def run_one_model(
    questions: list[str],
    model: str,
    settings: Settings,
) -> dict[str, object]:
    cfg = settings.model_copy(update={"openai_model": model})
    started = time.perf_counter()
    total_cost = 0.0
    total_retries = 0
    for text in questions:
        question = Question(question=text)
        answer = ask_llm(question, cfg)
        persist_answer(question, answer, cfg)
        total_cost += answer.cost_usd
        total_retries += answer.retries
    elapsed = time.perf_counter() - started
    n = len(questions)
    return {
        "model": model,
        "n": n,
        "total_cost_usd": round(total_cost, 6),
        "total_retries": total_retries,
        "elapsed_s": round(elapsed, 2),
        "avg_cost_usd": round(total_cost / n, 6) if n else 0.0,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare LLM models on the same questions")
    parser.add_argument("--models", default="gpt-4o-mini,gpt-4o")
    parser.add_argument("--questions", default="data/questions.csv")
    parser.add_argument("--db", default="data/answers.db")
    return parser.parse_args(argv)


def print_table(summaries: list[dict[str, object]]) -> None:
    print()
    print("=" * 60)
    print(f"{'Model':<20}  {'n':>4}  {'Total $':>10}  {'Avg $/q':>10}  {'Time':>8}")
    print("-" * 60)
    for summary in summaries:
        print(
            f"{summary['model']:<20}  {summary['n']:>4}  "
            f"{summary['total_cost_usd']:>10.6f}  "
            f"{summary['avg_cost_usd']:>10.6f}  "
            f"{summary['elapsed_s']:>7.2f}s"
        )
    if len(summaries) == 2:
        first, second = summaries
        first_cost = float(first["total_cost_usd"])
        second_cost = float(second["total_cost_usd"])
        ratio = second_cost / first_cost if first_cost else 0.0
        print()
        print(
            f"  → {second['model']} cost {ratio:.1f}× more than "
            f"{first['model']} on the same questions."
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_questions(Path(args.questions))
    if not questions:
        print("No questions found.", file=sys.stderr)
        return 1
    base = Settings()
    base = base.model_copy(update={"results_db": Path(args.db)})
    print(f"Running {len(questions)} questions through: {args.models}")
    print(f"Persisting to: {args.db}")
    print(f"USE_FAKE={base.use_fake}")
    print()
    summaries: list[dict[str, object]] = []
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    for model in models:
        print(f"  → {model} …")
        summary = run_one_model(questions, model, base)
        summaries.append(summary)
        print(
            f"    done — n={summary['n']}  "
            f"cost=${summary['total_cost_usd']}  "
            f"avg=${summary['avg_cost_usd']}/q  "
            f"retries={summary['total_retries']}  "
            f"time={summary['elapsed_s']}s"
        )
    print_table(summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
