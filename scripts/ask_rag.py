"""Ask a question against data/embeddings.json (A5). Does not start FastAPI.

Usage (repo root):

    USE_FAKE=true python scripts/ask_rag.py "How many annual leave days?"
    python scripts/ask_rag.py "How many annual leave days?"
    python scripts/ask_rag.py "How many annual leave days?" --k 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ask_rag import ask_rag
from src.rag.embeddings import load_index
from src.settings import Settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Naive RAG over data/embeddings.json")
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=3, help="How many chunks to retrieve")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(name)s: %(message)s",
    )
    index = load_index()
    if settings.use_fake:
        print("USE_FAKE=true: 8-d index + fake answer (not production quality)")
    answer = ask_rag(args.question, index, settings=settings, k=args.k)
    print(answer.content)
    print(f"sources={answer.sources} confidence={answer.confidence} cost_usd={answer.cost_usd}")


if __name__ == "__main__":
    main()
