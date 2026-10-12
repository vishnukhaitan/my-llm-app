"""Build data/embeddings.json from the corpus.

Usage (repo root):

    USE_FAKE=true python scripts/build_index.py
    USE_FAKE=true python scripts/build_index.py --source leave_policy
    python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.chunker import chunk_documents
from src.rag.corpus import load_corpus
from src.rag.embeddings import build_index, save_index
from src.settings import Settings


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed corpus chunks into data/embeddings.json")
    parser.add_argument(
        "--source",
        metavar="SOURCE_ID",
        help="Embed only this file stem, e.g. leave_policy (not all 12 docs)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    settings = Settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(name)s: %(message)s",
    )
    docs = load_corpus(source_id=args.source)
    if args.source and not docs:
        raise SystemExit(f"No corpus document with source_id={args.source!r}")
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents → {len(chunks)} chunks")
    if settings.use_fake:
        print("USE_FAKE=true: writing 8-d placeholder vectors (not for production retrieve)")
    build_index(chunks, settings=settings)
    path = save_index(chunks)
    dim = len(chunks[0]["vector"]) if chunks else 0
    print(f"Wrote {path} ({len(chunks)} chunks, dim={dim})")


if __name__ == "__main__":
    main()
