"""HTTP API: health, RAG ask_batched, and streaming ask (not RAG)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.main import persist_answer, stream_answer
from src.models import Answer, Question
from src.rag.ask_rag import ask_rag
from src.rag.debug_log import log_step
from src.rag.embeddings import DEFAULT_INDEX_PATH, load_index
from src.settings import Settings, get_settings

logger = logging.getLogger(__name__)

app = FastAPI(title="my-llm-app")

IndexLoader = Callable[[], list[dict]]


def get_api_settings() -> Settings:
    return get_settings()


def get_rag_index() -> list[dict]:
    """Load data/embeddings.json. Tests override this so they do not need the file."""
    path = Path(DEFAULT_INDEX_PATH)
    if not path.is_file():
        raise HTTPException(
            status_code=503,
            detail=f"RAG index missing: {path}. Run scripts/build_index.py first.",
        )
    log_step(logger, f"STEP api load_index ({path})")
    return load_index(path)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask_batched", response_model=Answer)
def ask_batched(
    q: Question,
    settings: Settings = Depends(get_api_settings),  # noqa: B008
    index: list[dict] = Depends(get_rag_index),  # noqa: B008
) -> Answer:
    """Retrieve from the JSON index, then generate. sources are chunk ids."""
    log_step(logger, "STEP api ask_batched")
    answer = ask_rag(q.question, index, settings=settings)
    persist_answer(q, answer, settings)
    return answer


@app.post("/ask")
def ask(
    q: Question,
    settings: Settings = Depends(get_api_settings),  # noqa: B008
) -> StreamingResponse:
    return StreamingResponse(
        stream_answer(q, settings),
        media_type="text/plain",
    )
