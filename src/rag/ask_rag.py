"""Naive RAG generate step: retrieve top-k chunks, then ask the chat model (A5)."""

from __future__ import annotations

import logging
from typing import Callable

from src.fake_llm import fake_ask_llm
from src.models import Answer, Question
from src.rag.debug_log import log_step
from src.rag.retrieve import DEFAULT_K, retrieve
from src.settings import Settings, get_settings
from src.tools import ANSWER_TOOL

logger = logging.getLogger(__name__)

SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you do "
    "not know rather than guessing. Put the chunk ids you used in sources."
)

LlmFn = Callable[[str, list[dict]], Answer]


def _format_context(hits: list[dict]) -> str:
    return "\n\n".join(f"[{hit['chunk_id']}]\n{hit['text']}" for hit in hits)


def _call_rag_llm(question: str, context: str, settings: Settings) -> Answer:
    from src.main import _answer_from_response, _make_client

    user = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    logger.debug("rag prompt chars=%s", len(user))
    response = _make_client(settings).chat.completions.create(
        model=settings.openai_model,
        temperature=0.0,
        max_tokens=settings.llm_max_output_tokens,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        tools=[ANSWER_TOOL],
        tool_choice={"type": "function", "function": {"name": "answer_question"}},
    )
    return _answer_from_response(response, settings)


def ask_rag(
    question: str,
    index: list[dict],
    settings: Settings | None = None,
    k: int = DEFAULT_K,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
    llm_fn: LlmFn | None = None,
) -> Answer:
    """Retrieve k chunks, generate an answer, set sources to retrieved chunk ids.

    Tests pass embed_fn and llm_fn so this never needs a live API key.
    FastAPI wiring is A6 — this function is the library entry point.
    """
    cfg = settings if settings is not None else get_settings()
    log_step(logger, f"STEP ask_rag START k={k}")
    hits = retrieve(question, index, k=k, settings=cfg, embed_fn=embed_fn)
    source_ids = [str(h["chunk_id"]) for h in hits]
    context = _format_context(hits)
    logger.debug("ask_rag retrieved=%s", source_ids)

    if llm_fn is not None:
        answer = llm_fn(question, hits)
    elif cfg.use_fake:
        answer = fake_ask_llm(Question(question=question), fail_rate=cfg.fail_rate)
    else:
        answer = _call_rag_llm(question, context, cfg)

    filled = answer.model_copy(update={"sources": source_ids})
    log_step(logger, "STEP ask_rag DONE sources=" + ",".join(source_ids))
    return filled
