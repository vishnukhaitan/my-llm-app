"""NumPy cosine retrieve over the JSON embedding index (A5). Qdrant comes later."""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from src.rag.debug_log import log_step
from src.rag.embeddings import embed_batch
from src.settings import Settings

logger = logging.getLogger(__name__)

DEFAULT_K = 3

EmbedFn = Callable[[list[str]], list[list[float]]]


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Zero-length vectors score 0.0."""
    va = np.asarray(a, dtype=float)
    vb = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0.0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def retrieve(
    query: str,
    index: list[dict],
    k: int = DEFAULT_K,
    settings: Settings | None = None,
    embed_fn: EmbedFn | None = None,
) -> list[dict]:
    """Embed the query with the same model as the index, rank by cosine, return top k.

    Each hit is a shallow copy of the chunk plus a score field. Query embedding
    must match index dim (1536 real, 8 when USE_FAKE=true).
    """
    log_step(logger, f"STEP retrieve START (k={k} index={len(index)})")
    if not query.strip():
        raise ValueError("query must not be empty")
    if k <= 0:
        raise ValueError("k must be > 0")
    if not index:
        log_step(logger, "STEP retrieve DONE (empty index)")
        return []

    texts_fn = embed_fn if embed_fn is not None else (lambda texts: embed_batch(texts, settings))
    q_vec = texts_fn([query])[0]
    matrix = np.asarray([row["vector"] for row in index], dtype=float)
    query_arr = np.asarray(q_vec, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != query_arr.shape[0]:
        raise ValueError(
            "query dim %s does not match index dim %s"
            % (query_arr.shape[0], matrix.shape[1] if matrix.ndim == 2 else None)
        )

    q_norm = float(np.linalg.norm(query_arr)) or 1.0
    row_norms = np.linalg.norm(matrix, axis=1)
    row_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    scores = (matrix @ query_arr) / (row_norms * q_norm)

    top_n = min(k, len(index))
    order = np.argsort(-scores)[:top_n]
    hits: list[dict] = []
    for idx in order:
        i = int(idx)
        hit = dict(index[i])
        hit["score"] = float(scores[i])
        logger.debug(
            "hit %s source_id=%s score=%.4f chars=%s",
            hit.get("chunk_id"),
            hit.get("source_id"),
            hit["score"],
            len(hit.get("text") or ""),
        )
        hits.append(hit)
    log_step(
        logger,
        "STEP retrieve DONE top=" + ",".join(str(h.get("chunk_id")) for h in hits),
    )
    return hits
