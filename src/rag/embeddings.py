"""Embed chunks with text-embedding-3-small (A4). Fake vectors when USE_FAKE=true."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Callable

from openai import OpenAI

from src.rag.debug_log import log_step
from src.settings import Settings, get_settings

logger = logging.getLogger(__name__)

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
FAKE_EMBED_DIM = 8
DEFAULT_INDEX_PATH = Path("data/embeddings.json")
BATCH_SIZE = 100

EmbedFn = Callable[[list[str]], list[list[float]]]


def _fake_vector(text: str, dim: int = FAKE_EMBED_DIM) -> list[float]:
    """Deterministic unit vector so tests and USE_FAKE do not call the API."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(digest[i % len(digest)] / 255.0) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def embed_batch(texts: list[str], settings: Settings | None = None) -> list[list[float]]:
    """Embed a list of strings. Same model must be used later for the query.

    Args:
        texts: Chunk strings.
        settings: App settings. USE_FAKE skips OpenAI and returns 8-d vectors.

    Returns:
        One vector per input string.
    """
    if not texts:
        logger.debug("embed_batch: empty input")
        return []
    cfg = settings or get_settings()
    if cfg.use_fake:
        log_step(logger, f"STEP embed_batch START (USE_FAKE n={len(texts)} dim={FAKE_EMBED_DIM})")
        vectors = [_fake_vector(t) for t in texts]
        log_step(logger, f"STEP embed_batch DONE ({len(vectors)} vectors)")
        return vectors
    log_step(logger, f"STEP embed_batch START (OpenAI model={EMBED_MODEL} n={len(texts)})")
    logger.debug("embed_batch: model=%s n=%s", EMBED_MODEL, len(texts))
    if cfg.openai_base_url:
        client = OpenAI(api_key=cfg.openai_api_key, base_url=cfg.openai_base_url)
    else:
        client = OpenAI(api_key=cfg.openai_api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        logger.debug("embed_batch OpenAI slice %s:%s", i, i + len(batch))
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vectors.extend(item.embedding for item in resp.data)
    dim = len(vectors[0]) if vectors else 0
    log_step(logger, f"STEP embed_batch DONE ({len(vectors)} vectors dim={dim})")
    return vectors


def build_index(
    chunks: list[dict],
    settings: Settings | None = None,
    embed_fn: EmbedFn | None = None,
) -> list[dict]:
    """Attach a vector field to each chunk. Mutates and returns the same list."""
    log_step(logger, f"STEP build_index START ({len(chunks)} chunks)")
    logger.debug("build_index: %s chunks", len(chunks))
    texts = [c["text"] for c in chunks]
    vectors = embed_fn(texts) if embed_fn is not None else embed_batch(texts, settings)
    if len(vectors) != len(chunks):
        raise RuntimeError("embed returned %s vectors for %s chunks" % (len(vectors), len(chunks)))
    for chunk, vec in zip(chunks, vectors, strict=True):
        chunk["vector"] = vec
        logger.debug("indexed %s dim=%s", chunk.get("chunk_id"), len(vec))
    log_step(logger, "STEP build_index DONE")
    return chunks


def save_index(index: list[dict], path: Path | None = None) -> Path:
    """Write chunks (including vectors) to JSON."""
    out = path or DEFAULT_INDEX_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    log_step(logger, f"STEP save_index START ({out})")
    out.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    log_step(logger, f"STEP save_index DONE ({len(index)} chunks)")
    return out


def load_index(path: Path | None = None) -> list[dict]:
    """Load a previously saved embeddings JSON file."""
    src = path or DEFAULT_INDEX_PATH
    log_step(logger, f"STEP load_index ({src})")
    return json.loads(src.read_text(encoding="utf-8"))
