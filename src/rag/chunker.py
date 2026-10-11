"""Sliding-window chunking (A3). Recursive / heading-aware split comes later."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

DEFAULT_SIZE = 500
DEFAULT_OVERLAP = 50


def chunk_text(
    text: str,
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[str]:
    """Split text into overlapping character windows.

    Args:
        text: Cleaned document text.
        size: Window length in characters.
        overlap: Characters reused at the start of the next window.
            Must be >= 0 and < size so the window always advances.

    Returns:
        Chunk strings in order. Empty text → empty list. Short text → one chunk.
    """
    if size <= 0:
        raise ValueError("size must be > 0")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be >= 0 and < size")
    if not text:
        logger.debug("chunk_text: empty input → 0 chunks")
        return []
    if len(text) <= size:
        logger.debug("chunk_text: %s chars fits in one window size=%s", len(text), size)
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        logger.debug("chunk_text window start=%s end=%s len=%s", start, end, end - start)
        if end == len(text):
            break
        start = end - overlap
    logger.debug("chunk_text: %s chars → %s chunks size=%s overlap=%s", len(text), len(chunks), size, overlap)
    return chunks


def chunk_documents(
    documents: list[dict[str, str]],
    size: int = DEFAULT_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict[str, str]]:
    """Turn loaded corpus docs into chunks with ids like leave_policy#0.

    Args:
        documents: Output of load_corpus (source_id + text).
        size: Window length.
        overlap: Window overlap.

    Returns:
        Flat list of {chunk_id, source_id, text}.
    """
    logger.debug(
        "chunk_documents: %s docs size=%s overlap=%s",
        len(documents),
        size,
        overlap,
    )
    chunks: list[dict[str, str]] = []
    for doc in documents:
        source_id = doc["source_id"]
        parts = chunk_text(doc["text"], size=size, overlap=overlap)
        logger.debug("source_id=%s → %s chunks (first id=%s#0)", source_id, len(parts), source_id)
        for idx, part in enumerate(parts):
            chunk_id = f"{source_id}#{idx}"
            logger.debug("  %s chars=%s", chunk_id, len(part))
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_id": source_id,
                    "text": part,
                }
            )
    logger.debug("chunk_documents done: %s chunks total", len(chunks))
    return chunks
