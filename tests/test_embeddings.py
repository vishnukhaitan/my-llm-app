from pathlib import Path

import pytest

from src.rag.chunker import chunk_documents
from src.rag.embeddings import (
    FAKE_EMBED_DIM,
    build_index,
    embed_batch,
    load_index,
    save_index,
)
from src.settings import Settings


def _fake_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("USE_FAKE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    return Settings()


def test_embed_batch_fake_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _fake_settings(monkeypatch)
    a = embed_batch(["annual leave is 18 days"], settings)
    b = embed_batch(["annual leave is 18 days"], settings)
    c = embed_batch(["vpn is required"], settings)
    assert len(a) == 1
    assert len(a[0]) == FAKE_EMBED_DIM
    assert a[0] == b[0]
    assert a[0] != c[0]


def test_build_and_save_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _fake_settings(monkeypatch)
    docs = [{"source_id": "leave_policy", "text": "Eighteen days of annual leave."}]
    chunks = chunk_documents(docs, size=500, overlap=50)
    indexed = build_index(chunks, settings=settings)
    assert indexed[0]["chunk_id"] == "leave_policy#0"
    assert len(indexed[0]["vector"]) == FAKE_EMBED_DIM
    path = save_index(indexed, tmp_path / "embeddings.json")
    loaded = load_index(path)
    assert loaded[0]["chunk_id"] == "leave_policy#0"
    assert loaded[0]["vector"] == indexed[0]["vector"]


def test_build_index_one_corpus_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.rag.corpus import load_corpus

    settings = _fake_settings(monkeypatch)
    docs = load_corpus(source_id="leave_policy")
    assert len(docs) == 1
    chunks = chunk_documents(docs)
    indexed = build_index(chunks, settings=settings)
    assert all(c["source_id"] == "leave_policy" for c in indexed)
    assert indexed[0]["chunk_id"].startswith("leave_policy#")
    assert len(indexed[0]["vector"]) == FAKE_EMBED_DIM


def test_build_index_custom_embed_fn() -> None:
    chunks = [{"chunk_id": "x#0", "source_id": "x", "text": "hello"}]
    build_index(chunks, embed_fn=lambda texts: [[1.0, 0.0] for _ in texts])
    assert chunks[0]["vector"] == [1.0, 0.0]
