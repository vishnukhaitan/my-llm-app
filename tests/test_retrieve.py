import pytest

from src.rag.retrieve import cosine, retrieve
from src.settings import Settings


def _fake_settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("USE_FAKE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    return Settings()


def test_cosine_identical_is_one() -> None:
    v = [1.0, 0.0, 0.0]
    assert cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_is_zero() -> None:
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_retrieve_ranks_matching_chunk_first() -> None:
    index = [
        {"chunk_id": "vpn#0", "source_id": "vpn", "text": "vpn", "vector": [0.0, 1.0]},
        {"chunk_id": "leave#0", "source_id": "leave", "text": "leave", "vector": [1.0, 0.0]},
    ]
    hits = retrieve(
        "how many leave days",
        index,
        k=1,
        embed_fn=lambda _texts: [[0.9, 0.1]],
    )
    assert len(hits) == 1
    assert hits[0]["chunk_id"] == "leave#0"
    assert hits[0]["score"] > 0.9


def test_retrieve_empty_index() -> None:
    assert retrieve("leave", [], k=3, embed_fn=lambda t: [[1.0]]) == []


def test_retrieve_rejects_dim_mismatch() -> None:
    index = [{"chunk_id": "x#0", "source_id": "x", "text": "x", "vector": [1.0, 0.0, 0.0]}]
    with pytest.raises(ValueError, match="dim"):
        retrieve("q", index, embed_fn=lambda _t: [[1.0, 0.0]])


def test_retrieve_fake_embed_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.rag.chunker import chunk_documents
    from src.rag.embeddings import build_index

    settings = _fake_settings(monkeypatch)
    docs = [{"source_id": "leave_policy", "text": "Eighteen days of annual leave."}]
    chunks = chunk_documents(docs)
    index = build_index(chunks, settings=settings)
    a = retrieve("eighteen days leave", index, k=1, settings=settings)
    b = retrieve("eighteen days leave", index, k=1, settings=settings)
    assert a[0]["chunk_id"] == b[0]["chunk_id"]
    assert a[0]["score"] == pytest.approx(b[0]["score"])
