import pytest

from src.rag.chunker import chunk_documents, chunk_text
from src.rag.corpus import load_corpus


def test_chunk_text_short_doc_is_one_chunk() -> None:
    assert chunk_text("eighteen days", size=500, overlap=50) == ["eighteen days"]


def test_chunk_text_empty() -> None:
    assert chunk_text("") == []


def test_chunk_text_overlap_advances() -> None:
    parts = chunk_text("abcdefghij", size=4, overlap=2)
    assert parts == ["abcd", "cdef", "efgh", "ghij"]
    assert all(len(p) <= 4 for p in parts)


def test_chunk_text_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("hello", size=4, overlap=4)


def test_chunk_documents_ids() -> None:
    docs = [{"source_id": "leave_policy", "text": "abcdefghij"}]
    chunks = chunk_documents(docs, size=4, overlap=2)
    assert [c["chunk_id"] for c in chunks] == [
        "leave_policy#0",
        "leave_policy#1",
        "leave_policy#2",
        "leave_policy#3",
    ]
    assert chunks[0]["source_id"] == "leave_policy"


def test_chunk_real_corpus_has_numbered_ids() -> None:
    docs = load_corpus()
    chunks = chunk_documents(docs, size=500, overlap=50)
    assert chunks
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert "leave_policy#0" in ids
    by_source: dict[str, list[int]] = {}
    for chunk in chunks:
        source, _, idx_s = chunk["chunk_id"].partition("#")
        assert chunk["source_id"] == source
        assert chunk["text"]
        by_source.setdefault(source, []).append(int(idx_s))
    for source, indexes in by_source.items():
        assert indexes == list(range(len(indexes)))
