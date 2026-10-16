from src.models import Answer
from src.rag.ask_rag import ask_rag


def test_ask_rag_sets_sources_from_retrieve() -> None:
    index = [
        {"chunk_id": "leave#0", "source_id": "leave", "text": "Eighteen days.", "vector": [1.0, 0.0]},
        {"chunk_id": "vpn#0", "source_id": "vpn", "text": "Use the VPN.", "vector": [0.0, 1.0]},
    ]
    captured: dict = {}

    def llm_fn(question: str, hits: list[dict]) -> Answer:
        captured["question"] = question
        captured["ids"] = [h["chunk_id"] for h in hits]
        return Answer(content="Eighteen days of leave.", confidence=0.9, sources=["ignored"])

    answer = ask_rag(
        "How many leave days?",
        index,
        k=1,
        embed_fn=lambda _t: [[1.0, 0.0]],
        llm_fn=llm_fn,
    )
    assert answer.content.startswith("Eighteen")
    assert answer.sources == ["leave#0"]
    assert captured["ids"] == ["leave#0"]
    assert captured["question"] == "How many leave days?"


def test_ask_rag_empty_index_has_empty_sources() -> None:
    answer = ask_rag(
        "anything",
        [],
        k=3,
        embed_fn=lambda _t: [[1.0]],
        llm_fn=lambda _q, _h: Answer(content="I do not know.", sources=["x"]),
    )
    assert answer.sources == []
    assert answer.content == "I do not know."
