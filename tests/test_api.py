from pathlib import Path

from fastapi.testclient import TestClient

import json

from src.api.main import app, get_api_settings, get_rag_index
from src.models import Answer
from src.rag.ask_rag import ask_rag
from src.rag.chunker import chunk_documents
from src.rag.embeddings import build_index, save_index
from src.settings import Settings
from src.store import connect, query_results


def test_health() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_batched_empty_question_is_422() -> None:
    with TestClient(app) as client:
        response = client.post("/ask_batched", json={"question": ""})
    assert response.status_code == 422


def _tiny_index(tmp_path: Path) -> Path:
    settings = Settings(use_fake=True, openai_api_key="", openai_model="gpt-4o-mini")
    docs = [{"source_id": "leave_policy", "text": "Eighteen days of annual leave."}]
    chunks = chunk_documents(docs)
    build_index(chunks, settings=settings)
    return save_index(chunks, tmp_path / "embeddings.json")


def test_ask_batched_fake_persists(tmp_path: Path) -> None:
    settings = Settings(
        use_fake=True,
        openai_api_key="",
        openai_model="gpt-4o-mini",
        results_db=tmp_path / "api.db",
        max_retries=0,
        retry_delay_s=0.0,
        fail_rate=0.0,
    )
    index_path = _tiny_index(tmp_path)
    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_rag_index] = lambda: json.loads(
        index_path.read_text(encoding="utf-8")
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/ask_batched",
                json={"question": "What is RAG?"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["content"].startswith("[FAKE]")
        assert "What is RAG?" in body["content"]
        assert body["schema_version"] == "v1"
        assert body["sources"] == ["leave_policy#0"]
        with connect(settings.results_db) as conn:
            rows = query_results(conn)
        assert len(rows) == 1
        assert rows[0]["question"] == "What is RAG?"
        assert "leave_policy#0" in str(rows[0]["sources_json"])
    finally:
        app.dependency_overrides.clear()


def test_ask_batched_missing_index_is_503(tmp_path: Path, monkeypatch) -> None:
    missing = tmp_path / "no-such-embeddings.json"
    monkeypatch.setattr("src.api.main.DEFAULT_INDEX_PATH", missing)
    app.dependency_overrides.pop(get_rag_index, None)
    with TestClient(app) as client:
        response = client.post("/ask_batched", json={"question": "leave days?"})
    assert response.status_code == 503
    assert "RAG index missing" in response.json()["detail"]


def test_ask_batched_uses_retrieved_sources(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        use_fake=False,
        openai_api_key="test-key",
        openai_model="gpt-4o-mini",
        results_db=tmp_path / "api.db",
        max_retries=0,
        retry_delay_s=0.0,
    )
    index = [
        {
            "chunk_id": "leave_policy#0",
            "source_id": "leave_policy",
            "text": "Eighteen days of annual leave.",
            "vector": [1.0, 0.0],
        }
    ]

    def fake_llm(question: str, hits: list[dict]) -> Answer:
        assert hits[0]["chunk_id"] == "leave_policy#0"
        return Answer(content=f"From context: {question}", confidence=0.8, sources=[])

    def routed(question: str, loaded: list[dict], settings: Settings) -> Answer:
        return ask_rag(
            question,
            loaded,
            settings=settings,
            embed_fn=lambda _texts: [[1.0, 0.0]],
            llm_fn=fake_llm,
        )

    monkeypatch.setattr("src.api.main.ask_rag", routed)
    app.dependency_overrides[get_api_settings] = lambda: settings
    app.dependency_overrides[get_rag_index] = lambda: index
    try:
        with TestClient(app) as client:
            response = client.post(
                "/ask_batched",
                json={"question": "How many leave days?"},
            )
        assert response.status_code == 200
        body = response.json()
        assert body["content"].startswith("From context:")
        assert body["sources"] == ["leave_policy#0"]
    finally:
        app.dependency_overrides.clear()


def test_ask_stream_fake_text(tmp_path: Path) -> None:
    settings = Settings(
        use_fake=True,
        openai_api_key="",
        openai_model="gpt-4o-mini",
        results_db=tmp_path / "stream.db",
        max_retries=0,
        retry_delay_s=0.0,
        fail_rate=0.0,
    )
    app.dependency_overrides[get_api_settings] = lambda: settings
    try:
        with (
            TestClient(app) as client,
            client.stream("POST", "/ask", json={"question": "What is RAG?"}) as response,
        ):
            assert response.status_code == 200
            assert "text/plain" in response.headers.get("content-type", "")
            text = "".join(response.iter_text())
        assert "[FAKE]" in text
        assert "What is RAG?" in text
        with connect(settings.results_db) as conn:
            assert query_results(conn) == []
    finally:
        app.dependency_overrides.clear()
