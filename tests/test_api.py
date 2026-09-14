from pathlib import Path

from fastapi.testclient import TestClient

from src.api.main import app, get_api_settings
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
    app.dependency_overrides[get_api_settings] = lambda: settings
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
        with connect(settings.results_db) as conn:
            rows = query_results(conn)
        assert len(rows) == 1
        assert rows[0]["question"] == "What is RAG?"
    finally:
        app.dependency_overrides.clear()
