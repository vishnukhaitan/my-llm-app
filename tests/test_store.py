import sqlite3
from pathlib import Path

from src.main import persist_answer
from src.models import Answer, Question
from src.settings import Settings
from src.store import connect, ensure_schema, query_results, save_answer


def test_ensure_schema_creates_w4_columns(tmp_path: Path) -> None:
    db = tmp_path / "answers.db"
    ensure_schema(db)
    conn = sqlite3.connect(str(db))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(answers)").fetchall()}
    conn.close()
    for name in (
        "content",
        "cost_usd",
        "retries",
        "model",
        "confidence",
        "sources_json",
        "schema_version",
    ):
        assert name in cols


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "answers.db"
    ensure_schema(db)
    ensure_schema(db)
    ensure_schema(db)


def test_save_and_query_roundtrip(tmp_path: Path) -> None:
    db = tmp_path / "answers.db"
    with connect(db) as conn:
        row_id = save_answer(
            conn,
            question="Q?",
            content="hi",
            retries=0,
            cost_usd=0.001,
            model="gpt-4o-mini",
            confidence=0.9,
            sources=["doc1"],
        )
        rows = query_results(conn)
    assert row_id == 1
    assert len(rows) == 1
    assert rows[0]["question"] == "Q?"
    assert rows[0]["content"] == "hi"
    assert rows[0]["schema_version"] == "v1"
    assert rows[0]["model"] == "gpt-4o-mini"


def test_query_can_filter_by_model(tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    with connect(db) as conn:
        save_answer(
            conn,
            question="a",
            content="1",
            retries=0,
            cost_usd=0.0,
            model="gpt-4o-mini",
            confidence=1.0,
            sources=[],
        )
        save_answer(
            conn,
            question="b",
            content="2",
            retries=0,
            cost_usd=0.0,
            model="gpt-4o",
            confidence=1.0,
            sources=[],
        )
        mini = query_results(conn, model="gpt-4o-mini")
    assert len(mini) == 1
    assert mini[0]["question"] == "a"


def test_migrate_script_creates_db(tmp_path: Path) -> None:
    from scripts.migrate_store import main

    db = tmp_path / "answers.db"
    assert main(["migrate_store.py", str(db)]) == 0
    assert db.exists()
    assert main(["migrate_store.py", str(db)]) == 0


def test_persist_answer_writes_row(tmp_path: Path) -> None:
    db = tmp_path / "cli.db"
    settings = Settings(
        use_fake=True,
        openai_api_key="",
        openai_model="gpt-4o-mini",
        results_db=db,
    )
    question = Question(question="What is RAG?")
    answer = Answer(content="[FAKE] What is RAG?", confidence=1.0, sources=[])
    row_id = persist_answer(question, answer, settings)
    with connect(db) as conn:
        rows = query_results(conn)
    assert row_id == 1
    assert rows[0]["question"] == "What is RAG?"
    assert rows[0]["content"] == "[FAKE] What is RAG?"
