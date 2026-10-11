from pathlib import Path

from scripts.compare_models import load_questions, main, parse_args, run_one_model
from src.settings import Settings
from src.store import connect, query_results


def test_load_questions_skips_header(tmp_path: Path) -> None:
    csv_path = tmp_path / "q.csv"
    csv_path.write_text("question\nWhat is RAG?\nWhat is a token?\n", encoding="utf-8")
    assert load_questions(csv_path) == ["What is RAG?", "What is a token?"]


def test_parse_args_defaults() -> None:
    args = parse_args([])
    assert "gpt-4o-mini" in args.models
    assert args.questions == "data/questions.csv"


def test_run_one_model_fake_persists(tmp_path: Path) -> None:
    db = tmp_path / "cmp.db"
    settings = Settings(
        use_fake=True,
        openai_api_key="",
        openai_model="gpt-4o-mini",
        results_db=db,
        max_retries=0,
        retry_delay_s=0.0,
        fail_rate=0.0,
    )
    summary = run_one_model(["What is RAG?"], "gpt-4o-mini", settings)
    assert summary["n"] == 1
    assert summary["total_cost_usd"] == 0.0
    with connect(db) as conn:
        rows = query_results(conn, model="gpt-4o-mini")
    assert len(rows) == 1
    assert "RAG" in str(rows[0]["question"])


def test_main_fake_two_models(tmp_path: Path, monkeypatch) -> None:
    csv_path = tmp_path / "q.csv"
    csv_path.write_text("question\nHello?\n", encoding="utf-8")
    db = tmp_path / "out.db"
    monkeypatch.setenv("USE_FAKE", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")
    rc = main(
        [
            "--questions",
            str(csv_path),
            "--db",
            str(db),
            "--models",
            "gpt-4o-mini,gpt-4o",
        ]
    )
    assert rc == 0
    with connect(db) as conn:
        rows = query_results(conn)
    assert len(rows) == 2
    assert {row["model"] for row in rows} == {"gpt-4o-mini", "gpt-4o"}
