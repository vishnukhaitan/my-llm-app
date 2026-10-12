from pathlib import Path

import pytest

from src.rag.corpus import clean_text, load_corpus, parse_document

CORPUS = Path("data/corpus")
EXPECTED_SOURCE_IDS = {
    "code_of_conduct",
    "data_retention",
    "expense_policy",
    "it_onboarding",
    "leave_policy",
    "llm_assistant_usage",
    "meeting_rooms",
    "parental_leave",
    "security_passwords",
    "sick_leave",
    "vpn_setup",
    "wfh_policy",
}


def test_clean_text_normalises_newlines_and_blank_runs() -> None:
    raw = "Leave policy\r\n\r\n\r\n  \nAnnual leave is 18 days.\n"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\n\n\n" not in cleaned
    assert cleaned.startswith("Leave policy")
    assert cleaned.endswith("Annual leave is 18 days.")


def test_parse_document_rejects_pdf() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        parse_document(Path("policies.pdf"))


def test_load_corpus_tmp_dir(tmp_path: Path) -> None:
    (tmp_path / "leave_policy.md").write_text("Eighteen days.\n\n\nMore text.\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("VPN is required.\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Ignore me.\n", encoding="utf-8")
    (tmp_path / "scan.pdf").write_text("not parsed", encoding="utf-8")
    docs = load_corpus(tmp_path)
    ids = {d["source_id"] for d in docs}
    assert ids == {"leave_policy", "notes"}
    leave = next(d for d in docs if d["source_id"] == "leave_policy")
    assert "Eighteen days." in leave["text"]
    assert "\n\n\n" not in leave["text"]


def test_load_corpus_one_source_id() -> None:
    docs = load_corpus(CORPUS, source_id="leave_policy")
    assert len(docs) == 1
    assert docs[0]["source_id"] == "leave_policy"
    assert docs[0]["text"].strip()


def test_load_real_corpus_has_twelve_nonempty_policies() -> None:
    docs = load_corpus(CORPUS)
    ids = {d["source_id"] for d in docs}
    assert ids == EXPECTED_SOURCE_IDS
    assert len(docs) == 12
    for doc in docs:
        assert doc["text"].strip()
