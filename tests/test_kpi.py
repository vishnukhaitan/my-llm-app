import pytest

from src.rag.kpi import load_golden, p50, retrieval_hit, summarize


def test_retrieval_hit_matches_chunk_prefix() -> None:
    assert retrieval_hit(["leave_policy#0", "vpn_setup#1"], "leave_policy")
    assert not retrieval_hit(["vpn_setup#0"], "leave_policy")


def test_p50_odd_and_even() -> None:
    assert p50([30, 10, 20]) == 20
    assert p50([10, 40]) == 25


def test_summarize_hit_rate_and_means() -> None:
    summary = summarize(
        [
            {"hit": True, "cost_usd": 0.0002, "latency_ms": 100},
            {"hit": False, "cost_usd": 0.0004, "latency_ms": 300},
        ]
    )
    assert summary["n"] == 2
    assert summary["hits"] == 1
    assert summary["hit_rate"] == 0.5
    assert summary["mean_cost_usd"] == pytest.approx(0.0003)
    assert summary["p50_latency_ms"] == 200


def test_golden_questions_cover_corpus_ids() -> None:
    from src.rag.corpus import load_corpus

    golden = load_golden()
    expected = {row["expected_source_id"] for row in golden}
    corpus_ids = {doc["source_id"] for doc in load_corpus()}
    assert expected == corpus_ids
    assert len(golden) == len(expected)
