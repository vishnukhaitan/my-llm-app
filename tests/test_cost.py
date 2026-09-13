import pytest

from src.cost import RATES, compute_cost_usd


def test_gpt_4o_mini_known_counts() -> None:
    # (100 * 0.15 + 50 * 0.60) / 1_000_000
    assert compute_cost_usd("gpt-4o-mini", 100, 50) == pytest.approx(45.0 / 1_000_000)


def test_gpt_4o_known_counts() -> None:
    # (100 * 2.50 + 50 * 10.00) / 1_000_000
    assert compute_cost_usd("gpt-4o", 100, 50) == pytest.approx(750.0 / 1_000_000)


def test_zero_tokens_costs_nothing() -> None:
    assert compute_cost_usd("gpt-4o-mini", 0, 0) == 0.0


def test_unknown_model_returns_zero() -> None:
    assert compute_cost_usd("not-a-real-model", 1000, 1000) == 0.0


def test_gpt_4o_costs_more_than_mini() -> None:
    mini = compute_cost_usd("gpt-4o-mini", 500, 200)
    full = compute_cost_usd("gpt-4o", 500, 200)
    assert full > mini
    assert full / mini > 5


def test_local_ollama_model_is_free() -> None:
    assert compute_cost_usd("llama3.2:3b", 10_000, 5_000) == 0.0


def test_rates_table_has_lab_models() -> None:
    assert "gpt-4o-mini" in RATES
    assert "gpt-4o" in RATES
    for rates in RATES.values():
        assert len(rates) == 2
        in_rate, out_rate = rates
        assert in_rate >= 0
        assert out_rate >= 0
