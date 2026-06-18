import pytest

from ai_cost_cutter.estimator import (
    CostEstimate,
    estimate,
    estimate_messages,
    estimate_tokens,
)
from ai_cost_cutter.pricing import ModelPrice


def test_estimate_tokens_matches_price_table():
    # gpt-4o: $2.50 / 1M input, $10.00 / 1M output.
    est = estimate_tokens("gpt-4o", input_tokens=1_000_000, output_tokens=1_000_000)
    assert est.input_cost == pytest.approx(2.50)
    assert est.output_cost == pytest.approx(10.00)
    assert est.total_cost == pytest.approx(12.50)
    assert est.total_tokens == 2_000_000


def test_cheaper_model_costs_less_for_same_tokens():
    big = estimate_tokens("gpt-4o", 10_000, 1_000)
    small = estimate_tokens("gpt-4o-mini", 10_000, 1_000)
    assert small.total_cost < big.total_cost


def test_estimate_from_prompt_counts_input_tokens():
    est = estimate("gpt-4o", "hello " * 100, expected_output_tokens=50)
    assert est.input_tokens > 0
    assert est.output_tokens == 50
    assert est.total_cost > 0


def test_estimate_messages():
    est = estimate_messages(
        "claude-3-5-haiku",
        [{"role": "user", "content": "Summarize this."}],
        expected_output_tokens=20,
    )
    assert est.input_tokens > 0
    assert est.output_tokens == 20


def test_negative_tokens_rejected():
    with pytest.raises(ValueError):
        estimate_tokens("gpt-4o", -1, 0)


def test_custom_prices_used():
    table = {"x": ModelPrice(1.0, 1.0, "x")}
    est = estimate_tokens("x", 1_000_000, 0, prices=table)
    assert est.total_cost == pytest.approx(1.0)


def test_local_model_is_free():
    est = estimate_tokens("local", 1_000_000, 1_000_000)
    assert est.total_cost == 0.0


def test_cost_estimate_as_dict_roundtrips_fields():
    est = CostEstimate("m", 10, 20, 0.1, 0.2)
    d = est.as_dict()
    assert d["total_cost"] == pytest.approx(0.3)
    assert d["input_tokens"] == 10
