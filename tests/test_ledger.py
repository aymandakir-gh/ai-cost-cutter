import pytest

from ai_cost_cutter.ledger import CallRecord, Ledger


def test_record_call_estimates_cost_when_missing():
    ledger = Ledger()
    rec = ledger.record_call("gpt-4o", input_tokens=1_000_000, output_tokens=0)
    assert rec.cost == pytest.approx(2.50)
    assert len(ledger.records()) == 1


def test_record_call_keeps_explicit_cost():
    ledger = Ledger()
    rec = ledger.record_call("gpt-4o", cost=0.01, baseline_cost=0.05)
    assert rec.cost == 0.01
    assert rec.savings == pytest.approx(0.04)


def test_unknown_model_cost_defaults_to_zero():
    ledger = Ledger()
    rec = ledger.record_call("mystery-model", input_tokens=100)
    assert rec.cost == 0.0


def test_cache_hit_record():
    ledger = Ledger()
    rec = ledger.record_cache_hit("gpt-4o", avoided_cost=0.02)
    assert rec.cached is True
    assert rec.cost == 0.0
    assert rec.savings == pytest.approx(0.02)


def test_effective_baseline_defaults_to_cost():
    rec = CallRecord("gpt-4o", cost=0.03)
    assert rec.effective_baseline == 0.03
    assert rec.savings == 0.0


def test_timestamp_assigned():
    ledger = Ledger(time_fn=lambda: 123.0)
    rec = ledger.record_call("gpt-4o", cost=0.0)
    assert rec.timestamp == 123.0


def test_jsonl_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger = Ledger(path)
    ledger.record_call("gpt-4o", input_tokens=10, output_tokens=5, cost=0.001)
    ledger.record_cache_hit("gpt-4o", avoided_cost=0.001)
    # A fresh ledger reading the same file recovers both records.
    reopened = Ledger(path)
    recs = reopened.records()
    assert len(recs) == 2
    assert recs[1].cached is True


def test_clear_removes_records(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger = Ledger(path)
    ledger.record_call("gpt-4o", cost=0.001)
    ledger.clear()
    assert ledger.records() == []


def test_record_route_result():
    from ai_cost_cutter.router import Router

    provider = lambda m, p: "The answer is 4."  # noqa: E731
    router = Router(["gpt-4o-mini", "gpt-4o"], provider)
    result = router.route("2+2?")
    ledger = Ledger()
    rec = ledger.record_route(result)
    assert rec.model == result.model
    assert rec.cost == pytest.approx(result.total_cost)
    assert rec.baseline_cost == pytest.approx(result.baseline_cost)
