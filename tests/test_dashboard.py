import json

import pytest

from ai_cost_cutter.dashboard import build_report
from ai_cost_cutter.ledger import CallRecord, Ledger


def sample_records():
    return [
        CallRecord("gpt-4o-mini", 100, 50, cost=0.001, baseline_cost=0.02),
        CallRecord("gpt-4o", 200, 100, cost=0.01),
        CallRecord("gpt-4o", 0, 0, cost=0.0, cached=True, baseline_cost=0.01),
    ]


def test_report_totals():
    report = build_report(sample_records())
    assert report.total_calls == 3
    assert report.cached_calls == 1
    assert report.total_cost == pytest.approx(0.011)
    # baseline: 0.02 + 0.01 (defaults to cost) + 0.01 = 0.04
    assert report.total_baseline_cost == pytest.approx(0.04)
    assert report.savings == pytest.approx(0.029)
    assert report.savings_pct == pytest.approx(0.029 / 0.04)
    assert report.cache_hit_rate == pytest.approx(1 / 3)


def test_report_by_model():
    report = build_report(sample_records())
    assert set(report.by_model) == {"gpt-4o", "gpt-4o-mini"}
    assert report.by_model["gpt-4o"].calls == 2
    assert report.by_model["gpt-4o-mini"].savings == pytest.approx(0.019)


def test_report_handles_empty():
    report = build_report([])
    assert report.total_calls == 0
    assert report.savings_pct == 0.0
    assert report.cache_hit_rate == 0.0


def test_render_text_contains_key_figures():
    text = build_report(sample_records()).render_text()
    assert "cost dashboard" in text.lower()
    assert "saved" in text.lower()
    assert "gpt-4o" in text


def test_render_html_is_valid_ish():
    html = build_report(sample_records()).render_html()
    assert html.startswith("<!doctype html>")
    assert "Cost dashboard" in html
    assert "gpt-4o" in html


def test_as_dict_is_json_serializable():
    d = build_report(sample_records()).as_dict()
    encoded = json.dumps(d)
    assert "by_model" in json.loads(encoded)


def test_build_report_from_ledger(tmp_path):
    ledger = Ledger(str(tmp_path / "l.jsonl"))
    ledger.record_call("gpt-4o", cost=0.01, baseline_cost=0.05)
    report = build_report(ledger)
    assert report.total_calls == 1
    assert report.savings == pytest.approx(0.04)
