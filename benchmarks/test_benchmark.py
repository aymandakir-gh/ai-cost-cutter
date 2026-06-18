"""The benchmark is part of CI: it must prove a >=30% cut and reproduce exactly."""

import pytest

from ai_cost_cutter.benchmark import build_workload, run_benchmark
from ai_cost_cutter.dashboard import build_report

# The headline contract for the project.
REQUIRED_CUT = 0.30


def test_benchmark_meets_required_cut():
    result = run_benchmark()
    assert result.savings_pct >= REQUIRED_CUT, (
        f"only cut {result.savings_pct:.1%}, need >= {REQUIRED_CUT:.0%}"
    )


def test_benchmark_is_reproducible():
    a = run_benchmark()
    b = run_benchmark()
    assert a.baseline_cost == b.baseline_cost
    assert a.breakdown == b.breakdown
    assert a.savings_pct == b.savings_pct


def test_each_mechanism_helps():
    result = run_benchmark()
    base = result.breakdown["baseline"]
    for config in ("compression_only", "routing_only", "cache_only"):
        assert result.breakdown[config] < base, f"{config} did not reduce cost"
    # Combining all three beats any single mechanism.
    assert result.breakdown["combined"] < result.breakdown["routing_only"]
    assert result.breakdown["combined"] < result.breakdown["cache_only"]
    assert result.breakdown["combined"] < result.breakdown["compression_only"]


def test_optimized_cheaper_than_baseline():
    result = run_benchmark()
    assert result.optimized_cost < result.baseline_cost
    assert result.savings > 0


def test_workload_is_nonempty_and_has_repeats():
    workload = build_workload()
    assert len(workload) > 10
    prompts = [r.prompt for r in workload]
    assert len(set(prompts)) < len(prompts)  # contains repeats (cache hits)


def test_dashboard_savings_match_benchmark():
    # The dashboard built from the combined-run ledger should report the same
    # savings the benchmark headline claims.
    result = run_benchmark()
    report = build_report(result.ledger)
    assert report.savings_pct == pytest.approx(result.savings_pct, rel=1e-9)
