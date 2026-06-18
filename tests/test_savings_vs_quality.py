import json

import pytest

from ai_cost_cutter.savings_vs_quality import (
    ComparisonReport,
    EvalConfig,
    EvalResult,
    Sample,
    compare_configs,
    contains_match,
    evaluate,
    exact_match,
    normalized_match,
    token_f1,
)

CHEAP = "gpt-4o-mini"
PREMIUM = "gpt-4o"


# --- scorers --------------------------------------------------------------


def test_exact_match():
    assert exact_match("Paris", "Paris") == 1.0
    assert exact_match("Paris", "paris") == 0.0


def test_normalized_match_ignores_case_and_punctuation():
    assert normalized_match("Paris.", "  paris ") == 1.0
    assert normalized_match("Paris", "London") == 0.0


def test_contains_match():
    assert contains_match("Paris", "The capital is Paris, of course.") == 1.0
    assert contains_match("Paris", "It is London.") == 0.0
    assert contains_match("", "anything") == 1.0


def test_token_f1_partial_credit():
    full = token_f1("the quick brown fox", "the quick brown fox")
    partial = token_f1("the quick brown fox", "the quick fox")
    none = token_f1("alpha beta", "gamma delta")
    assert full == pytest.approx(1.0)
    assert 0 < partial < 1
    assert none == 0.0


# --- workload + provider --------------------------------------------------

WORKLOAD = [
    Sample("What is the capital of France?", "Paris"),
    Sample("What is 2 + 2?", "4"),
    Sample("Prove sqrt(2) is irrational.", "irrational proof by contradiction"),
    Sample("Explain the halting problem.", "undecidable diagonalization argument"),
]

# Which prompts the cheap model gets wrong (gives a hedging, low-quality reply).
_CHEAP_FAILS = {
    "Prove sqrt(2) is irrational.",
    "Explain the halting problem.",
}
_ANSWERS = {s.prompt: s.reference for s in WORKLOAD}


def provider(model, prompt):
    # Match the (possibly compressed) prompt back to a known sample.
    norm = " ".join(prompt.split())
    sample = None
    for p, ref in _ANSWERS.items():
        if " ".join(p.split()) in norm:
            sample = (p, ref)
            break
    if sample is None:  # pragma: no cover - workload always matches
        return "unknown"
    p, ref = sample
    if model == CHEAP and p in _CHEAP_FAILS:
        return "I'm not sure, I don't know the exact answer."
    return ref


# --- evaluate -------------------------------------------------------------


def test_baseline_config_has_full_quality_and_no_savings():
    result = evaluate(WORKLOAD, provider, EvalConfig(model=PREMIUM), scorer=contains_match)
    assert result.samples == 4
    assert result.quality == pytest.approx(1.0)
    assert result.baseline_quality == pytest.approx(1.0)
    assert result.quality_retention == pytest.approx(1.0)
    # Comparing premium-to-premium baseline -> no savings.
    assert result.savings_pct == pytest.approx(0.0)


def test_single_cheap_model_saves_cost_but_drops_quality():
    result = evaluate(
        WORKLOAD,
        provider,
        EvalConfig(model=CHEAP),
        scorer=contains_match,
        baseline_model=PREMIUM,
    )
    # Cheaper than the premium baseline.
    assert result.savings_pct > 0
    # The cheap model flubs the two hard questions.
    assert result.quality == pytest.approx(0.5)
    assert result.quality_retention == pytest.approx(0.5)
    assert result.quality_drop == pytest.approx(0.5)


def test_routing_recovers_quality_while_still_saving():
    routed = evaluate(
        WORKLOAD,
        provider,
        EvalConfig(models=[CHEAP, PREMIUM]),
        scorer=contains_match,
    )
    # Escalation restores full quality...
    assert routed.quality == pytest.approx(1.0)
    assert routed.quality_retention == pytest.approx(1.0)
    # ...while still saving versus always-premium (easy ones answered cheap).
    assert routed.savings_pct > 0


def test_cache_counts_hits_and_lowers_cost():
    # Repeat the workload so the second pass is all cache hits.
    repeated = WORKLOAD + WORKLOAD
    cached = evaluate(
        repeated,
        provider,
        EvalConfig(model=PREMIUM, use_cache=True),
        scorer=contains_match,
        baseline_model=PREMIUM,
    )
    assert cached.cache_hits == len(WORKLOAD)
    # Half the calls were free, so well under the always-pay baseline.
    assert cached.savings_pct > 0.4
    # Quality unaffected by caching.
    assert cached.quality == pytest.approx(1.0)


def test_compression_is_applied_and_quality_tracked():
    bloated = [
        Sample(s.prompt + "\n\n\n   " + s.prompt, s.reference) for s in WORKLOAD
    ]
    result = evaluate(
        bloated,
        provider,
        EvalConfig(
            model=PREMIUM,
            compress_strategies=["strip_whitespace", "dedupe_lines"],
        ),
        scorer=contains_match,
        baseline_model=PREMIUM,
    )
    # Compression removes the duplicated/whitespace bloat -> cheaper than the
    # uncompressed premium baseline, with quality intact.
    assert result.savings_pct > 0
    assert result.quality == pytest.approx(1.0)


def test_empty_workload_raises():
    with pytest.raises(ValueError):
        evaluate([], provider, EvalConfig(model=PREMIUM))


def test_evalconfig_requires_exactly_one_routing_mode():
    with pytest.raises(ValueError):
        EvalConfig()  # neither
    with pytest.raises(ValueError):
        EvalConfig(model=PREMIUM, models=[CHEAP, PREMIUM])  # both


def test_quality_retention_handles_zero_baseline():
    r = EvalResult(
        name="x",
        samples=1,
        total_cost=0.0,
        baseline_cost=1.0,
        quality=0.0,
        baseline_quality=0.0,
    )
    assert r.quality_retention == 1.0
    r2 = EvalResult(
        name="y",
        samples=1,
        total_cost=0.0,
        baseline_cost=1.0,
        quality=0.5,
        baseline_quality=0.0,
    )
    assert r2.quality_retention == 0.0


# --- compare_configs ------------------------------------------------------


def test_compare_configs_shares_one_baseline():
    configs = {
        "premium": EvalConfig(model=PREMIUM),
        "cheap": EvalConfig(model=CHEAP),
        "routed": EvalConfig(models=[CHEAP, PREMIUM]),
    }
    report = compare_configs(WORKLOAD, provider, configs, scorer=contains_match)
    assert isinstance(report, ComparisonReport)
    by_name = {r.name: r for r in report.results}
    # All baselined against the same premium model.
    base_costs = {r.baseline_cost for r in report.results}
    assert len(base_costs) == 1
    # The tradeoff is visible: cheap saves most but loses quality; routed keeps
    # quality and still saves.
    assert by_name["cheap"].savings_pct > by_name["routed"].savings_pct > 0
    assert by_name["cheap"].quality_retention < 1.0
    assert by_name["routed"].quality_retention == pytest.approx(1.0)


def test_compare_configs_render_and_json():
    configs = {
        "premium": EvalConfig(model=PREMIUM),
        "routed": EvalConfig(models=[CHEAP, PREMIUM]),
    }
    report = compare_configs(WORKLOAD, provider, configs, scorer=contains_match)
    text = report.render_text()
    assert "savings vs. quality" in text
    assert "routed" in text
    payload = json.loads(json.dumps(report.as_dict()))
    assert len(payload["results"]) == 2
    assert "quality_retention" in payload["results"][0]


def test_compare_configs_empty_raises():
    with pytest.raises(ValueError):
        compare_configs(WORKLOAD, provider, {}, scorer=contains_match)
