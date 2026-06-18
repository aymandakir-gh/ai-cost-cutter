import pytest

from ai_cost_cutter.router import (
    Router,
    confidence_from_logprobs,
    heuristic_confidence,
)


def provider_from_map(responses):
    """Build a fake provider returning a canned reply per model."""

    calls = []

    def call(model, prompt):
        calls.append(model)
        value = responses[model]
        return value(prompt) if callable(value) else value

    call.calls = calls
    return call


# --- heuristic_confidence -------------------------------------------------


def test_confidence_high_for_direct_answer():
    assert heuristic_confidence("q", "The capital of France is Paris.") >= 0.9


def test_confidence_low_for_hedge():
    assert heuristic_confidence("q", "I'm not sure, it depends.") < 0.6


def test_confidence_zero_for_empty():
    assert heuristic_confidence("q", "") == 0.0
    assert heuristic_confidence("q", "   ") == 0.0


def test_confidence_low_for_clarifying_question():
    assert heuristic_confidence("q", "Which file do you mean?") < 0.6


def test_confidence_in_unit_range():
    score = heuristic_confidence("q", "I'm not sure. I don't know. No idea.")
    assert 0.0 <= score <= 1.0


def test_confidence_from_logprobs():
    import math

    # Two tokens each with prob ~0.9.
    lp = math.log(0.9)
    assert confidence_from_logprobs([lp, lp]) == pytest.approx(0.9, abs=1e-6)
    assert confidence_from_logprobs([]) == 0.0


# --- Router ---------------------------------------------------------------


def test_cheap_model_accepted_no_escalation():
    provider = provider_from_map(
        {
            "gpt-4o-mini": "The answer is 4.",
            "gpt-4o": "The answer is 4.",
        }
    )
    router = Router(["gpt-4o-mini", "gpt-4o"], provider)
    result = router.route("What is 2 + 2?")
    assert result.model == "gpt-4o-mini"
    assert result.escalated is False
    assert result.accepted is True
    assert provider.calls == ["gpt-4o-mini"]  # premium never called


def test_escalates_when_cheap_model_hedges():
    provider = provider_from_map(
        {
            "gpt-4o-mini": "I'm not sure, I don't know.",
            "gpt-4o": "The answer is definitely 42.",
        }
    )
    router = Router(["gpt-4o-mini", "gpt-4o"], provider)
    result = router.route("hard question")
    assert result.escalated is True
    assert result.model == "gpt-4o"
    assert result.accepted is True
    assert provider.calls == ["gpt-4o-mini", "gpt-4o"]


def test_returns_best_effort_when_nobody_confident():
    provider = provider_from_map(
        {
            "gpt-4o-mini": "I don't know.",
            "gpt-4o": "I'm not sure either.",
        }
    )
    router = Router(["gpt-4o-mini", "gpt-4o"], provider)
    result = router.route("impossible")
    assert result.accepted is False
    assert result.escalated is True
    assert result.model == "gpt-4o"  # most capable attempted


def test_cheap_path_saves_money_vs_baseline():
    provider = provider_from_map(
        {
            "gpt-4o-mini": "The answer is 4.",
            "gpt-4o": "The answer is 4.",
        }
    )
    router = Router(["gpt-4o-mini", "gpt-4o"], provider)
    result = router.route("What is 2 + 2?")
    assert result.baseline_cost > result.total_cost
    assert result.savings > 0
    assert 0 < result.savings_pct <= 1


def test_auto_order_sorts_cheapest_first():
    provider = provider_from_map(
        {
            "gpt-4o-mini": "Answer: yes.",
            "gpt-4o": "Answer: yes.",
        }
    )
    # Pass premium first; auto_order should still try the cheap one first.
    router = Router(["gpt-4o", "gpt-4o-mini"], provider)
    router.route("q")
    assert provider.calls[0] == "gpt-4o-mini"


def test_auto_order_disabled_respects_given_order():
    provider = provider_from_map(
        {
            "gpt-4o": "Answer: yes.",
            "gpt-4o-mini": "Answer: yes.",
        }
    )
    router = Router(["gpt-4o", "gpt-4o-mini"], provider, auto_order=False)
    router.route("q")
    assert provider.calls[0] == "gpt-4o"


def test_on_attempt_callback_fires_per_call():
    provider = provider_from_map(
        {
            "gpt-4o-mini": "I'm not sure.",
            "gpt-4o": "Confidently: 7.",
        }
    )
    seen = []
    router = Router(
        ["gpt-4o-mini", "gpt-4o"], provider, on_attempt=lambda a: seen.append(a.model)
    )
    router.route("q")
    assert seen == ["gpt-4o-mini", "gpt-4o"]


def test_custom_confidence_function():
    provider = provider_from_map({"gpt-4o-mini": "x", "gpt-4o": "y"})
    # Always-low confidence forces escalation regardless of text.
    router = Router(["gpt-4o-mini", "gpt-4o"], provider, confidence=lambda p, r: 0.0)
    result = router.route("q")
    assert result.escalated is True


def test_provider_can_return_dict():
    provider = provider_from_map({"gpt-4o-mini": {"content": "The answer is 4."}})
    router = Router(["gpt-4o-mini"], provider)
    result = router.route("q")
    assert result.response == "The answer is 4."


def test_empty_models_rejected():
    with pytest.raises(ValueError):
        Router([], lambda m, p: "x")


def test_unpriceable_model_rejected():
    from ai_cost_cutter.pricing import UnknownModelError

    with pytest.raises(UnknownModelError):
        Router(["totally-unknown-model"], lambda m, p: "x")


def test_attempts_recorded_with_costs():
    provider = provider_from_map(
        {"gpt-4o-mini": "I don't know.", "gpt-4o": "Answer: 4."}
    )
    router = Router(["gpt-4o-mini", "gpt-4o"], provider)
    result = router.route("q")
    assert len(result.attempts) == 2
    assert all(a.cost >= 0 for a in result.attempts)
    assert result.total_cost == pytest.approx(sum(a.cost for a in result.attempts))
