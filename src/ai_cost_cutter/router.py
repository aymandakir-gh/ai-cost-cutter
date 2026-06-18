"""Cheap-first model routing with confidence-based escalation.

Try the cheapest model first; only escalate to a stronger (pricier) model when
the cheap model's answer looks low-confidence. This is the single biggest lever
for cutting LLM spend when most requests are easy.

The router is provider-agnostic: you pass a ``call(model, prompt) -> str``
function, so OpenAI, Anthropic, or a local model all work the same way and the
whole thing is testable offline.

Example::

    from ai_cost_cutter.router import Router

    def call(model, prompt):
        ...  # your provider; return the model's text reply

    router = Router(["gpt-4o-mini", "gpt-4o"], call)
    result = router.route("What is 2 + 2?")
    print(result.model, result.response, f"saved {result.savings_pct:.0%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .estimator import estimate_tokens
from .pricing import ModelPrice, get_price
from .tokens import count_tokens

# A provider is any callable that maps (model, prompt) -> response text.
Provider = Callable[[str, str], object]
ConfidenceFn = Callable[[str, str], float]

DEFAULT_THRESHOLD = 0.6

# Phrases that signal a model is unsure / refusing / guessing.
HEDGE_PHRASES = (
    "i'm not sure",
    "i am not sure",
    "not entirely sure",
    "i'm not certain",
    "i am not certain",
    "not confident",
    "i don't know",
    "i do not know",
    "no idea",
    "i cannot determine",
    "can't determine",
    "cannot determine",
    "unable to determine",
    "i'm unable",
    "i am unable",
    "unable to",
    "i can't help",
    "i cannot help",
    "it's unclear",
    "it is unclear",
    "hard to say",
    "as an ai",
    "i apologize",
    "this is a guess",
    "i'm guessing",
    "i am guessing",
)


def heuristic_confidence(prompt: str, response: str) -> float:
    """Estimate a model's confidence in ``response`` from text signals only.

    Returns a score in ``[0, 1]``. Provider-agnostic and deterministic. This is
    the default; pass your own ``confidence`` to :class:`Router` to use logprobs
    or a judge model instead.
    """
    text = (response or "").strip()
    if not text:
        return 0.0
    low = text.lower()
    score = 1.0
    for phrase in HEDGE_PHRASES:
        if phrase in low:
            score -= 0.5
    words = text.split()
    if len(words) < 3:
        score -= 0.3
    # A reply that is only a clarifying question is not a confident answer.
    if text.endswith("?") and len(words) < 25:
        score -= 0.5
    # Long replies that end mid-sentence look truncated/incomplete.
    if len(words) >= 10 and text[-1] not in ".!?\"')]}”":
        score -= 0.2
    return max(0.0, min(1.0, score))


def confidence_from_logprobs(logprobs: Sequence[float]) -> float:
    """Confidence helper for providers that expose per-token logprobs.

    Returns the mean per-token probability (``exp(logprob)``) in ``[0, 1]``.
    Wrap your provider so ``confidence`` can reach the logprobs, then pass this.
    """
    import math

    values = [lp for lp in logprobs if lp is not None]
    if not values:
        return 0.0
    probs = [math.exp(lp) for lp in values]
    return max(0.0, min(1.0, sum(probs) / len(probs)))


def _as_text(raw: object) -> str:
    """Coerce a provider's return value into response text."""
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        for key in ("text", "content", "message", "output", "completion"):
            if key in raw:
                value = raw[key]
                return value if isinstance(value, str) else str(value)
        return str(raw)
    for attr in ("text", "content"):
        if hasattr(raw, attr):
            value = getattr(raw, attr)
            return value if isinstance(value, str) else str(value)
    return str(raw)


def _blended_price(model: str, prices: Optional[Dict[str, ModelPrice]]) -> float:
    price = get_price(model, prices)
    return price.input_per_1m + price.output_per_1m


@dataclass(frozen=True)
class Attempt:
    """A single model call made while routing."""

    model: str
    response: str
    confidence: float
    cost: float
    accepted: bool


@dataclass(frozen=True)
class RouteResult:
    """The outcome of routing one prompt."""

    response: str
    model: str
    confidence: float
    total_cost: float
    baseline_cost: float
    escalated: bool
    accepted: bool
    attempts: List[Attempt] = field(default_factory=list)

    @property
    def savings(self) -> float:
        """Dollars saved versus always using the most capable model."""
        return self.baseline_cost - self.total_cost

    @property
    def savings_pct(self) -> float:
        if self.baseline_cost <= 0:
            return 0.0
        return self.savings / self.baseline_cost


class Router:
    """Route prompts cheap-first, escalating only when confidence is low."""

    def __init__(
        self,
        models: Sequence[str],
        provider: Provider,
        confidence: ConfidenceFn = heuristic_confidence,
        threshold: float = DEFAULT_THRESHOLD,
        prices: Optional[Dict[str, ModelPrice]] = None,
        auto_order: bool = True,
        on_attempt: Optional[Callable[[Attempt], None]] = None,
    ) -> None:
        models = list(models)
        if not models:
            raise ValueError("models must be a non-empty list")
        # Validate every model is priceable so cost accounting always works.
        for model in models:
            get_price(model, prices)
        if auto_order:
            models.sort(key=lambda m: _blended_price(m, prices))
        self.models = models
        self.provider = provider
        self.confidence = confidence
        self.threshold = threshold
        self.prices = prices
        self.on_attempt = on_attempt

    def route(self, prompt: str) -> RouteResult:
        """Route ``prompt`` through the model chain and return a result."""
        attempts: List[Attempt] = []
        chosen: Optional[Attempt] = None

        for model in self.models:
            text = _as_text(self.provider(model, prompt))
            conf = float(self.confidence(prompt, text))
            in_tokens = count_tokens(prompt, model)
            out_tokens = count_tokens(text, model)
            cost = estimate_tokens(model, in_tokens, out_tokens, self.prices).total_cost
            accepted = conf >= self.threshold
            attempt = Attempt(model, text, conf, cost, accepted)
            attempts.append(attempt)
            if self.on_attempt is not None:
                self.on_attempt(attempt)
            if accepted:
                chosen = attempt
                break

        if chosen is None:
            # Nobody cleared the bar; keep the most capable answer we got.
            chosen = attempts[-1]

        total_cost = sum(a.cost for a in attempts)
        premium = self.models[-1]
        base_in = count_tokens(prompt, premium)
        base_out = count_tokens(chosen.response, premium)
        baseline_cost = estimate_tokens(
            premium, base_in, base_out, self.prices
        ).total_cost

        return RouteResult(
            response=chosen.response,
            model=chosen.model,
            confidence=chosen.confidence,
            total_cost=total_cost,
            baseline_cost=baseline_cost,
            escalated=len(attempts) > 1,
            accepted=chosen.accepted,
            attempts=attempts,
        )
