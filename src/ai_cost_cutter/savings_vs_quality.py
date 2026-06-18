"""Savings-vs-quality evaluation.

Cutting cost is only useful if quality holds up. This module measures **both**
sides of the tradeoff: given a labeled workload (prompts + reference answers)
and an injected quality scorer, it reports how much a configuration
(routing / compression / cache) saves *and* how much answer quality it retains
versus an unoptimized baseline.

Everything is provider-agnostic and fully offline: you inject a
``call(model, prompt) -> str`` function and a ``scorer(reference, answer) ->
float in [0, 1]``. A few ready-made scorers ship in the box.

Example::

    from ai_cost_cutter.savings_vs_quality import (
        Sample, EvalConfig, compare_configs, normalized_match,
    )

    workload = [
        Sample("What is 2 + 2?", "4"),
        Sample("Capital of France?", "Paris"),
    ]

    def call(model, prompt):
        ...  # your provider

    configs = {
        "baseline": EvalConfig(model="gpt-4o"),
        "routed":   EvalConfig(models=["gpt-4o-mini", "gpt-4o"]),
    }
    report = compare_configs(workload, call, configs, scorer=normalized_match)
    print(report.render_text())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from .cache import ResponseCache, _coerce_text
from .compression import compress
from .estimator import estimate_tokens
from .pricing import ModelPrice
from .router import Router, heuristic_confidence
from .tokens import count_tokens

Provider = Callable[[str, str], object]
Scorer = Callable[[str, str], float]
ConfidenceFn = Callable[[str, str], float]


# --- quality scorers ------------------------------------------------------

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    text = (text or "").lower()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def exact_match(reference: str, answer: str) -> float:
    """1.0 iff ``answer`` equals ``reference`` exactly, else 0.0."""
    return 1.0 if (answer or "") == (reference or "") else 0.0


def normalized_match(reference: str, answer: str) -> float:
    """1.0 iff case/punctuation/whitespace-normalized strings are equal."""
    return 1.0 if _normalize(reference) == _normalize(answer) else 0.0


def contains_match(reference: str, answer: str) -> float:
    """1.0 iff the normalized ``reference`` appears within the answer.

    Useful when the reference is a short key fact ("Paris") and the model
    replies in a sentence ("The capital of France is Paris.").
    """
    ref = _normalize(reference)
    if not ref:
        return 1.0
    return 1.0 if ref in _normalize(answer) else 0.0


def token_f1(reference: str, answer: str) -> float:
    """Token-overlap F1 between ``reference`` and ``answer`` in ``[0, 1]``.

    A soft scorer that credits partial answers, mirroring SQuAD-style F1.
    """
    ref_tokens = _normalize(reference).split()
    ans_tokens = _normalize(answer).split()
    if not ref_tokens and not ans_tokens:
        return 1.0
    if not ref_tokens or not ans_tokens:
        return 0.0
    # Multiset overlap.
    common = 0
    remaining = list(ans_tokens)
    for tok in ref_tokens:
        if tok in remaining:
            remaining.remove(tok)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(ans_tokens)
    recall = common / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


# --- workload + configs ---------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One labeled item of the evaluation workload."""

    prompt: str
    reference: str
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalConfig:
    """A configuration to evaluate.

    Pick exactly one routing mode:

    - ``model``  : always call this single model (no routing).
    - ``models`` : cheap-first route across these models (escalate on low
      confidence).

    Optionally layer on:

    - ``compress_strategies`` : compress each prompt before sending.
    - ``compress_max_tokens`` : cap each prompt to this token budget.
    - ``use_cache``           : serve repeated prompts from cache (free).
    """

    model: Optional[str] = None
    models: Optional[Sequence[str]] = None
    compress_strategies: Optional[Sequence[str]] = None
    compress_max_tokens: Optional[int] = None
    use_cache: bool = False
    threshold: float = 0.6
    confidence: ConfidenceFn = heuristic_confidence

    def __post_init__(self) -> None:
        if bool(self.model) == bool(self.models):
            raise ValueError("set exactly one of `model` or `models`")

    @property
    def premium_model(self) -> str:
        """The most capable model this config can use (for baselining)."""
        if self.model is not None:
            return self.model
        # The router auto-orders cheap->capable; the last is most capable.
        ordered = sorted(
            self.models or [],
            key=lambda m: _blended(m, None),
        )
        return ordered[-1]


def _blended(model: str, prices: Optional[Dict[str, ModelPrice]]) -> float:
    from .pricing import get_price

    p = get_price(model, prices)
    return p.input_per_1m + p.output_per_1m


# --- evaluation -----------------------------------------------------------


@dataclass(frozen=True)
class EvalResult:
    """Aggregate cost + quality for one config over the workload."""

    name: str
    samples: int
    total_cost: float
    baseline_cost: float
    quality: float
    baseline_quality: float
    per_sample_quality: List[float] = field(default_factory=list)
    cache_hits: int = 0

    @property
    def savings(self) -> float:
        return self.baseline_cost - self.total_cost

    @property
    def savings_pct(self) -> float:
        if self.baseline_cost <= 0:
            return 0.0
        return self.savings / self.baseline_cost

    @property
    def quality_retention(self) -> float:
        """Quality as a fraction of the baseline's quality (``1.0`` = no loss).

        When the baseline scores zero quality, retention is defined as ``1.0``
        if this config also scores zero, else ``0.0`` (avoids divide-by-zero).
        """
        if self.baseline_quality <= 0:
            return 1.0 if self.quality <= 0 else 0.0
        return self.quality / self.baseline_quality

    @property
    def quality_drop(self) -> float:
        return self.baseline_quality - self.quality

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "samples": self.samples,
            "total_cost": self.total_cost,
            "baseline_cost": self.baseline_cost,
            "savings": self.savings,
            "savings_pct": self.savings_pct,
            "quality": self.quality,
            "baseline_quality": self.baseline_quality,
            "quality_retention": self.quality_retention,
            "quality_drop": self.quality_drop,
            "cache_hits": self.cache_hits,
        }


def _prepare_prompt(prompt: str, config: EvalConfig, model: str) -> str:
    out = prompt
    if config.compress_strategies or config.compress_max_tokens is not None:
        out = compress(
            out,
            strategies=config.compress_strategies or ("strip_whitespace",),
            max_tokens=config.compress_max_tokens,
            model=model,
        ).compressed
    return out


def _cost_for(model: str, prompt: str, response: str,
              prices: Optional[Dict[str, ModelPrice]]) -> float:
    in_tokens = count_tokens(prompt, model)
    out_tokens = count_tokens(response, model)
    return estimate_tokens(model, in_tokens, out_tokens, prices).total_cost


def _baseline_quality_and_cost(
    workload: Sequence[Sample],
    provider: Provider,
    scorer: Scorer,
    premium: str,
    prices: Optional[Dict[str, ModelPrice]],
) -> "tuple[float, float]":
    """The unoptimized baseline: always premium, full prompt, no cache."""
    total_quality = 0.0
    total_cost = 0.0
    for sample in workload:
        response = _coerce_text(provider(premium, sample.prompt))
        total_cost += _cost_for(premium, sample.prompt, response, prices)
        total_quality += float(scorer(sample.reference, response))
    n = len(workload) or 1
    return total_quality / n, total_cost


def evaluate(
    workload: Sequence[Sample],
    provider: Provider,
    config: EvalConfig,
    scorer: Scorer = normalized_match,
    name: Optional[str] = None,
    prices: Optional[Dict[str, ModelPrice]] = None,
    baseline_model: Optional[str] = None,
) -> EvalResult:
    """Run ``config`` over ``workload`` and measure cost and quality.

    ``baseline_model`` is the model the savings/quality are compared against
    (defaults to the config's most capable model — i.e. "what if we always
    used the strong model, full prompt, no cache").
    """
    if not workload:
        raise ValueError("workload must be non-empty")

    premium = baseline_model or config.premium_model
    baseline_quality, baseline_cost = _baseline_quality_and_cost(
        workload, provider, scorer, premium, prices
    )

    cache = ResponseCache(prices=prices) if config.use_cache else None
    router = None
    if config.models is not None:
        router = Router(
            config.models,
            provider,
            confidence=config.confidence,
            threshold=config.threshold,
            prices=prices,
        )

    total_cost = 0.0
    per_quality: List[float] = []
    cache_hits = 0

    for sample in workload:
        model = config.model if config.model is not None else config.premium_model
        prompt = _prepare_prompt(sample.prompt, config, model)

        if cache is not None:
            hit = cache.get("eval", prompt)
            if hit is not None:
                cache_hits += 1
                per_quality.append(float(scorer(sample.reference, hit)))
                continue

        if router is not None:
            result = router.route(prompt)
            response = result.response
            cost = result.total_cost
            used_model = result.model
        else:
            response = _coerce_text(provider(config.model, prompt))
            cost = _cost_for(config.model, prompt, response, prices)
            used_model = config.model

        total_cost += cost
        per_quality.append(float(scorer(sample.reference, response)))
        if cache is not None:
            cache.set("eval", prompt, response, cost=cost)

    n = len(workload)
    quality = sum(per_quality) / n if n else 0.0
    return EvalResult(
        name=name or _config_name(config),
        samples=n,
        total_cost=total_cost,
        baseline_cost=baseline_cost,
        quality=quality,
        baseline_quality=baseline_quality,
        per_sample_quality=per_quality,
        cache_hits=cache_hits,
    )


def _config_name(config: EvalConfig) -> str:
    if config.model is not None:
        base = config.model
    else:
        base = "route(" + ",".join(config.models or []) + ")"
    extras = []
    if config.compress_strategies or config.compress_max_tokens is not None:
        extras.append("compress")
    if config.use_cache:
        extras.append("cache")
    return base + ("+" + "+".join(extras) if extras else "")


@dataclass
class ComparisonReport:
    """A table of :class:`EvalResult` rows sharing one baseline."""

    results: List[EvalResult] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {"results": [r.as_dict() for r in self.results]}

    def render_text(self) -> str:
        lines = [
            "ai-cost-cutter — savings vs. quality",
            "=" * 64,
            f"{'config':<26}{'cost':>10}{'saved':>8}{'quality':>9}{'retain':>9}",
            "-" * 64,
        ]
        for r in self.results:
            lines.append(
                f"{r.name[:25]:<26}"
                f"${r.total_cost:>8.4f}"
                f"{r.savings_pct:>7.0%}"
                f"{r.quality:>9.2f}"
                f"{r.quality_retention:>8.0%}"
            )
        lines.append("-" * 64)
        lines.append(
            "quality in [0,1] from the injected scorer; "
            "retain = quality / baseline quality"
        )
        return "\n".join(lines)


def compare_configs(
    workload: Sequence[Sample],
    provider: Provider,
    configs: Mapping[str, EvalConfig],
    scorer: Scorer = normalized_match,
    prices: Optional[Dict[str, ModelPrice]] = None,
    baseline_model: Optional[str] = None,
) -> ComparisonReport:
    """Evaluate several configs over the same workload and tabulate the tradeoff.

    All configs are baselined against the same model: ``baseline_model`` if
    given, else the most capable model across all configs.
    """
    if not configs:
        raise ValueError("configs must be non-empty")

    if baseline_model is None:
        premiums = [c.premium_model for c in configs.values()]
        baseline_model = max(premiums, key=lambda m: _blended(m, prices))

    results = [
        evaluate(
            workload,
            provider,
            config,
            scorer=scorer,
            name=name,
            prices=prices,
            baseline_model=baseline_model,
        )
        for name, config in configs.items()
    ]
    return ComparisonReport(results=results)
