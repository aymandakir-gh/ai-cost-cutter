"""A reproducible benchmark that quantifies the cost cut.

Everything here is deterministic and offline: a fixed sample workload and a
fake provider whose replies depend only on ``(model, prompt)``. The provider
returns *low-confidence* replies from the cheap model on hard questions, so the
router's escalation decisions emerge from the **real** confidence heuristic
rather than being hard-coded. Run it the same way twice and you get the same
numbers.

The benchmark compares a no-optimization baseline (always the premium model,
full prompt, no cache) against each mechanism in isolation and all three
combined:

- ``compression`` — shrink the prompt/context before sending.
- ``routing``     — answer easy prompts with the cheap model, escalate hard ones.
- ``cache``       — serve repeated prompts for free.

Run it::

    aicc benchmark
    python -m ai_cost_cutter.benchmark
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .cache import ResponseCache, _coerce_text
from .compression import compress
from .estimator import estimate_tokens
from .ledger import Ledger
from .router import Router
from .tokens import count_tokens

CHEAP_MODEL = "gpt-4o-mini"
PREMIUM_MODEL = "gpt-4o"
COMPRESSION_STRATEGIES = ("strip_whitespace", "dedupe_lines", "remove_filler")

# A deliberately bloated context block: duplicated lines, filler words, and
# trailing whitespace — exactly what compression is good at removing.
_CONTEXT = (
    "You are a helpful, accurate assistant. Please be concise.   \n"
    "You are a helpful, accurate assistant. Please be concise.\n"
    "Please note that you should use only the provided context to answer.\n"
    "Please note that you should use only the provided context to answer.\n"
    "In order to ensure correctness, kindly double-check your reasoning.   \n"
    "In order to ensure correctness, kindly double-check your reasoning.\n"
    "Basically, the user really just wants a direct and correct answer.\n"
    "Basically, the user really just wants a direct and correct answer.\n"
    "Do not include unnecessary preamble or filler in the response.   \n"
    "Do not include unnecessary preamble or filler in the response.\n"
)


@dataclass(frozen=True)
class Question:
    text: str
    difficulty: str  # "easy" or "hard"
    answer: str


QUESTIONS: List[Question] = [
    Question("What is the capital of France?", "easy", "The capital of France is Paris."),
    Question("What is 2 + 2?", "easy", "2 + 2 equals 4."),
    Question("Who wrote Romeo and Juliet?", "easy", "William Shakespeare wrote Romeo and Juliet."),
    Question("What color is a clear daytime sky?", "easy", "A clear daytime sky is blue."),
    Question(
        "What is the boiling point of water at sea level in Celsius?",
        "easy",
        "Water boils at 100 degrees Celsius at sea level.",
    ),
    Question("How many days are in a week?", "easy", "There are seven days in a week."),
    Question(
        "What is the chemical symbol for gold?",
        "easy",
        "The chemical symbol for gold is Au.",
    ),
    Question(
        "Which planet is known as the Red Planet?",
        "easy",
        "Mars is known as the Red Planet.",
    ),
    Question(
        "Prove that the square root of two is irrational.",
        "hard",
        "Assume sqrt(2) = a/b in lowest terms. Then a^2 = 2 b^2, so a is even, "
        "say a = 2k. Then 4k^2 = 2 b^2 gives b^2 = 2k^2, so b is even too, "
        "contradicting lowest terms. Hence sqrt(2) is irrational.",
    ),
    Question(
        "Explain why the sky appears red at sunset, citing the physics.",
        "hard",
        "At sunset light travels through more atmosphere, so short blue "
        "wavelengths are scattered away by Rayleigh scattering and the "
        "longer red wavelengths dominate what reaches the eye.",
    ),
    Question(
        "Derive the quadratic formula from a x^2 + b x + c = 0.",
        "hard",
        "Divide by a, complete the square: (x + b/2a)^2 = (b^2 - 4ac)/4a^2. "
        "Taking roots gives x = (-b +/- sqrt(b^2 - 4ac)) / (2a).",
    ),
    Question(
        "Explain the halting problem and why it is undecidable.",
        "hard",
        "No program H can decide whether an arbitrary program halts: assuming "
        "one exists, build D that loops iff H says it halts on itself, a "
        "contradiction. So halting is undecidable.",
    ),
]

# Recurring traffic: indices asked again, simulating FAQs/retries (cache hits).
_REPEAT_INDICES = [0, 1, 2, 4, 0, 1, 8, 5]


@dataclass(frozen=True)
class Request:
    prompt: str
    difficulty: str
    answer: str
    question: str


def _prompt_for(q: Question) -> str:
    return _CONTEXT + "\nQuestion: " + q.text


def build_workload() -> List[Request]:
    """Return the deterministic sample workload (unique calls + repeats)."""
    base = [_request_for(q) for q in QUESTIONS]
    repeats = [base[i] for i in _REPEAT_INDICES]
    return base + repeats


def _request_for(q: Question) -> Request:
    return Request(_prompt_for(q), q.difficulty, q.answer, q.text)


def make_provider(cheap_model: str = CHEAP_MODEL) -> Callable[[str, str], str]:
    """A deterministic fake provider.

    The cheap model hedges on hard questions (the real confidence heuristic then
    triggers escalation); every other case returns the canonical answer.
    """
    lookup = {" ".join(q.text.split()): q for q in QUESTIONS}

    def provider(model: str, prompt: str) -> str:
        norm = " ".join(prompt.split())
        question = None
        for key, q in lookup.items():
            if key in norm:
                question = q
                break
        if question is None:  # pragma: no cover - workload always matches
            return "The answer is clear."
        if question.difficulty == "hard" and model == cheap_model:
            return "I'm not sure, I don't know the exact answer to this."
        return question.answer

    return provider


def _baseline_cost(req: Request, premium: str) -> float:
    in_tokens = count_tokens(req.prompt, premium)
    out_tokens = count_tokens(req.answer, premium)
    return estimate_tokens(premium, in_tokens, out_tokens).total_cost


def _run_pipeline(
    workload: List[Request],
    provider: Callable[[str, str], str],
    use_compression: bool,
    use_routing: bool,
    use_cache: bool,
    cheap: str,
    premium: str,
    ledger: Optional[Ledger] = None,
) -> float:
    cache = ResponseCache() if use_cache else None
    router = Router([cheap, premium], provider) if use_routing else None
    total = 0.0

    for req in workload:
        baseline = _baseline_cost(req, premium)
        prompt = req.prompt
        if use_compression:
            prompt = compress(prompt, strategies=COMPRESSION_STRATEGIES).compressed

        if cache is not None:
            hit = cache.get("pipeline", prompt)
            if hit is not None:
                if ledger is not None:
                    ledger.record_call(
                        model="cache",
                        cost=0.0,
                        cached=True,
                        baseline_cost=baseline,
                        input_tokens=count_tokens(prompt),
                        output_tokens=count_tokens(req.answer),
                        metadata={"difficulty": req.difficulty},
                    )
                continue

        if router is not None:
            result = router.route(prompt)
            cost = result.total_cost
            model = result.model
            response = result.response
        else:
            response = _coerce_text(provider(premium, prompt))
            in_tokens = count_tokens(prompt, premium)
            out_tokens = count_tokens(response, premium)
            cost = estimate_tokens(premium, in_tokens, out_tokens).total_cost
            model = premium

        total += cost
        if cache is not None:
            cache.set("pipeline", prompt, response, cost=cost)
        if ledger is not None:
            ledger.record_call(
                model=model,
                cost=cost,
                baseline_cost=baseline,
                input_tokens=count_tokens(prompt, model),
                output_tokens=count_tokens(response, model),
                metadata={"difficulty": req.difficulty},
            )

    return total


@dataclass
class BenchmarkResult:
    requests: int
    baseline_cost: float
    breakdown: Dict[str, float] = field(default_factory=dict)
    ledger: Optional[Ledger] = None

    @property
    def optimized_cost(self) -> float:
        return self.breakdown["combined"]

    @property
    def savings(self) -> float:
        return self.baseline_cost - self.optimized_cost

    @property
    def savings_pct(self) -> float:
        if self.baseline_cost <= 0:
            return 0.0
        return self.savings / self.baseline_cost

    def savings_pct_for(self, config: str) -> float:
        if self.baseline_cost <= 0:
            return 0.0
        return (self.baseline_cost - self.breakdown[config]) / self.baseline_cost

    def render_text(self) -> str:
        lines = [
            "ai-cost-cutter — cost-cut benchmark",
            "=" * 44,
            f"workload: {self.requests} requests "
            f"(deterministic, offline, no API keys)",
            "",
            f"{'configuration':<22}{'cost':>12}{'saved':>10}",
            "-" * 44,
        ]
        labels = {
            "baseline": "baseline (premium)",
            "compression_only": "+ compression",
            "routing_only": "+ routing",
            "cache_only": "+ cache",
            "combined": "all combined",
        }
        for key in ["baseline", "compression_only", "routing_only", "cache_only", "combined"]:
            cost = self.breakdown[key]
            pct = self.savings_pct_for(key)
            saved = "—" if key == "baseline" else f"{pct:.0%}"
            lines.append(f"{labels[key]:<22}${cost:>10.4f}{saved:>10}")
        lines.append("-" * 44)
        lines.append(
            f"TOTAL CUT: {self.savings_pct:.1%}  "
            f"(${self.baseline_cost:.4f} -> ${self.optimized_cost:.4f})"
        )
        return "\n".join(lines)


def run_benchmark(
    workload: Optional[List[Request]] = None,
    cheap: str = CHEAP_MODEL,
    premium: str = PREMIUM_MODEL,
) -> BenchmarkResult:
    """Run the full benchmark and return a :class:`BenchmarkResult`."""
    workload = workload if workload is not None else build_workload()
    provider = make_provider(cheap)

    baseline = sum(_baseline_cost(req, premium) for req in workload)
    ledger = Ledger()

    def run(compression: bool, routing: bool, cache: bool, lg=None) -> float:
        return _run_pipeline(
            workload, provider, compression, routing, cache, cheap, premium, ledger=lg
        )

    breakdown = {
        "baseline": run(False, False, False),
        "compression_only": run(True, False, False),
        "routing_only": run(False, True, False),
        "cache_only": run(False, False, True),
        "combined": run(True, True, True, lg=ledger),
    }
    return BenchmarkResult(
        requests=len(workload),
        baseline_cost=baseline,
        breakdown=breakdown,
        ledger=ledger,
    )


def main() -> int:  # pragma: no cover - thin CLI wrapper
    print(run_benchmark().render_text())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
