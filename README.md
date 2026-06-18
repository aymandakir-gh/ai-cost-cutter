# ai-cost-cutter

A provider-agnostic toolkit to cut large-language-model spend — for a single
script or a whole platform. Four composable modules, **zero required runtime
dependencies**, and everything testable offline (no API keys needed).

```
  prompt ──▶ compress ──▶ cache? ──hit──▶ cached response   (≈100% saved)
                            │
                          miss
                            ▼
                      router (cheap model first)
                            │ low confidence?
                            ▼ escalate
                      stronger model ──▶ response ──▶ ledger ──▶ dashboard
```

## Modules

| Module | What it does | Status |
| --- | --- | --- |
| `estimator` | Count tokens and estimate per-call USD cost for any model. | v0.1 |
| `router` | Try a cheap model first; escalate to a stronger one only when confidence is low. | v0.1 |
| `cache` | Reuse responses for repeated prompts; track the dollars saved. | v0.2 |
| `compression` | Shrink prompts/context (whitespace, dedup, truncation, history pruning). | v0.2 |
| `dashboard` | Aggregate a usage ledger into a spend + savings report. | v0.3 |
| `benchmarks` | A reproducible workload proving a ≥30% cost cut. | v0.3 |

## Why provider-agnostic

You inject a `call(model, prompt) -> str` function. OpenAI, Anthropic, or a
local model all plug in the same way, so the routing/caching/accounting logic
never depends on a vendor SDK — and the entire test suite and benchmark run
without network access or API keys.

## Install

```bash
pip install ai-cost-cutter           # zero runtime dependencies
pip install "ai-cost-cutter[tiktoken]"   # optional: exact OpenAI token counts
```

## Quickstart

Estimate a call's cost:

```python
from ai_cost_cutter import estimate
est = estimate("gpt-4o", "Summarize this document...", expected_output_tokens=300)
print(f"${est.total_cost:.4f}")
```

Route cheap-first and escalate only when the cheap model is unsure:

```python
from ai_cost_cutter import Router

def call(model, prompt):
    ...  # your provider; return the model's reply as text

router = Router(["gpt-4o-mini", "gpt-4o"], call)   # ordered cheap -> capable
result = router.route("What is the capital of France?")
print(result.model, "|", result.response)
print(f"escalated={result.escalated} saved={result.savings_pct:.0%}")
```

`route()` calls the cheapest model first, scores its confidence (a pluggable
heuristic by default — swap in logprobs or a judge model), and escalates only
when confidence is below the threshold.

## Roadmap

- **v0.1** — `estimator` + `router`
- **v0.2** — `cache` + `compression`
- **v0.3** — `dashboard` + reproducible cost-cut benchmark

> Prices in `pricing.py` are approximate public list prices and are fully
> overridable. The savings the toolkit produces come from *mechanisms*
> (routing, caching, compression), not from any particular price table.

## License

[MIT](LICENSE)
