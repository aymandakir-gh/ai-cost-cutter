# ai-cost-cutter

A provider-agnostic toolkit to cut large-language-model spend — for a single
script or a whole platform. Four composable modules, **zero required runtime
dependencies**, and everything testable offline (no API keys needed).

> **Proven:** the bundled benchmark cuts cost **77.9%** on its sample workload
> (requirement was ≥30%) — deterministic and reproducible with `aicc benchmark`.

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
| `savings_vs_quality` | Measure **both** cost saved and quality retained for each config — see the tradeoff, not just the savings. | v0.4 |

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

Cache repeated calls so you never pay twice:

```python
from ai_cost_cutter.cache import ResponseCache

cache = ResponseCache()          # or SQLiteBackend(path) to persist
cached = cache.wrap(call)        # wraps your provider
cached("gpt-4o", "Hello")        # miss -> calls the model
cached("gpt-4o", "Hello")        # hit  -> free
print(f"hit rate {cache.stats.hit_rate:.0%}, saved ${cache.stats.saved_cost:.4f}")
```

Compress prompts and context before sending them:

```python
from ai_cost_cutter.compression import compress
result = compress(long_prompt, strategies=["strip_whitespace", "dedupe_lines"], max_tokens=2000)
print(f"{result.reduction:.0%} fewer tokens")   # then send result.compressed
```

Strategies include `strip_whitespace` (lossless, default), `dedupe_lines`,
`remove_filler` (lossy), and `collapse_json_whitespace` (losslessly minify JSON
blocks embedded in a prompt) — or pass your own `call(text) -> text` callable.

```bash
cat prompt.txt | aicc compress --dedupe --filler --max-tokens 2000
```

Track spend and savings with a ledger + dashboard:

```python
from ai_cost_cutter import Ledger, build_report

ledger = Ledger("usage.jsonl")           # persists across runs
ledger.record_route(router.route("..."))  # or record_call(...) / record_cache_hit(...)
print(build_report(ledger).render_text())
```

```bash
aicc dashboard --ledger usage.jsonl              # text report
aicc dashboard --ledger usage.jsonl --html out.html
```

Prove the savings on a deterministic, offline workload:

```bash
aicc benchmark        # see benchmarks/README.md for how it stays honest
```

Weigh savings against quality — cheaper is only better if answers hold up:

```python
from ai_cost_cutter import Sample, EvalConfig, compare_configs, contains_match

workload = [Sample("Capital of France?", "Paris"), Sample("2 + 2?", "4")]

def call(model, prompt):
    ...  # your provider

report = compare_configs(
    workload, call,
    {
        "baseline": EvalConfig(model="gpt-4o"),
        "routed":   EvalConfig(models=["gpt-4o-mini", "gpt-4o"]),
        "cheap":    EvalConfig(model="gpt-4o-mini"),
    },
    scorer=contains_match,            # inject any scorer(reference, answer) -> [0,1]
)
print(report.render_text())          # cost, % saved, quality, % quality retained
```

You inject the quality scorer (a few ship in the box: `exact_match`,
`normalized_match`, `contains_match`, `token_f1`), so the tradeoff is measured
on *your* definition of "good enough" — fully offline.

## Roadmap

- ✅ **v0.1** — `estimator` + `router`
- ✅ **v0.2** — `cache` + `compression`
- ✅ **v0.3** — `dashboard` + reproducible cost-cut benchmark (77.9% cut proven)
- ✅ **v0.4** — broader price tables, more compression strategies, and a
  `savings_vs_quality` eval module

> Prices in `pricing.py` are approximate public list prices and are fully
> overridable. The bundled table tags each model with its provider and covers
> OpenAI, Anthropic, Google (Gemini), Mistral, Cohere, DeepSeek, xAI (Grok),
> and Groq-hosted open models — plus a free `local` tier. The savings the
> toolkit produces come from *mechanisms* (routing, caching, compression), not
> from any particular price table.

## License

[MIT](LICENSE)
