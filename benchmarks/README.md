# Benchmark

A reproducible benchmark that proves the toolkit cuts cost by **well over 30%**
on a sample workload — with **no API keys and no network**.

## Reproduce it

```bash
pip install -e .
aicc benchmark
# or:
python -m ai_cost_cutter.benchmark
```

Example output:

```
ai-cost-cutter — cost-cut benchmark
============================================
workload: 20 requests (deterministic, offline, no API keys)

configuration                 cost     saved
--------------------------------------------
baseline (premium)    $    0.0122         —
+ compression         $    0.0073       40%
+ routing             $    0.0050       59%
+ cache               $    0.0077       37%
all combined          $    0.0027       78%
--------------------------------------------
TOTAL CUT: 77.9%  ($0.0122 -> $0.0027)
```

## How it stays honest

- **Deterministic, offline.** The "provider" is a fake whose replies depend
  only on `(model, prompt)`. No randomness, no clocks, no network — run it twice
  and the numbers are identical (enforced by `test_benchmark_is_reproducible`).
- **Escalation is earned, not scripted.** On hard questions the cheap model
  returns a hedged reply; the router escalates because the *real*
  `heuristic_confidence` scores it low — the routing decisions are not
  hard-coded.
- **Costs use the shared estimator** and the same price table for baseline and
  optimized runs, so the comparison is apples-to-apples. The prices cancel out
  of the ratio; the cut comes from the mechanisms, not the numbers.
- **Ablation breakdown.** Each mechanism is measured alone and combined, so you
  can see exactly where the savings come from.

The workload mixes easy and hard questions over a bloated, duplicated context,
with recurring requests (FAQs/retries) that the cache serves for free — a
realistic shape for production traffic.

## Where the savings come from

| Mechanism | Why it saves |
| --- | --- |
| compression | strips duplicated/filler context → fewer input tokens every call |
| routing | easy prompts answered by a model ~16× cheaper; premium only on escalation |
| cache | repeated prompts cost nothing |

CI runs `test_benchmark.py` on every push, so a regression that drops the cut
below 30% fails the build.
