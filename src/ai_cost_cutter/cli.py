"""Command-line interface for ai-cost-cutter.

Subcommands are added as modules land. v0.1 ships ``estimate`` and ``models``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from . import __version__
from . import tokens as _tokens
from .estimator import estimate, estimate_tokens
from .pricing import known_models


def _add_estimate_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("estimate", help="estimate the USD cost of a model call")
    p.add_argument("--model", "-m", required=True, help="model id, e.g. gpt-4o")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="prompt text to count input tokens from")
    src.add_argument(
        "--prompt-file", help="path to a file whose contents are the prompt"
    )
    src.add_argument(
        "--input-tokens", type=int, help="explicit input token count"
    )
    p.add_argument(
        "--output-tokens",
        type=int,
        help="explicit output token count (with --input-tokens)",
    )
    p.add_argument(
        "--expected-output",
        type=int,
        default=256,
        help="assumed output tokens when counting from a prompt (default: 256)",
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_estimate)


def _cmd_estimate(args: argparse.Namespace) -> int:
    if args.input_tokens is not None:
        est = estimate_tokens(
            args.model, args.input_tokens, args.output_tokens or 0
        )
    else:
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as fh:
                prompt = fh.read()
        elif args.prompt is not None:
            prompt = args.prompt
        else:
            prompt = sys.stdin.read()
        est = estimate(args.model, prompt, args.expected_output)

    if args.json:
        print(json.dumps(est.as_dict(), indent=2))
    else:
        print(f"model:         {est.model}")
        print(f"input tokens:  {est.input_tokens}")
        print(f"output tokens: {est.output_tokens}")
        print(f"input cost:    ${est.input_cost:.6f}")
        print(f"output cost:   ${est.output_cost:.6f}")
        print(f"total cost:    ${est.total_cost:.6f}")
        if not _tokens.using_tiktoken():
            print("(token counts are heuristic; install tiktoken for exact)")
    return 0


def _add_compress_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compress", help="compress a prompt and report token savings"
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="prompt text (else read stdin)")
    src.add_argument("--prompt-file", help="read the prompt from a file")
    p.add_argument("--model", "-m", help="model id (affects token counting)")
    p.add_argument("--max-tokens", type=int, help="cap output to N tokens")
    p.add_argument("--dedupe", action="store_true", help="drop duplicate lines")
    p.add_argument(
        "--dedupe-near",
        action="store_true",
        help="drop near-duplicate lines (case/punctuation/whitespace-insensitive)",
    )
    p.add_argument("--filler", action="store_true", help="remove filler phrases")
    p.add_argument(
        "--minify-json",
        action="store_true",
        help="losslessly minify JSON blocks embedded in the prompt",
    )
    p.add_argument(
        "--strip-comments",
        action="store_true",
        help="remove comments inside fenced code blocks (lossy)",
    )
    p.add_argument(
        "--stats-only",
        action="store_true",
        help="print only the savings summary (no compressed text on stdout)",
    )
    p.set_defaults(func=_cmd_compress)


def _cmd_compress(args: argparse.Namespace) -> int:
    from .compression import compress

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as fh:
            text = fh.read()
    elif args.prompt is not None:
        text = args.prompt
    else:
        text = sys.stdin.read()

    strategies = ["strip_whitespace"]
    if args.dedupe:
        strategies.append("dedupe_lines")
    if getattr(args, "dedupe_near", False):
        strategies.append("dedupe_near_lines")
    if args.filler:
        strategies.append("remove_filler")
    if getattr(args, "minify_json", False):
        strategies.append("collapse_json_whitespace")
    if getattr(args, "strip_comments", False):
        strategies.append("strip_code_comments")

    result = compress(
        text, strategies=strategies, max_tokens=args.max_tokens, model=args.model
    )
    if not args.stats_only:
        print(result.compressed)
    print(
        f"tokens {result.tokens_before} -> {result.tokens_after} "
        f"({result.reduction:.0%} smaller)",
        file=sys.stderr,
    )
    return 0


def _add_dashboard_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "dashboard", help="summarize spend and savings from a usage ledger"
    )
    p.add_argument(
        "--ledger", required=True, help="path to a JSONL ledger file"
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.add_argument("--html", help="write an HTML dashboard to this path")
    p.set_defaults(func=_cmd_dashboard)


def _cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import build_report
    from .ledger import Ledger

    report = build_report(Ledger(args.ledger))
    if args.html:
        with open(args.html, "w", encoding="utf-8") as fh:
            fh.write(report.render_html())
        print(f"wrote {args.html}", file=sys.stderr)
    if args.json:
        print(json.dumps(report.as_dict(), indent=2))
    elif not args.html:
        print(report.render_text())
    return 0


def _add_benchmark_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "benchmark", help="run the reproducible cost-cut benchmark"
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_benchmark)


def _cmd_benchmark(args: argparse.Namespace) -> int:
    from .benchmark import run_benchmark

    result = run_benchmark()
    if args.json:
        print(
            json.dumps(
                {
                    "requests": result.requests,
                    "baseline_cost": result.baseline_cost,
                    "optimized_cost": result.optimized_cost,
                    "savings_pct": result.savings_pct,
                    "breakdown": {
                        k: {
                            "cost": v,
                            "savings_pct": result.savings_pct_for(k),
                        }
                        for k, v in result.breakdown.items()
                    },
                },
                indent=2,
            )
        )
    else:
        print(result.render_text())
    return 0


def _add_price_compare_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "price-compare",
        help="estimate one prompt/workload across several models side-by-side",
    )
    p.add_argument(
        "--model",
        "-m",
        action="append",
        dest="models",
        metavar="MODEL",
        help="a model to compare (repeatable); default: all known models",
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument("--prompt", help="prompt text (else read stdin)")
    src.add_argument("--prompt-file", help="read the prompt from a file")
    src.add_argument(
        "--input-tokens", type=int, help="explicit input token count"
    )
    p.add_argument(
        "--output-tokens",
        type=int,
        help="explicit output token count (with --input-tokens)",
    )
    p.add_argument(
        "--expected-output",
        type=int,
        default=256,
        help="assumed output tokens when counting from a prompt (default: 256)",
    )
    p.add_argument(
        "--calls",
        type=int,
        default=1,
        help="number of identical calls to total the cost over (default: 1)",
    )
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_price_compare)


def _cmd_price_compare(args: argparse.Namespace) -> int:
    from .estimator import estimate, estimate_tokens
    from .pricing import UnknownModelError

    models = args.models or sorted(known_models())
    calls = max(1, args.calls)

    if args.input_tokens is not None:
        out_tokens = args.output_tokens or 0

        def make_estimate(model):
            return estimate_tokens(model, args.input_tokens, out_tokens)

    else:
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as fh:
                prompt = fh.read()
        elif args.prompt is not None:
            prompt = args.prompt
        else:
            prompt = sys.stdin.read()

        def make_estimate(model):
            return estimate(model, prompt, args.expected_output)

    rows = []
    for model in models:
        try:
            est = make_estimate(model)
        except UnknownModelError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        rows.append((model, est))

    # Cheapest first.
    rows.sort(key=lambda r: r[1].total_cost)

    if args.json:
        print(
            json.dumps(
                {
                    "calls": calls,
                    "models": [
                        {
                            "model": model,
                            "input_tokens": est.input_tokens,
                            "output_tokens": est.output_tokens,
                            "cost_per_call": est.total_cost,
                            "total_cost": est.total_cost * calls,
                        }
                        for model, est in rows
                    ],
                },
                indent=2,
            )
        )
        return 0

    width = max((len(m) for m, _ in rows), default=5)
    header = (
        f"{'model'.ljust(width)}  {'in tok':>7}  {'out tok':>7}  "
        f"{'$/call':>12}  {'$ total':>12}"
    )
    print(header)
    print("-" * len(header))
    cheapest = rows[0][1].total_cost if rows else 0.0
    for model, est in rows:
        per_call = est.total_cost
        total = per_call * calls
        marker = ""
        if cheapest > 0:
            mult = per_call / cheapest
            if mult > 1.0:
                marker = f"  ({mult:.1f}x)"
        print(
            f"{model.ljust(width)}  {est.input_tokens:>7}  "
            f"{est.output_tokens:>7}  ${per_call:>10.6f}  ${total:>10.6f}{marker}"
        )
    if calls > 1:
        print(f"(totals over {calls} identical calls)", file=sys.stderr)
    if not _tokens.using_tiktoken():
        print(
            "(token counts are heuristic; install tiktoken for exact)",
            file=sys.stderr,
        )
    return 0


def _add_models_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("models", help="list known models and their prices")
    p.add_argument("--json", action="store_true", help="emit JSON")
    p.set_defaults(func=_cmd_models)


def _cmd_models(args: argparse.Namespace) -> int:
    models = known_models()
    if args.json:
        print(
            json.dumps(
                {
                    name: {
                        "provider": mp.provider,
                        "input_per_1m": mp.input_per_1m,
                        "output_per_1m": mp.output_per_1m,
                    }
                    for name, mp in models.items()
                },
                indent=2,
            )
        )
        return 0
    width = max((len(n) for n in models), default=5)
    header = f"{'model'.ljust(width)}  {'provider':<10}  {'in/1M':>8}  {'out/1M':>8}"
    print(header)
    print("-" * len(header))
    for name in sorted(models):
        mp = models[name]
        print(
            f"{name.ljust(width)}  {mp.provider:<10}  "
            f"{mp.input_per_1m:>8.2f}  {mp.output_per_1m:>8.2f}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aicc",
        description="Provider-agnostic toolkit to cut LLM spend.",
    )
    parser.add_argument(
        "--version", action="version", version=f"ai-cost-cutter {__version__}"
    )
    sub = parser.add_subparsers(dest="command")
    _add_estimate_parser(sub)
    _add_price_compare_parser(sub)
    _add_compress_parser(sub)
    _add_dashboard_parser(sub)
    _add_benchmark_parser(sub)
    _add_models_parser(sub)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
