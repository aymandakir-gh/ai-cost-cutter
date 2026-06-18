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
    p.add_argument("--filler", action="store_true", help="remove filler phrases")
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
    if args.filler:
        strategies.append("remove_filler")

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
    _add_compress_parser(sub)
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
