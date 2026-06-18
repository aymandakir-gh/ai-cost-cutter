"""Prompt and context compression.

Shrink prompts and context windows so you send (and pay for) fewer input
tokens. Strategies range from lossless tidying to opt-in lossy trimming, plus
chat-history pruning that respects a token budget.

Strategies
----------
- ``strip_whitespace`` : remove trailing whitespace and collapse blank-line
  runs. Lossless for content; on by default.
- ``dedupe_lines``     : drop repeated identical lines (common in stuffed
  context). Opt-in.
- ``remove_filler``    : remove/shorten common filler phrases. Opt-in, lossy.
- ``truncate_middle``  : cap text to a token budget by keeping the head and
  tail and omitting the middle.

Example::

    from ai_cost_cutter.compression import compress
    result = compress(long_prompt, strategies=["strip_whitespace", "dedupe_lines"],
                      max_tokens=2000)
    print(result.reduction)   # e.g. 0.41 -> 41% fewer tokens
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, List, Mapping, Optional, Sequence, Union

from .tokens import count_messages_tokens, count_tokens

Strategy = Union[str, Callable[[str], str]]

# Curated, conservative filler reductions (case-insensitive). Order matters:
# multi-word phrases first.
_FILLER_RULES = [
    (r"\bdue to the fact that\b", "because"),
    (r"\bin order to\b", "to"),
    (r"\bat this point in time\b", "now"),
    (r"\bin the event that\b", "if"),
    (r"\bit is important to note that\b", ""),
    (r"\bplease note that\b", ""),
    (r"\bneedless to say\b", ""),
    (r"\bas a matter of fact\b", ""),
    (r"\bfor all intents and purposes\b", ""),
    (r"\bplease\b", ""),
    (r"\bkindly\b", ""),
    (r"\bbasically\b", ""),
    (r"\bactually\b", ""),
    (r"\bvery\b", ""),
    (r"\breally\b", ""),
    (r"\bjust\b", ""),
]
_COMPILED_FILLER = [(re.compile(pat, re.IGNORECASE), repl) for pat, repl in _FILLER_RULES]


def strip_whitespace(text: str) -> str:
    """Trim trailing whitespace per line and collapse blank-line runs."""
    if not text:
        return ""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    out: List[str] = []
    blanks = 0
    for line in lines:
        if line == "":
            blanks += 1
            if blanks <= 1:
                out.append(line)
        else:
            blanks = 0
            out.append(line)
    return "\n".join(out).strip()


def dedupe_lines(text: str) -> str:
    """Remove repeated identical non-empty lines, keeping first occurrence."""
    seen = set()
    out: List[str] = []
    for line in text.split("\n"):
        key = line.strip()
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(line)
    return "\n".join(out)


def remove_filler(text: str) -> str:
    """Remove or shorten common filler phrases, then tidy spacing."""
    out = text
    for pattern, repl in _COMPILED_FILLER:
        out = pattern.sub(repl, out)
    # Collapse spaces left behind and fix spacing before punctuation.
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r" +([,.;:!?])", r"\1", out)
    return out.strip()


def truncate_middle(
    text: str,
    max_tokens: int,
    model: Optional[str] = None,
    marker: str = "\n... [{n} words omitted] ...\n",
) -> str:
    """Cap ``text`` to ``max_tokens`` by keeping the head and tail.

    Uses a binary search over how many words to keep so the result is at or
    under the budget regardless of which token counter is active.
    """
    if max_tokens <= 0:
        return ""
    if count_tokens(text, model) <= max_tokens:
        return text
    words = text.split()
    total = len(words)

    def build(keep: int) -> str:
        if keep >= total:
            return text
        head = keep // 2
        tail = keep - head
        omitted = total - head - tail
        mk = marker.format(n=omitted)
        return " ".join(words[:head]) + mk + " ".join(words[-tail:])

    lo, hi = 0, total
    best = build(0)
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = build(mid)
        if count_tokens(candidate, model) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


STRATEGIES = {
    "strip_whitespace": strip_whitespace,
    "dedupe_lines": dedupe_lines,
    "remove_filler": remove_filler,
}

DEFAULT_STRATEGIES = ("strip_whitespace",)


@dataclass(frozen=True)
class CompressionResult:
    original: str
    compressed: str
    tokens_before: int
    tokens_after: int

    @property
    def saved_tokens(self) -> int:
        return self.tokens_before - self.tokens_after

    @property
    def ratio(self) -> float:
        """Fraction of tokens kept (compressed / original)."""
        if self.tokens_before == 0:
            return 1.0
        return self.tokens_after / self.tokens_before

    @property
    def reduction(self) -> float:
        """Fraction of tokens removed (1 - ratio)."""
        return 1.0 - self.ratio


def _resolve(strategy: Strategy) -> Callable[[str], str]:
    if callable(strategy):
        return strategy
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise ValueError(
            f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)} "
            f"or pass a callable"
        )


def compress(
    text: str,
    strategies: Sequence[Strategy] = DEFAULT_STRATEGIES,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> CompressionResult:
    """Run ``text`` through ``strategies`` (then optional truncation)."""
    before = count_tokens(text, model)
    out = text or ""
    for strategy in strategies:
        out = _resolve(strategy)(out)
    if max_tokens is not None:
        out = truncate_middle(out, max_tokens, model=model)
    after = count_tokens(out, model)
    return CompressionResult(text or "", out, before, after)


@dataclass(frozen=True)
class MessagePruneResult:
    messages: List[Mapping[str, str]] = field(default_factory=list)
    tokens_before: int = 0
    tokens_after: int = 0
    dropped: int = 0

    @property
    def saved_tokens(self) -> int:
        return self.tokens_before - self.tokens_after


def prune_messages(
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    model: Optional[str] = None,
    keep_system: bool = True,
    keep_recent: int = 2,
) -> MessagePruneResult:
    """Drop oldest messages until the history fits ``max_tokens``.

    System messages (when ``keep_system``) and the most recent ``keep_recent``
    messages are protected and never dropped.
    """
    msgs = list(messages)
    n = len(msgs)
    before = count_messages_tokens(msgs, model)
    if before <= max_tokens or n == 0:
        return MessagePruneResult(msgs, before, before, 0)

    protected = set()
    if keep_system:
        protected.update(
            i for i, m in enumerate(msgs) if m.get("role") == "system"
        )
    protected.update(range(max(0, n - keep_recent), n))

    droppable = [i for i in range(n) if i not in protected]  # oldest first
    removed = set()
    for idx in droppable:
        kept = [m for j, m in enumerate(msgs) if j not in removed]
        if count_messages_tokens(kept, model) <= max_tokens:
            break
        removed.add(idx)

    result_msgs = [m for j, m in enumerate(msgs) if j not in removed]
    after = count_messages_tokens(result_msgs, model)
    return MessagePruneResult(result_msgs, before, after, len(removed))
