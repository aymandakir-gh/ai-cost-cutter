"""Prompt and context compression.

Shrink prompts and context windows so you send (and pay for) fewer input
tokens. Strategies range from lossless tidying to opt-in lossy trimming, plus
chat-history pruning that respects a token budget.

Strategies
----------
- ``strip_whitespace``          : remove trailing whitespace and collapse
  blank-line runs. Lossless for content; on by default.
- ``dedupe_lines``              : drop repeated identical lines (common in
  stuffed context). Opt-in.
- ``dedupe_near_lines``         : drop near-duplicate lines via normalized
  comparison (case/punctuation/whitespace-insensitive, configurable). Opt-in.
- ``remove_filler``             : remove/shorten common filler phrases. Opt-in,
  lossy.
- ``collapse_json_whitespace``  : minify JSON objects/arrays embedded in the
  prompt (lossless — semantically identical JSON). Opt-in.
- ``strip_code_comments``       : remove comments inside fenced code blocks,
  language-aware. Opt-in, lossy.
- ``truncate_middle``           : cap text to a token budget by keeping the head
  and tail and omitting the middle.

Example::

    from ai_cost_cutter.compression import compress
    result = compress(long_prompt, strategies=["strip_whitespace", "dedupe_lines"],
                      max_tokens=2000)
    print(result.reduction)   # e.g. 0.41 -> 41% fewer tokens
"""

from __future__ import annotations

import json
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


_NORMALIZE_PUNCT_RE = re.compile(r"[^\w\s]")
_NORMALIZE_WS_RE = re.compile(r"\s+")


def _normalize_line(
    line: str, case_insensitive: bool, ignore_punctuation: bool
) -> str:
    """Normalize a line for near-duplicate comparison."""
    norm = line
    if case_insensitive:
        norm = norm.lower()
    if ignore_punctuation:
        norm = _NORMALIZE_PUNCT_RE.sub(" ", norm)
    norm = _NORMALIZE_WS_RE.sub(" ", norm).strip()
    return norm


def dedupe_near_lines(
    text: str,
    case_insensitive: bool = True,
    ignore_punctuation: bool = True,
    min_length: int = 1,
) -> str:
    """Drop near-duplicate lines, keeping the first occurrence of each.

    Two lines are "near-duplicates" when their *normalized* forms are equal.
    Normalization (configurable) collapses internal whitespace and, by default,
    lowercases and strips punctuation — so ``"Step 1: do it."`` and
    ``"step 1  do it"`` collapse to one. The original text of the first
    occurrence is preserved verbatim; later near-duplicates are removed.

    Parameters
    ----------
    case_insensitive:
        Lowercase before comparing (default True).
    ignore_punctuation:
        Treat punctuation as insignificant when comparing (default True).
    min_length:
        Lines whose normalized form is shorter than this many characters are
        never deduped (so short structural lines like ``"---"`` or ``"}"`` are
        kept). Default 1 (only fully-empty normalized lines are exempt).
    """
    seen = set()
    out: List[str] = []
    for line in (text or "").split("\n"):
        norm = _normalize_line(line, case_insensitive, ignore_punctuation)
        if not norm or len(norm) < min_length:
            out.append(line)
            continue
        if norm in seen:
            continue
        seen.add(norm)
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


def _find_json_spans(text: str) -> List[tuple]:
    """Return ``(start, end)`` spans of top-level balanced ``{}``/``[]`` blocks.

    Brackets inside JSON string literals are ignored, so a quoted ``"{"`` does
    not throw off the balance count. Only the outermost blocks are returned;
    nested structures are handled by JSON parsing the whole span.
    """
    spans: List[tuple] = []
    i = 0
    n = len(text)
    openers = {"{": "}", "[": "]"}
    while i < n:
        ch = text[i]
        if ch in openers:
            close = openers[ch]
            depth = 0
            j = i
            in_str = False
            escape = False
            while j < n:
                c = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_str = False
                elif c == '"':
                    in_str = True
                elif c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        spans.append((i, j + 1))
                        break
                j += 1
            # Resume scanning after this block (or after this char if unbalanced).
            i = (j + 1) if (spans and spans[-1][0] == i) else (i + 1)
        else:
            i += 1
    return spans


def collapse_json_whitespace(text: str) -> str:
    """Losslessly minify JSON objects/arrays embedded in ``text``.

    Finds balanced ``{...}`` / ``[...]`` spans, and for each that parses as
    valid JSON, replaces it with its compact form (no insignificant
    whitespace). Non-JSON brackets and surrounding prose are left untouched, so
    the transform is lossless: the minified JSON is semantically identical.
    """
    if not text or ("{" not in text and "[" not in text):
        return text
    spans = _find_json_spans(text)
    if not spans:
        return text
    out: List[str] = []
    last = 0
    for start, end in spans:
        chunk = text[start:end]
        try:
            parsed = json.loads(chunk)
        except (ValueError, RecursionError):
            continue
        compact = json.dumps(parsed, separators=(",", ":"), ensure_ascii=False)
        out.append(text[last:start])
        out.append(compact)
        last = end
    if not out:
        return text
    out.append(text[last:])
    return "".join(out)


# --- code-comment stripping -----------------------------------------------

# Single-line comment markers per language family.
_HASH_LANGS = {
    "python", "py", "ruby", "rb", "bash", "sh", "shell", "zsh", "yaml", "yml",
    "toml", "ini", "perl", "pl", "r", "makefile", "make", "dockerfile",
    "nim", "elixir", "ex", "exs", "coffee", "coffeescript",
}
_SLASH_LANGS = {
    "javascript", "js", "jsx", "typescript", "ts", "tsx", "java", "c", "cpp",
    "c++", "cc", "h", "hpp", "cs", "csharp", "go", "golang", "rust", "rs",
    "swift", "kotlin", "kt", "scala", "php", "dart", "objective-c", "objc",
}  # support // line comments and /* */ block comments
_DASH_LANGS = {"sql", "lua", "haskell", "hs", "ada", "elm"}

# Map a language to its (line_marker, supports_block_comment) profile.
def _comment_profile(lang: str):
    lang = (lang or "").strip().lower()
    if lang in _HASH_LANGS:
        return ("#", False)
    if lang in _SLASH_LANGS:
        return ("//", True)
    if lang in _DASH_LANGS:
        return ("--", False)
    return None


# Matches string literals (single, double, or backtick) so we can skip over
# comment markers that appear inside them.
_STRING_RE = re.compile(
    r"""'(?:\\.|[^'\\])*'|"(?:\\.|[^"\\])*"|`(?:\\.|[^`\\])*`""",
    re.DOTALL,
)


def _strip_line_comment(line: str, marker: str) -> str:
    """Remove a trailing ``marker`` comment from ``line``, respecting strings."""
    if marker not in line:
        return line
    # Walk the line, tracking string state, to find the first marker that is
    # not inside a string literal.
    i = 0
    n = len(line)
    mlen = len(marker)
    while i < n:
        ch = line[i]
        if ch in "'\"`":
            m = _STRING_RE.match(line, i)
            if m:
                i = m.end()
                continue
            i += 1
            continue
        if line.startswith(marker, i):
            return line[:i].rstrip()
        i += 1
    return line


def _strip_block_comments(code: str) -> str:
    """Remove ``/* ... */`` block comments outside of string literals."""
    if "/*" not in code:
        return code
    out: List[str] = []
    i = 0
    n = len(code)
    while i < n:
        ch = code[i]
        if ch in "'\"`":
            m = _STRING_RE.match(code, i)
            if m:
                out.append(m.group(0))
                i = m.end()
                continue
            out.append(ch)
            i += 1
            continue
        if code.startswith("/*", i):
            end = code.find("*/", i + 2)
            if end == -1:
                # Unterminated block comment: drop the rest defensively.
                break
            i = end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _strip_comments_in_code(code: str, lang: str) -> str:
    profile = _comment_profile(lang)
    if profile is None:
        return code
    marker, has_block = profile
    if has_block:
        code = _strip_block_comments(code)
    out_lines: List[str] = []
    for line in code.split("\n"):
        stripped = _strip_line_comment(line, marker)
        # Drop lines that became empty *only* because they were a full comment.
        if stripped.strip() == "" and line.strip() != "":
            continue
        out_lines.append(stripped)
    return "\n".join(out_lines)


_FENCE_RE = re.compile(
    r"(?P<fence>^[ \t]*(?:```+|~~~+))[ \t]*(?P<info>[^\n`]*)\n"
    r"(?P<body>.*?)"
    r"(?P<close>^[ \t]*(?:```+|~~~+)[ \t]*$)",
    re.DOTALL | re.MULTILINE,
)


def strip_code_comments(text: str) -> str:
    """Remove comments inside fenced code blocks (language-aware).

    Opt-in and **lossy**: comments can carry intent, so only enable this when
    you are confident the model does not need them. Only fenced blocks
    (```` ```lang ````` … ```` ``` ````) with a recognised language are touched;
    prose and unknown languages are left untouched. Comment markers inside
    string literals are preserved.
    """
    if not text or ("```" not in text and "~~~" not in text):
        return text

    def repl(m: "re.Match") -> str:
        info = m.group("info").strip()
        lang = info.split()[0] if info else ""
        body = m.group("body")
        cleaned = _strip_comments_in_code(body, lang)
        return f"{m.group('fence')}{m.group('info')}\n{cleaned}{m.group('close')}"

    return _FENCE_RE.sub(repl, text)


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
    "dedupe_near_lines": dedupe_near_lines,
    "remove_filler": remove_filler,
    "collapse_json_whitespace": collapse_json_whitespace,
    "strip_code_comments": strip_code_comments,
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
