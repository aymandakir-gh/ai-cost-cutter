"""Token counting.

Uses ``tiktoken`` when it is installed (exact for OpenAI models, a good
approximation for others). Falls back to a deterministic, dependency-free
heuristic so the toolkit works fully offline.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

try:  # pragma: no cover - exercised indirectly depending on environment
    import tiktoken as _tiktoken
except Exception:  # pragma: no cover
    _tiktoken = None

# Per-message overhead used by chat APIs (priming tokens for role/formatting).
_PER_MESSAGE_OVERHEAD = 4
_REPLY_PRIMING = 3


def using_tiktoken() -> bool:
    """Return True if exact token counting via tiktoken is available."""
    return _tiktoken is not None


def _heuristic_tokens(text: str) -> int:
    """Deterministic offline estimate of token count.

    Approximates BPE tokenizers: roughly four characters per token, but never
    fewer tokens than whitespace-delimited words (so short, word-dense text is
    not undercounted).
    """
    if not text:
        return 0
    char_estimate = len(text) / 4.0
    word_estimate = len(text.split())
    return max(1, round(max(char_estimate, word_estimate)))


def _encoding_for(model: Optional[str]):
    """Return a tiktoken encoding for ``model`` or None."""
    if _tiktoken is None:
        return None
    try:
        if model:
            return _tiktoken.encoding_for_model(model)
    except Exception:
        pass
    try:
        return _tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover
        return None


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Count the number of tokens in ``text``.

    When ``tiktoken`` is available it is used (exact for the given model);
    otherwise a deterministic heuristic is used.
    """
    if text is None:
        return 0
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    enc = _encoding_for(model)
    if enc is not None:
        try:
            return len(enc.encode(text))
        except Exception:  # pragma: no cover - defensive
            pass
    return _heuristic_tokens(text)


def count_messages_tokens(
    messages: Iterable[Mapping[str, str]], model: Optional[str] = None
) -> int:
    """Estimate tokens for a list of chat ``messages``.

    Each message is ``{"role": ..., "content": ...}``. Adds a small per-message
    overhead plus reply priming, mirroring how chat APIs bill formatting.
    """
    total = 0
    count = 0
    for msg in messages:
        count += 1
        total += _PER_MESSAGE_OVERHEAD
        content = msg.get("content", "") if isinstance(msg, Mapping) else ""
        total += count_tokens(content or "", model)
        role = msg.get("role", "") if isinstance(msg, Mapping) else ""
        if role:
            total += count_tokens(role, model)
    if count:
        total += _REPLY_PRIMING
    return total
