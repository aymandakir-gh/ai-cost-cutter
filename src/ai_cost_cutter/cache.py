"""Response cache.

Reuse a model's response for repeated prompts instead of paying for the same
call twice. Tracks the dollars saved on every cache hit.

Two backends ship in the box:

- :class:`MemoryBackend` (default) — fast, process-local.
- :class:`SQLiteBackend` — persists across processes/runs, stdlib only.

Example::

    from ai_cost_cutter.cache import ResponseCache

    cache = ResponseCache()
    cached = cache.wrap(call)          # call(model, prompt) -> str
    cached("gpt-4o", "Hello")          # miss -> calls provider
    cached("gpt-4o", "Hello")          # hit  -> free
    print(cache.stats.hit_rate, cache.stats.saved_cost)

Key normalization (opt-in, deterministic) raises the hit rate by mapping
trivially-different prompts to the same key::

    # casefold + strip punctuation/accents (and optionally stem):
    cache = ResponseCache(normalize="aggressive")
    cache = ResponseCache(normalize="stemming")
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

from .estimator import estimate_tokens
from .pricing import ModelPrice
from .tokens import count_tokens

Provider = Callable[[str, str], object]
Normalizer = Callable[[str], str]


def default_normalize(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip.

    Improves hit rate without changing meaning for typical prompts. Pass
    ``normalize=None`` to :class:`ResponseCache` to key on the raw text.
    """
    return " ".join((text or "").split())


_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Conservative, deterministic suffix stripping. Order matters: longest first.
_STEM_SUFFIXES = ("ingly", "edly", "ies", "ing", "ed", "ly", "es", "s")


def _stem_word(word: str) -> str:
    """A tiny, deterministic suffix stripper (not a full stemmer).

    Trims a few common English inflectional suffixes so ``"running"`` and
    ``"run"`` collide. Conservative: never shortens a word below three letters.
    After stripping ``-ing``/``-ed`` it collapses a doubled final consonant
    (``running`` -> ``runn`` -> ``run``).
    """
    for suffix in _STEM_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            stem = word[: -len(suffix)]
            if suffix in ("ing", "ed") and len(stem) >= 3:
                # Undo consonant doubling: "runn" -> "run", "stopp" -> "stop".
                if (
                    stem[-1] == stem[-2]
                    and stem[-1] not in "aeiou"
                ):
                    stem = stem[:-1]
            return stem
    return word


def normalize_key(
    text: str,
    *,
    casefold: bool = True,
    strip_punctuation: bool = True,
    strip_accents: bool = True,
    stem: bool = False,
) -> str:
    """Deterministically normalize ``text`` for cache keying.

    Designed to raise the cache hit rate by mapping trivially-different prompts
    ("What is 2+2?" / "what is 2 + 2") to the same key. Every step is opt-out:

    - ``casefold``         : Unicode-aware lowercasing (default on).
    - ``strip_punctuation``: drop punctuation/symbols (default on).
    - ``strip_accents``    : fold accents (café -> cafe) (default on).
    - ``stem``             : trim common English suffixes (default off).

    Whitespace is always collapsed. Purely textual and deterministic, so the
    same input always yields the same key across processes and runs.
    """
    out = text or ""
    # Normalize Unicode form first so casefold/accents behave consistently.
    out = unicodedata.normalize("NFKC", out)
    if strip_accents:
        out = "".join(
            ch
            for ch in unicodedata.normalize("NFKD", out)
            if not unicodedata.combining(ch)
        )
    if casefold:
        out = out.casefold()
    if strip_punctuation:
        out = _PUNCT_RE.sub(" ", out)
    words = out.split()
    if stem:
        words = [_stem_word(w) for w in words]
    return " ".join(words)


def aggressive_normalize(text: str) -> str:
    """Strong normalizer: casefold + strip punctuation/accents (no stemming)."""
    return normalize_key(text)


def stemming_normalize(text: str) -> str:
    """Strongest built-in normalizer: :func:`aggressive_normalize` plus stemming."""
    return normalize_key(text, stem=True)


# Named, deterministic normalization modes selectable on :class:`ResponseCache`.
_NORMALIZE_MODES: Dict[str, Optional[Normalizer]] = {
    "none": None,
    "whitespace": default_normalize,
    "aggressive": aggressive_normalize,
    "stemming": stemming_normalize,
}


def _coerce_text(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        for key in ("text", "content", "message", "output", "completion"):
            if key in raw:
                value = raw[key]
                return value if isinstance(value, str) else str(value)
        return str(raw)
    for attr in ("text", "content"):
        if hasattr(raw, attr):
            value = getattr(raw, attr)
            return value if isinstance(value, str) else str(value)
    return str(raw)


# --- backends -------------------------------------------------------------


class MemoryBackend:
    """In-process dict backend."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Dict]:
        with self._lock:
            rec = self._data.get(key)
            return dict(rec) if rec is not None else None

    def set(self, key: str, record: Dict) -> None:
        with self._lock:
            self._data[key] = dict(record)

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)


class SQLiteBackend:
    """Persistent backend backed by a SQLite file (stdlib only)."""

    def __init__(self, path: str) -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        with self._lock, self._connect() as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS cache ("
                "key TEXT PRIMARY KEY, value TEXT, created_at REAL, "
                "cost REAL, hits INTEGER)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def get(self, key: str) -> Optional[Dict]:
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT value, created_at, cost, hits FROM cache WHERE key=?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return {"value": row[0], "created_at": row[1], "cost": row[2], "hits": row[3]}

    def set(self, key: str, record: Dict) -> None:
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO cache (key, value, created_at, cost, hits) "
                "VALUES (?, ?, ?, ?, ?)",
                (key, record["value"], record["created_at"], record["cost"], record["hits"]),
            )

    def delete(self, key: str) -> None:
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM cache WHERE key=?", (key,))

    def clear(self) -> None:
        with self._lock, self._connect() as con:
            con.execute("DELETE FROM cache")

    def __len__(self) -> int:
        with self._lock, self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM cache").fetchone()[0]


# --- cache ----------------------------------------------------------------


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    saved_cost: float = 0.0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0


class ResponseCache:
    """A response cache with savings accounting."""

    def __init__(
        self,
        backend: Optional[object] = None,
        ttl: Optional[float] = None,
        namespace: str = "",
        normalize: Union[str, Normalizer, None] = default_normalize,
        prices: Optional[Dict[str, ModelPrice]] = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.backend = backend if backend is not None else MemoryBackend()
        self.ttl = ttl
        self.namespace = namespace
        self.normalize = self._resolve_normalize(normalize)
        self.prices = prices
        self._time = time_fn
        self.stats = CacheStats()

    @staticmethod
    def _resolve_normalize(
        normalize: Union[str, Normalizer, None]
    ) -> Normalizer:
        """Resolve ``normalize`` (a mode name, a callable, or None) to a fn.

        Modes: ``"none"`` (raw), ``"whitespace"`` (default), ``"aggressive"``
        (casefold + strip punctuation/accents), ``"stemming"`` (aggressive plus
        light suffix stripping). Each mode is deterministic.
        """
        if isinstance(normalize, str):
            try:
                fn = _NORMALIZE_MODES[normalize]
            except KeyError:
                raise ValueError(
                    f"unknown normalize mode {normalize!r}; choose from "
                    f"{sorted(_NORMALIZE_MODES)} or pass a callable/None"
                )
            return fn or (lambda x: x)
        return normalize or (lambda x: x)

    def make_key(self, model: str, prompt: str, **params: object) -> str:
        norm = self.normalize(prompt)
        payload = json.dumps(
            [self.namespace, model, norm, sorted(params.items())],
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _expired(self, record: Dict) -> bool:
        if self.ttl is None:
            return False
        return (self._time() - record["created_at"]) > self.ttl

    def get(self, model: str, prompt: str, **params: object) -> Optional[str]:
        key = self.make_key(model, prompt, **params)
        rec = self.backend.get(key)
        if rec is None or self._expired(rec):
            if rec is not None:
                self.backend.delete(key)
            self.stats.misses += 1
            return None
        rec["hits"] = rec.get("hits", 0) + 1
        self.backend.set(key, rec)
        self.stats.hits += 1
        self.stats.saved_cost += rec.get("cost", 0.0) or 0.0
        return rec["value"]

    def set(
        self,
        model: str,
        prompt: str,
        response: str,
        cost: Optional[float] = None,
        **params: object,
    ) -> None:
        key = self.make_key(model, prompt, **params)
        if cost is None:
            cost = self._estimate_cost(model, prompt, response)
        self.backend.set(
            key,
            {"value": response, "created_at": self._time(), "cost": cost, "hits": 0},
        )

    def _estimate_cost(self, model: str, prompt: str, response: str) -> float:
        try:
            in_tokens = count_tokens(prompt, model)
            out_tokens = count_tokens(response, model)
            return estimate_tokens(model, in_tokens, out_tokens, self.prices).total_cost
        except Exception:
            return 0.0

    def wrap(self, provider: Provider) -> Provider:
        """Return a cached version of ``provider`` (``call(model, prompt)``)."""

        def cached(model: str, prompt: str) -> str:
            hit = self.get(model, prompt)
            if hit is not None:
                return hit
            response = _coerce_text(provider(model, prompt))
            self.set(model, prompt, response)
            return response

        return cached

    def clear(self) -> None:
        self.backend.clear()
        self.stats = CacheStats()
