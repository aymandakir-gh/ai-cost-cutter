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
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

from .estimator import estimate_tokens
from .pricing import ModelPrice
from .tokens import count_tokens

Provider = Callable[[str, str], object]


def default_normalize(text: str) -> str:
    """Collapse all runs of whitespace to single spaces and strip.

    Improves hit rate without changing meaning for typical prompts. Pass
    ``normalize=None`` to :class:`ResponseCache` to key on the raw text.
    """
    return " ".join((text or "").split())


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
        normalize: Optional[Callable[[str], str]] = default_normalize,
        prices: Optional[Dict[str, ModelPrice]] = None,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self.backend = backend if backend is not None else MemoryBackend()
        self.ttl = ttl
        self.namespace = namespace
        self.normalize = normalize or (lambda x: x)
        self.prices = prices
        self._time = time_fn
        self.stats = CacheStats()

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
