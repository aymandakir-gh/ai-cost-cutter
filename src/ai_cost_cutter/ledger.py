"""Usage ledger.

A simple append-only log of model calls that the dashboard reads. Records can
be kept in memory or persisted as JSON Lines so spend accrues across runs and
processes.

Each :class:`CallRecord` carries the actual ``cost`` and, where known, the
``baseline_cost`` — what the call *would* have cost without optimization (e.g.
the premium model, or a cache miss). The difference is your savings.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional

from .estimator import estimate_tokens


@dataclass(frozen=True)
class CallRecord:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float = 0.0
    cached: bool = False
    baseline_cost: Optional[float] = None
    timestamp: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def effective_baseline(self) -> float:
        """The cost avoided is measured against this. Defaults to ``cost``."""
        return self.cost if self.baseline_cost is None else self.baseline_cost

    @property
    def savings(self) -> float:
        return self.effective_baseline - self.cost

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CallRecord":
        fields = {
            "model",
            "input_tokens",
            "output_tokens",
            "cost",
            "cached",
            "baseline_cost",
            "timestamp",
            "metadata",
        }
        return cls(**{k: v for k, v in data.items() if k in fields})


class Ledger:
    """An append-only record of model calls (in-memory or JSONL-backed)."""

    def __init__(self, path: Optional[str] = None, time_fn=time.time) -> None:
        self.path = str(path) if path else None
        self._time = time_fn
        self._mem: List[CallRecord] = []

    def record(self, record: CallRecord) -> CallRecord:
        if record.timestamp is None:
            record = replace(record, timestamp=self._time())
        if self.path:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.as_dict()) + "\n")
        else:
            self._mem.append(record)
        return record

    def record_call(
        self,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost: Optional[float] = None,
        cached: bool = False,
        baseline_cost: Optional[float] = None,
        metadata: Optional[Dict[str, object]] = None,
    ) -> CallRecord:
        if cost is None:
            try:
                cost = estimate_tokens(model, input_tokens, output_tokens).total_cost
            except Exception:
                cost = 0.0
        return self.record(
            CallRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached=cached,
                baseline_cost=baseline_cost,
                metadata=metadata or {},
            )
        )

    def record_route(self, result) -> CallRecord:
        """Record a :class:`~ai_cost_cutter.router.RouteResult`."""
        return self.record_call(
            model=result.model,
            cost=result.total_cost,
            baseline_cost=result.baseline_cost,
            metadata={
                "escalated": result.escalated,
                "accepted": result.accepted,
                "confidence": result.confidence,
            },
        )

    def record_cache_hit(self, model: str, avoided_cost: float) -> CallRecord:
        """Record a cache hit: zero cost, ``avoided_cost`` saved."""
        return self.record_call(
            model=model, cost=0.0, cached=True, baseline_cost=avoided_cost
        )

    def records(self) -> List[CallRecord]:
        if self.path:
            out: List[CallRecord] = []
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            out.append(CallRecord.from_dict(json.loads(line)))
            return out
        return list(self._mem)

    def clear(self) -> None:
        if self.path and os.path.exists(self.path):
            os.remove(self.path)
        self._mem = []
