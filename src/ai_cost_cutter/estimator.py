"""Per-call cost estimation.

Combine a price table (:mod:`ai_cost_cutter.pricing`) with token counts
(:mod:`ai_cost_cutter.tokens`) to estimate the USD cost of a call before or
after you make it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

from . import tokens as _tokens
from .pricing import ModelPrice, get_price

# Fallback assumption when the caller does not specify expected output length.
DEFAULT_EXPECTED_OUTPUT_TOKENS = 256


@dataclass(frozen=True)
class CostEstimate:
    """The estimated cost of a single model call."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> Dict[str, float]:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total_cost,
        }


def estimate_tokens(
    model: str,
    input_tokens: int,
    output_tokens: int = 0,
    prices: Optional[Dict[str, ModelPrice]] = None,
) -> CostEstimate:
    """Estimate cost from explicit token counts."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    price = get_price(model, prices)
    input_cost = input_tokens / 1_000_000 * price.input_per_1m
    output_cost = output_tokens / 1_000_000 * price.output_per_1m
    return CostEstimate(
        model=model,
        input_tokens=int(input_tokens),
        output_tokens=int(output_tokens),
        input_cost=input_cost,
        output_cost=output_cost,
    )


def estimate(
    model: str,
    prompt: str,
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS,
    prices: Optional[Dict[str, ModelPrice]] = None,
) -> CostEstimate:
    """Estimate cost for a text ``prompt`` (input tokens are counted for you)."""
    input_tokens = _tokens.count_tokens(prompt, model)
    return estimate_tokens(model, input_tokens, expected_output_tokens, prices)


def estimate_messages(
    model: str,
    messages: Iterable[Mapping[str, str]],
    expected_output_tokens: int = DEFAULT_EXPECTED_OUTPUT_TOKENS,
    prices: Optional[Dict[str, ModelPrice]] = None,
) -> CostEstimate:
    """Estimate cost for a list of chat ``messages``."""
    input_tokens = _tokens.count_messages_tokens(messages, model)
    return estimate_tokens(model, input_tokens, expected_output_tokens, prices)
