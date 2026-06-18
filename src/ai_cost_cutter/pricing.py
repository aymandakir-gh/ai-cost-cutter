"""Model price tables.

Prices are approximate public list prices in **USD per 1,000,000 tokens** and
are meant to be overridden. The toolkit's savings come from *mechanisms*
(routing, caching, compression), not from any particular price table, so feel
free to plug in your negotiated or self-hosted numbers.

Example::

    from ai_cost_cutter.pricing import register_price, ModelPrice, get_price
    register_price("my-llm", ModelPrice(input_per_1m=0.1, output_per_1m=0.2))
    price = get_price("gpt-4o")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


class UnknownModelError(KeyError):
    """Raised when a model has no known price and no override is supplied."""


@dataclass(frozen=True)
class ModelPrice:
    """Price of a model in USD per 1,000,000 tokens."""

    input_per_1m: float
    output_per_1m: float
    provider: str = "unknown"

    def __post_init__(self) -> None:
        if self.input_per_1m < 0 or self.output_per_1m < 0:
            raise ValueError("prices must be non-negative")


# Approximate public list prices (USD per 1M tokens). Override as needed.
DEFAULT_PRICES: Dict[str, ModelPrice] = {
    # --- OpenAI ---
    "gpt-4o": ModelPrice(2.50, 10.00, "openai"),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, "openai"),
    "gpt-4-turbo": ModelPrice(10.00, 30.00, "openai"),
    "gpt-4": ModelPrice(30.00, 60.00, "openai"),
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50, "openai"),
    # --- Anthropic ---
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00, "anthropic"),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00, "anthropic"),
    "claude-3-opus": ModelPrice(15.00, 75.00, "anthropic"),
    "claude-3-sonnet": ModelPrice(3.00, 15.00, "anthropic"),
    "claude-3-haiku": ModelPrice(0.25, 1.25, "anthropic"),
    # --- Local / self-hosted (compute cost approximated as ~0) ---
    "local": ModelPrice(0.0, 0.0, "local"),
    "local-small": ModelPrice(0.0, 0.0, "local"),
    "local-large": ModelPrice(0.0, 0.0, "local"),
}

# Runtime-registered overrides / additions.
_CUSTOM_PRICES: Dict[str, ModelPrice] = {}


def register_price(model: str, price: ModelPrice) -> None:
    """Register or override the price for ``model`` (process-global)."""
    if not isinstance(price, ModelPrice):
        raise TypeError("price must be a ModelPrice")
    _CUSTOM_PRICES[model] = price


def known_models() -> Dict[str, ModelPrice]:
    """Return all known models (defaults plus registered overrides)."""
    merged = dict(DEFAULT_PRICES)
    merged.update(_CUSTOM_PRICES)
    return merged


def _resolve_key(model: str, table: Dict[str, ModelPrice]) -> Optional[str]:
    """Resolve a model id to a known key.

    Tries an exact match first, then falls back to the longest known key that
    is a prefix of ``model`` (so dated ids like ``gpt-4o-2024-08-06`` resolve
    to ``gpt-4o``).
    """
    if model in table:
        return model
    candidates = [k for k in table if model.startswith(k)]
    if candidates:
        return max(candidates, key=len)
    return None


def get_price(model: str, prices: Optional[Dict[str, ModelPrice]] = None) -> ModelPrice:
    """Return the :class:`ModelPrice` for ``model``.

    ``prices`` may be supplied to use a custom table; otherwise the default
    table plus any registered overrides is used. Raises
    :class:`UnknownModelError` if the model cannot be resolved.
    """
    table = prices if prices is not None else known_models()
    key = _resolve_key(model, table)
    if key is None:
        raise UnknownModelError(
            f"no price for model {model!r}; register one with "
            f"register_price({model!r}, ModelPrice(...)) or pass a custom "
            f"`prices` table"
        )
    return table[key]
