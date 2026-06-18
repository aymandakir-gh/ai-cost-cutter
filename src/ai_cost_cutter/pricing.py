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


# Approximate public list prices (USD per 1M tokens). These are best-effort
# snapshots of vendors' published list prices and **WILL drift** — treat them as
# approximate and override with your own (negotiated/self-hosted) numbers via
# ``register_price`` or a custom ``prices`` table. The toolkit's savings come
# from mechanisms (routing, caching, compression), not from this table.
DEFAULT_PRICES: Dict[str, ModelPrice] = {
    # --- OpenAI ---
    "gpt-4o": ModelPrice(2.50, 10.00, "openai"),
    "gpt-4o-mini": ModelPrice(0.15, 0.60, "openai"),
    "gpt-4.1": ModelPrice(2.00, 8.00, "openai"),
    "gpt-4.1-mini": ModelPrice(0.40, 1.60, "openai"),
    "gpt-4.1-nano": ModelPrice(0.10, 0.40, "openai"),
    "gpt-4-turbo": ModelPrice(10.00, 30.00, "openai"),
    "gpt-4": ModelPrice(30.00, 60.00, "openai"),
    "gpt-3.5-turbo": ModelPrice(0.50, 1.50, "openai"),
    "o1": ModelPrice(15.00, 60.00, "openai"),
    "o1-mini": ModelPrice(1.10, 4.40, "openai"),
    "o3": ModelPrice(2.00, 8.00, "openai"),
    "o3-mini": ModelPrice(1.10, 4.40, "openai"),
    "o4-mini": ModelPrice(1.10, 4.40, "openai"),
    # --- Anthropic ---
    "claude-opus-4": ModelPrice(15.00, 75.00, "anthropic"),
    "claude-sonnet-4": ModelPrice(3.00, 15.00, "anthropic"),
    "claude-3-7-sonnet": ModelPrice(3.00, 15.00, "anthropic"),
    "claude-3-5-sonnet": ModelPrice(3.00, 15.00, "anthropic"),
    "claude-3-5-haiku": ModelPrice(0.80, 4.00, "anthropic"),
    "claude-3-opus": ModelPrice(15.00, 75.00, "anthropic"),
    "claude-3-sonnet": ModelPrice(3.00, 15.00, "anthropic"),
    "claude-3-haiku": ModelPrice(0.25, 1.25, "anthropic"),
    # --- Google (Gemini) ---
    "gemini-2.5-pro": ModelPrice(1.25, 10.00, "google"),
    "gemini-2.5-flash": ModelPrice(0.30, 2.50, "google"),
    "gemini-2.0-flash": ModelPrice(0.10, 0.40, "google"),
    "gemini-2.0-flash-lite": ModelPrice(0.075, 0.30, "google"),
    "gemini-1.5-pro": ModelPrice(1.25, 5.00, "google"),
    "gemini-1.5-flash": ModelPrice(0.075, 0.30, "google"),
    "gemini-1.5-flash-8b": ModelPrice(0.0375, 0.15, "google"),
    # --- Mistral ---
    "mistral-large": ModelPrice(2.00, 6.00, "mistral"),
    "mistral-small": ModelPrice(0.20, 0.60, "mistral"),
    "mistral-nemo": ModelPrice(0.15, 0.15, "mistral"),
    "codestral": ModelPrice(0.30, 0.90, "mistral"),
    "open-mistral-7b": ModelPrice(0.25, 0.25, "mistral"),
    "open-mixtral-8x7b": ModelPrice(0.70, 0.70, "mistral"),
    # --- Cohere ---
    "command-a": ModelPrice(2.50, 10.00, "cohere"),
    "command-r-plus": ModelPrice(2.50, 10.00, "cohere"),
    "command-r": ModelPrice(0.15, 0.60, "cohere"),
    "command-r7b": ModelPrice(0.0375, 0.15, "cohere"),
    # --- DeepSeek ---
    "deepseek-chat": ModelPrice(0.27, 1.10, "deepseek"),
    "deepseek-reasoner": ModelPrice(0.55, 2.19, "deepseek"),
    # --- xAI (Grok) ---
    "grok-4": ModelPrice(3.00, 15.00, "xai"),
    "grok-3": ModelPrice(3.00, 15.00, "xai"),
    "grok-3-mini": ModelPrice(0.30, 0.50, "xai"),
    "grok-2": ModelPrice(2.00, 10.00, "xai"),
    # --- Groq (hosted open models) ---
    "llama-3.3-70b-versatile": ModelPrice(0.59, 0.79, "groq"),
    "llama-3.1-8b-instant": ModelPrice(0.05, 0.08, "groq"),
    "llama3-70b-8192": ModelPrice(0.59, 0.79, "groq"),
    "llama3-8b-8192": ModelPrice(0.05, 0.08, "groq"),
    "gemma2-9b-it": ModelPrice(0.20, 0.20, "groq"),
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


def known_providers() -> "set[str]":
    """Return the set of provider tags present in the known price table."""
    return {mp.provider for mp in known_models().values()}


def models_for_provider(provider: str) -> Dict[str, ModelPrice]:
    """Return the known models whose ``provider`` tag matches ``provider``."""
    return {
        name: mp
        for name, mp in known_models().items()
        if mp.provider == provider
    }


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
