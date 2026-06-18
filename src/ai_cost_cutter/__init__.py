"""ai-cost-cutter: a provider-agnostic toolkit to cut LLM spend.

Modules
-------
- ``pricing``     : provider/model price tables (overridable).
- ``tokens``      : offline-friendly token counting.
- ``estimator``   : per-call cost estimation.
- ``router``      : cheap-first model routing with confidence-based escalation.
- ``cache``       : response caching with savings accounting.
- ``compression`` : prompt/context compression strategies.
- ``ledger``      : a usage log shared across modules.
- ``dashboard``   : cost dashboard built from the ledger.

Everything is provider-agnostic: you inject a ``call(model, prompt) -> str``
function, so OpenAI, Anthropic, or a local model all work the same way and the
whole toolkit is testable offline.
"""

__version__ = "0.1.0"

from .estimator import (
    CostEstimate,
    estimate,
    estimate_messages,
    estimate_tokens,
)
from .pricing import (
    DEFAULT_PRICES,
    ModelPrice,
    UnknownModelError,
    get_price,
    known_models,
    register_price,
)
from .tokens import count_messages_tokens, count_tokens

__all__ = [
    "__version__",
    # estimator
    "CostEstimate",
    "estimate",
    "estimate_messages",
    "estimate_tokens",
    # pricing
    "DEFAULT_PRICES",
    "ModelPrice",
    "UnknownModelError",
    "get_price",
    "known_models",
    "register_price",
    # tokens
    "count_tokens",
    "count_messages_tokens",
]
