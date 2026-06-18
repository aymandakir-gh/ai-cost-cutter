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
from .cache import (
    CacheStats,
    MemoryBackend,
    ResponseCache,
    SQLiteBackend,
)
from .compression import (
    CompressionResult,
    MessagePruneResult,
    compress,
    dedupe_lines,
    prune_messages,
    remove_filler,
    strip_whitespace,
    truncate_middle,
)
from .pricing import (
    DEFAULT_PRICES,
    ModelPrice,
    UnknownModelError,
    get_price,
    known_models,
    register_price,
)
from .router import (
    Attempt,
    RouteResult,
    Router,
    confidence_from_logprobs,
    heuristic_confidence,
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
    # router
    "Router",
    "RouteResult",
    "Attempt",
    "heuristic_confidence",
    "confidence_from_logprobs",
    # cache
    "ResponseCache",
    "CacheStats",
    "MemoryBackend",
    "SQLiteBackend",
    # compression
    "compress",
    "CompressionResult",
    "strip_whitespace",
    "dedupe_lines",
    "remove_filler",
    "truncate_middle",
    "prune_messages",
    "MessagePruneResult",
]
