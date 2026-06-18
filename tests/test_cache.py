import pytest

from ai_cost_cutter.cache import (
    MemoryBackend,
    ResponseCache,
    SQLiteBackend,
    aggressive_normalize,
    default_normalize,
    normalize_key,
    stemming_normalize,
)


def test_miss_then_hit():
    cache = ResponseCache()
    assert cache.get("gpt-4o", "hello") is None
    cache.set("gpt-4o", "hello", "hi there")
    assert cache.get("gpt-4o", "hello") == "hi there"
    assert cache.stats.hits == 1
    assert cache.stats.misses == 1


def test_normalization_collapses_whitespace():
    assert default_normalize("  a   b\n\tc ") == "a b c"
    cache = ResponseCache()
    cache.set("gpt-4o", "what  is\n2+2?", "4")
    # Different whitespace, same normalized prompt -> hit.
    assert cache.get("gpt-4o", "what is 2+2?") == "4"


def test_normalization_can_be_disabled():
    cache = ResponseCache(normalize=None)
    cache.set("gpt-4o", "what  is 2+2?", "4")
    assert cache.get("gpt-4o", "what is 2+2?") is None  # raw text differs


# --- key normalization modes ----------------------------------------------


def test_normalize_key_casefold_punctuation_accents():
    assert normalize_key("What is 2+2?") == "what is 2 2"
    assert normalize_key("Café!") == "cafe"
    # Idempotent and deterministic.
    assert normalize_key("Hello,  WORLD") == normalize_key("hello world")


def test_normalize_key_toggles():
    # With punctuation kept, only case/whitespace/accents are touched.
    assert normalize_key("Don't!", strip_punctuation=False) == "don't!"
    assert normalize_key("CAFÉ", casefold=False, strip_accents=True) == "CAFE"


def test_aggressive_and_stemming_normalizers():
    assert aggressive_normalize("Running, Fast!") == "running fast"
    # "running" -> strip "ing" -> "runn" -> undo doubling -> "run"; "quickly" -> "quick".
    assert stemming_normalize("running quickly") == "run quick"
    # Stemming never shortens below three letters.
    assert stemming_normalize("is as") == "is as"


def test_aggressive_mode_collapses_punctuation_and_case_to_one_key():
    cache = ResponseCache(normalize="aggressive")
    cache.set("gpt-4o", "What is 2+2?", "4")
    # Different punctuation/case/spacing -> same normalized key -> hit.
    assert cache.get("gpt-4o", "what  is 2 + 2") == "4"


def test_aggressive_mode_raises_hit_rate_vs_whitespace():
    # A set of prompts that differ only in case/punctuation/whitespace.
    variants = [
        "What is the capital of France?",
        "what is the capital of france",
        "What  is the Capital of France???",
        "WHAT IS THE CAPITAL OF FRANCE",
    ]

    def run(mode):
        cache = ResponseCache(normalize=mode)
        cached = cache.wrap(lambda m, p: "Paris")
        for v in variants:
            cached("gpt-4o", v)
        return cache.stats.hit_rate

    whitespace_rate = run("whitespace")
    aggressive_rate = run("aggressive")
    # Whitespace-only keying treats the case/punctuation variants as distinct.
    assert whitespace_rate == 0.0
    # Aggressive normalization collapses them: 1 miss + 3 hits.
    assert aggressive_rate == pytest.approx(3 / 4)


def test_stemming_mode_collapses_inflections():
    cache = ResponseCache(normalize="stemming")
    cache.set("gpt-4o", "summarize the running logs", "ok")
    assert cache.get("gpt-4o", "Summarize the run log!") == "ok"


def test_normalize_mode_none_uses_raw_text():
    cache = ResponseCache(normalize="none")
    cache.set("gpt-4o", "Hello", "hi")
    assert cache.get("gpt-4o", "hello") is None  # raw text differs


def test_unknown_normalize_mode_raises():
    with pytest.raises(ValueError):
        ResponseCache(normalize="bogus-mode")


def test_callable_normalizer_still_supported():
    cache = ResponseCache(normalize=str.upper)
    cache.set("gpt-4o", "hello", "hi")
    assert cache.get("gpt-4o", "HELLO") == "hi"


def test_different_model_is_a_different_key():
    cache = ResponseCache()
    cache.set("gpt-4o", "hello", "A")
    assert cache.get("gpt-4o-mini", "hello") is None


def test_params_differentiate_keys():
    cache = ResponseCache()
    cache.set("gpt-4o", "hello", "warm", temperature=0.9)
    assert cache.get("gpt-4o", "hello", temperature=0.1) is None
    assert cache.get("gpt-4o", "hello", temperature=0.9) == "warm"


def test_ttl_expiry():
    clock = {"t": 1000.0}
    cache = ResponseCache(ttl=60, time_fn=lambda: clock["t"])
    cache.set("gpt-4o", "hello", "hi")
    clock["t"] = 1050.0  # within ttl
    assert cache.get("gpt-4o", "hello") == "hi"
    clock["t"] = 1100.0  # past ttl
    assert cache.get("gpt-4o", "hello") is None


def test_hit_rate_and_saved_cost():
    cache = ResponseCache()
    provider_calls = []

    def provider(model, prompt):
        provider_calls.append(prompt)
        return "a real, confident answer to the question"

    cached = cache.wrap(provider)
    cached("gpt-4o", "q1")
    cached("gpt-4o", "q1")  # hit
    cached("gpt-4o", "q1")  # hit
    assert len(provider_calls) == 1
    assert cache.stats.hits == 2
    assert cache.stats.misses == 1
    assert cache.stats.hit_rate == pytest.approx(2 / 3)
    assert cache.stats.saved_cost > 0  # gpt-4o is not free


def test_wrap_coerces_dict_response():
    cache = ResponseCache()
    cached = cache.wrap(lambda m, p: {"content": "structured answer"})
    assert cached("gpt-4o", "q") == "structured answer"


def test_explicit_cost_is_used():
    cache = ResponseCache()
    cache.set("local", "q", "free model answer", cost=0.0)
    cache.get("local", "q")
    assert cache.stats.saved_cost == 0.0


def test_clear_resets_store_and_stats():
    cache = ResponseCache()
    cache.set("gpt-4o", "hello", "hi")
    cache.get("gpt-4o", "hello")
    cache.clear()
    assert cache.get("gpt-4o", "hello") is None
    assert cache.stats.hits == 0


def test_sqlite_backend_persists(tmp_path):
    db = str(tmp_path / "cache.db")
    cache1 = ResponseCache(backend=SQLiteBackend(db))
    cache1.set("gpt-4o", "persist me", "stored value")
    # A fresh cache over the same file sees the entry.
    cache2 = ResponseCache(backend=SQLiteBackend(db))
    assert cache2.get("gpt-4o", "persist me") == "stored value"


def test_sqlite_backend_len_and_clear(tmp_path):
    backend = SQLiteBackend(str(tmp_path / "c.db"))
    cache = ResponseCache(backend=backend)
    cache.set("gpt-4o", "a", "1")
    cache.set("gpt-4o", "b", "2")
    assert len(backend) == 2
    backend.clear()
    assert len(backend) == 0


def test_memory_backend_len():
    backend = MemoryBackend()
    cache = ResponseCache(backend=backend)
    cache.set("gpt-4o", "a", "1")
    assert len(backend) == 1
