import pytest

from ai_cost_cutter.cache import (
    MemoryBackend,
    ResponseCache,
    SQLiteBackend,
    default_normalize,
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
