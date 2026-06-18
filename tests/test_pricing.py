import pytest

from ai_cost_cutter.estimator import estimate_tokens
from ai_cost_cutter.pricing import (
    DEFAULT_PRICES,
    ModelPrice,
    UnknownModelError,
    get_price,
    known_models,
    known_providers,
    models_for_provider,
    register_price,
)


def test_default_table_has_each_provider_tier():
    providers = {mp.provider for mp in DEFAULT_PRICES.values()}
    assert {"openai", "anthropic", "local"} <= providers


def test_get_price_exact_match():
    price = get_price("gpt-4o")
    assert price.input_per_1m == 2.50
    assert price.output_per_1m == 10.00
    assert price.provider == "openai"


def test_get_price_resolves_dated_model_id_by_prefix():
    # An exact key is absent but the prefix resolves.
    assert get_price("gpt-4o-2024-08-06") is get_price("gpt-4o")


def test_get_price_prefers_longest_prefix():
    # "gpt-4o-mini-2024" should resolve to gpt-4o-mini, not gpt-4o.
    assert get_price("gpt-4o-mini-2024-07-18") is get_price("gpt-4o-mini")


def test_unknown_model_raises():
    with pytest.raises(UnknownModelError):
        get_price("does-not-exist-xyz")


def test_register_and_override_price():
    register_price("my-custom-llm", ModelPrice(0.1, 0.2, "custom"))
    assert "my-custom-llm" in known_models()
    assert get_price("my-custom-llm").input_per_1m == 0.1


def test_custom_prices_table_argument():
    table = {"x": ModelPrice(1.0, 2.0, "x")}
    assert get_price("x", prices=table).output_per_1m == 2.0
    with pytest.raises(UnknownModelError):
        get_price("gpt-4o", prices=table)


def test_negative_price_rejected():
    with pytest.raises(ValueError):
        ModelPrice(-1.0, 0.0)


def test_local_models_are_free():
    assert get_price("local").input_per_1m == 0.0
    assert get_price("local").output_per_1m == 0.0


# --- expanded multi-provider price table ----------------------------------

EXPECTED_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "deepseek",
    "xai",
    "groq",
    "local",
}


def test_default_table_covers_all_major_providers():
    assert EXPECTED_PROVIDERS <= known_providers()


# A representative model from each newly added provider.
_SAMPLE_MODELS = [
    "gpt-4.1",
    "o3-mini",
    "claude-opus-4",
    "claude-3-7-sonnet",
    "gemini-2.5-pro",
    "gemini-1.5-flash",
    "mistral-large",
    "codestral",
    "command-r-plus",
    "deepseek-chat",
    "grok-4",
    "llama-3.3-70b-versatile",
    "gemma2-9b-it",
]


@pytest.mark.parametrize("model", _SAMPLE_MODELS)
def test_new_models_resolve(model):
    price = get_price(model)
    assert isinstance(price, ModelPrice)
    assert price.provider != "unknown"
    assert price.input_per_1m >= 0
    assert price.output_per_1m >= 0


@pytest.mark.parametrize("model", _SAMPLE_MODELS)
def test_new_models_cost_estimate(model):
    # 1M input + 1M output tokens should equal input_per_1m + output_per_1m.
    price = get_price(model)
    est = estimate_tokens(model, 1_000_000, 1_000_000)
    assert est.total_cost == pytest.approx(price.input_per_1m + price.output_per_1m)


def test_new_models_resolve_dated_suffixes():
    # Dated/variant ids resolve to the base entry by longest-prefix match.
    assert get_price("gemini-1.5-flash-002") is get_price("gemini-1.5-flash")
    assert get_price("claude-3-5-sonnet-20241022") is get_price("claude-3-5-sonnet")


def test_known_providers_and_models_for_provider_agree():
    for provider in known_providers():
        models = models_for_provider(provider)
        assert models  # every advertised provider has at least one model
        assert all(mp.provider == provider for mp in models.values())


def test_models_for_provider_filters():
    google = models_for_provider("google")
    assert "gemini-2.5-pro" in google
    assert "gpt-4o" not in google


def test_every_default_price_is_well_formed():
    for name, mp in DEFAULT_PRICES.items():
        assert isinstance(name, str) and name
        assert mp.input_per_1m >= 0 and mp.output_per_1m >= 0
        assert mp.provider
