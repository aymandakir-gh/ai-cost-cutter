import pytest

from ai_cost_cutter.pricing import (
    DEFAULT_PRICES,
    ModelPrice,
    UnknownModelError,
    get_price,
    known_models,
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
