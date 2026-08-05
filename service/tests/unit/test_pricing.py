"""Tests for src/pricing.py's pure functions (SPEC-cost-accounting.md §5) —
prefix precedence, provider scoping, provider-only fallback, no-match -> None,
zero-rate vs NULL distinction, and malformed table validation."""
import pytest
from pydantic import ValidationError

from src.models.pricing import PriceEntry, PriceMatch, PricingConfig
from src.pricing import Rate, resolve_rate, step_cost


def _table(*entries: dict, currency: str = "USD") -> PricingConfig:
    return PricingConfig.model_validate({"currency": currency, "models": list(entries)})


# ---------------------------------------------------------------------------
# resolve_rate
# ---------------------------------------------------------------------------

def test_resolve_rate_no_table_is_none():
    assert resolve_rate(None, "anthropic", "claude-sonnet-4-6") is None


def test_resolve_rate_no_match_is_none():
    table = _table({"match": {"provider": "anthropic", "model": "claude-sonnet-4-6"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0})
    assert resolve_rate(table, "openai", "gpt-5") is None


def test_resolve_rate_exact_and_prefix_match():
    table = _table({"match": {"provider": "anthropic", "model": "claude-sonnet-4-6"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0})
    assert resolve_rate(table, "anthropic", "claude-sonnet-4-6") == Rate(3.0, 15.0)
    # prefix match — a dated/suffixed model string still resolves
    assert resolve_rate(table, "anthropic", "claude-sonnet-4-6-20260101") == Rate(3.0, 15.0)


def test_resolve_rate_longest_prefix_wins():
    table = _table(
        {"match": {"provider": "anthropic", "model": "claude"}, "input_per_mtok": 1.0, "output_per_mtok": 1.0},
        {"match": {"provider": "anthropic", "model": "claude-haiku"}, "input_per_mtok": 2.0, "output_per_mtok": 2.0},
    )
    # both "claude" and "claude-haiku" are valid prefixes of "claude-haiku-4-5" —
    # the longer, more specific entry must win.
    assert resolve_rate(table, "anthropic", "claude-haiku-4-5") == Rate(2.0, 2.0)


def test_resolve_rate_provider_scoping_prevents_cross_provider_match():
    table = _table({"match": {"provider": "anthropic", "model": "claude"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0})
    # same model prefix, different provider — must not match
    assert resolve_rate(table, "openrouter", "claude-via-openrouter") is None


def test_resolve_rate_provider_only_fallback():
    table = _table(
        {"match": {"provider": "anthropic", "model": "claude-sonnet-4-6"}, "input_per_mtok": 3.0, "output_per_mtok": 15.0},
        {"match": {"provider": "openrouter"}, "input_per_mtok": 2.0, "output_per_mtok": 8.0},
    )
    assert resolve_rate(table, "openrouter", "some/random-model") == Rate(2.0, 8.0)
    # a specific model-prefix entry still beats a provider-only fallback for the same provider
    assert resolve_rate(table, "anthropic", "claude-sonnet-4-6-latest") == Rate(3.0, 15.0)


def test_resolve_rate_fully_generic_fallback_no_provider_constraint():
    table = _table({"match": {}, "input_per_mtok": 0.5, "output_per_mtok": 1.0})
    assert resolve_rate(table, "anything", "whatever-model") == Rate(0.5, 1.0)


def test_resolve_rate_zero_is_a_real_rate_not_none():
    table = _table({"match": {"provider": "ollama"}, "input_per_mtok": 0.0, "output_per_mtok": 0.0})
    rate = resolve_rate(table, "ollama", "llama-local")
    assert rate == Rate(0.0, 0.0)
    assert rate is not None


# ---------------------------------------------------------------------------
# step_cost
# ---------------------------------------------------------------------------

def test_step_cost_no_rate_is_none():
    assert step_cost(None, 1000, 500) is None


def test_step_cost_no_token_data_is_none_even_with_a_rate():
    rate = Rate(3.0, 15.0)
    assert step_cost(rate, None, None) is None


def test_step_cost_computes_from_per_mtok_rate():
    rate = Rate(3.0, 15.0)
    # 1000 input @ $3/mtok + 500 output @ $15/mtok
    assert step_cost(rate, 1000, 500) == pytest.approx(1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000)


def test_step_cost_zero_rate_is_zero_not_none():
    rate = Rate(0.0, 0.0)
    assert step_cost(rate, 1000, 500) == 0.0


def test_step_cost_zero_tokens_with_priced_rate_is_zero():
    rate = Rate(3.0, 15.0)
    assert step_cost(rate, 0, 0) == 0.0


# ---------------------------------------------------------------------------
# Malformed table validation (config-load time)
# ---------------------------------------------------------------------------

def test_malformed_pricing_table_negative_rate_raises():
    with pytest.raises(ValidationError):
        PricingConfig.model_validate({
            "currency": "USD",
            "models": [{"match": {"provider": "anthropic"}, "input_per_mtok": -1.0, "output_per_mtok": 5.0}],
        })


def test_malformed_pricing_table_missing_required_field_raises():
    with pytest.raises(ValidationError):
        PricingConfig.model_validate({
            "currency": "USD",
            "models": [{"match": {"provider": "anthropic"}, "input_per_mtok": 1.0}],
        })


def test_pricing_config_defaults_to_usd_and_empty_models():
    table = PricingConfig.model_validate({})
    assert table.currency == "USD"
    assert table.models == []
    assert table.team_budgets == {}
