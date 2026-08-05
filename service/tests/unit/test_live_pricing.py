"""Tests for src/live_pricing.py's pure functions (SPEC-live-pricing.md):
fuzzy matching of (provider, model) against OpenRouter's catalog, and
approx_step_cost. Also covers refresh_catalog's failure handling."""
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.live_pricing import (
    ApproxRate,
    approx_step_cost,
    refresh_catalog,
    resolve_approx_rate,
)

_CATALOG = [
    {"id": "anthropic/claude-3.5-sonnet", "pricing": {"prompt": "0.000003", "completion": "0.000015"}},
    {"id": "anthropic/claude-3.5-haiku", "pricing": {"prompt": "0.000001", "completion": "0.000005"}},
    {"id": "openai/gpt-4o", "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
]


# ---------------------------------------------------------------------------
# resolve_approx_rate
# ---------------------------------------------------------------------------

def test_resolve_approx_rate_no_catalog_is_none():
    assert resolve_approx_rate(None, "anthropic", "claude-sonnet-4-6") is None


def test_resolve_approx_rate_no_model_is_none():
    assert resolve_approx_rate(_CATALOG, "anthropic", None) is None


def test_resolve_approx_rate_matches_versioned_model_name():
    # Our internal name ("claude-sonnet-4-6") differs from OpenRouter's
    # ("claude-3.5-sonnet") in exactly where the version digits sit — the family
    # key (digit-stripped) must still connect them.
    rate = resolve_approx_rate(_CATALOG, "anthropic", "claude-sonnet-4-6")
    assert rate == ApproxRate(3.0, 15.0, "anthropic/claude-3.5-sonnet")


def test_resolve_approx_rate_distinguishes_sonnet_from_haiku():
    rate = resolve_approx_rate(_CATALOG, "anthropic", "claude-haiku-4-5")
    assert rate.openrouter_id == "anthropic/claude-3.5-haiku"


def test_resolve_approx_rate_scopes_by_provider():
    # Same-ish model name, wrong provider — must not cross-match to anthropic's catalog entry.
    rate = resolve_approx_rate(_CATALOG, "someothervendor", "claude-sonnet-4-6")
    assert rate is None


def test_resolve_approx_rate_openrouter_provider_not_scoped_by_vendor_prefix():
    # provider == "openrouter" means the model string may already be (close to) the
    # catalog id itself — scoping should be skipped, not filtered to a "openrouter" vendor segment.
    rate = resolve_approx_rate(_CATALOG, "openrouter", "anthropic/claude-3.5-sonnet")
    assert rate == ApproxRate(3.0, 15.0, "anthropic/claude-3.5-sonnet")


def test_resolve_approx_rate_no_plausible_match_is_none():
    rate = resolve_approx_rate(_CATALOG, "mystery", "totally-unrelated-model-xyz")
    assert rate is None


def test_resolve_approx_rate_rejects_short_coincidental_overlap():
    # A short substring match (e.g. shared "gpt") shouldn't count as identification.
    rate = resolve_approx_rate(_CATALOG, "openai", "gp")
    assert rate is None


def test_resolve_approx_rate_malformed_pricing_entry_is_none():
    bad_catalog = [{"id": "vendor/model-x", "pricing": {"prompt": "not-a-number", "completion": "0.00001"}}]
    assert resolve_approx_rate(bad_catalog, "vendor", "model-x") is None


# ---------------------------------------------------------------------------
# approx_step_cost
# ---------------------------------------------------------------------------

def test_approx_step_cost_no_rate_is_none():
    assert approx_step_cost(None, 1000, 500) is None


def test_approx_step_cost_no_token_data_is_none():
    rate = ApproxRate(3.0, 15.0, "anthropic/claude-3.5-sonnet")
    assert approx_step_cost(rate, None, None) is None


def test_approx_step_cost_computes_from_per_mtok_rate():
    rate = ApproxRate(3.0, 15.0, "anthropic/claude-3.5-sonnet")
    assert approx_step_cost(rate, 1000, 500) == pytest.approx(1000 * 3.0 / 1_000_000 + 500 * 15.0 / 1_000_000)


# ---------------------------------------------------------------------------
# refresh_catalog — never raises, keeps previous snapshot on failure
# ---------------------------------------------------------------------------

async def test_refresh_catalog_success_populates_cache():
    import src.live_pricing as live_pricing_module

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"data": _CATALOG})

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.live_pricing.httpx.AsyncClient", return_value=mock_client):
        await refresh_catalog()

    assert live_pricing_module.get_catalog() == _CATALOG
    assert live_pricing_module.last_refreshed() is not None


async def test_refresh_catalog_network_failure_keeps_previous_snapshot():
    import src.live_pricing as live_pricing_module

    live_pricing_module._catalog = _CATALOG
    live_pricing_module._fetched_at = None

    with patch("src.live_pricing.httpx.AsyncClient", side_effect=httpx.ConnectError("boom")):
        await refresh_catalog()  # must not raise

    assert live_pricing_module.get_catalog() == _CATALOG  # unchanged


async def test_refresh_catalog_bad_response_shape_keeps_previous_snapshot():
    import src.live_pricing as live_pricing_module

    live_pricing_module._catalog = _CATALOG

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"data": "not-a-list"})

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("src.live_pricing.httpx.AsyncClient", return_value=mock_client):
        await refresh_catalog()  # must not raise

    assert live_pricing_module.get_catalog() == _CATALOG
