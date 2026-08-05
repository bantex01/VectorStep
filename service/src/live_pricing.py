"""Best-effort approximate cost from OpenRouter's public model catalog
(`GET https://openrouter.ai/api/v1/models`, no auth required).

This is deliberately kept separate from pricing.py's authoritative, operator-
maintained rate table: nothing here ever writes to pipeline_steps.cost. It only
fills in a display-time (and, if a pipeline/step opts in, budget-accumulator-
time) *approximation* for steps that have no manual pricing.models entry —
labeled as such everywhere it's shown, because a fuzzy-matched, third-party
catalog price is not the same kind of fact as an operator's own configured
rate (SPEC-live-pricing.md).

Same pure/IO split as pricing.py: refresh_catalog() is the only I/O, a
scheduled background job (see main.py) that replaces the cached catalog and
never raises — a transient OpenRouter outage must not crash the refresh job or
take down approximate pricing entirely, it just leaves the previous snapshot
in place. resolve_approx_rate()/approx_step_cost() are pure and take the
catalog as an explicit argument.
"""
import logging
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

from .utils import utc_now

logger = logging.getLogger(__name__)

_OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Below this normalized-character overlap, a "match" is more likely a coincidental
# short substring (e.g. "gpt" inside "gpt-5" AND "gpt-oss") than a real identification —
# treated the same as no match at all (see resolve_approx_rate).
_MIN_MATCH_SCORE = 4

_catalog: list[dict] | None = None
_fetched_at: datetime | None = None


async def refresh_catalog() -> None:
    """Fetch and replace the cached OpenRouter model catalog. Never raises."""
    global _catalog, _fetched_at
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(_OPENROUTER_MODELS_URL)
            resp.raise_for_status()
            body = resp.json()
        catalog = body.get("data")
        if not isinstance(catalog, list):
            raise ValueError(f"unexpected response shape: {type(catalog)}")
        _catalog = catalog
        _fetched_at = utc_now()
        logger.info("OpenRouter catalog refreshed: %d model(s)", len(catalog))
    except Exception as exc:
        logger.warning(
            "OpenRouter catalog refresh failed (keeping previous snapshot, last "
            "refreshed %s): %s", _fetched_at, exc,
        )


def get_catalog() -> list[dict] | None:
    return _catalog


def last_refreshed() -> datetime | None:
    return _fetched_at


@dataclass(frozen=True)
class ApproxRate:
    input_per_mtok: float
    output_per_mtok: float
    openrouter_id: str  # which catalog entry this was matched to, for display/debugging


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _family_key(s: str) -> str:
    """Digit-stripped normalized form — version numbers (4-6, 3.5, ...) land in
    different positions across providers/gateways ("claude-sonnet-4-6" vs
    "claude-3.5-sonnet"), so comparing on the alphabetic "family" name catches
    same-model-different-version cases a plain substring check would miss.
    This is exactly the kind of looseness that makes the result approximate,
    not exact — a version mismatch can mean a real price difference."""
    return re.sub(r"[0-9]", "", _normalize(s))


def resolve_approx_rate(
    catalog: list[dict] | None, provider: str | None, model: str | None,
) -> ApproxRate | None:
    """Best-effort fuzzy match of (provider, model) against the OpenRouter
    catalog's "<vendor>/<slug>" ids. Scopes to entries whose vendor segment
    plausibly matches `provider` (skipped when provider is "openrouter" itself,
    or unset, since the model string may already closely resemble the catalog
    id directly); among those, picks the entry whose family key has the longest
    overlap with `model`'s. Returns None — never a weak guess — when nothing
    clears _MIN_MATCH_SCORE, since this feeds a number that's shown as a real
    (if approximate) price, not a shot in the dark."""
    if not catalog or not model:
        return None
    norm_model = _family_key(model)
    if not norm_model:
        return None

    candidates = catalog
    if provider and provider != "openrouter":
        # Hard filter, not a soft preference: if the provider doesn't correspond to
        # any vendor segment in the catalog, there's no plausible match — falling
        # back to the full (wrong-vendor) catalog would risk pricing e.g. an
        # Anthropic call off an OpenAI model that merely shares a short name.
        norm_provider = _normalize(provider)
        candidates = [c for c in catalog if norm_provider in _normalize(c.get("id", "").split("/")[0])]

    best_entry: dict | None = None
    best_score = 0
    for entry in candidates:
        entry_id = entry.get("id", "")
        slug = entry_id.split("/", 1)[-1]
        norm_slug = _family_key(slug)
        if not norm_slug:
            continue
        if norm_model in norm_slug or norm_slug in norm_model:
            score = min(len(norm_model), len(norm_slug))
            if score > best_score:
                best_entry, best_score = entry, score

    if best_entry is None or best_score < _MIN_MATCH_SCORE:
        return None

    pricing = best_entry.get("pricing") or {}
    try:
        input_per_mtok = float(pricing["prompt"]) * 1_000_000
        output_per_mtok = float(pricing["completion"]) * 1_000_000
    except (KeyError, TypeError, ValueError):
        return None
    return ApproxRate(input_per_mtok, output_per_mtok, best_entry["id"])


def approx_step_cost(
    rate: ApproxRate | None, input_tokens: int | None, output_tokens: int | None,
) -> float | None:
    if rate is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    return (
        (input_tokens or 0) * rate.input_per_mtok / 1_000_000
        + (output_tokens or 0) * rate.output_per_mtok / 1_000_000
    )
