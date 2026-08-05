"""Cost accounting (SPEC-cost-accounting.md): converts persisted token counts to
money at the rate in force when the tokens were bought.

Split in two, same pure/IO shape as readiness.py: resolve_rate/step_cost are
pure and take the pricing table as an explicit argument (fully unit-testable
with no config file on disk); configure()/get_table() are the only I/O-adjacent
part — a module-level holder set once at startup and again on every
/reload or SIGHUP, so already-persisted costs never change but future steps
price against the fresh table.

NULL vs zero (do not conflate): resolve_rate returning None means "no pricing
entry matches — cost is unknown," which step_cost propagates as cost=NULL.
A rate of 0.0 (an operator explicitly pricing a provider, e.g. local Ollama,
at zero) is a real answer and must produce cost=0.0, not NULL.
"""
from dataclasses import dataclass

from .models.pricing import PricingConfig


@dataclass(frozen=True)
class Rate:
    input_per_mtok: float
    output_per_mtok: float


_table: PricingConfig | None = None


def configure(raw: dict | None) -> None:
    """Parse and install the pricing table from config.yaml's `pricing:` block.

    Raises pydantic.ValidationError on a malformed table — fails config load/
    reload loudly rather than silently running unpriced, matching how the rest
    of config.yaml's Pydantic-modeled sections behave.
    """
    global _table
    _table = PricingConfig.model_validate(raw) if raw else None


def get_table() -> PricingConfig | None:
    return _table


def resolve_rate(table: PricingConfig | None, provider: str | None, model: str | None) -> Rate | None:
    """Longest-prefix match on `model`, scoped by `provider` first.

    Candidate entries are those whose match.provider is unset (applies to any
    provider) or equals `provider`. Among those, the entry with the longest
    match.model that is a prefix of `model` wins. If none has a model prefix
    match, a provider-only entry (match.model unset) is the fallback — one
    scoped to this exact provider is preferred over a fully generic one.
    No match at all → None (unpriced), never a zero rate.
    """
    if table is None:
        return None

    applicable = [
        entry for entry in table.models
        if entry.match.provider is None or entry.match.provider == provider
    ]

    best = None
    if model:
        for entry in applicable:
            if entry.match.model is None:
                continue
            if model.startswith(entry.match.model):
                if best is None or len(entry.match.model) > len(best.match.model):
                    best = entry
    if best is not None:
        return Rate(best.input_per_mtok, best.output_per_mtok)

    fallback = next(
        (e for e in applicable if e.match.model is None and e.match.provider == provider), None
    )
    if fallback is None:
        fallback = next(
            (e for e in applicable if e.match.model is None and e.match.provider is None), None
        )
    return Rate(fallback.input_per_mtok, fallback.output_per_mtok) if fallback else None


def step_cost(rate: Rate | None, input_tokens: int | None, output_tokens: int | None) -> float | None:
    """cost in currency units, or None when unpriced (no rate) or when there's
    no token data at all (both counts None — e.g. the openclaw executor)."""
    if rate is None:
        return None
    if input_tokens is None and output_tokens is None:
        return None
    return (
        (input_tokens or 0) * rate.input_per_mtok / 1_000_000
        + (output_tokens or 0) * rate.output_per_mtok / 1_000_000
    )
