from pydantic import BaseModel, Field, field_validator


class PriceMatch(BaseModel):
    provider: str | None = None   # None = matches any provider (used only on provider-only fallback entries)
    model: str | None = None      # prefix match against the step's persisted model string; None = provider-only fallback


class PriceEntry(BaseModel):
    match: PriceMatch
    input_per_mtok: float   # currency units per 1,000,000 input tokens
    output_per_mtok: float  # currency units per 1,000,000 output tokens

    @field_validator("input_per_mtok", "output_per_mtok")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("per-mtok rates must be >= 0 (use 0 for genuinely free, e.g. local Ollama)")
        return v


class LivePricingConfig(BaseModel):
    """Approximate, best-effort pricing from OpenRouter's public catalog
    (SPEC-live-pricing.md) — never the source of pipeline_steps.cost, only a
    disclosed display-time (and opt-in budget-accumulator) approximation for
    steps with no manual pricing.models entry."""
    enabled: bool = False
    refresh_interval_seconds: int = 3600


class PricingConfig(BaseModel):
    currency: str = "USD"          # display label only — no FX conversion, see SPEC-cost-accounting.md
    models: list[PriceEntry] = Field(default_factory=list)
    team_budgets: dict[str, float] = Field(default_factory=dict)  # currency units per calendar month, UTC — advisory only
    live_pricing: LivePricingConfig = Field(default_factory=LivePricingConfig)
