from pydantic import BaseModel, ConfigDict


class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="allow")

    confidence: float
    proceed: bool = True          # false = pipeline closes cleanly at this step; no further steps run
    proceed_reason: str | None = None  # why proceed was set to true or false
    summary: str
    next_step_context: str
    reasoning: dict | None = None
    model: str | None = None      # populated from executor metadata, not agent self-report
    provider: str | None = None   # populated from executor metadata (gateway executor only)
    agent_version: str | None = None  # populated from executor metadata (gateway executor only)
    raw_response: dict = {}
