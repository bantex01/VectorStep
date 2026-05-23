from datetime import datetime
from pydantic import BaseModel, Field


class NormalisedContext(BaseModel):
    source: str
    pipeline: str
    severity: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    summary: str | None = None
    raw: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=datetime.utcnow)
