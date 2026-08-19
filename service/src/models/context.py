from datetime import datetime
from pydantic import BaseModel, Field

from ..utils import utc_now


class NormalisedContext(BaseModel):
    source: str
    pipeline: str | None = None  # explicit routing — unset falls through to trigger.match
    severity: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)
    summary: str | None = None
    fingerprint: str | None = None  # dedup key — see README §3a
    team: str | None = None  # owning team, resolved from the auth token — see README §3b
    raw: dict = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)
    received_at: datetime = Field(default_factory=utc_now)
