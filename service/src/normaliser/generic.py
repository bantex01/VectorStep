from datetime import datetime

from pydantic import BaseModel, ValidationError

from ..models.context import NormalisedContext
from ..utils import utc_now
from .base import BaseParser


class GenericPayload(BaseModel):
    pipeline: str
    event: str | None = None
    source: str = "generic"
    summary: str | None = None
    idempotency_key: str | None = None  # dedup key — see README §3a
    data: dict = {}


class GenericParser(BaseParser):
    async def parse(self, payload: dict) -> NormalisedContext:
        try:
            parsed = GenericPayload(**payload)
        except ValidationError as exc:
            raise ValueError(f"Invalid generic payload: {exc}") from exc

        labels: dict[str, str] = {}
        if parsed.event:
            labels["event"] = parsed.event

        return NormalisedContext(
            source=parsed.source,
            pipeline=parsed.pipeline,
            labels=labels,
            summary=parsed.summary,
            fingerprint=parsed.idempotency_key,
            raw=payload,
            metadata=parsed.data,
            received_at=utc_now(),
        )
