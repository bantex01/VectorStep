import hashlib
import json
from datetime import datetime
from typing import Literal
from .base import BaseParser
from ..models.context import NormalisedContext
from ..utils import utc_now

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}

AlertmanagerStrategy = Literal["most_severe", "common_labels"]


def _fingerprint(base: str | None, labels: dict, status: str | None) -> str:
    """Build a dedup fingerprint from an Alertmanager-supplied base identifier.

    Falls back to a hash of the labels if Alertmanager didn't supply one (older
    versions omit per-alert `fingerprint`). The alert `status` (firing/resolved) is
    appended so a resolved notification isn't suppressed as a duplicate of the
    firing run for the same alert.
    """
    key = base or hashlib.sha256(json.dumps(labels, sort_keys=True).encode()).hexdigest()[:16]
    return f"{key}:{status}" if status else key


class AlertmanagerParser(BaseParser):
    def __init__(self, strategy: AlertmanagerStrategy = "most_severe"):
        self.strategy = strategy

    async def parse(self, payload: dict) -> NormalisedContext:
        alerts = payload.get("alerts", [])

        if self.strategy == "common_labels":
            return self._parse_common_labels(payload, alerts)
        else:
            return self._parse_most_severe(payload, alerts)

    def _parse_most_severe(self, payload: dict, alerts: list) -> NormalisedContext:
        alert = min(
            alerts,
            key=lambda a: SEVERITY_ORDER.get(a.get("labels", {}).get("severity", ""), 99),
            default={},
        )

        labels = alert.get("labels", {})
        annotations = alert.get("annotations", {})
        severity = labels.get("severity")
        pipeline = labels.get("pipeline")  # unset falls through to trigger.match
        summary = annotations.get("summary") or annotations.get("description")

        return NormalisedContext(
            source="alertmanager",
            pipeline=pipeline,
            severity=severity,
            labels=labels,
            summary=summary,
            fingerprint=_fingerprint(alert.get("fingerprint"), labels, payload.get("status")),
            raw=payload,
            metadata=self._common_metadata(payload, alerts, alert),
            received_at=utc_now(),
        )

    def _parse_common_labels(self, payload: dict, alerts: list) -> NormalisedContext:
        labels = payload.get("commonLabels", {})
        annotations = payload.get("commonAnnotations", {})
        severity = labels.get("severity")
        pipeline = labels.get("pipeline")  # unset falls through to trigger.match
        summary = annotations.get("summary") or annotations.get("description")

        return NormalisedContext(
            source="alertmanager",
            pipeline=pipeline,
            severity=severity,
            labels=labels,
            summary=summary,
            fingerprint=_fingerprint(payload.get("groupKey"), labels, payload.get("status")),
            raw=payload,
            metadata=self._common_metadata(payload, alerts, alerts[0] if alerts else {}),
            received_at=utc_now(),
        )

    def _common_metadata(self, payload: dict, alerts: list, alert: dict) -> dict:
        return {
            "status": payload.get("status"),
            "strategy": self.strategy,
            "group_labels": payload.get("groupLabels", {}),
            "common_labels": payload.get("commonLabels", {}),
            "common_annotations": payload.get("commonAnnotations", {}),
            "external_url": payload.get("externalURL"),
            "alert_count": len(alerts),
            "starts_at": alert.get("startsAt"),
            "ends_at": alert.get("endsAt"),
        }
