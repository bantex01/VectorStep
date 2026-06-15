from src.normaliser.alertmanager import AlertmanagerParser, _fingerprint
from src.normaliser.generic import GenericParser


# ----------------------------------------------------------------------
# _fingerprint helper
# ----------------------------------------------------------------------

def test_fingerprint_appends_status_to_provided_base():
    assert _fingerprint("abc123", {"service": "x"}, "firing") == "abc123:firing"


def test_fingerprint_no_status_returns_base_unchanged():
    assert _fingerprint("abc123", {}, None) == "abc123"


def test_fingerprint_falls_back_to_label_hash_when_base_missing():
    fp1 = _fingerprint(None, {"service": "x"}, "firing")
    fp2 = _fingerprint(None, {"service": "x"}, "firing")
    fp3 = _fingerprint(None, {"service": "y"}, "firing")

    assert fp1 == fp2  # deterministic for identical labels
    assert fp1 != fp3  # differs when labels differ
    assert fp1.endswith(":firing")


# ----------------------------------------------------------------------
# AlertmanagerParser
# ----------------------------------------------------------------------

async def test_most_severe_uses_alert_fingerprint_with_status_suffix():
    payload = {
        "status": "firing",
        "commonLabels": {"alertname": "HighErrorRate"},
        "commonAnnotations": {},
        "groupLabels": {},
        "alerts": [
            {
                "fingerprint": "fp-1234",
                "status": "firing",
                "labels": {"severity": "critical", "service": "payments-api"},
                "annotations": {"summary": "Error rate high"},
            }
        ],
    }

    ctx = await AlertmanagerParser(strategy="most_severe").parse(payload)

    assert ctx.fingerprint == "fp-1234:firing"


async def test_most_severe_falls_back_to_label_hash_when_no_alert_fingerprint():
    payload = {
        "status": "firing",
        "commonLabels": {},
        "commonAnnotations": {},
        "groupLabels": {},
        "alerts": [
            {
                "status": "firing",
                "labels": {"severity": "warning", "service": "payments-api"},
                "annotations": {"summary": "Something"},
            }
        ],
    }

    ctx = await AlertmanagerParser(strategy="most_severe").parse(payload)

    assert ctx.fingerprint is not None
    assert ctx.fingerprint.endswith(":firing")


async def test_common_labels_uses_group_key():
    payload = {
        "status": "firing",
        "groupKey": '{}:{alertname="HighErrorRate"}',
        "commonLabels": {"severity": "warning", "service": "x"},
        "commonAnnotations": {},
        "groupLabels": {},
        "alerts": [{"labels": {"severity": "warning"}}],
    }

    ctx = await AlertmanagerParser(strategy="common_labels").parse(payload)

    assert ctx.fingerprint == '{}:{alertname="HighErrorRate"}:firing'


async def test_resolved_status_produces_different_fingerprint_than_firing():
    base = {
        "groupKey": "gk",
        "commonLabels": {},
        "commonAnnotations": {},
        "groupLabels": {},
        "alerts": [],
    }
    parser = AlertmanagerParser(strategy="common_labels")

    firing_ctx = await parser.parse({**base, "status": "firing"})
    resolved_ctx = await parser.parse({**base, "status": "resolved"})

    assert firing_ctx.fingerprint != resolved_ctx.fingerprint
    assert firing_ctx.fingerprint == "gk:firing"
    assert resolved_ctx.fingerprint == "gk:resolved"


# ----------------------------------------------------------------------
# GenericParser
# ----------------------------------------------------------------------

async def test_generic_idempotency_key_becomes_fingerprint():
    ctx = await GenericParser().parse({"pipeline": "new-order", "idempotency_key": "order-12345"})

    assert ctx.fingerprint == "order-12345"


async def test_generic_omitted_idempotency_key_disables_dedup():
    ctx = await GenericParser().parse({"pipeline": "new-order"})

    assert ctx.fingerprint is None
