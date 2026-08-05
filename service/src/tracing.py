"""OpenTelemetry tracing setup and helpers for per-run pipeline traces.

Disabled by default. The OTel SDK's default TracerProvider is a no-op, so every
``tracer.start_as_current_span(...)`` call throughout the runner and executors is
free until ``setup_tracing()`` installs a real exporter via config.
"""
from __future__ import annotations

import logging

from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

logger = logging.getLogger(__name__)

tracer = trace.get_tracer("vectorstep")

_DEFAULT_SERVICE_NAME = "vectorstep-service"
_DEFAULT_OTLP_ENDPOINT = "http://localhost:4318/v1/traces"

_PRIMITIVE_TYPES = (str, int, float, bool)


def setup_tracing(config: dict) -> None:
    """Install a real TracerProvider if observability.otel.enabled is true.

    Leaves the SDK's default no-op provider in place otherwise.
    """
    otel_cfg = (config.get("observability") or {}).get("otel") or {}
    if not otel_cfg.get("enabled"):
        return

    service_name = otel_cfg.get("service_name", _DEFAULT_SERVICE_NAME)
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)

    exporter_kind = otel_cfg.get("exporter", "otlp")
    if exporter_kind == "console":
        exporter = ConsoleSpanExporter()
    else:
        endpoint = otel_cfg.get("endpoint", _DEFAULT_OTLP_ENDPOINT)
        exporter = OTLPSpanExporter(endpoint=endpoint)

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("OpenTelemetry tracing enabled: service=%s exporter=%s", service_name, exporter_kind)


def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider, if a real SDK provider was installed."""
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()


def _clean(attrs: dict) -> dict:
    """Drop None values and coerce non-primitive values to strings for OTel attributes."""
    cleaned: dict = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if isinstance(value, _PRIMITIVE_TYPES):
            cleaned[key] = value
        elif isinstance(value, (list, tuple)) and all(isinstance(v, _PRIMITIVE_TYPES) for v in value):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def record_event(name: str, attributes: dict | None = None) -> None:
    """Add an event to the current span. A no-op if there is no active span."""
    trace.get_current_span().add_event(name, attributes=_clean(attributes or {}))


def start_root_span(name: str, attributes: dict | None = None):
    """Start a span detached from any current context — always a new trace root.

    Pipeline runs are long-lived background tasks and shouldn't become children
    of the short-lived webhook request span.
    """
    return tracer.start_as_current_span(
        name, context=otel_context.Context(), attributes=_clean(attributes or {})
    )


def gen_ai_attrs_from_meta(meta: dict) -> dict:
    """Extract gen_ai.* span attributes from a gateway/OpenClaw response 'meta' dict."""
    attrs: dict = {}
    model = (meta.get("agentMeta") or {}).get("model")
    if model:
        attrs["gen_ai.response.model"] = model
    duration_ms = meta.get("durationMs")
    if duration_ms is not None:
        attrs["vectorstep.gateway.duration_ms"] = duration_ms
    return attrs


def inject_traceparent(params: dict) -> dict:
    """Inject the current trace context (traceparent/tracestate) into an outbound params dict.

    Forward-compatible with a future VectorStep Gateway that adds its own OTel spans —
    ignored by the current gateway, which passes unknown params through untouched.
    """
    propagate.inject(params)
    return params
