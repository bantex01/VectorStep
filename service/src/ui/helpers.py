from fastapi.templating import Jinja2Templates
from .. import live_pricing
from .. import pricing
from ..db.models import PipelineStep
from ..db.models import RunFeedback
from ..executors.human import pending_count as _pending_approval_count
from ..gateway import gateway_call_safe
from ..models.pipeline import FanOutGroupConfig
from ..models.pipeline import ParallelGroupConfig
from ..models.pipeline import PipelineConfig
from ..utils import utc_now
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from sqlalchemy import select
import asyncio
import httpx
import json
import os
import re
import yaml


templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "..", "..", "templates")
)


# --- lines 49-60 ---
def _status_classes(status: str) -> str:
    return {
        "completed":   "bg-green-950 text-green-400 ring-green-800",
        "running":     "bg-blue-950 text-blue-400 ring-blue-800",
        "escalated":   "bg-amber-950 text-amber-400 ring-amber-800",
        "aborted":     "bg-orange-950 text-orange-400 ring-orange-800",
        "failed":      "bg-red-950 text-red-400 ring-red-800",
        "stopped":     "bg-purple-950 text-purple-400 ring-purple-800",
        "interrupted": "bg-zinc-700 text-zinc-300 ring-zinc-500",
    }.get(status or "", "bg-zinc-800 text-zinc-400 ring-zinc-600")



# --- lines 61-71 ---
def _readiness_verdict_classes(verdict: str) -> str:
    """Badge classes for a pipeline/step readiness verdict (SPEC-readiness-criteria.md §9)."""
    return {
        "ready":           "bg-green-950 text-green-400 ring-green-800",
        "not_ready":       "bg-red-950 text-red-400 ring-red-800",
        "building":        "bg-amber-950 text-amber-400 ring-amber-800",
        "no_data":         "bg-zinc-800 text-zinc-400 ring-zinc-700",
        "not_configured":  "bg-zinc-800 text-zinc-600 ring-zinc-700",
    }.get(verdict, "bg-zinc-800 text-zinc-400 ring-zinc-700")



# --- lines 72-82 ---
def _readiness_tier_classes(verdict: str) -> str:
    """Badge classes for one tier chip within a readiness step row."""
    return {
        "pass":               "bg-green-950 text-green-400 ring-green-800",
        "fail":                "bg-red-950 text-red-400 ring-red-800",
        "insufficient_data":  "bg-amber-950 text-amber-400 ring-amber-800",
        "not_current_config": "bg-zinc-800 text-zinc-600 ring-zinc-700",
        "not_configured":     "bg-zinc-800 text-zinc-600 ring-zinc-700",
    }.get(verdict, "bg-zinc-800 text-zinc-600 ring-zinc-700")



# --- lines 83-102 ---
def _readiness_observed_status(observed_combos: list) -> dict:
    """Roll up a step's observed_combos (service-default calibration snapshot,
    computed regardless of whether any tier is configured) into a single status —
    the §11 backward-compat guarantee: a pipeline with no readiness: block must
    still show observed calibration evidence, not a vanished signal. Mirrors the
    old get_pipeline_promotion_readiness ready/flagged/building/no_data rollup."""
    if not observed_combos:
        return {"status": "no_data", "label": "no observed data yet"}
    flagged = [c for c in observed_combos if c["observed"]["recommendation"]]
    if flagged:
        return {"status": "not_ready", "label": f"flagged — {flagged[0]['observed']['recommendation']}"}
    validated = [c for c in observed_combos if c["observed"]["validated"]]
    if validated:
        best = max(validated, key=lambda c: c["observed"]["total_n"])
        return {"status": "ready", "label": f"ready — n={best['observed']['total_n']}"}
    if any(c["observed"]["total_n"] for c in observed_combos):
        return {"status": "building", "label": "building"}
    return {"status": "no_data", "label": "no observed data yet"}



# --- lines 103-142 ---
def _readiness_tier_label(tier_name: str, tier: dict) -> str:
    """Short chip text for one tier — the calibration label deliberately always
    says 'N/n_min in top band', never a bare 'N/n_min' (SPEC-readiness-criteria.md
    §12.5 — the single most misread knob in the system)."""
    verdict = tier.get("verdict")
    if verdict == "not_configured":
        return "—"
    icon = "✓" if verdict == "pass" else "✗" if verdict == "fail" else "…"

    if tier_name == "operational":
        return f"runs {tier['runs_acceptable']}/{tier['min_runs']} {icon}"
    if tier_name == "confidence":
        pct = f"{tier['mean_confidence']:.0%}" if tier.get("mean_confidence") is not None else "—"
        return f"conf {pct} {icon}"
    if tier_name == "accuracy":
        pct = f"{tier['accuracy']:.0%}" if tier.get("accuracy") is not None else "—"
        return f"acc {pct} {icon}"
    if tier_name == "calibration":
        fullest = 0
        for c in tier.get("combos", []):
            for view in (c.get("own"), c.get("production")):
                if view:
                    fullest = max(fullest, max((b["n"] for b in view["bins"]), default=0))
        return f"calib {fullest}/{tier['n_min']} in top band {icon}"
    return icon


_CHART_PALETTE = ["#6366f1", "#22d3ee", "#fbbf24", "#34d399", "#fb7185", "#a78bfa", "#71717a"]

_STATUS_HEX = {
    "completed":   "#4ade80",
    "running":     "#60a5fa",
    "escalated":   "#fbbf24",
    "aborted":     "#fb923c",
    "failed":      "#f87171",
    "stopped":     "#c084fc",
    "interrupted": "#a1a1aa",
}



# --- lines 143-156 ---
def _format_seconds(secs: float | None) -> str:
    """Like _format_duration, but takes a raw seconds value (e.g. an average) rather than two datetimes."""
    if secs is None:
        return "—"
    secs = int(secs)
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"



# --- lines 157-166 ---
def _confidence_bar_color(c: float | None) -> str:
    if c is None:
        return "bg-gray-300"
    if c >= 0.75:
        return "bg-green-500"
    if c >= 0.5:
        return "bg-amber-400"
    return "bg-red-400"



# --- lines 167-181 ---
def _format_duration(start: datetime, end: datetime | None) -> str:
    if end is None:
        return "in progress"
    secs = int((end.replace(tzinfo=None) - start.replace(tzinfo=None)).total_seconds())
    if secs < 0:
        return "—"
    if secs < 60:
        return f"{secs}s"
    m, s = divmod(secs, 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m"



# --- lines 182-209 ---
def _format_ago(dt: datetime) -> str:
    secs = int((utc_now() - dt.replace(tzinfo=None)).total_seconds())
    if secs < 5:
        return "just now"
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"


_TELEGRAM_HTML_TAG_RESTORE = [
    (re.compile(r"&lt;b&gt;(.*?)&lt;/b&gt;", re.IGNORECASE | re.DOTALL), r"<b>\1</b>"),
    (re.compile(r"&lt;strong&gt;(.*?)&lt;/strong&gt;", re.IGNORECASE | re.DOTALL), r"<strong>\1</strong>"),
    (re.compile(r"&lt;i&gt;(.*?)&lt;/i&gt;", re.IGNORECASE | re.DOTALL), r"<i>\1</i>"),
    (re.compile(r"&lt;em&gt;(.*?)&lt;/em&gt;", re.IGNORECASE | re.DOTALL), r"<em>\1</em>"),
    (re.compile(r"&lt;code&gt;(.*?)&lt;/code&gt;", re.IGNORECASE | re.DOTALL), r"<code>\1</code>"),
]
# Only http(s) links are restored — never javascript:/data: schemes — and only from the
# escaped form, so a raw unescaped "<a href=...>" smuggled in via {{summary}} or an
# agent's own output never matches (it's not literal &lt;a href=&#34;...&gt; text).
_TELEGRAM_HTML_LINK_RE = re.compile(
    r'&lt;a href=&#34;(https?://[^&"]*)&#34;&gt;(.*?)&lt;/a&gt;', re.IGNORECASE | re.DOTALL
)



# --- lines 210-234 ---
def _telegram_html_to_safe_html(text: str):
    """Render the small Telegram HTML parse-mode subset used in `human` step
    prompt_templates (<b>, <i>, <code>, <a href>) as real HTML on the /ui/approvals
    pages, without trusting arbitrary content.

    The message text is a fully-rendered Jinja2 prompt_template — it can embed
    {{summary}}, {{steps.x.field}}, etc., which ultimately trace back to webhook
    payloads or LLM output, neither of which is trusted input. So this escapes the
    *entire* string first (exactly like Jinja2's default autoescape would), then
    re-opens only the small whitelist of tags above from their escaped form. Anything
    else — including an attempt to smuggle in a real <script> tag — stays inert
    escaped text rather than becoming live markup.
    """
    from markupsafe import Markup, escape

    escaped = str(escape(text))
    for pattern, replacement in _TELEGRAM_HTML_TAG_RESTORE:
        escaped = pattern.sub(replacement, escaped)
    escaped = _TELEGRAM_HTML_LINK_RE.sub(
        lambda m: f'<a href="{m.group(1)}" target="_blank" rel="noopener noreferrer">{m.group(2)}</a>',
        escaped,
    )
    return Markup(escaped)



# --- lines 235-240 ---
def _source_label(source: str) -> str:
    return {"alertmanager": "Alertmanager", "scheduler": "Scheduler", "generic": "Generic"}.get(
        source, source
    )



# --- lines 241-260 ---
def _provider_from_model(model: str | None) -> str:
    """Best-effort provider guess from a model string's prefix, for PipelineStep
    rows recorded before the `provider` column existed (it's populated directly
    from the Gateway's agentMeta.provider for every step since).

    This is NOT reliable in general: `PipelineStep.model` stores whatever the
    underlying LLM API reports as its own "model" field, which for most
    providers is NOT the same as the Gateway's routing prefix — e.g. OpenRouter
    reports "deepseek/deepseek-v4-pro-...", not "openrouter/deepseek/...", so
    this would incorrectly bucket it under "deepseek". Only Azure's provider
    happens to prepend its own prefix, and bare Anthropic model strings have no
    prefix, so those two are the only cases this can guess correctly.
    """
    if not model:
        return "unknown"
    if "/" in model:
        return model.split("/", 1)[0]
    return "anthropic"



# --- lines 261-282 ---
def _qualified_model(provider: str | None, model: str | None) -> str:
    """Prefix a bare model string with its provider, e.g. "claude-sonnet-5" ->
    "anthropic/claude-sonnet-5", so two agents/steps using the same model name
    through different providers aren't conflated.

    Only uses a real `provider` column value — deliberately does NOT fall back to
    guessing from the model string (see _provider_from_model). `provider` is only
    ever populated for `executor: gateway` steps; OpenClaw-executed steps (and any
    other executor) never set it, so guessing would confidently mislabel every one
    of those as a specific provider with zero actual evidence. No signal beats a
    wrong signal here. Avoids double-prefixing when the model string already
    carries the same prefix (e.g. Azure deployments, which come back as
    "azure/<deployment>" already)."""
    if not model:
        return "—"
    if not provider:
        return model
    if model.startswith(provider + "/"):
        return model
    return f"{provider}/{model}"



# --- lines 510-513 ---
def _ts_resolution(time_range: str) -> str:
    return "hour" if time_range == "24h" else "week" if time_range == "all" else "day"



# --- lines 514-523 ---
def _ts_bucket(dt: datetime, resolution: str) -> str:
    d = dt.replace(tzinfo=None)
    if resolution == "hour":
        return d.strftime("%d %H:00")
    if resolution == "week":
        monday = d - timedelta(days=d.weekday())
        return monday.strftime("%b %d")
    return d.strftime("%b %d")



# --- lines 524-547 ---
def _ts_all_buckets(start: datetime, end: datetime, resolution: str) -> list[str]:
    """All bucket keys from start to end, so empty buckets render as zero."""
    buckets: list[str] = []
    seen: set[str] = set()
    cur = start.replace(tzinfo=None, second=0, microsecond=0)
    if resolution == "hour":
        cur = cur.replace(minute=0)
        step = timedelta(hours=1)
    elif resolution == "week":
        cur = (cur - timedelta(days=cur.weekday())).replace(hour=0, minute=0)
        step = timedelta(weeks=1)
    else:
        cur = cur.replace(hour=0, minute=0)
        step = timedelta(days=1)
    end_clean = end.replace(tzinfo=None)
    while cur <= end_clean + step:
        key = _ts_bucket(cur, resolution)
        if key not in seen:
            seen.add(key)
            buckets.append(key)
        cur += step
    return buckets



# --- lines 548-588 ---
def _build_ts(
    rows: list,
    now: datetime,
    cutoff: datetime | None,
    time_range: str,
    dim_fn,
    val_fn=None,
    top_n: int = 7,
) -> dict:
    """Build a Chart.js multi-series line-chart dict from raw (timestamp, ...) rows.

    dim_fn(row) -> series name (string)
    val_fn(row) -> numeric value (default: 1 per row, i.e. counts)
    """
    if val_fn is None:
        val_fn = lambda _r: 1

    resolution = _ts_resolution(time_range)
    start = (cutoff or (min((r[0] for r in rows), default=now) if rows else now - timedelta(days=7)))
    bucket_labels = _ts_all_buckets(start, now, resolution)

    accumulator: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        key = _ts_bucket(row[0], resolution)
        dim = str(dim_fn(row)) if dim_fn(row) is not None else "—"
        accumulator[dim][key] += val_fn(row)

    # Keep only the top_n most active series
    dims = sorted(accumulator, key=lambda d: sum(accumulator[d].values()), reverse=True)[:top_n]

    datasets = [
        {
            "label": dim,
            "data": [accumulator[dim].get(b, 0) for b in bucket_labels],
            "color": _CHART_PALETTE[i % len(_CHART_PALETTE)],
        }
        for i, dim in enumerate(dims)
    ]
    return {"labels": bucket_labels, "datasets": datasets}



# --- lines 656-659 ---
class _LiteralBlockDumper(yaml.Dumper):
    pass



# --- lines 660-668 ---
def _literal_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralBlockDumper.add_representer(str, _literal_str)



# --- lines 669-675 ---
def _to_yaml(obj) -> str:
    return yaml.dump(
        obj, Dumper=_LiteralBlockDumper,
        default_flow_style=False, allow_unicode=True, sort_keys=False,
    )



# --- lines 676-679 ---
def _to_json(obj, indent=None) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)



# --- lines 697-703 ---
_OUTCOME_CLASSES = {
    "correct":   "bg-green-950 text-green-400 ring-green-800",
    "partial":   "bg-amber-950 text-amber-400 ring-amber-800",
    "incorrect": "bg-red-950 text-red-400 ring-red-800",
}



# --- lines 704-717 ---
def _approx_cost_for_step(step: PipelineStep) -> tuple[float | None, bool]:
    """Best-effort OpenRouter-catalog cost for a step with no real (manual)
    price — never computed for an already-priced step. Returns
    (approx_cost, is_native): is_native means step.provider genuinely IS
    "openrouter" (a live price for the exact API that was called, not a
    cross-provider guess against a different provider's model) — used to color
    the badge green rather than amber (SPEC-live-pricing.md)."""
    if step.cost is not None:
        return None, False
    rate = live_pricing.resolve_approx_rate(live_pricing.get_catalog(), step.provider, step.model)
    cost = live_pricing.approx_step_cost(rate, step.input_tokens, step.output_tokens)
    return cost, (cost is not None and step.provider == "openrouter")



# --- lines 718-730 ---
async def _feedback_by_run_id(session, run_ids: list[str]) -> dict[str, str]:
    """outcome for each run_id that has feedback — absent key means unmarked.

    Used by every template that renders a runs table (dashboard, /runs, pipeline
    detail) to show the feedback_badge macro without a per-row query.
    """
    if not run_ids:
        return {}
    rows = await session.execute(
        select(RunFeedback.run_id, RunFeedback.outcome).where(RunFeedback.run_id.in_(run_ids))
    )
    return dict(rows.all())


# --- lines 731-745 ---
def _format_cost(value: float | None, precision: int = 2) -> str | None:
    """`$12.34` for USD, `12.34 EUR` otherwise (SPEC-cost-accounting.md §4 — currency
    is a display label, not a conversion). None (unpriced, or no rate configured)
    passes through as None so a template can choose its own "not priced" wording
    rather than this filter silently rendering "$0.00" for an unknown cost.
    precision=4 is used only on the step-detail sub-cent case; everywhere else
    is 2dp."""
    if value is None:
        return None
    table = pricing.get_table()
    currency = table.currency if table else "USD"
    formatted = f"{value:,.{precision}f}"
    return f"${formatted}" if currency == "USD" else f"{formatted} {currency}"



# --- lines 746-768 ---
templates.env.filters["to_yaml"] = _to_yaml
templates.env.filters["tojson"] = _to_json
templates.env.filters["format_number"] = lambda n: f"{int(n):,}"
templates.env.filters["format_cost"] = _format_cost
templates.env.filters["telegram_html"] = _telegram_html_to_safe_html
templates.env.globals.update({
    "status_classes": _status_classes,
    "confidence_bar_color": _confidence_bar_color,
    "format_duration": _format_duration,
    "format_seconds": _format_seconds,
    "format_ago": _format_ago,
    "pending_approval_count": _pending_approval_count,
    "source_label": _source_label,
    "outcome_classes": lambda o: _OUTCOME_CLASSES.get(o or "", "bg-zinc-800 text-zinc-400 ring-zinc-600"),
    "readiness_verdict_classes": _readiness_verdict_classes,
    "readiness_tier_classes": _readiness_tier_classes,
    "readiness_tier_label": _readiness_tier_label,
    "readiness_observed_status": _readiness_observed_status,
})


# ── Routes ────────────────────────────────────────────────────────────────────


# --- lines 1312-1348 --- (moved from pipelines.py: also called directly by
# pipelines.py's pipeline-detail route, not just by _agents_in_pipeline below)
def _agent_usage_in_pipeline(p: PipelineConfig) -> list[dict]:
    """Every (step, role, executor:agent) usage in a pipeline, read straight from config.

    role is "primary" or "verifier" — the same agent can be a primary in one step
    and a verifier (critic or independent — both use the same VerifierConfig
    shape, see README §6) in another, so a given agent can carry both roles.
    Powers the pipeline detail page's Agents card. helpers._agents_in_pipeline collapses
    this down to a flat set of `executor:agent` keys for badges/filtering.
    """
    usage: list[dict] = []

    def _add(step_name: str, executor: str, agent: str | None, role: str) -> None:
        if agent:
            usage.append({
                "step": step_name, "executor": executor, "agent": agent,
                "role": role, "key": f"{executor}:{agent}",
            })

    def _add_verifier(step_name: str, verifier) -> None:
        if verifier is not None:
            _add(step_name, verifier.executor, verifier.executor_config.get("agent"), "verifier")

    for step in p.steps:
        if isinstance(step, ParallelGroupConfig):
            for s in step.parallel.steps:
                branch_name = f"{step.parallel.name}/{s.name}"
                _add(branch_name, s.executor, s.executor_config.get("agent"), "primary")
                _add_verifier(branch_name, s.verifier)
        elif isinstance(step, FanOutGroupConfig):
            _add(step.fan_out.name, step.fan_out.executor, step.fan_out.executor_config.get("agent"), "primary")
            _add_verifier(step.fan_out.name, step.fan_out.verifier)
        else:
            _add(step.name, step.executor, step.executor_config.get("agent"), "primary")
            _add_verifier(step.name, step.verifier)
    return usage


# --- lines 1349-1362 ---
def _agents_in_pipeline(p: PipelineConfig) -> list[str]:
    """Distinct `executor:agent` identifiers referenced by a pipeline's steps, read straight from config.

    Prefixed with the executor (openclaw/gateway/etc.) for the same reason the Agent
    Library page namespaces agents that way — the same agent name can exist on
    multiple backends and they are not the same agent (see README §Agent Library).

    Reliable and available even for pipelines that have never run — unlike the model
    actually used, which depends on the agent's own backend config and is only known
    from run history (see ui_pipelines for why models are deliberately left off).
    """
    return sorted({u["key"] for u in _agent_usage_in_pipeline(p)})



# --- lines 3975-3995 ---
# ── Agent pages ────────────────────────────────────────────────────────────

# Per-executor URLs — overridden at startup by configure() called from main.py lifespan.
# Defaults allow the service to start without config and fall back gracefully.
_vectorstep_gateway_base: str = os.environ.get("VECTORSTEP_GATEWAY_URL", "http://localhost:18780")
_openclaw_ws_url: str = "ws://127.0.0.1:18789/rpc"
_openclaw_enabled: bool = True  # set False when executors.openclaw is absent from config
_team_count: int = 0  # number of teams configured under auth.teams — see README §3b


def configure(openclaw_ws_url: str = "", vectorstep_gateway_base: str = "", team_count: int = 0) -> None:
    """Set agent source URLs and team count from config.yaml values. Call from main.py lifespan."""
    global _openclaw_ws_url, _vectorstep_gateway_base, _openclaw_enabled, _team_count
    _openclaw_enabled = bool(openclaw_ws_url)
    if openclaw_ws_url:
        _openclaw_ws_url = openclaw_ws_url
    if vectorstep_gateway_base:
        _vectorstep_gateway_base = vectorstep_gateway_base
    _team_count = team_count



# --- lines 4016-4025 ---
async def _fetch_openclaw_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the OpenClaw Gateway WS API (agents.list RPC)."""
    if not _openclaw_enabled:
        return [], None
    result = await gateway_call_safe("agents.list", {}, gateway_url=_openclaw_ws_url)
    if result is None:
        return [], f"Could not reach OpenClaw Gateway at {_openclaw_ws_url} — is it running?"
    return result.get("agents") or [], None



# --- lines 4026-4037 ---
async def _fetch_vectorstep_gateway_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the VectorStep Gateway REST /agents endpoint."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{_vectorstep_gateway_base}/agents")
            resp.raise_for_status()
            return resp.json().get("agents", []), None
    except Exception as exc:
        logger.debug("VectorStep Gateway /agents failed: %s", exc)
        return [], f"Could not reach VectorStep Gateway at {_vectorstep_gateway_base} — is it running?"



# --- lines 4038-4057 ---
async def _fetch_vectorstep_gateway_mcp() -> tuple[dict, dict, str | None]:
    """Fetch the MCP tool registry + server status from the VectorStep Gateway REST API.

    GET /mcp/tools returns {server_name: [{name, registeredName, description, inputSchema}, ...]}
    GET /mcp/servers returns {server_name: {running, pid, restart_count}}
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            tools_resp, servers_resp = await asyncio.gather(
                client.get(f"{_vectorstep_gateway_base}/mcp/tools"),
                client.get(f"{_vectorstep_gateway_base}/mcp/servers"),
            )
            tools_resp.raise_for_status()
            servers_resp.raise_for_status()
            return tools_resp.json(), servers_resp.json(), None
    except Exception as exc:
        logger.debug("VectorStep Gateway MCP endpoints failed: %s", exc)
        return {}, {}, f"Could not reach VectorStep Gateway at {_vectorstep_gateway_base} — is it running?"



# --- lines 4058-4074 ---
def _agent_provider_model_pairs(agents: list[dict]) -> list[tuple[str | None, str]]:
    """Parse each agent's live primary model string into (provider, bare_model) —
    the same "vendor/model" prefix split _dashboard_status_panels uses for the
    Models Configured panel, except an unprefixed model yields provider=None here
    (so resolve_approx_rate fuzzy-matches the whole catalog) rather than that
    panel's "unspecified" grouping label, which is a display convenience and not
    a real provider value that should ever hard-filter a match away."""
    pairs = []
    for a in agents:
        model = a.get("model")
        if not model:
            continue
        provider, sep, bare = model.partition("/")
        pairs.append((provider if sep else None, bare if sep else model))
    return pairs



# --- lines 4075-4118 ---
async def _compute_live_pricing_rows(agents: list[dict]) -> dict:
    """Live-pricing reference rows (SPEC-live-pricing.md) — shared by the dashboard's
    compact panel and the full Insights Overview panel. Only rendered when
    pricing.live_pricing.enabled and the catalog has actually been fetched at least
    once. Purely informational — a flat "what OpenRouter currently lists for the
    models you use" reference, not tied to any actual paid cost, so unlike the
    per-step badges on run detail (_approx_cost_for_step) these rows carry no
    real/approx color coding, and no provider column — that's per-step detail this
    aggregate list isn't trying to give.

    Matched against each agent's *configured* live primary model (same source as
    the Models Configured panel), not run history — this is "models we're set up
    to use," available immediately on a fresh deployment, not gated on any run
    having actually executed yet. Deduped to one row per matched OpenRouter
    catalog entry: many agents/providers landing on the same OpenRouter model are
    the same reference price, so they collapse to one row.
    """
    live_pricing_rows: list[dict] = []
    _table = pricing.get_table()
    live_pricing_enabled = bool(_table and _table.live_pricing.enabled)
    if live_pricing_enabled:
        catalog = live_pricing.get_catalog()
        rates_by_openrouter_id: dict[str, live_pricing.ApproxRate] = {}
        for provider, model in _agent_provider_model_pairs(agents):
            rate = live_pricing.resolve_approx_rate(catalog, provider, model)
            if rate is None:
                continue
            rates_by_openrouter_id[rate.openrouter_id] = rate
        live_pricing_rows = [
            {
                "openrouter_id": rate.openrouter_id,
                "input_per_mtok": rate.input_per_mtok,
                "output_per_mtok": rate.output_per_mtok,
            }
            for rate in sorted(rates_by_openrouter_id.values(), key=lambda r: r.openrouter_id)
        ]

    return {
        "live_pricing_rows": live_pricing_rows,
        "live_pricing_enabled": live_pricing_enabled,
        "live_pricing_last_refreshed": live_pricing.last_refreshed(),
    }



