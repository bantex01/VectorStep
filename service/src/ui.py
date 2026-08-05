import asyncio
import glob
import hashlib
import json
import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta

import httpx
import yaml
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, select
from sqlalchemy.orm import selectinload

from .analytics import ALL_STEP_STATUSES, _pipeline_rollup, _production_only, _step_rollup, _time_range_cutoff
from .analytics import get_agent_versions as _get_agent_versions
from .db.database import get_session_factory
from .db.models import AgentVersionSnapshot, PipelineRun, PipelineStep, RunFeedback, StepFeedback
from .executors.human import pending_count as _pending_approval_count
from .gateway import gateway_call_safe
from .models.pipeline import FanOutGroupConfig, ParallelGroupConfig, PipelineConfig
from .readiness import READINESS_KNOB_HELP
from .readiness import builder_seed as _builder_seed
from .readiness import evaluate_readiness as _evaluate_readiness
from .readiness import gather_readiness_evidence as _gather_readiness_evidence
from .utils import utc_now
from . import run_events

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ui")

templates = Jinja2Templates(
    directory=os.path.join(os.path.dirname(__file__), "..", "templates")
)


# ── Template helpers ──────────────────────────────────────────────────────────
# _production_only / _time_range_cutoff now live in analytics.py (imported above)
# so the JSON /stats endpoints and these HTML pages share one implementation.


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


def _readiness_verdict_classes(verdict: str) -> str:
    """Badge classes for a pipeline/step readiness verdict (SPEC-readiness-criteria.md §9)."""
    return {
        "ready":           "bg-green-950 text-green-400 ring-green-800",
        "not_ready":       "bg-red-950 text-red-400 ring-red-800",
        "building":        "bg-amber-950 text-amber-400 ring-amber-800",
        "no_data":         "bg-zinc-800 text-zinc-400 ring-zinc-700",
        "not_configured":  "bg-zinc-800 text-zinc-600 ring-zinc-700",
    }.get(verdict, "bg-zinc-800 text-zinc-400 ring-zinc-700")


def _readiness_tier_classes(verdict: str) -> str:
    """Badge classes for one tier chip within a readiness step row."""
    return {
        "pass":               "bg-green-950 text-green-400 ring-green-800",
        "fail":                "bg-red-950 text-red-400 ring-red-800",
        "insufficient_data":  "bg-amber-950 text-amber-400 ring-amber-800",
        "not_current_config": "bg-zinc-800 text-zinc-600 ring-zinc-700",
        "not_configured":     "bg-zinc-800 text-zinc-600 ring-zinc-700",
    }.get(verdict, "bg-zinc-800 text-zinc-600 ring-zinc-700")


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


def _confidence_bar_color(c: float | None) -> str:
    if c is None:
        return "bg-gray-300"
    if c >= 0.75:
        return "bg-green-500"
    if c >= 0.5:
        return "bg-amber-400"
    return "bg-red-400"


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


def _source_label(source: str) -> str:
    return {"alertmanager": "Alertmanager", "scheduler": "Scheduler", "generic": "Generic"}.get(
        source, source
    )


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


def _bucket_reset_narrative(bucket_reset: dict, n: int, n_min: int) -> str:
    """Plain-language line for the bucket_reset key on a calibration report
    (SPEC-prompt-versioning.md §4h/§6c) — the single most valuable line in this
    whole feature: it converts "this step suddenly started escalating everything
    and I don't know why" into "this step's calibration reset when you edited its
    prompt on 3 Jul; it needs 20 marked results at the new version."."""
    reason_text = {
        "prompt_changed": "this step's prompt changed",
        "agent_changed": "the agent's configuration changed",
        "both_changed": "this step's prompt and the agent's configuration both changed",
    }.get(bucket_reset["reason"], "this step's configuration changed")
    when = ""
    last_seen = bucket_reset.get("previous_version_last_seen")
    if last_seen:
        try:
            when = f" on {datetime.fromisoformat(last_seen).strftime('%b %d')}"
        except ValueError:
            when = ""
    previous_n = bucket_reset["previous_validated_n"]
    return (
        f"Calibration was reset when {reason_text}{when}. The previous version had "
        f"{previous_n} marked results; this one has {n} of the {n_min} needed."
    )


def _confidence_narrative(trust: dict, status: str) -> list[str]:
    """Plain-language, numbers-first walkthrough of how this ONE run's trust score was
    derived — S -> verifier combine -> calibration -> grounding -> deterministic checks
    -> final gate decision. No config keys or jargon; every figure is real data from
    this run, not a description of how the mechanism works in general."""
    sig = trust["signals"]
    lines: list[str] = []

    s = sig["S"]
    lines.append(f"This step's agent reported {s:.0%} confidence in its own answer.")

    seed = s
    if sig.get("V") is not None:
        v = sig["V"]
        s_after_v = sig["S_after_V"]
        blind = sig.get("V_mode") == "independent"
        checker = "A second, blind check (no sight of the primary's answer)" if blind else "A second check that reviewed the primary's own reasoning"
        veto_floor = sig.get("V_veto_floor")
        if s_after_v < s:
            lines.append(
                f"{checker} scored it at {v:.0%}. A verifier can only lower or hold "
                f"confidence, never raise it, so this pulled the working score down to {s_after_v:.0%}."
            )
        elif v < s:
            # s_after_v == s despite v < s is only reachable when the veto floor wasn't
            # tripped — under a plain minimum-of-the-two strategy this branch can't
            # happen at all, since that always takes the lower value unconditionally.
            if veto_floor is not None:
                lines.append(
                    f"{checker} scored it at {v:.0%}. This step uses a 'veto' rule: the "
                    f"verifier only overrides the primary's score when it scores below "
                    f"{veto_floor:.0%} — since {v:.0%} cleared that bar, the {s:.0%} "
                    f"confidence stood unchanged."
                )
            else:
                lines.append(
                    f"{checker} scored it at {v:.0%} — not low enough to change anything "
                    f"here, so the {s:.0%} confidence stood."
                )
        else:
            lines.append(
                f"{checker} scored it at {v:.0%} — at or above the {s:.0%} self-report, "
                f"so there was nothing to pull down."
            )
        seed = s_after_v

    calibration = trust.get("calibration")
    if calibration:
        n, n_min = calibration["n"], calibration["n_min"]
        if calibration["validated"]:
            calibrated = calibration["calibrated"]
            lines.append(
                f"This exact combination of step, agent, and model has a track record: "
                f"based on {n} past marked results, it's actually correct about "
                f"{calibrated:.0%} of the time when it reports around this confidence "
                f"level. That measured figure replaced the {seed:.0%} score above."
            )
            seed = calibrated
        elif calibration["on_uncalibrated"] == "escalate":
            lines.append(
                f"This combination doesn't have enough history yet to trust "
                f"({n}/{n_min} marked results) — since it's configured to play it safe "
                f"until proven, this run's confidence was treated as 0%."
            )
            seed = 0.0
        else:
            lines.append(
                f"This combination doesn't have enough history yet to calibrate "
                f"({n}/{n_min} marked results), so the {seed:.0%} score above was used as-is."
            )
        bucket_reset = calibration.get("bucket_reset")
        if bucket_reset:
            lines.append(_bucket_reset_narrative(bucket_reset, n, n_min))

    grounding = trust.get("grounding")
    if grounding and grounding.get("computed"):
        g = grounding["score"]
        enforce = grounding.get("enforce")
        if enforce is None:
            # Historical row from before the grounding report recorded its own enforce
            # flag — computed and reported either way, so its presence alone can't tell
            # us whether it actually gated this specific run. Say so rather than guess.
            lines.append(
                f"A separate check found {g:.0%} of this run's load-bearing claims "
                f"supported by its own evidence (whether this affected the final trust "
                f"isn't recorded for this older run)."
            )
        elif enforce:
            if g < seed:
                lines.append(
                    f"A separate check looked at whether this run's own evidence backs "
                    f"up what it claimed, and found only {g:.0%} of the load-bearing "
                    f"claims supported — this capped the confidence at {g:.0%}."
                )
                seed = g
            else:
                lines.append(
                    f"A separate check found {g:.0%} of this run's load-bearing claims "
                    f"supported by its own evidence — at or above the confidence already "
                    f"reached, so it didn't change anything."
                )
        else:
            lines.append(
                f"A separate check found {g:.0%} of this run's load-bearing claims "
                f"supported by its own evidence — recorded for visibility only, it "
                f"wasn't configured to affect this run's outcome."
            )
    elif grounding and not grounding.get("computed"):
        lines.append("A grounding check was configured but couldn't complete this run, so it had no effect.")

    det = trust.get("deterministic_checks")
    if det:
        failed = [c["name"] for c in det if not c["passed"]]
        if failed:
            lines.append(
                f"A hard, automated check failed ({', '.join(failed)}) — this overrides "
                f"everything above and forces the confidence straight to 0%."
            )
            seed = 0.0
        else:
            lines.append("All of this run's automated checks passed — no additional effect on the confidence above.")

    combined = trust["combined_trust"]
    outcome = {
        "completed": "completed normally",
        "escalated": "was escalated for a human to review",
        "aborted": "was aborted",
        "stopped": "stopped the pipeline here",
        "failed": "failed",
    }.get(status, status)
    threshold = trust.get("gate", {}).get("confidence_threshold")
    if threshold is not None:
        cleared = "cleared" if combined >= threshold else "fell short of"
        lines.append(
            f"Put together, this run's final confidence came out at {combined:.0%}, "
            f"which {cleared} this step's {threshold:.0%} threshold — so it {outcome}."
        )
    else:
        lines.append(f"Put together, this run's final confidence came out at {combined:.0%}, and the step {outcome}.")
    return lines


def _step_config_summary(trust: dict) -> list[str]:
    """Plain-language summary of how this step is set up to be judged — the gate
    threshold plus whichever of verifier/grounding/deterministic/calibration are
    configured. Complements _confidence_narrative (what happened this run) with what's
    configured to happen on every run, so a reviewer has the full picture in one place
    without going to find the pipeline's YAML."""
    lines: list[str] = []
    gate = trust.get("gate") or {}
    threshold = gate.get("confidence_threshold")
    if threshold is not None:
        lines.append(f"Confidence threshold: {threshold:.0%} (on low confidence: {gate.get('on_low_confidence')}).")

    sig = trust["signals"]
    if sig.get("V") is not None:
        mode = sig.get("V_mode") or "configured"
        strategy = sig.get("V_combination_strategy")
        floor = sig.get("V_veto_floor")
        if strategy == "veto" and floor is not None:
            detail = f" (veto rule, floor {floor:.0%})"
        elif strategy:
            detail = f" ({strategy} combination)"
        else:
            detail = ""
        lines.append(f"Verifier: {mode}{detail}.")

    grounding = trust.get("grounding")
    if grounding:
        agent = grounding.get("agent")
        enforce = grounding.get("enforce")
        if enforce is True:
            lines.append(f"Grounding: enforced (agent: {agent}).")
        elif enforce is False:
            lines.append(f"Grounding: shadow only, recorded but not enforced (agent: {agent}).")
        else:
            lines.append(f"Grounding: configured (agent: {agent}) — enforcement not recorded for this older run.")

    det = trust.get("deterministic_checks")
    if det:
        names = ", ".join(f"{c['name']} ({c['type']})" for c in det)
        lines.append(f"Deterministic checks: {names}.")

    calibration = trust.get("calibration")
    if calibration:
        line = (
            f"Calibration: enforced (needs {calibration['n_min']} marked results; "
            f"on_uncalibrated: {calibration['on_uncalibrated']})."
        )
        bucket_reset = calibration.get("bucket_reset")
        if bucket_reset:
            reason_label = {
                "prompt_changed": "prompt changed",
                "agent_changed": "agent changed",
                "both_changed": "prompt and agent changed",
            }.get(bucket_reset["reason"], "configuration changed")
            line += f" Reset: {reason_label}, previous version had {bucket_reset['previous_validated_n']} marked results."
        lines.append(line)

    return lines


def _ts_resolution(time_range: str) -> str:
    return "hour" if time_range == "24h" else "week" if time_range == "all" else "day"


def _ts_bucket(dt: datetime, resolution: str) -> str:
    d = dt.replace(tzinfo=None)
    if resolution == "hour":
        return d.strftime("%d %H:00")
    if resolution == "week":
        monday = d - timedelta(days=d.weekday())
        return monday.strftime("%b %d")
    return d.strftime("%b %d")


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


async def _fetch_step_agent_model_combo(
    cutoff: datetime | None,
) -> dict[tuple[str | None, str, str, str | None, str | None, str | None], dict]:
    """Raw (team, pipeline_name, step_name, agent, provider, model) -> counters (total,
    failed, tokens, duration), scoped to production and an optional time cutoff.
    provider/model are kept unqualified (raw DB values) so callers can group by real
    provider or real model directly (Providers/Models Insights) as well as by
    pipeline/step/agent/team (Pipelines/Steps/Agents/Teams Insights) — each computes its
    own display-qualified model label at aggregation time via _qualified_model(provider,
    model) rather than baking it into the cache key.

    Shared by every Insights drilldown that breaks down by (team, pipeline, step, agent,
    model) — each re-aggregates this same data along a different axis rather than
    re-querying it."""
    sf = get_session_factory()
    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.team,
                PipelineRun.pipeline_name,
                PipelineStep.step_name,
                PipelineStep.agent,
                PipelineStep.provider,
                PipelineStep.model,
                PipelineStep.status,
                func.count().label("n"),
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
                func.avg(PipelineStep.duration_ms),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(
                PipelineRun.team, PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.agent, PipelineStep.provider, PipelineStep.model, PipelineStep.status,
            )
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        raw = rows.all()

    combo: dict[tuple[str | None, str, str, str | None, str | None, str | None], dict] = {}
    for team, pipeline_name, step_name, agent, provider, model, status, n, in_tok, out_tok, avg_dur_ms in raw:
        key = (team, pipeline_name, step_name, agent, provider, model)
        c = combo.setdefault(key, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "duration_sum_ms": 0.0, "duration_n": 0,
        })
        c["total"] += n
        if status == "failed":
            c["failed"] += n
        c["input_tokens"] += in_tok
        c["output_tokens"] += out_tok
        if avg_dur_ms is not None:
            # Postgres/asyncpg returns avg(integer_column) as decimal.Decimal, not
            # float (SQLite returns a plain float) — cast so this doesn't crash
            # mixing Decimal into a float accumulator.
            c["duration_sum_ms"] += float(avg_dur_ms) * n
            c["duration_n"] += n
    return combo


class _LiteralBlockDumper(yaml.Dumper):
    pass


def _literal_str(dumper, data):
    if "\n" in data:
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


_LiteralBlockDumper.add_representer(str, _literal_str)


def _to_yaml(obj) -> str:
    return yaml.dump(
        obj, Dumper=_LiteralBlockDumper,
        default_flow_style=False, allow_unicode=True, sort_keys=False,
    )


def _to_json(obj, indent=None) -> str:
    return json.dumps(obj, indent=indent, ensure_ascii=False)


def _config_fingerprint(steps: list) -> str:
    """12-char SHA-256 of the ordered (step_name, agent, model) sequence for a run."""
    key = tuple(
        (s.step_name, s.agent or "", s.model or "")
        for s in sorted(steps, key=lambda x: x.step_index)
    )
    return hashlib.sha256(str(key).encode()).hexdigest()[:12]


def _config_description(steps: list) -> dict:
    """Human-readable summary of the config for display in the feedback breakdown."""
    sorted_steps = sorted(steps, key=lambda s: s.step_index)
    step_names = [s.step_name for s in sorted_steps if "/" not in s.step_name]
    models = sorted({s.model for s in sorted_steps if s.model})
    return {"steps": step_names, "models": models}


_OUTCOME_CLASSES = {
    "correct":   "bg-green-950 text-green-400 ring-green-800",
    "partial":   "bg-amber-950 text-amber-400 ring-amber-800",
    "incorrect": "bg-red-950 text-red-400 ring-red-800",
}


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

templates.env.filters["to_yaml"] = _to_yaml
templates.env.filters["tojson"] = _to_json
templates.env.filters["format_number"] = lambda n: f"{int(n):,}"
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

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def ui_dashboard(request: Request):
    sf = get_session_factory()
    cutoff_24h = utc_now() - timedelta(hours=24)
    cutoff_7d  = utc_now() - timedelta(days=7)

    status_panels = await _dashboard_status_panels()

    async with sf() as session:
        rows = await session.execute(_production_only(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.status)
        ))
        counts_24h: dict[str, int] = dict(rows.all())

        rows = await session.execute(_production_only(
            select(PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.status)
        ))
        counts_all: dict[str, int] = dict(rows.all())

        rows = await session.execute(_production_only(
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_7d)
            .group_by(PipelineRun.pipeline_name)
            .order_by(func.count().desc())
        ))
        pipeline_activity = rows.all()

        rows = await session.execute(_production_only(
            select(PipelineRun.pipeline_name, PipelineRun.team, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_7d)
            .group_by(PipelineRun.pipeline_name, PipelineRun.team)
        ))
        teams_by_pipeline: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for name, team, n in rows.all():
            teams_by_pipeline[name].append((team or "unattributed", n))

        rows = await session.execute(_production_only(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0)
                + func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.input_tokens.is_not(None))
            .group_by(PipelineRun.pipeline_name)
        ))
        tokens_by_pipeline: dict[str, int] = dict(rows.all())

        rows = await session.execute(_production_only(
            select(PipelineRun.source, func.count().label("n"))
            .where(PipelineRun.triggered_at >= cutoff_24h)
            .group_by(PipelineRun.source)
        ))
        source_counts = dict(rows.all())

        # Browse surface — shows testing runs too (badged in the template), unlike
        # every aggregate query above/below.
        rows = await session.execute(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(10)
        )
        recent_runs = rows.scalars().all()
        feedback_by_run = await _feedback_by_run_id(session, [r.id for r in recent_runs])

        rows = await session.execute(_production_only(
            select(PipelineRun.triggered_at, PipelineRun.status)
            .where(PipelineRun.triggered_at >= cutoff_24h)
        ))
        runs_ts_raw = rows.all()

        rows = await session.execute(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        )
        feedback_agg: dict[str, dict[str, int]] = {}
        for name, outcome, n in rows.all():
            d = feedback_agg.setdefault(name, {"correct": 0, "partial": 0, "incorrect": 0, "total": 0})
            d[outcome] = n
            d["total"] += n

        rows = await session.execute(_production_only(
            select(PipelineStep.executor, PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.agent.is_not(None))
            .group_by(PipelineStep.executor, PipelineStep.agent)
            .order_by(func.count().desc())
            .limit(5)
        ))
        top_agents = rows.all()

        rows = await session.execute(_production_only(
            select(PipelineStep.agent_trace)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineRun.triggered_at >= cutoff_7d, PipelineStep.agent_trace.is_not(None))
        ))
        tool_call_counts: Counter = Counter()
        for (trace_json,) in rows.all():
            try:
                events = json.loads(trace_json)
            except (TypeError, ValueError):
                continue
            for event in events or []:
                if isinstance(event, dict) and event.get("type") == "tool_call" and event.get("name"):
                    tool_call_counts[event["name"]] += 1
        top_tools = tool_call_counts.most_common(5)

    accuracy_list = [
        {"pipeline": name, "pct": round(d["correct"] / d["total"] * 100), "total": d["total"]}
        for name, d in feedback_agg.items() if d["total"] > 0
    ]
    most_accurate = sorted(accuracy_list, key=lambda x: (-x["pct"], -x["total"]))[:5]
    least_accurate = sorted(accuracy_list, key=lambda x: (x["pct"], -x["total"]))[:5]

    # Enrich pipeline_activity (7d run counts) with per-pipeline team breakdown (7d),
    # accuracy (all-time, same data as most/least accurate above), and tokens (7d).
    pipeline_activity = [
        {
            "name": name,
            "count": count,
            "teams": sorted(teams_by_pipeline.get(name, []), key=lambda t: -t[1]),
            "accuracy": (
                {"pct": round(feedback_agg[name]["correct"] / feedback_agg[name]["total"] * 100),
                 "total": feedback_agg[name]["total"]}
                if feedback_agg.get(name, {}).get("total") else None
            ),
            "tokens": tokens_by_pipeline.get(name, 0),
        }
        for name, count in pipeline_activity
    ]

    total_24h = sum(counts_24h.values())
    non_terminal_24h = counts_24h.get("running", 0) + counts_24h.get("interrupted", 0)
    terminal_24h = total_24h - non_terminal_24h
    success_rate = (
        round((terminal_24h - counts_24h.get("failed", 0)) / terminal_24h * 100)
        if terminal_24h > 0 else None
    )
    pipelines = getattr(request.app.state, "pipelines", [])

    # Status donut (24h)
    status_donut = {
        "labels": list(counts_24h.keys()),
        "data": list(counts_24h.values()),
        "colors": [_STATUS_HEX.get(s, "#71717a") for s in counts_24h.keys()],
    }

    # 24h runs timeseries bucketed by hour and status
    now = utc_now()
    bucket_labels = _ts_all_buckets(cutoff_24h, now, "hour")
    status_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for triggered_at, status in runs_ts_raw:
        status_buckets[status][_ts_bucket(triggered_at, "hour")] += 1
    runs_ts = {
        "labels": bucket_labels,
        "datasets": [
            {
                "label": status,
                "data": [status_buckets[status].get(b, 0) for b in bucket_labels],
                "color": _STATUS_HEX.get(status, "#71717a"),
            }
            for status in sorted(status_buckets, key=lambda s: sum(status_buckets[s].values()), reverse=True)
        ],
    }

    return templates.TemplateResponse(request, "dashboard.html", {
        "counts_24h": counts_24h,
        "counts_all": counts_all,
        "total_24h": total_24h,
        "success_rate": success_rate,
        "pipeline_activity": pipeline_activity,
        "source_counts": source_counts,
        "recent_runs": recent_runs,
        "feedback_by_run": feedback_by_run,
        "pipeline_count": len(pipelines),
        "scheduled_count": sum(1 for p in pipelines if p.schedule),
        "team_count": _team_count,
        "status_donut": status_donut,
        "runs_ts": runs_ts,
        "most_accurate": most_accurate,
        "least_accurate": least_accurate,
        "top_agents": top_agents,
        "top_tools": top_tools,
        "active_page": "dashboard",
        **status_panels,
    })


@router.get("/runs", response_class=HTMLResponse)
async def ui_runs(
    request: Request,
    status: str | None = None,
    pipeline: str | None = None,
    team: str | None = None,
    stage: str | None = None,
    time_range: str = "all",
    limit: int = 50,
    offset: int = 0,
):
    cutoff, range_label = _time_range_cutoff(time_range)

    def _filtered(q):
        if status:
            q = q.where(PipelineRun.status == status)
        if pipeline:
            q = q.where(PipelineRun.pipeline_name == pipeline)
        if team:
            q = q.where(PipelineRun.team == team)
        if stage:
            q = q.where(PipelineRun.stage == stage)
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        return q

    sf = get_session_factory()
    async with sf() as session:
        # Browse surface — shows testing runs too (badged in the template) unless the
        # user explicitly filters by stage; only the stat cards below are always
        # scoped to production, regardless of the stage filter.
        q = _filtered(select(PipelineRun).order_by(PipelineRun.triggered_at.desc()))
        rows = await session.execute(q.limit(limit).offset(offset))
        runs = rows.scalars().all()
        feedback_by_run = await _feedback_by_run_id(session, [r.id for r in runs])

        rows = await session.execute(
            select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name)
        )
        pipeline_names = [r[0] for r in rows.all()]

        rows = await session.execute(
            select(PipelineRun.team).distinct().where(PipelineRun.team.is_not(None)).order_by(PipelineRun.team)
        )
        team_names = [r[0] for r in rows.all()]

        # Stat cards — aggregated over every run matching the active filters, not just the
        # current page, and always scoped to production (see _production_only).
        q = _production_only(_filtered(select(PipelineRun.status, func.count()).group_by(PipelineRun.status)))
        rows = await session.execute(q)
        status_counts: dict[str, int] = dict(rows.all())

        total_matching = sum(status_counts.values())
        non_terminal = status_counts.get("running", 0) + status_counts.get("interrupted", 0)
        terminal = total_matching - non_terminal
        success_rate = (
            round((terminal - status_counts.get("failed", 0)) / terminal * 100)
            if terminal > 0 else None
        )
        escalated_failed = (
            status_counts.get("escalated", 0)
            + status_counts.get("failed", 0)
            + status_counts.get("aborted", 0)
        )

        q = _production_only(_filtered(
            select(RunFeedback.outcome, func.count())
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.outcome)
        ))
        rows = await session.execute(q)
        feedback_counts: dict[str, int] = dict(rows.all())
        marked_total = sum(feedback_counts.values())
        accuracy_pct = (
            round(feedback_counts.get("correct", 0) / marked_total * 100)
            if marked_total > 0 else None
        )

        q = _production_only(_filtered(
            select(func.coalesce(func.sum(PipelineStep.input_tokens), 0) + func.coalesce(func.sum(PipelineStep.output_tokens), 0))
            .select_from(PipelineRun)
            .join(PipelineStep, PipelineStep.run_id == PipelineRun.id)
        ))
        token_total = (await session.execute(q)).scalar() or 0

    return templates.TemplateResponse(request, "runs.html", {
        "runs": runs,
        "feedback_by_run": feedback_by_run,
        "pipeline_names": pipeline_names,
        "team_names": team_names,
        "selected_status": status or "",
        "selected_pipeline": pipeline or "",
        "selected_team": team or "",
        "selected_stage": stage or "",
        "stage_values": ["testing", "production"],
        "time_range": time_range,
        "range_label": range_label,
        "limit": limit,
        "offset": offset,
        "statuses": ["completed", "running", "escalated", "aborted", "failed", "stopped", "interrupted"],
        "active_page": "runs",
        "total_matching": total_matching,
        "success_rate": success_rate,
        "escalated_failed": escalated_failed,
        "marked_total": marked_total,
        "accuracy_pct": accuracy_pct,
        "token_total": token_total,
    })


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def ui_run_detail(request: Request, run_id: str):
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.id == run_id)
            .options(selectinload(PipelineRun.steps))
        )
        run = rows.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    sorted_steps = sorted(run.steps, key=lambda s: s.step_index)

    display_items: list[dict] = []
    seen_groups: set[str] = set()

    for step in sorted_steps:
        parsed = json.loads(step.parsed_output) if step.parsed_output else {}
        pretty = json.dumps(parsed, indent=2) if parsed else ""

        verifier_parsed = json.loads(step.verifier_output) if step.verifier_output else {}
        verifier_pretty = json.dumps(verifier_parsed, indent=2) if verifier_parsed else ""
        verifier_label = "Independent" if step.verifier_mode in ("challenger", "independent") else "Critic"

        trace = json.loads(step.agent_trace) if step.agent_trace else []

        if "/" in step.step_name:
            group_name, branch_name = step.step_name.split("/", 1)
            if group_name not in seen_groups:
                seen_groups.add(group_name)
                display_items.append({"type": "group_header", "name": group_name})
            display_items.append({
                "type": "branch",
                "group": group_name,
                "name": branch_name,
                "step": step,
                "parsed": parsed,
                "pretty": pretty,
                "verifier_pretty": verifier_pretty,
                "verifier_label": verifier_label,
                "trace": trace,
            })
        else:
            trust = json.loads(step.trust_report) if step.trust_report else None
            display_items.append({
                "type": "step",
                "name": step.step_name,
                "step": step,
                "parsed": parsed,
                "pretty": pretty,
                "verifier_pretty": verifier_pretty,
                "verifier_label": verifier_label,
                "trace": trace,
                "trust": trust,
                "confidence_narrative": _confidence_narrative(trust, step.status) if trust else None,
                "step_config_summary": _step_config_summary(trust) if trust else None,
            })

    normalised = json.loads(run.normalised_context) if run.normalised_context else {}
    run_log = json.loads(run.logs) if run.logs else []

    total_input_tokens = sum(s.input_tokens or 0 for s in run.steps)
    total_output_tokens = sum(s.output_tokens or 0 for s in run.steps)

    # Re-run support: load prior steps from the original run so the UI can show
    # the full picture (replayed steps + new steps together).
    original_run = None
    rerun_prior_items: list[dict] = []
    rerun_from_step: str | None = None

    if normalised.get("source") == "rerun":
        rerun_meta = normalised.get("metadata", {})
        original_run_id = rerun_meta.get("original_run_id")
        rerun_from_step = rerun_meta.get("from_step")
        if original_run_id:
            async with sf() as session:
                orig_result = await session.execute(
                    select(PipelineRun)
                    .where(PipelineRun.id == original_run_id)
                    .options(selectinload(PipelineRun.steps))
                )
                original_run = orig_result.scalar_one_or_none()

        if original_run and rerun_from_step:
            orig_sorted = sorted(original_run.steps, key=lambda s: s.step_index)
            prior_seen_groups: set[str] = set()
            for step in orig_sorted:
                if step.step_name == rerun_from_step:
                    break
                parsed = json.loads(step.parsed_output) if step.parsed_output else {}
                pretty = json.dumps(parsed, indent=2) if parsed else ""
                verifier_parsed = json.loads(step.verifier_output) if step.verifier_output else {}
                verifier_pretty = json.dumps(verifier_parsed, indent=2) if verifier_parsed else ""
                verifier_label = "Independent" if step.verifier_mode in ("challenger", "independent") else "Critic"
                trace = json.loads(step.agent_trace) if step.agent_trace else []
                if "/" in step.step_name:
                    group_name, branch_name = step.step_name.split("/", 1)
                    if group_name not in prior_seen_groups:
                        prior_seen_groups.add(group_name)
                        rerun_prior_items.append({"type": "group_header", "name": group_name})
                    rerun_prior_items.append({
                        "type": "branch", "group": group_name, "name": branch_name,
                        "step": step, "parsed": parsed, "pretty": pretty,
                        "verifier_pretty": verifier_pretty, "verifier_label": verifier_label,
                        "trace": trace,
                    })
                else:
                    rerun_prior_items.append({
                        "type": "step", "name": step.step_name,
                        "step": step, "parsed": parsed, "pretty": pretty,
                        "verifier_pretty": verifier_pretty, "verifier_label": verifier_label,
                        "trace": trace,
                    })

    # Attach each step's agent-version snapshot (soul.md / agent.yaml as they existed
    # at run time) so the run detail page can show *why* the agent behaved as it did
    # without navigating away to the agent's version history.
    agent_versions_needed = {
        item["step"].agent_version
        for item in (*display_items, *rerun_prior_items)
        if item.get("step") is not None and item["step"].agent_version
    }
    agent_snapshot_by_version: dict[str, AgentVersionSnapshot] = {}
    if agent_versions_needed:
        async with sf() as session:
            snap_rows = await session.execute(
                select(AgentVersionSnapshot)
                .where(AgentVersionSnapshot.agent_version.in_(agent_versions_needed))
            )
            agent_snapshot_by_version = {s.agent_version: s for s in snap_rows.scalars().all()}
    for item in (*display_items, *rerun_prior_items):
        step = item.get("step")
        if step is not None and step.agent_version:
            item["agent_snapshot"] = agent_snapshot_by_version.get(step.agent_version)

    feedback = None
    async with sf() as session:
        result = await session.execute(select(RunFeedback).where(RunFeedback.run_id == run_id))
        feedback = result.scalar_one_or_none()

    step_feedback_by_name: dict[str, dict] = {}
    async with sf() as session:
        result = await session.execute(
            select(StepFeedback.step_name, StepFeedback.outcome, StepFeedback.notes)
            .where(StepFeedback.run_id == run_id)
        )
        for name, outcome, notes in result.all():
            step_feedback_by_name[name] = {"outcome": outcome, "notes": notes or ""}

    from .executors.human import get_pending_for_run
    pending_approvals = get_pending_for_run(run_id) if run.status == "running" else []

    return templates.TemplateResponse(request, "run_detail.html", {
        "run": run,
        "display_items": display_items,
        "rerun_prior_items": rerun_prior_items,
        "rerun_from_step": rerun_from_step,
        "original_run": original_run,
        "normalised": normalised,
        "run_log": run_log,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "feedback": feedback,
        "step_feedback_by_name": step_feedback_by_name,
        "pending_approvals": pending_approvals,
        "active_page": "runs",
    })


@router.get("/runs/{run_id}/stream")
async def ui_run_stream(request: Request, run_id: str):
    """Server-Sent Events endpoint for live run tailing.

    - Run still in progress: streams events as they happen, closes on completion.
    - Run already finished: replays the stored log and closes immediately.
    """
    sf = get_session_factory()
    async with sf() as session:
        run = await session.get(PipelineRun, run_id)

    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    async def _generate():
        # Flush connection immediately so the browser fires onopen.
        yield ": connected\n\n"

        # Subscribe and snapshot history atomically (no await between these two
        # calls, so no events can slip through the gap).
        q, history = run_events.subscribe(run_id)
        try:
            # Re-read run status now that we're subscribed.
            async with sf() as session:
                fresh_run = await session.get(PipelineRun, run_id)

            if not fresh_run or fresh_run.status != "running":
                # Run already finished — replay from the DB log then close.
                status = fresh_run.status if fresh_run else "unknown"
                if fresh_run and fresh_run.logs:
                    for event in json.loads(fresh_run.logs):
                        yield f"data: {json.dumps(event)}\n\n"
                yield f"data: {json.dumps({'type': 'run_complete', 'status': status})}\n\n"
                return

            # Run still active — replay everything that happened before we
            # connected, then stream new events from the queue.
            for event in history:
                yield f"data: {json.dumps(event)}\n\n"

            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") == "run_complete":
                    break
        finally:
            run_events.unsubscribe(run_id, q)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _agent_usage_in_pipeline(p: PipelineConfig) -> list[dict]:
    """Every (step, role, executor:agent) usage in a pipeline, read straight from config.

    role is "primary" or "verifier" — the same agent can be a primary in one step
    and a verifier (critic or independent — both use the same VerifierConfig
    shape, see README §6) in another, so a given agent can carry both roles.
    Powers the pipeline detail page's Agents card. _agents_in_pipeline collapses
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


@router.get("/pipelines", response_class=HTMLResponse)
async def ui_pipelines(request: Request, tag: str | None = None, agent: str | None = None):
    all_pipelines = getattr(request.app.state, "pipelines", [])
    all_tags = sorted({t for p in all_pipelines for t in p.tags})
    agents_by_pipeline = {p.name: _agents_in_pipeline(p) for p in all_pipelines}
    all_agents = sorted({a for agents in agents_by_pipeline.values() for a in agents})

    pipelines = all_pipelines
    if tag:
        pipelines = [p for p in pipelines if tag in p.tags]
    if agent:
        pipelines = [p for p in pipelines if agent in agents_by_pipeline.get(p.name, [])]

    sf = get_session_factory()

    async with sf() as session:
        # All aggregates on this page (last-run status, run counts, success rate, avg
        # tokens, accuracy) are rollups, not a browse list — scoped to production.
        rows = await session.execute(_production_only(
            select(PipelineRun)
            .order_by(PipelineRun.triggered_at.desc())
        ))
        all_runs = rows.scalars().all()

        fb_rows = await session.execute(_production_only(
            select(RunFeedback.pipeline_name, RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .group_by(RunFeedback.pipeline_name, RunFeedback.outcome)
        ))
        feedback_by_pipeline: dict[str, dict[str, int]] = {}
        for name, outcome, n in fb_rows.all():
            if name not in feedback_by_pipeline:
                feedback_by_pipeline[name] = {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
            feedback_by_pipeline[name][outcome] = n
            feedback_by_pipeline[name]["total"] += n

        token_rows = await session.execute(_production_only(
            select(
                PipelineRun.pipeline_name,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineStep, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineRun.pipeline_name)
        ))
        tokens_by_pipeline: dict[str, tuple[int, int]] = {
            name: (inp, out) for name, inp, out in token_rows.all()
        }

    last_run: dict[str, datetime] = {}
    last_status: dict[str, str] = {}
    run_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    for run in all_runs:
        run_counts[run.pipeline_name] = run_counts.get(run.pipeline_name, 0) + 1
        if run.status == "failed":
            failed_counts[run.pipeline_name] = failed_counts.get(run.pipeline_name, 0) + 1
        if run.pipeline_name not in last_run:
            last_run[run.pipeline_name] = run.triggered_at
            last_status[run.pipeline_name] = run.status

    success_rate_by_pipeline: dict[str, int] = {}
    avg_tokens_by_pipeline: dict[str, tuple[int, int]] = {}
    for name, n in run_counts.items():
        success_rate_by_pipeline[name] = round((n - failed_counts.get(name, 0)) / n * 100)
        inp, out = tokens_by_pipeline.get(name, (0, 0))
        if inp or out:
            avg_tokens_by_pipeline[name] = (round(inp / n), round(out / n))

    # ── Header stat cards — scoped to the filtered pipeline set ─────────────
    filtered_names = [p.name for p in pipelines]
    scheduled_count = sum(1 for p in pipelines if p.schedule)

    total_runs = sum(run_counts.get(name, 0) for name in filtered_names)
    total_failed = sum(failed_counts.get(name, 0) for name in filtered_names)
    overall_success_rate = (
        round((total_runs - total_failed) / total_runs * 100) if total_runs else None
    )

    fb_correct = sum(feedback_by_pipeline.get(name, {}).get("correct", 0) for name in filtered_names)
    fb_total = sum(feedback_by_pipeline.get(name, {}).get("total", 0) for name in filtered_names)
    overall_accuracy_pct = round(fb_correct / fb_total * 100) if fb_total else None

    agents_in_view_count = len({a for name in filtered_names for a in agents_by_pipeline.get(name, [])})

    return templates.TemplateResponse(request, "pipelines.html", {
        "pipelines": pipelines,
        "last_run": last_run,
        "last_status": last_status,
        "run_counts": run_counts,
        "feedback_by_pipeline": feedback_by_pipeline,
        "agents_by_pipeline": agents_by_pipeline,
        "success_rate_by_pipeline": success_rate_by_pipeline,
        "avg_tokens_by_pipeline": avg_tokens_by_pipeline,
        "all_tags": all_tags,
        "all_agents": all_agents,
        "selected_tag": tag or "",
        "selected_agent": agent or "",
        "total_pipeline_count": len(pipelines),
        "scheduled_count": scheduled_count,
        "overall_success_rate": overall_success_rate,
        "overall_accuracy_pct": overall_accuracy_pct,
        "agents_in_view_count": agents_in_view_count,
        "active_page": "pipelines",
    })


@router.get("/insights", response_class=HTMLResponse)
async def ui_insights_overview(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Every query below is a rollup for this insights page — scoped to production.
    async with sf() as session:
        q = _production_only(select(PipelineRun.status, func.count().label("n")).group_by(PipelineRun.status))
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts: dict[str, int] = dict(rows.all())

        q = _production_only(select(PipelineRun.team, func.count().label("n")).group_by(PipelineRun.team))
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        team_counts = rows.all()

        q = _production_only(
            select(
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        total_input_tokens, total_output_tokens = (await session.execute(q)).one()

        # Fetch model + trace together so we can count both tool calls and LLM
        # call iterations per model in a single pass over the blobs.
        q = _production_only(
            select(PipelineStep.model, PipelineStep.agent_trace)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway")
            .where(PipelineStep.agent_trace.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        model_traces = rows.all()

        q = _production_only(
            select(PipelineRun.pipeline_name, func.count().label("n"))
            .group_by(PipelineRun.pipeline_name)
            .order_by(func.count().desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        pipeline_run_counts = rows.all()

        q = _production_only(
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(PipelineStep.agent)
            .order_by(func.count().desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        agent_step_counts = rows.all()

        q = _production_only(
            select(
                PipelineStep.model,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
            .group_by(PipelineStep.model)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        tokens_by_model = rows.all()

        q = (
            select(RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(PipelineRun.stage == "production")
            .group_by(RunFeedback.outcome)
        )
        if cutoff:
            q = q.where(RunFeedback.submitted_at >= cutoff)
        rows = await session.execute(q)
        feedback_by_outcome: dict[str, int] = dict(rows.all())

        # Raw rows for timeseries bucketing — one consolidated fetch per level.
        q = _production_only(
            select(PipelineRun.triggered_at, PipelineRun.status, PipelineRun.team, PipelineRun.pipeline_name)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_ts_rows = rows.all()

        q = _production_only(
            select(
                PipelineRun.triggered_at,
                PipelineStep.agent,
                PipelineStep.model,
                PipelineStep.input_tokens,
                PipelineStep.output_tokens,
                PipelineStep.agent_trace,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        step_ts_rows = rows.all()

    tool_counter: Counter = Counter()
    llm_call_counter: Counter = Counter()
    for model, trace_json in model_traces:
        try:
            events = json.loads(trace_json)
        except (TypeError, ValueError):
            continue
        for event in events:
            t = event.get("type")
            if t == "tool_call":
                tool_counter[event.get("name") or "unknown"] += 1
            elif t == "llm_call":
                llm_call_counter[model or "Unknown model"] += 1
    top_tools = tool_counter.most_common(10)

    llm_calls_sorted = llm_call_counter.most_common()

    total_runs = sum(status_counts.values())
    failed_count = status_counts.get("failed", 0)
    failure_rate = round(failed_count / total_runs * 100) if total_runs else None

    runs_by_team = sorted(
        ((team or "Unattributed", n) for team, n in team_counts),
        key=lambda t: t[1], reverse=True,
    )
    team_count = len(runs_by_team)

    tokens_by_model_sorted = sorted(
        ((model or "Unknown model", i, o) for model, i, o in tokens_by_model),
        key=lambda t: t[1] + t[2], reverse=True,
    )

    def _chart(labels, data, colors, extra_rows=None):
        """Build a chart dict with pre-zipped rows for Jinja2 table rendering."""
        rows = [{"label": l, "value": v, "color": c} for l, v, c in zip(labels, data, colors)]
        if extra_rows:
            for row, extra in zip(rows, extra_rows):
                row.update(extra)
        return {"labels": labels, "data": data, "colors": colors, "rows": rows}

    status_labels = list(status_counts.keys())
    status_data = list(status_counts.values())
    status_chart = _chart(
        status_labels, status_data,
        [_STATUS_HEX.get(s, "#71717a") for s in status_labels],
    )

    team_labels = [t for t, _ in runs_by_team]
    team_data = [n for _, n in runs_by_team]
    team_chart = _chart(
        team_labels, team_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(team_labels))],
    )

    pipeline_labels = [name for name, _ in pipeline_run_counts]
    pipeline_data = [n for _, n in pipeline_run_counts]
    pipeline_chart = _chart(
        pipeline_labels, pipeline_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(pipeline_labels))],
    )

    agent_labels = [agent or "—" for agent, _ in agent_step_counts]
    agent_data = [n for _, n in agent_step_counts]
    agent_step_chart = _chart(
        agent_labels, agent_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(agent_labels))],
    )

    llm_labels = [model for model, _ in llm_calls_sorted]
    llm_data = [count for _, count in llm_calls_sorted]
    llm_calls_chart = _chart(
        llm_labels, llm_data,
        [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(llm_labels))],
    )

    # Tokens by model: donut slices are totals; table rows carry input/output detail.
    model_labels = [m for m, _, _ in tokens_by_model_sorted]
    model_totals = [i + o for _, i, o in tokens_by_model_sorted]
    model_colors = [_CHART_PALETTE[i % len(_CHART_PALETTE)] for i in range(len(model_labels))]
    model_token_chart = _chart(
        model_labels, model_totals, model_colors,
        extra_rows=[{"input": i, "output": o} for _, i, o in tokens_by_model_sorted],
    )

    tool_chart = {
        "labels": [name for name, _ in top_tools],
        "data": [count for _, count in top_tools],
    }

    now = utc_now()
    ts_kw = {"now": now, "cutoff": cutoff, "time_range": time_range}
    status_ts  = _build_ts(run_ts_rows,  dim_fn=lambda r: r[1],  **ts_kw)   # status
    team_ts    = _build_ts(run_ts_rows,  dim_fn=lambda r: r[2] or "Unattributed", **ts_kw)   # team
    pipeline_ts = _build_ts(run_ts_rows, dim_fn=lambda r: r[3],  **ts_kw)   # pipeline
    agent_ts   = _build_ts(step_ts_rows, dim_fn=lambda r: r[1] or "—", **ts_kw)   # agent
    llm_ts     = _build_ts(
        step_ts_rows,
        dim_fn=lambda r: r[2] or "Unknown model",
        val_fn=lambda r: sum(1 for e in (json.loads(r[5]) if r[5] else []) if e.get("type") == "llm_call"),
        **ts_kw,
    )
    tokens_ts  = _build_ts(
        [r for r in step_ts_rows if r[3] is not None],
        dim_fn=lambda r: r[2] or "Unknown model",
        val_fn=lambda r: (r[3] or 0) + (r[4] or 0),
        **ts_kw,
    )

    feedback_total = sum(feedback_by_outcome.values())
    feedback_correct = feedback_by_outcome.get("correct", 0)

    return templates.TemplateResponse(request, "insights_overview.html", {
        "time_range": time_range,
        "range_label": range_label,
        "total_runs": total_runs,
        "failed_count": failed_count,
        "failure_rate": failure_rate,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "team_count": team_count,
        "feedback_total": feedback_total,
        "feedback_correct": feedback_correct,
        "feedback_by_outcome": feedback_by_outcome,
        "status_chart": status_chart, "status_ts": status_ts,
        "team_chart": team_chart, "team_ts": team_ts,
        "pipeline_chart": pipeline_chart, "pipeline_ts": pipeline_ts,
        "agent_step_chart": agent_step_chart, "agent_ts": agent_ts,
        "llm_calls_chart": llm_calls_chart, "llm_ts": llm_ts,
        "model_token_chart": model_token_chart, "tokens_ts": tokens_ts,
        "tool_chart": tool_chart,
        "active_page": "insights_overview",
    })


@router.get("/insights/pipelines", response_class=HTMLResponse)
async def ui_insights_pipelines(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Per-pipeline run/token/duration/team/feedback rollup — shared with the
    # /pipelines/{name}/stats and /stats/pipelines JSON endpoints (analytics.py)
    # so this page and those endpoints can never disagree.
    rollup = await _pipeline_rollup(sf, time_range, "production")
    run_counts = {name: r["runs_total"] for name, r in rollup.items()}
    failed_counts = {name: r["status_counts"].get("failed", 0) for name, r in rollup.items()}
    token_totals = {name: (r["tokens"]["input"], r["tokens"]["output"]) for name, r in rollup.items()}
    teams_by_pipeline: dict[str, list[str]] = {name: list(r["teams"]) for name, r in rollup.items()}
    avg_duration_by_pipeline = {
        name: r["duration_seconds"]["avg"] for name, r in rollup.items()
        if r["duration_seconds"]["avg"] is not None
    }
    status_counts_by_pipeline: dict[str, dict[str, int]] = {name: dict(r["status_counts"]) for name, r in rollup.items()}
    feedback_raw: dict[str, dict[str, int]] = {
        name: {
            "correct": r["accuracy"]["correct"],
            "partial": r["accuracy"]["partial"],
            "incorrect": r["accuracy"]["incorrect"],
        }
        for name, r in rollup.items()
    }

    # Per-run rows — needed for the timeseries/recent-runs drilldown, which the
    # shared rollup (aggregated) doesn't expose at this granularity.
    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.id,
                PipelineRun.pipeline_name,
                PipelineRun.status,
                PipelineRun.triggered_at,
                PipelineRun.completed_at,
            )
            .order_by(PipelineRun.triggered_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_runs_raw = rows.all()

    step_combo = await _fetch_step_agent_model_combo(cutoff)

    # ── Per-pipeline aggregates ───────────────────────────────────────────────
    # avg_duration_by_pipeline comes from the shared rollup above.

    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_runs_raw:
        oldest = min(r.triggered_at.replace(tzinfo=None) for r in all_runs_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_pipeline: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_pipeline: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_pipeline: dict[str, list] = defaultdict(list)

    for row in all_runs_raw:
        bucket = _ts_bucket(row.triggered_at, resolution)
        runs_by_bucket_pipeline[row.pipeline_name][bucket] += 1
        if row.completed_at:
            secs = (row.completed_at.replace(tzinfo=None) - row.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                durations_by_bucket_pipeline[row.pipeline_name][bucket].append(secs)
        if len(recent_by_pipeline[row.pipeline_name]) < 5:
            recent_by_pipeline[row.pipeline_name].append(row)

    # ── Per-pipeline step breakdown (agents/models/tokens involved) ───────────
    # step_combo (fetched above) is keyed by (pipeline_name, step_name, agent, provider,
    # model) — re-aggregate it here indexed by pipeline for this page's drilldown. The
    # Steps/Models/Providers insights pages reuse the same combo, indexed differently.
    step_breakdown_by_pipeline: dict[str, list[dict]] = defaultdict(list)
    for (_team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        step_breakdown_by_pipeline[pipeline_name].append({
            "step_name": step_name,
            "agent": agent,
            "model": _qualified_model(provider, model),
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": _format_seconds(avg_duration_secs),
        })
    for rows_ in step_breakdown_by_pipeline.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────

    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_pipeline.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_pipeline.get(name)
        inp, out = token_totals.get(name, (0, 0))

        runs_ts_data = [runs_by_bucket_pipeline[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_pipeline[name][b]) / len(durations_by_bucket_pipeline[name][b]))
            if durations_by_bucket_pipeline[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_pipeline.get(name, []):
            dur_str = None
            if r.completed_at:
                secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
                dur_str = _format_seconds(secs)
            recent.append({
                "id": str(r.id),
                "status": r.status,
                "ago": _format_ago(r.triggered_at),
                "duration": dur_str,
            })

        fb = feedback_raw.get(name, {})
        fb_correct = fb.get("correct", 0)
        fb_partial = fb.get("partial", 0)
        fb_incorrect = fb.get("incorrect", 0)
        fb_total = fb_correct + fb_partial + fb_incorrect

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": step_breakdown_by_pipeline.get(name, []),
            "feedback_total": fb_total,
            "feedback_correct": fb_correct,
            "feedback_partial": fb_partial,
            "feedback_incorrect": fb_incorrect,
            "accuracy_pct": round(fb_correct / fb_total * 100) if fb_total else None,
        }

    # ── Headline accuracy across all pipelines ────────────────────────────────

    insights_feedback_total = sum(sum(fb.values()) for fb in feedback_raw.values())
    insights_feedback_correct = sum(fb.get("correct", 0) for fb in feedback_raw.values())
    insights_accuracy_pct = round(insights_feedback_correct / insights_feedback_total * 100) if insights_feedback_total else None

    # ── Headline stats ────────────────────────────────────────────────────────

    all_durations_flat = []
    for r in all_runs_raw:
        if r.completed_at:
            secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                all_durations_flat.append(secs)
    min_duration_secs = min(all_durations_flat) if all_durations_flat else None
    overall_avg_duration_secs = (sum(all_durations_flat) / len(all_durations_flat)) if all_durations_flat else None
    max_duration_secs = max(all_durations_flat) if all_durations_flat else None

    # ── Chart data ────────────────────────────────────────────────────────────

    pipeline_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        fb = feedback_raw.get(name, {})
        fb_total = sum(fb.values())
        fb_correct = fb.get("correct", 0)
        pipeline_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_pipeline.get(name),
            "input_tokens": token_totals.get(name, (0, 0))[0],
            "output_tokens": token_totals.get(name, (0, 0))[1],
            "teams": sorted(teams_by_pipeline.get(name, [])),
            "feedback_total": fb_total,
            "accuracy_pct": round(fb_correct / fb_total * 100) if fb_total else None,
        })

    duration_chart_rows = sorted(
        ((name, secs) for name, secs in avg_duration_by_pipeline.items()),
        key=lambda t: t[1], reverse=True,
    )
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    token_chart_rows = sorted(
        pipeline_rows, key=lambda r: r["input_tokens"] + r["output_tokens"], reverse=True,
    )
    token_chart_rows = [r for r in token_chart_rows if r["input_tokens"] or r["output_tokens"]]
    token_chart = {
        "labels": [r["name"] for r in token_chart_rows],
        "input": [r["input_tokens"] for r in token_chart_rows],
        "output": [r["output_tokens"] for r in token_chart_rows],
    }

    return templates.TemplateResponse(request, "insights_pipelines.html", {
        "time_range": time_range,
        "range_label": range_label,
        "pipeline_rows": pipeline_rows,
        "duration_chart": duration_chart,
        "token_chart": token_chart,
        "total_pipeline_count": len(run_counts),
        "min_duration_secs": min_duration_secs,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "max_duration_secs": max_duration_secs,
        "drilldown_data": drilldown_data,
        "insights_feedback_total": insights_feedback_total,
        "insights_accuracy_pct": insights_accuracy_pct,
        "active_page": "insights_pipelines",
    })


def _buckets_matching(calibration_buckets: dict, step_name, agent, model, provider) -> list:
    """Every calibration bucket matching the first four key components, regardless
    of prompt_hash/agent_version. See SPEC-prompt-versioning.md §4g — the bucket key
    grew to 6 components, but several call sites (like the ones below) only have the
    original 4 to key off of."""
    return [
        b for (s, a, m, p, _ph, _av), b in calibration_buckets.items()
        if (s, a, m, p) == (step_name, agent, model, provider)
    ]


def _largest_bucket_matching(calibration_buckets: dict, step_name, agent, model, provider):
    """The matching bucket with the most samples — used for the Calibration bins
    display, so a brand-new (tiny) version doesn't blank out a rich, informative
    history the moment a prompt or agent changes."""
    matches = _buckets_matching(calibration_buckets, step_name, agent, model, provider)
    if not matches:
        return None
    return max(matches, key=lambda b: b.total_n)


def _most_recent_bucket_matching(calibration_buckets: dict, step_name, agent, model, provider):
    """The matching bucket most recently added to — used for the Version chips.
    Deliberately NOT the same choice as _largest_bucket_matching: right after a
    prompt/agent edit, the legacy or previous-version bucket is almost always
    still the *largest* (it's had longer to accumulate), but the Version column's
    whole job is to say what's CURRENT, and showing the largest bucket's hash
    there would keep pointing at a stale version for a long time. A bucket with
    no last_seen_at (shouldn't happen post-migration, but defensive) sorts last."""
    matches = _buckets_matching(calibration_buckets, step_name, agent, model, provider)
    if not matches:
        return None
    return max(matches, key=lambda b: b.last_seen_at or datetime.min)


@router.get("/insights/steps", response_class=HTMLResponse)
async def ui_insights_steps(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Per-step run/token/duration rollup — shared with the /steps/{name}/stats
    # JSON endpoint (analytics.py) so this page and that endpoint can never
    # disagree.
    rollup = await _step_rollup(sf, time_range, "production")
    run_counts = {name: r["runs_total"] for name, r in rollup.items()}
    failed_counts = {name: r["status_counts"].get("failed", 0) for name, r in rollup.items()}
    token_totals = {name: (r["tokens"]["input"], r["tokens"]["output"]) for name, r in rollup.items()}
    avg_duration_by_step = {
        name: r["duration_seconds"]["avg"] for name, r in rollup.items()
        if r["duration_seconds"]["avg"] is not None
    }
    status_counts_by_step: dict[str, dict[str, int]] = {name: dict(r["status_counts"]) for name, r in rollup.items()}

    # Per-step-execution rows and the finer (step, agent, model, provider) feedback
    # breakdown below stay as page-specific queries — granularity the shared
    # rollup (aggregated to step_name only) doesn't expose.
    async with sf() as session:
        # All step executions in range — used for per-step timeseries and recent-list
        q = _production_only(
            select(
                PipelineStep.step_name, PipelineRun.pipeline_name, PipelineStep.run_id,
                PipelineStep.status, PipelineStep.executed_at, PipelineStep.duration_ms,
                PipelineStep.input_tokens, PipelineStep.output_tokens,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_steps_raw = rows.all()

        q = _production_only(
            select(
                PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.model, PipelineStep.provider,
                StepFeedback.outcome, func.count().label("n"),
            )
            .join(PipelineStep, StepFeedback.step_id == PipelineStep.id)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .group_by(
                PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.model, PipelineStep.provider, StepFeedback.outcome,
            )
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        step_feedback_rows = (await session.execute(q)).all()

    _all_step_durations_secs = [r.duration_ms / 1000 for r in all_steps_raw if r.duration_ms is not None]
    overall_avg_duration_secs = (
        sum(_all_step_durations_secs) / len(_all_step_durations_secs) if _all_step_durations_secs else None
    )

    # step_combo: (pipeline_name, step_name, agent, qualified_model) -> counters, shared
    # with the Pipelines Insights drilldown (see _fetch_step_agent_model_combo).
    step_combo = await _fetch_step_agent_model_combo(cutoff)

    from .pipeline.calibration import calibration_recommendation, compute_calibration_buckets

    calibration_buckets = await compute_calibration_buckets(sf)

    feedback_by_step: dict[str, dict] = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0})
    feedback_by_combo: dict[tuple, dict] = defaultdict(lambda: {"correct": 0, "partial": 0, "incorrect": 0})
    for step_name, agent, model, provider, outcome, n in step_feedback_rows:
        feedback_by_step[step_name][outcome] += n
        feedback_by_combo[(step_name, agent, _qualified_model(provider, model))][outcome] += n

    def _acc(d: dict) -> dict:
        total = d["correct"] + d["partial"] + d["incorrect"]
        return {**d, "total": total,
                "accuracy_pct": round(d["correct"] / total * 100) if total else None}

    feedback_by_step = {k: _acc(v) for k, v in feedback_by_step.items()}
    feedback_by_combo = {k: _acc(v) for k, v in feedback_by_combo.items()}

    # ── Per-step aggregates ────────────────────────────────────────────────────

    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_steps_raw:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_steps_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_step: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_step: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_step: dict[str, list] = defaultdict(list)

    for row in all_steps_raw:
        bucket = _ts_bucket(row.executed_at, resolution)
        runs_by_bucket_step[row.step_name][bucket] += 1
        if row.duration_ms is not None:
            durations_by_bucket_step[row.step_name][bucket].append(row.duration_ms / 1000)
        if len(recent_by_step[row.step_name]) < 5:
            recent_by_step[row.step_name].append(row)

    # ── Per-step pipeline/agent/model breakdown ────────────────────────────────
    # Re-aggregates the shared step_combo indexed by step instead of by pipeline (compare
    # ui_insights_pipelines's step_breakdown_by_pipeline, built from the same combo).
    pipeline_breakdown_by_step: dict[str, list[dict]] = defaultdict(list)
    distinct_pipelines_by_step: dict[str, set[str]] = defaultdict(set)
    for (_team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        distinct_pipelines_by_step[step_name].add(pipeline_name)
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        fb = feedback_by_combo.get((step_name, agent, _qualified_model(provider, model)))
        # step_combo isn't scoped by prompt_hash/agent_version, but calibration_buckets
        # now is (SPEC-prompt-versioning.md §4g) — one (step, agent, model, provider)
        # combo can match several buckets (one per prompt/agent version). The
        # Calibration column shows the LARGEST bucket (most informative — a brand-new
        # version's near-empty bins shouldn't blank out a rich history), but the
        # Version column shows the MOST RECENT bucket's hash (what's actually current
        # right now) — using the largest bucket for both would keep pointing the
        # Version chip at a stale prompt for a long time after every edit. §6a's
        # prompt-history view is where every version gets its own row regardless.
        bins_bucket = _largest_bucket_matching(calibration_buckets, step_name, agent, model, provider)
        version_bucket = _most_recent_bucket_matching(calibration_buckets, step_name, agent, model, provider)
        calibration_summary = calibration_recommendation(bins_bucket) if bins_bucket else None
        pipeline_breakdown_by_step[step_name].append({
            "pipeline_name": pipeline_name,
            "agent": agent,
            "model": _qualified_model(provider, model),
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": _format_seconds(avg_duration_secs),
            "accuracy_pct": fb["accuracy_pct"] if fb else None,
            "marked_total": fb["total"] if fb else 0,
            "calibration_bins": [
                {"lo": b.lo, "hi": b.hi, "n": b.n, "mean_label": b.mean_label, "validated": b.validated}
                for b in bins_bucket.bins
            ] if bins_bucket else [],
            "calibration_recommendation": calibration_summary,
            "prompt_hash": version_bucket.prompt_hash if version_bucket else None,
            "agent_version": version_bucket.agent_version if version_bucket else None,
        })
    for rows_ in pipeline_breakdown_by_step.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────

    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_step.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_step.get(name)
        inp, out = token_totals.get(name, (0, 0))

        runs_ts_data = [runs_by_bucket_step[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_step[name][b]) / len(durations_by_bucket_step[name][b]))
            if durations_by_bucket_step[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_step.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "status": r.status,
                "ago": _format_ago(r.executed_at),
                "duration": _format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "pipeline_breakdown": pipeline_breakdown_by_step.get(name, []),
            "accuracy_pct": feedback_by_step.get(name, {}).get("accuracy_pct"),
            "marked_total": feedback_by_step.get(name, {}).get("total", 0),
            "feedback_breakdown": {
                k: feedback_by_step.get(name, {}).get(k, 0)
                for k in ("correct", "partial", "incorrect")
            },
        }

    # ── Headline stats ────────────────────────────────────────────────────────

    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────

    step_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        step_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_step.get(name),
            "input_tokens": token_totals.get(name, (0, 0))[0],
            "output_tokens": token_totals.get(name, (0, 0))[1],
            "pipelines": sorted(distinct_pipelines_by_step.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_step.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = _build_ts(
        [(r.executed_at, r.step_name) for r in all_steps_raw], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = _build_ts(
        [(r.executed_at, r.step_name, (r.input_tokens or 0) + (r.output_tokens or 0)) for r in all_steps_raw],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_steps.html", {
        "time_range": time_range,
        "range_label": range_label,
        "step_rows": step_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_step_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "drilldown_data": drilldown_data,
        "active_page": "insights_steps",
    })


@router.get("/insights/agents", response_class=HTMLResponse)
async def ui_insights_agents(request: Request, time_range: str = "7d"):
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    # Every query below is a rollup for this insights page — scoped to production.
    # PipelineStep.agent already carries the executor prefix (f"{executor}:{agent}" —
    # see runner.py), so it alone is a stable, unique key without also grouping by executor.
    async with sf() as session:
        q = _production_only(
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_counts = dict(rows.all())

        q = _production_only(
            select(PipelineStep.agent, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None), PipelineStep.status == "failed")
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        failed_counts = dict(rows.all())

        q = _production_only(
            select(
                PipelineStep.agent,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        token_totals = {name: (i, o) for name, i, o in rows.all()}

        q = _production_only(
            select(PipelineStep.agent, func.avg(PipelineStep.duration_ms))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        avg_duration_by_agent = {name: ms / 1000 for name, ms in rows.all() if ms is not None}

        q = _production_only(
            select(func.avg(PipelineStep.duration_ms))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        overall_avg_duration_ms = (await session.execute(q)).scalar()
        overall_avg_duration_secs = overall_avg_duration_ms / 1000 if overall_avg_duration_ms is not None else None

        q = _production_only(
            select(PipelineStep.agent, PipelineStep.status, func.count().label("n"))
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent, PipelineStep.status)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts_by_agent: dict[str, dict[str, int]] = defaultdict(dict)
        for name, status, n in rows.all():
            status_counts_by_agent[name][status] = n

        # All steps in range — used for per-agent timeseries and recent-list
        q = _production_only(
            select(
                PipelineStep.agent, PipelineRun.pipeline_name, PipelineStep.step_name,
                PipelineStep.run_id, PipelineStep.status, PipelineStep.executed_at,
                PipelineStep.duration_ms, PipelineStep.input_tokens, PipelineStep.output_tokens,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent.isnot(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_steps_raw = rows.all()

    # step_combo: (pipeline_name, step_name, agent, qualified_model) -> counters, shared
    # with the Pipelines/Steps Insights drilldowns (see _fetch_step_agent_model_combo).
    step_combo = await _fetch_step_agent_model_combo(cutoff)

    # ── Per-agent aggregates ──────────────────────────────────────────────────

    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_steps_raw:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_steps_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_agent: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_agent: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_agent: dict[str, list] = defaultdict(list)

    for row in all_steps_raw:
        bucket = _ts_bucket(row.executed_at, resolution)
        runs_by_bucket_agent[row.agent][bucket] += 1
        if row.duration_ms is not None:
            durations_by_bucket_agent[row.agent][bucket].append(row.duration_ms / 1000)
        if len(recent_by_agent[row.agent]) < 5:
            recent_by_agent[row.agent].append(row)

    # ── Per-agent step/pipeline/model breakdown ────────────────────────────────
    # Re-aggregates the shared step_combo indexed by agent instead of by pipeline/step
    # (compare ui_insights_pipelines / ui_insights_steps, built from the same combo).
    step_breakdown_by_agent: dict[str, list[dict]] = defaultdict(list)
    models_by_agent: dict[str, set[str]] = defaultdict(set)
    for (_team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        if not agent:
            continue
        qualified_model = _qualified_model(provider, model)
        models_by_agent[agent].add(qualified_model)
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        step_breakdown_by_agent[agent].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "model": qualified_model,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": _format_seconds(avg_duration_secs),
        })
    for rows_ in step_breakdown_by_agent.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────

    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_agent.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_agent.get(name)
        inp, out = token_totals.get(name, (0, 0))

        runs_ts_data = [runs_by_bucket_agent[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_agent[name][b]) / len(durations_by_bucket_agent[name][b]))
            if durations_by_bucket_agent[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_agent.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "step_name": r.step_name,
                "status": r.status,
                "ago": _format_ago(r.executed_at),
                "duration": _format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": step_breakdown_by_agent.get(name, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────

    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────

    agent_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        executor, _, bare_name = name.partition(":")
        agent_rows.append({
            "name": name,
            "executor": executor,
            "bare_name": bare_name or name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_agent.get(name),
            "input_tokens": token_totals.get(name, (0, 0))[0],
            "output_tokens": token_totals.get(name, (0, 0))[1],
            "models": sorted(models_by_agent.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_agent.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = _build_ts(
        [(r.executed_at, r.agent) for r in all_steps_raw], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = _build_ts(
        [(r.executed_at, r.agent, (r.input_tokens or 0) + (r.output_tokens or 0)) for r in all_steps_raw],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_agents.html", {
        "time_range": time_range,
        "range_label": range_label,
        "agent_rows": agent_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_agent_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "drilldown_data": drilldown_data,
        "active_page": "insights_agents",
    })


@router.get("/insights/providers", response_class=HTMLResponse)
async def ui_insights_providers(request: Request, time_range: str = "7d"):
    """Token/call spend grouped by LLM provider (gateway executor only). Unlike every
    other Insights page, this one falls back to a best-effort provider guess from the
    model string (see _provider_from_model) when the `provider` column is NULL — this
    page's entire purpose is grouping by provider, so a best-effort bucket for
    pre-migration rows beats losing that history from the page entirely. Contrast with
    _qualified_model (used for per-row display elsewhere), which never guesses."""
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.provider, PipelineStep.model, PipelineStep.status,
                PipelineStep.run_id, PipelineStep.executed_at, PipelineStep.duration_ms,
                PipelineStep.input_tokens, PipelineStep.output_tokens,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway", PipelineStep.model.is_not(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_rows = rows.all()

    def eff_provider(provider: str | None, model: str | None) -> str:
        return provider or _provider_from_model(model)

    run_counts: dict[str, int] = defaultdict(int)
    failed_counts: dict[str, int] = defaultdict(int)
    token_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    duration_sum: dict[str, float] = defaultdict(float)
    duration_n: dict[str, int] = defaultdict(int)
    status_counts_by_provider: dict[str, dict[str, int]] = defaultdict(dict)
    models_by_provider: dict[str, set[str]] = defaultdict(set)
    breakdown_combo: dict[tuple[str, str, str, str | None, str], dict] = {}

    for r in all_rows:
        provider = eff_provider(r.provider, r.model)
        run_counts[provider] += 1
        if r.status == "failed":
            failed_counts[provider] += 1
        token_totals[provider][0] += r.input_tokens or 0
        token_totals[provider][1] += r.output_tokens or 0
        if r.duration_ms is not None:
            duration_sum[provider] += r.duration_ms
            duration_n[provider] += 1
        status_counts_by_provider[provider][r.status] = status_counts_by_provider[provider].get(r.status, 0) + 1
        models_by_provider[provider].add(r.model)

        bkey = (provider, r.pipeline_name, r.step_name, r.agent, r.model)
        bc = breakdown_combo.setdefault(bkey, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "duration_sum_ms": 0.0, "duration_n": 0,
        })
        bc["total"] += 1
        if r.status == "failed":
            bc["failed"] += 1
        bc["input_tokens"] += r.input_tokens or 0
        bc["output_tokens"] += r.output_tokens or 0
        if r.duration_ms is not None:
            bc["duration_sum_ms"] += r.duration_ms
            bc["duration_n"] += 1

    avg_duration_by_provider = {p: duration_sum[p] / duration_n[p] / 1000 for p in duration_n if duration_n[p]}
    overall_avg_duration_secs = (
        sum(duration_sum.values()) / sum(duration_n.values()) / 1000 if sum(duration_n.values()) else None
    )

    # ── Timeseries + recent-list buckets ───────────────────────────────────────
    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_rows:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_rows)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_provider: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_provider: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_provider: dict[str, list] = defaultdict(list)

    for r in all_rows:
        provider = eff_provider(r.provider, r.model)
        bucket = _ts_bucket(r.executed_at, resolution)
        runs_by_bucket_provider[provider][bucket] += 1
        if r.duration_ms is not None:
            durations_by_bucket_provider[provider][bucket].append(r.duration_ms / 1000)
        if len(recent_by_provider[provider]) < 5:
            recent_by_provider[provider].append(r)

    # ── Per-provider pipeline/step/agent/model breakdown ───────────────────────
    breakdown_by_provider: dict[str, list[dict]] = defaultdict(list)
    for (provider, pipeline_name, step_name, agent, model), c in breakdown_combo.items():
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        breakdown_by_provider[provider].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "agent": agent,
            "model": model,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": _format_seconds(avg_duration_secs),
        })
    for rows_ in breakdown_by_provider.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_provider.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_provider.get(name)
        inp, out = token_totals.get(name, [0, 0])

        runs_ts_data = [runs_by_bucket_provider[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_provider[name][b]) / len(durations_by_bucket_provider[name][b]))
            if durations_by_bucket_provider[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_provider.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "step_name": r.step_name,
                "status": r.status,
                "ago": _format_ago(r.executed_at),
                "duration": _format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": breakdown_by_provider.get(name, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────
    provider_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        provider_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_provider.get(name),
            "input_tokens": token_totals.get(name, [0, 0])[0],
            "output_tokens": token_totals.get(name, [0, 0])[1],
            "models": sorted(models_by_provider.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_provider.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = _build_ts(
        [(r.executed_at, eff_provider(r.provider, r.model)) for r in all_rows], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = _build_ts(
        [(r.executed_at, eff_provider(r.provider, r.model), (r.input_tokens or 0) + (r.output_tokens or 0))
         for r in all_rows],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_providers.html", {
        "time_range": time_range,
        "range_label": range_label,
        "provider_rows": provider_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_provider_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "drilldown_data": drilldown_data,
        "active_page": "insights_providers",
    })


@router.get("/insights/models", response_class=HTMLResponse)
async def ui_insights_models(request: Request, time_range: str = "7d"):
    """Success rate/duration/tokens grouped by model (gateway executor only).

    Grouped by the display-qualified model identity (see _qualified_model) rather than the
    bare model string — deliberately does NOT guess a provider for rows missing one (unlike
    Insights > Providers, whose whole point is provider bucketing). A bare pre-migration
    "claude-sonnet-5" and a qualified "anthropic/claude-sonnet-5" are kept as distinct rows
    here rather than merged on a guess — same reasoning as _qualified_model everywhere else.
    """
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.provider, PipelineStep.model, PipelineStep.status,
                PipelineStep.run_id, PipelineStep.executed_at, PipelineStep.duration_ms,
                PipelineStep.input_tokens, PipelineStep.output_tokens,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.executor == "gateway", PipelineStep.model.is_not(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_rows = rows.all()

    run_counts: dict[str, int] = defaultdict(int)
    failed_counts: dict[str, int] = defaultdict(int)
    token_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    duration_sum: dict[str, float] = defaultdict(float)
    duration_n: dict[str, int] = defaultdict(int)
    status_counts_by_model: dict[str, dict[str, int]] = defaultdict(dict)
    agents_by_model: dict[str, set[str]] = defaultdict(set)
    breakdown_combo: dict[tuple[str, str, str, str | None], dict] = {}

    for r in all_rows:
        model = _qualified_model(r.provider, r.model)
        run_counts[model] += 1
        if r.status == "failed":
            failed_counts[model] += 1
        token_totals[model][0] += r.input_tokens or 0
        token_totals[model][1] += r.output_tokens or 0
        if r.duration_ms is not None:
            duration_sum[model] += r.duration_ms
            duration_n[model] += 1
        status_counts_by_model[model][r.status] = status_counts_by_model[model].get(r.status, 0) + 1
        if r.agent:
            agents_by_model[model].add(r.agent)

        bkey = (model, r.pipeline_name, r.step_name, r.agent)
        bc = breakdown_combo.setdefault(bkey, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0,
            "duration_sum_ms": 0.0, "duration_n": 0,
        })
        bc["total"] += 1
        if r.status == "failed":
            bc["failed"] += 1
        bc["input_tokens"] += r.input_tokens or 0
        bc["output_tokens"] += r.output_tokens or 0
        if r.duration_ms is not None:
            bc["duration_sum_ms"] += r.duration_ms
            bc["duration_n"] += 1

    avg_duration_by_model = {m: duration_sum[m] / duration_n[m] / 1000 for m in duration_n if duration_n[m]}
    overall_avg_duration_secs = (
        sum(duration_sum.values()) / sum(duration_n.values()) / 1000 if sum(duration_n.values()) else None
    )

    # ── Timeseries + recent-list buckets ───────────────────────────────────────
    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_rows:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_rows)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_model: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_model: dict[str, list] = defaultdict(list)

    for r in all_rows:
        model = _qualified_model(r.provider, r.model)
        bucket = _ts_bucket(r.executed_at, resolution)
        runs_by_bucket_model[model][bucket] += 1
        if r.duration_ms is not None:
            durations_by_bucket_model[model][bucket].append(r.duration_ms / 1000)
        if len(recent_by_model[model]) < 5:
            recent_by_model[model].append(r)

    # ── Per-model pipeline/step/agent breakdown ────────────────────────────────
    breakdown_by_model: dict[str, list[dict]] = defaultdict(list)
    for (model, pipeline_name, step_name, agent), c in breakdown_combo.items():
        total = c["total"]
        avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
        breakdown_by_model[model].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "agent": agent,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "avg_duration": _format_seconds(avg_duration_secs),
        })
    for rows_ in breakdown_by_model.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_model.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_model.get(name)
        inp, out = token_totals.get(name, [0, 0])

        runs_ts_data = [runs_by_bucket_model[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_model[name][b]) / len(durations_by_bucket_model[name][b]))
            if durations_by_bucket_model[name].get(b) else None
            for b in bucket_labels
        ]

        recent = []
        for r in recent_by_model.get(name, []):
            recent.append({
                "id": str(r.run_id),
                "pipeline_name": r.pipeline_name,
                "step_name": r.step_name,
                "status": r.status,
                "ago": _format_ago(r.executed_at),
                "duration": _format_seconds(r.duration_ms / 1000) if r.duration_ms is not None else None,
            })

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent,
            "step_breakdown": breakdown_by_model.get(name, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_step_executions = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_step_executions - total_failed) / total_step_executions * 100) if total_step_executions else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())

    # ── Chart data ────────────────────────────────────────────────────────────
    model_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        model_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_model.get(name),
            "input_tokens": token_totals.get(name, [0, 0])[0],
            "output_tokens": token_totals.get(name, [0, 0])[1],
            "agents": sorted(agents_by_model.get(name, [])),
        })

    duration_chart_rows = sorted(avg_duration_by_model.items(), key=lambda t: t[1], reverse=True)
    duration_chart = {
        "labels": [name for name, _ in duration_chart_rows],
        "data": [round(secs) for _, secs in duration_chart_rows],
    }

    steps_ts = _build_ts(
        [(r.executed_at, _qualified_model(r.provider, r.model)) for r in all_rows], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = _build_ts(
        [(r.executed_at, _qualified_model(r.provider, r.model), (r.input_tokens or 0) + (r.output_tokens or 0))
         for r in all_rows],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_models.html", {
        "time_range": time_range,
        "range_label": range_label,
        "model_rows": model_rows,
        "duration_chart": duration_chart,
        "steps_ts": steps_ts,
        "tokens_ts": tokens_ts,
        "total_model_count": len(run_counts),
        "total_step_executions": total_step_executions,
        "overall_success_rate": overall_success_rate,
        "overall_avg_duration_secs": overall_avg_duration_secs,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "drilldown_data": drilldown_data,
        "active_page": "insights_models",
    })


@router.get("/insights/mcp", response_class=HTMLResponse)
async def ui_insights_mcp(request: Request, time_range: str = "7d"):
    """MCP tool call usage — extracted from PipelineStep.agent_trace (gateway executor
    only; OpenClaw doesn't expose intermediate events, so agent_trace is always NULL for
    those steps and they're naturally excluded). Tool names are namespaced by the Gateway
    as "{server}__{tool}" (see MCPManager) — split on that to group by server too.

    Unlike every other Insights page, tool calls have no token/duration concept of their
    own (those belong to the step/LLM call as a whole) — "errors" (tool_result.is_error)
    takes the place of "failures" as the reliability signal.
    """
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    async with sf() as session:
        q = _production_only(
            select(
                PipelineRun.pipeline_name, PipelineStep.step_name, PipelineStep.agent,
                PipelineStep.run_id, PipelineStep.executed_at, PipelineStep.agent_trace,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.agent_trace.is_not(None))
            .order_by(PipelineStep.executed_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_rows = rows.all()

    # Per-row tool usage: {tool_name: {"calls": n, "errors": n}} extracted from that
    # step's trace. A single step can call several tools, each several times.
    row_tool_usage: list[tuple] = []
    for r in all_rows:
        try:
            events = json.loads(r.agent_trace)
        except (TypeError, ValueError):
            continue
        usage: dict[str, dict] = {}
        for event in events:
            if not isinstance(event, dict):
                continue
            name = event.get("name")
            if not name:
                continue
            t = event.get("type")
            if t == "tool_call":
                usage.setdefault(name, {"calls": 0, "errors": 0})["calls"] += 1
            elif t == "tool_result":
                u = usage.setdefault(name, {"calls": 0, "errors": 0})
                if event.get("is_error"):
                    u["errors"] += 1
        if usage:
            row_tool_usage.append((r, usage))

    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_rows:
        oldest = min(r.executed_at.replace(tzinfo=None) for r in all_rows)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    call_counts: dict[str, int] = defaultdict(int)
    error_counts: dict[str, int] = defaultdict(int)
    pipelines_by_tool: dict[str, set[str]] = defaultdict(set)
    calls_by_bucket_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    errors_by_bucket_tool: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    recent_by_tool: dict[str, list] = defaultdict(list)
    breakdown_combo: dict[tuple[str, str, str, str | None], dict] = {}

    for r, usage in row_tool_usage:
        bucket = _ts_bucket(r.executed_at, resolution)
        for tool, u in usage.items():
            call_counts[tool] += u["calls"]
            error_counts[tool] += u["errors"]
            pipelines_by_tool[tool].add(r.pipeline_name)
            calls_by_bucket_tool[tool][bucket] += u["calls"]
            errors_by_bucket_tool[tool][bucket] += u["errors"]
            if len(recent_by_tool[tool]) < 5:
                recent_by_tool[tool].append({
                    "id": str(r.run_id),
                    "pipeline_name": r.pipeline_name,
                    "step_name": r.step_name,
                    "calls": u["calls"],
                    "errors": u["errors"],
                    "ago": _format_ago(r.executed_at),
                })
            bkey = (tool, r.pipeline_name, r.step_name, r.agent)
            bc = breakdown_combo.setdefault(bkey, {"calls": 0, "errors": 0})
            bc["calls"] += u["calls"]
            bc["errors"] += u["errors"]

    # ── Per-tool pipeline/step/agent breakdown ─────────────────────────────────
    breakdown_by_tool: dict[str, list[dict]] = defaultdict(list)
    for (tool, pipeline_name, step_name, agent), c in breakdown_combo.items():
        calls = c["calls"]
        breakdown_by_tool[tool].append({
            "pipeline_name": pipeline_name,
            "step_name": step_name,
            "agent": agent,
            "calls": calls,
            "errors": c["errors"],
            "success_rate": round((calls - c["errors"]) / calls * 100) if calls else None,
        })
    for rows_ in breakdown_by_tool.values():
        rows_.sort(key=lambda r: r["calls"], reverse=True)

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for tool, calls in call_counts.items():
        errors = error_counts.get(tool, 0)
        success_rate = round((calls - errors) / calls * 100) if calls else None
        drilldown_data[tool] = {
            "calls": calls,
            "errors": errors,
            "success_rate": success_rate,
            "calls_ts": {"labels": bucket_labels, "data": [calls_by_bucket_tool[tool].get(b, 0) for b in bucket_labels]},
            "errors_ts": {"labels": bucket_labels, "data": [errors_by_bucket_tool[tool].get(b, 0) for b in bucket_labels]},
            "recent": recent_by_tool.get(tool, []),
            "breakdown": breakdown_by_tool.get(tool, []),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_calls = sum(call_counts.values())
    total_errors = sum(error_counts.values())
    overall_success_rate = round((total_calls - total_errors) / total_calls * 100) if total_calls else None
    distinct_servers = len({tool.partition("__")[0] for tool in call_counts})

    # ── Chart data ────────────────────────────────────────────────────────────
    tool_rows = []
    for tool, calls in sorted(call_counts.items(), key=lambda t: t[1], reverse=True):
        server, _, bare_tool = tool.partition("__")
        errors = error_counts.get(tool, 0)
        tool_rows.append({
            "name": tool,
            "server": server,
            "tool": bare_tool or tool,
            "calls": calls,
            "errors": errors,
            "success_rate": round((calls - errors) / calls * 100) if calls else None,
            "pipelines": sorted(pipelines_by_tool.get(tool, [])),
        })

    calls_by_server: dict[str, int] = defaultdict(int)
    for tool, calls in call_counts.items():
        calls_by_server[tool.partition("__")[0]] += calls
    server_chart_rows = sorted(calls_by_server.items(), key=lambda t: t[1], reverse=True)
    server_chart = {
        "labels": [name for name, _ in server_chart_rows],
        "data": [n for _, n in server_chart_rows],
    }

    calls_ts_rows = [(r.executed_at, tool, u["calls"]) for r, usage in row_tool_usage for tool, u in usage.items()]
    errors_ts_rows = [(r.executed_at, tool, u["errors"]) for r, usage in row_tool_usage for tool, u in usage.items()]
    calls_ts = _build_ts(calls_ts_rows, now, cutoff, time_range, dim_fn=lambda r: r[1], val_fn=lambda r: r[2])
    errors_ts = _build_ts(errors_ts_rows, now, cutoff, time_range, dim_fn=lambda r: r[1], val_fn=lambda r: r[2])

    return templates.TemplateResponse(request, "insights_mcp.html", {
        "time_range": time_range,
        "range_label": range_label,
        "tool_rows": tool_rows,
        "server_chart": server_chart,
        "calls_ts": calls_ts,
        "errors_ts": errors_ts,
        "total_tool_count": len(call_counts),
        "total_calls": total_calls,
        "total_errors": total_errors,
        "overall_success_rate": overall_success_rate,
        "distinct_servers": distinct_servers,
        "drilldown_data": drilldown_data,
        "active_page": "insights_mcp",
    })


@router.get("/insights/teams", response_class=HTMLResponse)
async def ui_insights_teams(request: Request, time_range: str = "7d"):
    """Per-team rollup — which pipelines/steps/agents/models a team uses, and its token
    spend, so a team can see a complete picture of what they're running and make informed
    cost decisions. `team` is a PipelineRun attribute (resolved from the webhook auth
    token — see README §3b); NULL is bucketed as "Unattributed" rather than dropped, since
    unattributed spend is exactly the kind of thing this page exists to surface.

    Run counts/success-rate/duration come from PipelineRun directly (a "run" is a run,
    not a step); tokens and the "what they're using" breakdown come from the shared
    per-step combo (see _fetch_step_agent_model_combo), which also carries `team`.
    """
    cutoff, range_label = _time_range_cutoff(time_range)
    sf = get_session_factory()

    def norm_team(t: str | None) -> str:
        return t or "Unattributed"

    async with sf() as session:
        q = _production_only(select(PipelineRun.team, func.count().label("n")).group_by(PipelineRun.team))
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        run_counts_raw = rows.all()

        q = _production_only(
            select(PipelineRun.team, func.count().label("n"))
            .where(PipelineRun.status == "failed")
            .group_by(PipelineRun.team)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        failed_counts_raw = rows.all()

        q = _production_only(
            select(PipelineRun.team, PipelineRun.status, func.count().label("n"))
            .group_by(PipelineRun.team, PipelineRun.status)
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        status_counts_raw = rows.all()

        q = _production_only(select(PipelineRun.team, PipelineRun.pipeline_name).distinct())
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        pipelines_by_team_raw = rows.all()

        # No duration column on pipeline_runs — compute from timestamps, same approach as
        # ui_insights_pipelines. Only terminal runs (completed_at set) count.
        q = _production_only(
            select(PipelineRun.team, PipelineRun.triggered_at, PipelineRun.completed_at)
            .where(PipelineRun.completed_at.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        duration_rows_raw = rows.all()

        # All runs in range — used for per-team timeseries and recent-runs list
        q = _production_only(
            select(
                PipelineRun.id, PipelineRun.team, PipelineRun.pipeline_name,
                PipelineRun.status, PipelineRun.triggered_at, PipelineRun.completed_at,
            )
            .order_by(PipelineRun.triggered_at.desc())
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        all_runs_raw = rows.all()

        # Per-step token usage with timestamps — for the tokens-over-time chart (tokens
        # live on PipelineStep, not PipelineRun, so this needs its own query).
        q = _production_only(
            select(PipelineRun.team, PipelineStep.executed_at, PipelineStep.input_tokens, PipelineStep.output_tokens)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.input_tokens.is_not(None))
        )
        if cutoff:
            q = q.where(PipelineRun.triggered_at >= cutoff)
        rows = await session.execute(q)
        token_ts_raw = rows.all()

    # step_combo (shared with Pipelines/Steps/Agents Insights) now also carries team —
    # re-aggregated here indexed by team for tokens and the "what they're using" breakdown.
    step_combo = await _fetch_step_agent_model_combo(cutoff)

    run_counts: dict[str, int] = defaultdict(int)
    for team, n in run_counts_raw:
        run_counts[norm_team(team)] += n

    failed_counts: dict[str, int] = defaultdict(int)
    for team, n in failed_counts_raw:
        failed_counts[norm_team(team)] += n

    status_counts_by_team: dict[str, dict[str, int]] = defaultdict(dict)
    for team, status, n in status_counts_raw:
        tk = norm_team(team)
        status_counts_by_team[tk][status] = status_counts_by_team[tk].get(status, 0) + n

    pipelines_by_team: dict[str, set[str]] = defaultdict(set)
    for team, pipeline_name in pipelines_by_team_raw:
        pipelines_by_team[norm_team(team)].add(pipeline_name)

    durations_by_team: dict[str, list[float]] = defaultdict(list)
    for team, triggered_at, completed_at in duration_rows_raw:
        secs = (completed_at.replace(tzinfo=None) - triggered_at.replace(tzinfo=None)).total_seconds()
        if secs >= 0:
            durations_by_team[norm_team(team)].append(secs)
    avg_duration_by_team = {t: sum(v) / len(v) for t, v in durations_by_team.items() if v}

    token_totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    breakdown_combo_by_team: dict[str, list] = defaultdict(list)
    for (team, pipeline_name, step_name, agent, provider, model), c in step_combo.items():
        tk = norm_team(team)
        token_totals[tk][0] += c["input_tokens"]
        token_totals[tk][1] += c["output_tokens"]
        breakdown_combo_by_team[tk].append((pipeline_name, step_name, agent, provider, model, c))

    breakdown_by_team: dict[str, list[dict]] = defaultdict(list)
    for tk, entries in breakdown_combo_by_team.items():
        for pipeline_name, step_name, agent, provider, model, c in entries:
            total = c["total"]
            avg_duration_secs = (c["duration_sum_ms"] / c["duration_n"] / 1000) if c["duration_n"] else None
            breakdown_by_team[tk].append({
                "pipeline_name": pipeline_name,
                "step_name": step_name,
                "agent": agent,
                "model": _qualified_model(provider, model),
                "total": total,
                "success_rate": round((total - c["failed"]) / total * 100) if total else None,
                "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
                "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
                "avg_duration": _format_seconds(avg_duration_secs),
            })
    for rows_ in breakdown_by_team.values():
        rows_.sort(key=lambda r: r["total"], reverse=True)

    # ── Timeseries + recent-list buckets ───────────────────────────────────────
    now = utc_now()
    resolution = _ts_resolution(time_range)
    if all_runs_raw:
        oldest = min(r.triggered_at.replace(tzinfo=None) for r in all_runs_raw)
    else:
        oldest = (cutoff or (now - timedelta(days=7)))
    ts_start = cutoff or oldest
    bucket_labels = _ts_all_buckets(ts_start, now, resolution)

    runs_by_bucket_team: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations_by_bucket_team: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    recent_by_team: dict[str, list] = defaultdict(list)

    for r in all_runs_raw:
        tk = norm_team(r.team)
        bucket = _ts_bucket(r.triggered_at, resolution)
        runs_by_bucket_team[tk][bucket] += 1
        if r.completed_at:
            secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
            if secs >= 0:
                durations_by_bucket_team[tk][bucket].append(secs)
        if len(recent_by_team[tk]) < 5:
            dur_str = None
            if r.completed_at:
                secs = (r.completed_at.replace(tzinfo=None) - r.triggered_at.replace(tzinfo=None)).total_seconds()
                dur_str = _format_seconds(secs)
            recent_by_team[tk].append({
                "id": str(r.id), "pipeline_name": r.pipeline_name, "status": r.status,
                "ago": _format_ago(r.triggered_at), "duration": dur_str,
            })

    # ── Drilldown payload (serialised to JSON for JS) ─────────────────────────
    drilldown_data: dict[str, dict] = {}
    for name, n in run_counts.items():
        sc = status_counts_by_team.get(name, {})
        failed = failed_counts.get(name, 0)
        escalated = sc.get("escalated", 0)
        success_rate = round((n - failed) / n * 100) if n else None
        escalation_rate = round(escalated / n * 100) if n else None
        avg_dur = avg_duration_by_team.get(name)
        inp, out = token_totals.get(name, [0, 0])

        runs_ts_data = [runs_by_bucket_team[name].get(b, 0) for b in bucket_labels]
        duration_ts_data = [
            round(sum(durations_by_bucket_team[name][b]) / len(durations_by_bucket_team[name][b]))
            if durations_by_bucket_team[name].get(b) else None
            for b in bucket_labels
        ]

        drilldown_data[name] = {
            "run_count": n,
            "failed_count": failed,
            "success_rate": success_rate,
            "escalation_rate": escalation_rate,
            "avg_duration": _format_seconds(avg_dur),
            "input_tokens": inp,
            "output_tokens": out,
            "status_breakdown": sc,
            "runs_ts": {"labels": bucket_labels, "data": runs_ts_data},
            "duration_ts": {"labels": bucket_labels, "data": duration_ts_data},
            "recent_runs": recent_by_team.get(name, []),
            "step_breakdown": breakdown_by_team.get(name, []),
            "pipelines": sorted(pipelines_by_team.get(name, [])),
        }

    # ── Headline stats ────────────────────────────────────────────────────────
    total_runs = sum(run_counts.values())
    total_failed = sum(failed_counts.values())
    overall_success_rate = round((total_runs - total_failed) / total_runs * 100) if total_runs else None
    total_input_tokens = sum(v[0] for v in token_totals.values())
    total_output_tokens = sum(v[1] for v in token_totals.values())
    total_pipelines_used = len({p for pset in pipelines_by_team.values() for p in pset})

    # ── Chart data ────────────────────────────────────────────────────────────
    team_rows = []
    for name, n in sorted(run_counts.items(), key=lambda t: t[1], reverse=True):
        inp, out = token_totals.get(name, [0, 0])
        team_rows.append({
            "name": name,
            "run_count": n,
            "failed_count": failed_counts.get(name, 0),
            "avg_duration_secs": avg_duration_by_team.get(name),
            "input_tokens": inp,
            "output_tokens": out,
            "pipelines": sorted(pipelines_by_team.get(name, [])),
        })

    token_chart_rows = sorted(
        (r for r in team_rows if r["input_tokens"] or r["output_tokens"]),
        key=lambda r: r["input_tokens"] + r["output_tokens"], reverse=True,
    )
    token_chart = {
        "labels": [r["name"] for r in token_chart_rows],
        "input": [r["input_tokens"] for r in token_chart_rows],
        "output": [r["output_tokens"] for r in token_chart_rows],
    }

    runs_ts = _build_ts(
        [(r.triggered_at, norm_team(r.team)) for r in all_runs_raw], now, cutoff, time_range,
        dim_fn=lambda r: r[1],
    )
    tokens_ts = _build_ts(
        [(r.executed_at, norm_team(r.team), (r.input_tokens or 0) + (r.output_tokens or 0)) for r in token_ts_raw],
        now, cutoff, time_range,
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    return templates.TemplateResponse(request, "insights_teams.html", {
        "time_range": time_range,
        "range_label": range_label,
        "team_rows": team_rows,
        "token_chart": token_chart,
        "runs_ts": runs_ts,
        "tokens_ts": tokens_ts,
        "total_team_count": len(run_counts),
        "total_runs": total_runs,
        "overall_success_rate": overall_success_rate,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_pipelines_used": total_pipelines_used,
        "drilldown_data": drilldown_data,
        "active_page": "insights_teams",
    })


@router.get("/pipelines/{name}", response_class=HTMLResponse)
async def ui_pipeline_detail(request: Request, name: str):
    pipelines = getattr(request.app.state, "pipelines", [])
    pipeline = next((p for p in pipelines if p.name == name), None)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    pipeline_dir = getattr(request.app.state, "pipeline_dir", "./pipelines")
    raw_yaml = _read_pipeline_yaml(pipeline_dir, name) or ""

    sf = get_session_factory()
    async with sf() as session:
        # Browse surface — shows testing runs too (badged in the template).
        rows = await session.execute(
            select(PipelineRun)
            .where(PipelineRun.pipeline_name == name)
            .order_by(PipelineRun.triggered_at.desc())
            .limit(10)
        )
        recent_runs = rows.scalars().all()
        feedback_by_run = await _feedback_by_run_id(session, [r.id for r in recent_runs])

        # Success-rate / accuracy bars below are rollups — scoped to production.
        rows = await session.execute(_production_only(
            select(PipelineRun.status, func.count().label("n"))
            .where(PipelineRun.pipeline_name == name)
            .group_by(PipelineRun.status)
        ))
        status_counts = dict(rows.all())

        rows = await session.execute(_production_only(
            select(RunFeedback.outcome, func.count().label("n"))
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(RunFeedback.pipeline_name == name)
            .group_by(RunFeedback.outcome)
        ))
        feedback_counts = dict(rows.all())

    feedback_total = sum(feedback_counts.values())

    promotion_readiness = None
    readiness_builder_seed = None
    if pipeline.stage == "testing":
        readiness_evidence = await _gather_readiness_evidence(sf, pipeline)
        promotion_readiness = _evaluate_readiness(readiness_evidence, pipeline)
        readiness_builder_seed = _builder_seed(pipeline)

    # Group this pipeline's agent usage by executor:agent, then enrich with each
    # agent's live-configured model + fallbacks (only known from the backend's
    # own config, unlike everything else here which comes straight from the
    # pipeline YAML — see _agent_usage_in_pipeline).
    usage = _agent_usage_in_pipeline(pipeline)
    agents_grouped: dict[str, dict] = {}
    for u in usage:
        g = agents_grouped.setdefault(u["key"], {
            "executor": u["executor"], "agent": u["agent"], "roles": set(), "steps": [],
        })
        g["roles"].add(u["role"])
        if u["step"] not in g["steps"]:
            g["steps"].append(u["step"])

    live_by_key: dict[str, dict] = {}
    if agents_grouped:
        (oc_agents, _), (gw_agents, _) = await asyncio.gather(
            _fetch_openclaw_agents(), _fetch_vectorstep_gateway_agents(),
        )
        for a in oc_agents:
            live_by_key[f"openclaw:{a.get('id') or a.get('name')}"] = a
        for a in gw_agents:
            live_by_key[f"gateway:{a.get('id') or a.get('name')}"] = a

    pipeline_agents = []
    for key, g in sorted(agents_grouped.items()):
        live = live_by_key.get(key)
        pipeline_agents.append({
            "key": key,
            "executor": g["executor"],
            "agent": g["agent"],
            "roles": sorted(g["roles"]),
            "steps": g["steps"],
            "found": live is not None,
            "model": live.get("model") if live else None,
            "model_fallbacks": (live.get("model_fallbacks") or []) if live else [],
        })

    return templates.TemplateResponse(request, "pipeline_detail.html", {
        "pipeline": pipeline,
        "raw_yaml": raw_yaml,
        "recent_runs": recent_runs,
        "feedback_by_run": feedback_by_run,
        "status_counts": status_counts,
        "total_runs": sum(status_counts.values()),
        "feedback_counts": feedback_counts,
        "feedback_total": feedback_total,
        "pipeline_agents": pipeline_agents,
        "promotion_readiness": promotion_readiness,
        "readiness_builder_seed": readiness_builder_seed,
        "readiness_knob_help": READINESS_KNOB_HELP,
        "active_page": "pipelines",
    })


@router.get("/pipelines/{name}/feedback", response_class=HTMLResponse)
async def ui_pipeline_feedback(request: Request, name: str):
    pipelines = getattr(request.app.state, "pipelines", [])
    pipeline = next((p for p in pipelines if p.name == name), None)
    if not pipeline:
        raise HTTPException(status_code=404, detail="Pipeline not found")

    sf = get_session_factory()
    async with sf() as session:
        # Chronological browse list — includes testing runs, badged (see stage below).
        # The config-fingerprint comparison further down is production-only, since
        # mixing testing and production runs would corrupt "compare accuracy before
        # vs after a pipeline change."
        rows = await session.execute(
            select(RunFeedback, PipelineRun.triggered_at, PipelineRun.status, PipelineRun.stage)
            .join(PipelineRun, RunFeedback.run_id == PipelineRun.id)
            .where(RunFeedback.pipeline_name == name)
            .order_by(RunFeedback.submitted_at.desc())
        )
        feedback_rows = rows.all()

        run_ids = [fb.run_id for fb, _, _, _ in feedback_rows]
        steps_by_run: dict[str, list] = defaultdict(list)
        if run_ids:
            step_rows = await session.execute(
                select(PipelineStep)
                .where(PipelineStep.run_id.in_(run_ids))
                .order_by(PipelineStep.step_index)
            )
            for step in step_rows.scalars().all():
                steps_by_run[step.run_id].append(step)

    # Build per-run records annotated with config fingerprint
    marked_runs = []
    for fb, triggered_at, run_status, stage in feedback_rows:
        steps = steps_by_run.get(fb.run_id, [])
        fp = _config_fingerprint(steps) if steps else "unknown"
        desc = _config_description(steps) if steps else {"steps": [], "models": []}
        marked_runs.append({
            "run_id": fb.run_id,
            "outcome": fb.outcome,
            "notes": fb.notes,
            "submitted_at": fb.submitted_at,
            "triggered_at": triggered_at,
            "run_status": run_status,
            "stage": stage,
            "fingerprint": fp,
            "config": desc,
        })

    # Group by config fingerprint — summary cards and this comparison are rollups,
    # scoped to production only (unlike the chronological marked_runs list above).
    production_marked_runs = [r for r in marked_runs if r["stage"] == "production"]
    config_groups: dict[str, dict] = {}
    for r in production_marked_runs:
        fp = r["fingerprint"]
        if fp not in config_groups:
            config_groups[fp] = {
                "fingerprint": fp,
                "config": r["config"],
                "runs": [],
                "correct": 0, "partial": 0, "incorrect": 0,
                "first_seen": r["triggered_at"],
                "last_seen": r["triggered_at"],
            }
        g = config_groups[fp]
        g["runs"].append(r)
        g[r["outcome"]] = g.get(r["outcome"], 0) + 1
        if r["triggered_at"] and (g["first_seen"] is None or r["triggered_at"] < g["first_seen"]):
            g["first_seen"] = r["triggered_at"]
        if r["triggered_at"] and (g["last_seen"] is None or r["triggered_at"] > g["last_seen"]):
            g["last_seen"] = r["triggered_at"]

    # Sort groups: most recently active first
    sorted_groups = sorted(
        config_groups.values(),
        key=lambda g: g["last_seen"] or datetime.min,
        reverse=True,
    )
    for g in sorted_groups:
        total = g["correct"] + g["partial"] + g["incorrect"]
        g["total"] = total
        g["correct_pct"] = round(g["correct"] / total * 100) if total else 0

    total_marked = len(production_marked_runs)
    total_correct = sum(1 for r in production_marked_runs if r["outcome"] == "correct")
    total_partial = sum(1 for r in production_marked_runs if r["outcome"] == "partial")
    total_incorrect = sum(1 for r in production_marked_runs if r["outcome"] == "incorrect")

    return templates.TemplateResponse(request, "pipeline_feedback.html", {
        "pipeline": pipeline,
        "marked_runs": marked_runs,
        "config_groups": sorted_groups,
        "total_marked": total_marked,
        "total_correct": total_correct,
        "total_partial": total_partial,
        "total_incorrect": total_incorrect,
        "active_page": "pipelines",
    })


@router.get("/marking-queue", response_class=HTMLResponse)
async def ui_marking_queue(
    request: Request,
    pipeline: str | None = None,
    team: str | None = None,
    stage: str = "testing",
):
    """Cross-pipeline review queue: every step with no HUMAN accuracy feedback
    (StepFeedback) — the exact gap `readiness.accuracy.min_human_marked` checks
    for. A step that already has an automatic label (a failed deterministic
    check, or an inherited run-level rating) still shows up here, tagged with
    where that label came from, since neither counts as `human_marked` — see
    CONFIDENCE-EXPLAINED.md §12.2/§9.

    Filters are additive (pipeline AND team AND stage, all optional). stage
    defaults to "testing" — the pre-promotion review case — but production
    pipelines using calibration independently of the testing->production gate
    need this too, so stage is a real filter, not a hard scope."""
    def _filtered(q):
        if pipeline:
            q = q.where(PipelineRun.pipeline_name == pipeline)
        if team:
            q = q.where(PipelineRun.team == team)
        if stage:
            q = q.where(PipelineRun.stage == stage)
        return q

    sf = get_session_factory()
    async with sf() as session:
        q = _filtered(
            select(
                PipelineStep.run_id, PipelineStep.step_name, PipelineStep.executed_at,
                PipelineStep.deterministic_passed,
                PipelineRun.pipeline_name, PipelineRun.team, PipelineRun.stage,
            )
            .select_from(PipelineStep)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .outerjoin(StepFeedback, StepFeedback.step_id == PipelineStep.id)
            .where(StepFeedback.id.is_(None))
            .where(PipelineStep.status.in_(ALL_STEP_STATUSES))
            .order_by(PipelineStep.executed_at.asc())
        )
        unmarked_rows = (await session.execute(q)).all()

        run_feedback_ids: set[str] = set()
        run_ids = {r.run_id for r in unmarked_rows}
        if run_ids:
            rf_rows = await session.execute(
                select(RunFeedback.run_id).where(RunFeedback.run_id.in_(run_ids))
            )
            run_feedback_ids = {r[0] for r in rf_rows.all()}

        q = _filtered(
            select(func.count())
            .select_from(PipelineStep)
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .where(PipelineStep.status.in_(ALL_STEP_STATUSES))
        )
        total_terminal_steps = (await session.execute(q)).scalar() or 0

        # Unfiltered so switching one filter doesn't collapse the others' options.
        rows = await session.execute(select(PipelineRun.pipeline_name).distinct().order_by(PipelineRun.pipeline_name))
        pipeline_names = [r[0] for r in rows.all()]
        rows = await session.execute(
            select(PipelineRun.team).distinct().where(PipelineRun.team.is_not(None)).order_by(PipelineRun.team)
        )
        team_names = [r[0] for r in rows.all()]

    # Group: pipeline -> step-group (fan-out/parallel branches collapse to their
    # group name, same as readiness._bucket_name) -> individual unmarked items,
    # oldest first (rows are already ordered that way from the query above).
    pipelines_by_name: dict[str, dict] = {}
    for r in unmarked_rows:
        if r.deterministic_passed is False:
            provenance = "failed check"
        elif r.run_id in run_feedback_ids:
            provenance = "run feedback"
        else:
            provenance = None

        p = pipelines_by_name.setdefault(r.pipeline_name, {
            "name": r.pipeline_name, "stage": r.stage, "steps": {}, "run_ids": set(),
        })
        p["run_ids"].add(r.run_id)
        step_group = r.step_name.split("/", 1)[0]
        s = p["steps"].setdefault(step_group, {"name": step_group, "items": []})
        s["items"].append({"run_id": r.run_id, "executed_at": r.executed_at, "provenance": provenance})

    pipeline_list = []
    total_unmarked_steps = 0
    for p in pipelines_by_name.values():
        step_list = []
        for s in p["steps"].values():
            s["count"] = len(s["items"])
            s["oldest"] = s["items"][0]["executed_at"]
            s["preview"] = s["items"][:5]
            s["more"] = max(0, s["count"] - 5)
            step_list.append(s)
        step_list.sort(key=lambda s: s["count"], reverse=True)
        step_count = sum(s["count"] for s in step_list)
        total_unmarked_steps += step_count
        pipeline_list.append({
            "name": p["name"], "stage": p["stage"], "steps": step_list,
            "step_count": step_count, "run_count": len(p["run_ids"]),
        })
    pipeline_list.sort(key=lambda p: p["step_count"], reverse=True)

    marked_pct = (
        round((total_terminal_steps - total_unmarked_steps) / total_terminal_steps * 100)
        if total_terminal_steps else None
    )

    return templates.TemplateResponse(request, "marking_queue.html", {
        "pipelines": pipeline_list,
        "pipeline_names": pipeline_names,
        "team_names": team_names,
        "selected_pipeline": pipeline or "",
        "selected_team": team or "",
        "selected_stage": stage or "",
        "stage_values": ["testing", "production"],
        "total_pipelines": len(pipeline_list),
        "total_runs": len({r.run_id for r in unmarked_rows}),
        "total_unmarked_steps": total_unmarked_steps,
        "total_terminal_steps": total_terminal_steps,
        "marked_pct": marked_pct,
        "active_page": "marking_queue",
    })


@router.get("/schedules", response_class=HTMLResponse)
async def ui_schedules(request: Request):
    scheduler = getattr(request.app.state, "scheduler", None)
    pipelines = getattr(request.app.state, "pipelines", [])

    jobs = []
    if scheduler:
        for job in scheduler.get_jobs():
            if not job.id.startswith("pipeline:"):
                continue
            p = next((p for p in pipelines if p.name == job.name), None)
            jobs.append({
                "name": job.name,
                "cron": p.schedule.cron if p and p.schedule else "—",
                "summary": p.schedule.summary if p and p.schedule else "",
                "severity": p.schedule.severity if p and p.schedule else "",
                "next_run": job.next_run_time,
            })

    return templates.TemplateResponse(request, "schedules.html", {
        "jobs": jobs,
        "active_page": "schedules",
    })



# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_pipeline_yaml(pipeline_dir: str, pipeline_name: str) -> str | None:
    for path in glob.glob(os.path.join(pipeline_dir, "*.yaml")):
        try:
            with open(path) as f:
                content = f.read()
            data = yaml.safe_load(content)
            if isinstance(data, dict) and data.get("name") == pipeline_name:
                return content
        except Exception:
            continue
    return None


def _read_step_yaml(steps_dir: str, step_name: str) -> str | None:
    """Same lookup as _read_pipeline_yaml, for the step library directory."""
    for path in glob.glob(os.path.join(steps_dir, "*.yaml")):
        try:
            with open(path) as f:
                content = f.read()
            data = yaml.safe_load(content)
            if isinstance(data, dict) and data.get("name") == step_name:
                return content
        except Exception:
            continue
    return None


# ── Step library ─────────────────────────────────────────────────────────────

def _iter_all_raw_steps(steps: list):
    """Yield every step dict from a raw YAML steps list, including parallel inner steps."""
    for step in steps:
        if isinstance(step, dict) and "parallel" in step:
            yield from _iter_all_raw_steps(step["parallel"].get("steps", []))
        else:
            yield step


def _compute_step_usage(pipeline_dir: str, library: dict) -> dict[str, list[str]]:
    """Scan pipeline YAMLs to find which pipelines reference each library step."""
    usage: dict[str, list[str]] = {name: [] for name in library}
    for path in glob.glob(os.path.join(pipeline_dir, "*.yaml")):
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
            pipeline_name = raw.get("name", os.path.basename(path))
            for step in _iter_all_raw_steps(raw.get("steps", [])):
                use = step.get("use") if isinstance(step, dict) else None
                if use and use in usage:
                    usage[use].append(pipeline_name)
        except Exception:
            pass
    return usage


def _compute_step_runtime_names(pipeline_dir: str, library: dict) -> dict[str, set[str]]:
    """Map each library step name to the runtime `step_name`(s) it executes under.

    A pipeline step that does `use: <library-name>` inherits the library step's
    `name` field unless it overrides `name:` locally (see
    `pipeline.loader._resolve_step_references`) — so the DB's `pipeline_steps.step_name`
    usually matches the library name directly, but can differ if a pipeline renames it.
    Always includes the library name itself so unused/never-renamed steps still match.
    """
    runtime_names: dict[str, set[str]] = {name: {name} for name in library}
    for path in glob.glob(os.path.join(pipeline_dir, "*.yaml")):
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)
            for step in _iter_all_raw_steps(raw.get("steps", [])):
                if not isinstance(step, dict):
                    continue
                use = step.get("use")
                if use and use in runtime_names:
                    runtime_names[use].add(step.get("name") or use)
        except Exception:
            pass
    return runtime_names


async def _fetch_step_model_stats(
    runtime_names: dict[str, set[str]],
) -> dict[str, list[dict]]:
    """Per-library-step breakdown of run history by (pipeline, agent, model): success rate
    and average token usage, aggregated across every runtime step_name the library step is
    known to execute under (see _compute_step_runtime_names). Scoped to production runs
    only, matching the rest of the app's rollup surfaces (see _production_only).

    Grouped by pipeline as well as agent/model — a step used by several pipelines can be
    wired to a different agent/model in each, and folding them together would hide that."""
    all_names = {n for names in runtime_names.values() for n in names}
    if not all_names:
        return {}

    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.step_name,
                    PipelineRun.pipeline_name,
                    PipelineStep.agent,
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.status,
                    func.count().label("n"),
                    func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                    func.coalesce(func.sum(PipelineStep.output_tokens), 0),
                    func.max(PipelineStep.executed_at).label("last_run"),
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.step_name.in_(all_names))
                .group_by(
                    PipelineStep.step_name, PipelineRun.pipeline_name, PipelineStep.agent,
                    PipelineStep.model, PipelineStep.provider, PipelineStep.status,
                )
            )
        )
        db_rows = rows.all()

    # (step_name, pipeline_name, agent, qualified_model) -> aggregated counters. Qualifying
    # the model with its provider (see _qualified_model) up front means two providers that
    # happen to report the same bare model string aren't silently merged together.
    combo_stats: dict[tuple[str, str, str | None, str], dict] = {}
    for step_name, pipeline_name, agent, model, provider, status, n, in_tok, out_tok, last_run in db_rows:
        key = (step_name, pipeline_name, agent, _qualified_model(provider, model))
        c = combo_stats.setdefault(key, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "last_run": None,
        })
        c["total"] += n
        if status == "failed":
            c["failed"] += n
        c["input_tokens"] += in_tok
        c["output_tokens"] += out_tok
        if last_run and (c["last_run"] is None or last_run > c["last_run"]):
            c["last_run"] = last_run

    result: dict[str, list[dict]] = {}
    for lib_name, names in runtime_names.items():
        # Re-aggregate by (pipeline, agent, model) across every runtime name for this
        # library step.
        by_pipeline_agent_model: dict[tuple[str, str | None, str], dict] = {}
        for (step_name, pipeline_name, agent, model), c in combo_stats.items():
            if step_name not in names:
                continue
            row = by_pipeline_agent_model.setdefault(
                (pipeline_name, agent, model),
                {"total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "last_run": None},
            )
            row["total"] += c["total"]
            row["failed"] += c["failed"]
            row["input_tokens"] += c["input_tokens"]
            row["output_tokens"] += c["output_tokens"]
            if c["last_run"] and (row["last_run"] is None or c["last_run"] > row["last_run"]):
                row["last_run"] = c["last_run"]

        if not by_pipeline_agent_model:
            continue

        rows_out = []
        for (pipeline_name, agent, model), row in by_pipeline_agent_model.items():
            total = row["total"]
            rows_out.append({
                "pipeline_name": pipeline_name,
                "agent": agent,
                "model": model,
                "total": total,
                "success_rate": round((total - row["failed"]) / total * 100) if total else None,
                "avg_input_tokens": round(row["input_tokens"] / total) if total else None,
                "avg_output_tokens": round(row["output_tokens"] / total) if total else None,
                "last_run": row["last_run"],
            })
        rows_out.sort(key=lambda r: r["total"], reverse=True)
        result[lib_name] = rows_out

    return result


@router.get("/steps", response_class=HTMLResponse)
async def ui_steps(request: Request, tag: str | None = None):
    step_library: dict = getattr(request.app.state, "step_library", {})
    pipeline_dir: str = getattr(request.app.state, "pipeline_dir", "./pipelines")

    step_usage = _compute_step_usage(pipeline_dir, step_library)
    runtime_names = _compute_step_runtime_names(pipeline_dir, step_library)
    step_model_stats = await _fetch_step_model_stats(runtime_names)
    all_steps = sorted(step_library.values(), key=lambda s: s.get("name", ""))
    all_tags = sorted({t for s in all_steps for t in s.get("tags") or []})
    steps = [s for s in all_steps if tag in (s.get("tags") or [])] if tag else all_steps

    return templates.TemplateResponse(request, "steps.html", {
        "steps": steps,
        "step_usage": step_usage,
        "step_model_stats": step_model_stats,
        "all_tags": all_tags,
        "selected_tag": tag or "",
        "active_page": "steps",
    })


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


def _first_description_line(content: str | None) -> str | None:
    """Return the first non-heading paragraph from soul/description content."""
    if not content:
        return None
    paragraph: list[str] = []
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if paragraph:
                break
            continue
        if stripped:
            paragraph.append(stripped)
        elif paragraph:
            break
    return " ".join(paragraph) or None


# ── Per-executor agent discovery ──────────────────────────────────────────────

async def _fetch_openclaw_agents() -> tuple[list[dict], str | None]:
    """Fetch agents list from the OpenClaw Gateway WS API (agents.list RPC)."""
    if not _openclaw_enabled:
        return [], None
    result = await gateway_call_safe("agents.list", {}, gateway_url=_openclaw_ws_url)
    if result is None:
        return [], f"Could not reach OpenClaw Gateway at {_openclaw_ws_url} — is it running?"
    return result.get("agents") or [], None


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


async def _dashboard_status_panels() -> dict:
    """Backend/MCP/model status for the dashboard's status panels.

    Deliberately reuses the same reachability checks as /ui/agents and /ui/mcp rather
    than a fabricated per-provider (Anthropic/OpenRouter/...) health dot — VectorStep never
    calls providers directly, so "online" here means "the Gateway that talks to them
    responded," which is the only thing this service can honestly claim to know.
    """
    (gw_agents, gw_error), (oc_agents, oc_error), (mcp_tools, mcp_servers, mcp_error) = await asyncio.gather(
        _fetch_vectorstep_gateway_agents(),
        _fetch_openclaw_agents(),
        _fetch_vectorstep_gateway_mcp(),
    )

    backends = [{
        "name": "VectorStep Gateway",
        "online": gw_error is None,
        "agent_count": len(gw_agents),
        "error": gw_error,
    }]
    if _openclaw_enabled:
        backends.append({
            "name": "OpenClaw",
            "online": oc_error is None,
            "agent_count": len(oc_agents),
            "error": oc_error,
        })

    mcp_server_rows = [
        {
            "name": name,
            "running": (mcp_servers.get(name) or {}).get("running"),
            "restart_count": (mcp_servers.get(name) or {}).get("restart_count", 0),
        }
        for name in sorted(set(mcp_tools) | set(mcp_servers))
    ]
    mcp_running_count = sum(1 for s in mcp_server_rows if s["running"])

    # "Configured" = each agent's live primary model, not its failover fallbacks —
    # fallbacks only ever run when the primary fails, so folding them in here would
    # overstate what's actually driving day-to-day traffic.
    models_by_provider: dict[str, Counter] = defaultdict(Counter)
    for a in gw_agents + oc_agents:
        model = a.get("model")
        if not model:
            continue
        provider, sep, bare = model.partition("/")
        if not sep:
            provider, bare = "unspecified", model
        models_by_provider[provider][bare] += 1

    models_configured = [
        {"provider": provider, "models": [{"name": m, "agent_count": n} for m, n in counter.most_common()]}
        for provider, counter in sorted(models_by_provider.items())
    ]
    total_model_count = sum(len(counter) for counter in models_by_provider.values())

    return {
        "backends": backends,
        "mcp_server_rows": mcp_server_rows,
        "mcp_running_count": mcp_running_count,
        "mcp_error": mcp_error,
        "models_configured": models_configured,
        "total_model_count": total_model_count,
    }


async def _fetch_openclaw_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch SOUL.md / TOOLS.md / IDENTITY.md from the OpenClaw Gateway WS API."""
    def _content(payload: dict | None) -> str | None:
        return ((payload or {}).get("file") or {}).get("content") or None

    soul_r, tools_r, identity_r = await asyncio.gather(
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "SOUL.md"}, gateway_url=_openclaw_ws_url),
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "TOOLS.md"}, gateway_url=_openclaw_ws_url),
        gateway_call_safe("agents.files.get", {"agentId": agent_id, "name": "IDENTITY.md"}, gateway_url=_openclaw_ws_url),
    )
    return {
        "soul": _content(soul_r),
        "tools": _content(tools_r),
        "identity": _content(identity_r),
    }


async def _fetch_vectorstep_gateway_agent_files(agent_id: str) -> dict[str, str | None]:
    """Fetch soul and agent.yaml from the VectorStep Gateway REST API."""
    result: dict[str, str | None] = {"soul": None, "tools": None, "identity": None, "agent_file": None}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            soul_resp, agent_resp = await asyncio.gather(
                client.get(f"{_vectorstep_gateway_base}/agents/{agent_id}/soul"),
                client.get(f"{_vectorstep_gateway_base}/agents/{agent_id}/agent"),
            )
            if soul_resp.status_code == 200:
                result["soul"] = soul_resp.json().get("content") or soul_resp.text
            if agent_resp.status_code == 200:
                result["agent_file"] = agent_resp.json().get("content") or agent_resp.text
    except Exception:
        pass
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/agents", response_class=HTMLResponse)
async def ui_agents(request: Request, executor: str | None = None, model: str | None = None):
    # Fetch from both backends concurrently
    (oc_agents, oc_error), (gw_agents, gw_error) = await asyncio.gather(
        _fetch_openclaw_agents(),
        _fetch_vectorstep_gateway_agents(),
    )

    # Tag every entry with its executor source so the template and URL routing
    # can distinguish agents that share names across backends.
    all_agents: list[dict] = []
    for a in oc_agents:
        all_agents.append({**a, "executor": "openclaw"})
    for a in gw_agents:
        all_agents.append({**a, "executor": "gateway"})

    # Batch-fetch the first SOUL.md line for openclaw agents as a list-page preview.
    if oc_agents and _openclaw_enabled:
        oc_ids = [a.get("id") or a.get("name") for a in oc_agents]
        soul_results = await asyncio.gather(*[
            gateway_call_safe("agents.files.get", {"agentId": aid, "name": "SOUL.md"}, gateway_url=_openclaw_ws_url)
            for aid in oc_ids
        ], return_exceptions=True)
        for agent, soul_result in zip(all_agents[:len(oc_agents)], soul_results):
            if isinstance(soul_result, Exception) or soul_result is None:
                continue
            content = ((soul_result or {}).get("file") or {}).get("content")
            desc = _first_description_line(content)
            if desc:
                agent["description"] = desc

    # Batch-fetch soul descriptions for gateway agents.
    if gw_agents:
        gw_ids = [a.get("id") or a.get("name") for a in gw_agents]

        async def _fetch_gw_soul(agent_id: str) -> str | None:
            try:
                async with httpx.AsyncClient(timeout=5) as client:
                    resp = await client.get(f"{_vectorstep_gateway_base}/agents/{agent_id}/soul")
                    if resp.status_code == 200:
                        return resp.json().get("content") or resp.text
            except Exception:
                pass
            return None

        gw_soul_results = await asyncio.gather(*[_fetch_gw_soul(aid) for aid in gw_ids], return_exceptions=True)
        gw_agent_entries = all_agents[len(oc_agents):]
        for agent, soul_content in zip(gw_agent_entries, gw_soul_results):
            if isinstance(soul_content, Exception) or not soul_content:
                continue
            desc = _first_description_line(soul_content)
            if desc:
                agent["description"] = desc

    # Collect non-None error messages
    gateway_errors = [e for e in [oc_error, gw_error] if e]

    # DB stats — agent column now stores "executor:name" prefixed values.
    # Group by the full prefixed key so openclaw:X and gateway:X are counted separately.
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            select(
                PipelineStep.agent,
                PipelineStep.status,
                func.count().label("n"),
                func.max(PipelineStep.executed_at).label("last_run"),
            )
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent, PipelineStep.status)
        )
        step_rows = rows.all()

        token_rows = await session.execute(
            select(
                PipelineStep.agent,
                func.coalesce(func.sum(PipelineStep.input_tokens), 0),
                func.coalesce(func.sum(PipelineStep.output_tokens), 0),
            )
            .where(PipelineStep.agent.isnot(None))
            .group_by(PipelineStep.agent)
        )
        tokens_by_agent: dict[str, tuple[int, int]] = {
            agent: (inp, out) for agent, inp, out in token_rows.all()
        }

        duration_rows = await session.execute(
            select(PipelineStep.agent, func.avg(PipelineStep.duration_ms))
            .where(PipelineStep.agent.isnot(None), PipelineStep.duration_ms.isnot(None))
            .group_by(PipelineStep.agent)
        )
        avg_duration_by_agent: dict[str, float] = dict(duration_rows.all())

    agent_stats: dict[str, dict] = {}
    for row in step_rows:
        s = agent_stats.setdefault(
            row.agent,
            {"succeeded": 0, "failed": 0, "total": 0, "last_run": None, "success_rate": None,
             "avg_input_tokens": None, "avg_output_tokens": None, "avg_duration_secs": None},
        )
        s["total"] += row.n
        if row.status == "failed":
            s["failed"] += row.n
        else:
            s["succeeded"] += row.n
        if row.last_run and (s["last_run"] is None or row.last_run > s["last_run"]):
            s["last_run"] = row.last_run

    for key, s in agent_stats.items():
        if s["total"] > 0:
            s["success_rate"] = round(s["succeeded"] / s["total"] * 100, 1)
            inp, out = tokens_by_agent.get(key, (0, 0))
            if inp or out:
                s["avg_input_tokens"] = round(inp / s["total"])
                s["avg_output_tokens"] = round(out / s["total"])
            avg_duration_ms = avg_duration_by_agent.get(key)
            if avg_duration_ms is not None:
                s["avg_duration_secs"] = avg_duration_ms / 1000

    # If both backends are unreachable, synthesise stub entries from DB history
    # so the page still renders something useful.  Only synthesise for rows that
    # carry a recognisable "executor:agent" prefix — bare legacy entries are ignored.
    if not all_agents:
        seen: set[str] = set()
        for key in sorted(agent_stats.keys()):
            if ":" in key:
                stub_executor, _, aid = key.partition(":")
                stub_key = f"{stub_executor}:{aid}"
                if stub_key not in seen:
                    seen.add(stub_key)
                    all_agents.append({"id": aid, "name": aid, "executor": stub_executor})

    # Reverse lookup — which pipelines reference each "executor:agent", read from config
    # (same source as the Pipelines page's agent badges).
    all_pipelines = getattr(request.app.state, "pipelines", [])
    pipelines_by_agent: dict[str, list[str]] = defaultdict(list)
    for p in all_pipelines:
        for key in _agents_in_pipeline(p):
            pipelines_by_agent[key].append(p.name)

    # ── Filter option universes (unfiltered) ──────────────────────────────────
    all_executors = sorted({a.get("executor", "") for a in all_agents if a.get("executor")})
    all_models = sorted({
        m for a in all_agents
        for m in ([a["model"]] if a.get("model") else []) + (a.get("model_fallbacks") or [])
    })

    if executor:
        all_agents = [a for a in all_agents if a.get("executor") == executor]
    if model:
        all_agents = [
            a for a in all_agents
            if a.get("model") == model or model in (a.get("model_fallbacks") or [])
        ]

    # ── Header stat cards — scoped to the filtered agent set ─────────────────
    filtered_keys = [
        f"{a.get('executor', '')}:{a.get('id') or a.get('agentId') or a.get('name') or ''}"
        for a in all_agents
    ]
    executor_counts = Counter(a.get("executor", "") for a in all_agents)
    total_steps = sum(agent_stats.get(k, {}).get("total", 0) for k in filtered_keys)
    total_succeeded = sum(agent_stats.get(k, {}).get("succeeded", 0) for k in filtered_keys)
    overall_success_rate = round(total_succeeded / total_steps * 100) if total_steps else None
    total_input_tokens = sum(tokens_by_agent.get(k, (0, 0))[0] for k in filtered_keys)
    total_output_tokens = sum(tokens_by_agent.get(k, (0, 0))[1] for k in filtered_keys)

    return templates.TemplateResponse(request, "agents.html", {
        "agents": all_agents,
        "agent_stats": agent_stats,
        "pipelines_by_agent": pipelines_by_agent,
        "gateway_errors": gateway_errors,
        "all_executors": all_executors,
        "all_models": all_models,
        "selected_executor": executor or "",
        "selected_model": model or "",
        "total_agent_count": len(all_agents),
        "executor_counts": dict(executor_counts),
        "overall_success_rate": overall_success_rate,
        "total_steps": total_steps,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "active_page": "agents",
    })


@router.get("/providers", response_class=RedirectResponse)
async def ui_providers(time_range: str = "7d"):
    """Folded into Insights — /ui/providers now redirects there. Kept as a route (rather
    than removed outright) so old bookmarks/links keep working."""
    return RedirectResponse(url=f"/ui/insights/providers?time_range={time_range}", status_code=307)


@router.get("/agents/{executor}/{agent_id}", response_class=HTMLResponse)
async def ui_agent_detail(request: Request, executor: str, agent_id: str):
    prefixed_key = f"{executor}:{agent_id}"

    # Fetch live config + file contents from the appropriate backend
    if executor == "openclaw":
        agents_raw, _ = await _fetch_openclaw_agents()
        agent_files = await _fetch_openclaw_agent_files(agent_id)
    else:  # gateway (or any future executor with REST discovery)
        agents_raw, _ = await _fetch_vectorstep_gateway_agents()
        agent_files = await _fetch_vectorstep_gateway_agent_files(agent_id)

    agent_config = next(
        (a for a in agents_raw if a.get("name") == agent_id or a.get("id") == agent_id),
        None,
    )

    # DB run stats scoped to this executor:agent pair, production runs only (see _production_only)
    sf = get_session_factory()
    async with sf() as session:
        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.status,
                    func.count().label("n"),
                    func.coalesce(func.sum(PipelineStep.input_tokens), 0).label("input_tokens"),
                    func.coalesce(func.sum(PipelineStep.output_tokens), 0).label("output_tokens"),
                    func.avg(PipelineStep.duration_ms).label("avg_duration_ms"),
                    func.max(PipelineStep.executed_at).label("last_run"),
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
                .group_by(PipelineStep.model, PipelineStep.provider, PipelineStep.status)
            )
        )
        model_status_rows = rows.all()

        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.step_name,
                    PipelineRun.pipeline_name,
                    PipelineStep.model,
                    PipelineStep.provider,
                    PipelineStep.status,
                    func.count().label("n"),
                    func.coalesce(func.sum(PipelineStep.input_tokens), 0).label("input_tokens"),
                    func.coalesce(func.sum(PipelineStep.output_tokens), 0).label("output_tokens"),
                    func.max(PipelineStep.executed_at).label("last_run"),
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
                .group_by(
                    PipelineStep.step_name, PipelineRun.pipeline_name,
                    PipelineStep.model, PipelineStep.provider, PipelineStep.status,
                )
            )
        )
        step_status_rows = rows.all()

        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.executed_at, PipelineStep.model, PipelineStep.provider,
                    PipelineStep.input_tokens, PipelineStep.output_tokens,
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
            )
        )
        ts_rows = rows.all()

        rows = await session.execute(
            _production_only(
                select(
                    PipelineStep.run_id, PipelineStep.step_name, PipelineStep.model,
                    PipelineStep.provider, PipelineStep.status, PipelineStep.effective_confidence,
                    PipelineStep.duration_ms, PipelineStep.input_tokens,
                    PipelineStep.output_tokens, PipelineStep.executed_at,
                    PipelineRun.pipeline_name,
                )
                .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
                .where(PipelineStep.agent == prefixed_key)
                .order_by(PipelineStep.executed_at.desc())
                .limit(15)
            )
        )
        recent_rows = rows.all()

    # ── By-model breakdown — runs, success rate, avg duration, avg tokens ────
    model_stats: dict[str, dict] = {}
    for row in model_status_rows:
        m = _qualified_model(row.provider, row.model)
        s = model_stats.setdefault(m, {
            "succeeded": 0, "failed": 0, "total": 0, "last_run": None,
            "input_tokens": 0, "output_tokens": 0, "duration_sum_ms": 0.0, "duration_n": 0,
        })
        s["total"] += row.n
        if row.status == "failed":
            s["failed"] += row.n
        else:
            s["succeeded"] += row.n
        s["input_tokens"] += row.input_tokens
        s["output_tokens"] += row.output_tokens
        if row.avg_duration_ms is not None:
            s["duration_sum_ms"] += row.avg_duration_ms * row.n
            s["duration_n"] += row.n
        if row.last_run and (s["last_run"] is None or row.last_run > s["last_run"]):
            s["last_run"] = row.last_run

    for s in model_stats.values():
        total = s["total"]
        s["success_rate"] = round(s["succeeded"] / total * 100, 1) if total else None
        s["avg_input_tokens"] = round(s["input_tokens"] / total) if total and s["input_tokens"] else None
        s["avg_output_tokens"] = round(s["output_tokens"] / total) if total and s["output_tokens"] else None
        s["avg_duration_secs"] = (s["duration_sum_ms"] / s["duration_n"] / 1000) if s["duration_n"] else None

    # ── By-step breakdown — which pipeline steps this agent runs, per pipeline/model ──
    # Keyed by (step_name, pipeline_name, model) — the same step name can be wired to a
    # different model in different pipelines, so folding pipelines together would hide that.
    step_combo: dict[tuple[str, str, str], dict] = {}
    for row in step_status_rows:
        key = (row.step_name, row.pipeline_name, _qualified_model(row.provider, row.model))
        c = step_combo.setdefault(key, {
            "total": 0, "failed": 0, "input_tokens": 0, "output_tokens": 0, "last_run": None,
        })
        c["total"] += row.n
        if row.status == "failed":
            c["failed"] += row.n
        c["input_tokens"] += row.input_tokens
        c["output_tokens"] += row.output_tokens
        if row.last_run and (c["last_run"] is None or row.last_run > c["last_run"]):
            c["last_run"] = row.last_run

    step_stats = []
    for (step_name, pipeline_name, model), c in step_combo.items():
        total = c["total"]
        step_stats.append({
            "step_name": step_name,
            "pipeline_name": pipeline_name,
            "model": model,
            "total": total,
            "success_rate": round((total - c["failed"]) / total * 100) if total else None,
            "avg_input_tokens": round(c["input_tokens"] / total) if total else None,
            "avg_output_tokens": round(c["output_tokens"] / total) if total else None,
            "last_run": c["last_run"],
        })
    step_stats.sort(key=lambda r: r["total"], reverse=True)

    # ── Usage over time — run count and token volume, split by model ────────
    now = utc_now()
    runs_ts = _build_ts(
        [(r.executed_at, _qualified_model(r.provider, r.model)) for r in ts_rows], now, None, "all",
        dim_fn=lambda r: r[1],
    )
    tokens_ts = _build_ts(
        [(r.executed_at, _qualified_model(r.provider, r.model),
          (r.input_tokens or 0) + (r.output_tokens or 0)) for r in ts_rows],
        now, None, "all",
        dim_fn=lambda r: r[1],
        val_fn=lambda r: r[2],
    )

    # ── Recent activity — last 15 steps this agent ran, across any pipeline ──
    recent_activity = [{
        "run_id": r.run_id,
        "pipeline_name": r.pipeline_name,
        "step_name": r.step_name,
        "model": _qualified_model(r.provider, r.model),
        "status": r.status,
        "confidence": r.effective_confidence,
        "duration_secs": (r.duration_ms / 1000) if r.duration_ms is not None else None,
        "input_tokens": r.input_tokens,
        "output_tokens": r.output_tokens,
        "ago": _format_ago(r.executed_at),
    } for r in recent_rows]

    # Version history is only meaningful for gateway agents — VectorStep has no
    # equivalent content-hash mechanism for openclaw (SPEC-prompt-versioning.md §6d).
    agent_versions = await _get_agent_versions(sf, agent_id) if executor == "gateway" else []

    return templates.TemplateResponse(request, "agent_detail.html", {
        "agent_id": agent_id,
        "executor": executor,
        "agent_config": agent_config,
        "soul": agent_files["soul"],
        "tools": agent_files["tools"],
        "identity": agent_files["identity"],
        "agent_file": agent_files.get("agent_file"),
        "model_stats": model_stats,
        "step_stats": step_stats,
        "runs_ts": runs_ts,
        "tokens_ts": tokens_ts,
        "recent_activity": recent_activity,
        "agent_versions": agent_versions,
        "active_page": "agents",
    })


# ── MCP tools ─────────────────────────────────────────────────────────────────

@router.get("/mcp", response_class=HTMLResponse)
async def ui_mcp(request: Request):
    """Browse the MCP tool registry exposed by the VectorStep Gateway.

    OpenClaw isn't included here — it has no REST endpoint for tool
    introspection (the gateway executor is the only one that exposes
    GET /mcp/tools and GET /mcp/servers).
    """
    tools_by_server, servers_status, error = await _fetch_vectorstep_gateway_mcp()

    servers = []
    for name in sorted(set(tools_by_server) | set(servers_status)):
        status = servers_status.get(name, {})
        servers.append({
            "name": name,
            "tools": sorted(tools_by_server.get(name, []), key=lambda t: t.get("name", "")),
            "running": status.get("running"),
            "pid": status.get("pid"),
            "restart_count": status.get("restart_count", 0),
        })

    return templates.TemplateResponse(request, "mcp.html", {
        "servers": servers,
        "gateway_error": error,
        "server_count": len(servers),
        "total_tools": sum(len(s["tools"]) for s in servers),
        "active_page": "mcp",
    })


# ── Human-in-the-loop approvals ────────────────────────────────────────────────
#
# /approvals is a normal dashboard page (sidebar chrome) listing every pending
# approval regardless of channel — a universal fallback so a team isn't stuck if
# their primary chat channel (Slack/Telegram) is unreachable. /approvals/{token} is
# a standalone (no sidebar) page reached via a direct token link — used by the
# Teams approval channel, which posts this link instead of an in-chat button
# since Teams interactive cards need a public Bot Framework callback endpoint
# this deployment doesn't expose. See executors/human.py TeamsApprovalChannel.

@router.get("/approvals", response_class=HTMLResponse)
async def ui_approvals_list(request: Request):
    from .executors.human import list_pending

    return templates.TemplateResponse(request, "approvals_list.html", {
        "pending": list_pending(),
        "active_page": "approvals",
    })


@router.get("/approvals/{token}", response_class=HTMLResponse)
async def ui_approval(request: Request, token: str):
    from .executors.human import get_pending_meta

    meta = get_pending_meta(token)
    if meta is None:
        return templates.TemplateResponse(request, "approval.html", {
            "state": "not_found",
            "token": token,
        })

    return templates.TemplateResponse(request, "approval.html", {
        "state": "pending",
        "token": token,
        "meta": meta,
    })


@router.post("/approvals/{token}/approve", response_class=HTMLResponse)
async def ui_approval_approve(request: Request, token: str):
    return _decide(request, token, approved=True)


@router.post("/approvals/{token}/reject", response_class=HTMLResponse)
async def ui_approval_reject(request: Request, token: str):
    return _decide(request, token, approved=False)


def _decide(request: Request, token: str, approved: bool):
    from .executors.human import resolve_approval

    if not resolve_approval(token, approved):
        return templates.TemplateResponse(request, "approval.html", {
            "state": "not_found",
            "token": token,
        })

    return templates.TemplateResponse(request, "approval.html", {
        "state": "decided",
        "token": token,
        "decision": "approve" if approved else "reject",
    })
