"""Owner-defined promotion readiness (SPEC-readiness-criteria.md): config
flattening + tier merge, evidence gathering, pure evaluation, narrative
generation and knob help text.

Split deliberately in two: gather_readiness_evidence (the only I/O, two DB
queries) and evaluate_readiness (pure, no I/O). This is what lets the preview
endpoint (§8b) re-evaluate on every keystroke against one cached evidence
gather, and it collapses nearly every tier-verdict test into a pure-function
test with no DB fixture.

Readiness does not call pipeline.calibration.compute_calibration_buckets() at
all — it composes the pieces that function itself is built from
(resolve_label, bins_from_samples, fetch_label_rows)."""
import asyncio
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Literal

from sqlalchemy.ext.asyncio import async_sessionmaker

from .models.pipeline import (
    FanOutGroupConfig,
    ParallelGroupConfig,
    PipelineConfig,
    ReadinessAccuracyConfig,
    ReadinessCalibrationConfig,
    ReadinessConfidenceConfig,
    ReadinessConfig,
    ReadinessOperationalConfig,
)
from .pipeline.calibration import (
    LABEL_DETERMINISTIC,
    LABEL_HUMAN,
    LABEL_RUN_FALLBACK,
    CalibrationBucket,
    bins_from_samples,
    calibration_recommendation,
    fetch_label_rows,
    resolve_label,
)
from .pipeline.versioning import prompt_hash as _prompt_hash

READINESS_TIERS = ("operational", "confidence", "accuracy", "calibration")

_ALL_STEP_STATUSES = ("completed", "failed", "aborted", "escalated", "stopped")
_LABEL_TO_OUTCOME = {1.0: "correct", 0.5: "partial", 0.0: "incorrect"}


# ---------------------------------------------------------------------------
# §5 — tier-level merge
# ---------------------------------------------------------------------------

def resolve_step_readiness(
    pipeline_level: ReadinessConfig | None,
    step_level: ReadinessConfig | None,   # already library-merged by the loader
    step_wrote_readiness_explicitly: bool = False,
) -> tuple[ReadinessConfig | None, dict[str, str]]:
    """Tier-level merge: the pipeline sets a house standard, a step ADDS tiers to it or
    REPLACES individual tiers. Tier contents are replaced wholesale, never field-merged,
    so a step's `accuracy:` block is always exactly what's written in the YAML.

    Explicit null at step level removes an inherited tier:
        readiness: {calibration: null}   # this step: no calibration bar
    Explicit null for the whole block opts the step out entirely:
        readiness: null

    `step_wrote_readiness_explicitly` is the raw loader-side signal
    `"readiness" in step_model.model_fields_set` — the caller (step_specs) has
    access to the original StepConfig/ParallelGroupInner/FanOutConfig object;
    this function can't recover that signal from `step_level` alone once it has
    already collapsed to None.

    Returns (effective_config, {tier_name: "pipeline" | "step"}) — the second element is
    provenance for the UI, so an owner can see where each tier's value came from.

    Documented wart (do not fix here): when a library step and the pipeline-level
    block define the SAME tier, the library step wins, because after the loader's
    'use:' resolution there is no way to tell library-provided from
    locally-written — see SPEC-readiness-criteria.md §5.
    """
    if step_level is None and step_wrote_readiness_explicitly:
        return None, {}

    merged: dict = {}
    source: dict[str, str] = {}
    for tier in READINESS_TIERS:
        if step_level is not None and tier in step_level.model_fields_set:
            value = getattr(step_level, tier)      # may be None -> explicit removal
            if value is not None:
                merged[tier] = value
                source[tier] = "step"
        elif pipeline_level is not None and getattr(pipeline_level, tier) is not None:
            merged[tier] = getattr(pipeline_level, tier)
            source[tier] = "pipeline"

    if not merged:
        return None, {}
    return ReadinessConfig(**merged), source


# ---------------------------------------------------------------------------
# §7b — config-side flattening
# ---------------------------------------------------------------------------

@dataclass
class StepSpec:
    """One evaluable unit, keyed by the COLLAPSED bucket name — fan-out and parallel
    branches roll up to their group name, matching step_name.split('/', 1)[0]."""
    name: str
    kind: Literal["step", "parallel", "fan_out"]
    executor: str | None            # None for parallel groups (branches can differ)
    confidence_threshold: float
    when: str | None                # surfaced so "never fired" is explicable (§12.8)
    config_prompt_hash: str | None  # prompt_hash(prompt_template) straight from the YAML
    readiness: ReadinessConfig | None
    readiness_source: dict[str, str]


def step_specs(pipeline: PipelineConfig) -> list[StepSpec]:
    """Replaces analytics._confidence_thresholds_by_step. Flattens parallel/fan-out
    groups the same way, taking the GROUP-level confidence_threshold and readiness
    (never a branch's own — the bucket already pools every branch together).

    For a parallel group config_prompt_hash is None: branches have different
    templates, so there is no single hash for the group. Fan-out groups have one
    prompt_template on FanOutConfig, so they behave like plain steps."""
    specs: list[StepSpec] = []
    for step in pipeline.steps:
        if isinstance(step, ParallelGroupConfig):
            inner = step.parallel
            readiness, source = resolve_step_readiness(
                pipeline.readiness, inner.readiness, "readiness" in inner.model_fields_set,
            )
            specs.append(StepSpec(
                name=inner.name, kind="parallel", executor=None,
                confidence_threshold=inner.confidence_threshold, when=inner.when,
                config_prompt_hash=None, readiness=readiness, readiness_source=source,
            ))
        elif isinstance(step, FanOutGroupConfig):
            fo = step.fan_out
            readiness, source = resolve_step_readiness(
                pipeline.readiness, fo.readiness, "readiness" in fo.model_fields_set,
            )
            specs.append(StepSpec(
                name=fo.name, kind="fan_out", executor=fo.executor,
                confidence_threshold=fo.confidence_threshold, when=fo.when,
                config_prompt_hash=_prompt_hash(fo.prompt_template),
                readiness=readiness, readiness_source=source,
            ))
        else:  # StepConfig
            readiness, source = resolve_step_readiness(
                pipeline.readiness, step.readiness, "readiness" in step.model_fields_set,
            )
            specs.append(StepSpec(
                name=step.name, kind="step", executor=step.executor,
                confidence_threshold=step.confidence_threshold, when=step.when,
                config_prompt_hash=_prompt_hash(step.prompt_template),
                readiness=readiness, readiness_source=source,
            ))
    return specs


# ---------------------------------------------------------------------------
# §7c — evidence gathering (the only I/O)
# ---------------------------------------------------------------------------

@dataclass
class ComboEvidence:
    agent: str | None
    model: str | None
    provider: str | None
    prompt_hash: str | None
    agent_version: str | None
    samples: list[tuple[float, float, str]] = field(default_factory=list)   # (predicted, label, label_source)
    rows: int = 0
    runs: int = 0
    last_seen_at: datetime | None = None


@dataclass
class StepEvidence:
    rows: list[dict] = field(default_factory=list)          # all rows for this collapsed step name
    runs: dict[str, list[dict]] = field(default_factory=dict)   # run_id -> that run's rows for this step
    own_combos: dict[tuple, ComboEvidence] = field(default_factory=dict)  # this pipeline, its own stage
    prod_combos: dict[tuple, ComboEvidence] = field(default_factory=dict)  # production, cross-pipeline
    prod_pipelines: dict[tuple, set] = field(default_factory=dict)  # combo -> contributing pipeline names
    latest_agent_version: str | None = None
    latest_prompt_hash: str | None = None


@dataclass
class ReadinessEvidence:
    pipeline_name: str
    evidence_stage: str
    gathered_at: datetime
    by_step: dict[str, StepEvidence]


def _bucket_name(step_name: str) -> str:
    return step_name.split("/", 1)[0]


async def gather_readiness_evidence(
    session_factory: async_sessionmaker, pipeline: PipelineConfig,
) -> ReadinessEvidence:
    """Exactly two queries, run concurrently."""
    specs = step_specs(pipeline)
    names = {s.name for s in specs}
    evidence_stage = pipeline.stage   # NOT hardcoded "testing" — see §2

    own_rows, prod_rows = await asyncio.gather(
        fetch_label_rows(session_factory, stage=evidence_stage, pipeline_name=pipeline.name,
                          step_names=names, require_confidence=False),
        fetch_label_rows(session_factory, stage="production", step_names=names,
                          require_confidence=True),
    )

    by_step: dict[str, StepEvidence] = {name: StepEvidence() for name in names}
    latest_at: dict[str, datetime] = {}
    latest_prompt_at: dict[str, datetime] = {}

    for (step_name, agent, model, provider, p_hash, a_version,
         predicted, det_passed, executed_at, run_id, status, pipeline_name,
         step_outcome, run_outcome) in own_rows:
        bname = _bucket_name(step_name)
        if bname not in by_step:
            continue
        se = by_step[bname]

        resolved = resolve_label(step_outcome, det_passed, run_outcome)
        label, label_source = resolved if resolved is not None else (None, None)

        row = {
            "run_id": run_id, "status": status, "executed_at": executed_at,
            "predicted": predicted, "label": label, "label_source": label_source,
            "agent": agent, "model": model, "provider": provider,
            "prompt_hash": p_hash, "agent_version": a_version,
        }
        se.rows.append(row)
        se.runs.setdefault(run_id, []).append(row)

        combo_key = (agent, model, provider, p_hash, a_version)
        combo = se.own_combos.setdefault(combo_key, ComboEvidence(
            agent=agent, model=model, provider=provider, prompt_hash=p_hash, agent_version=a_version,
        ))
        combo.rows += 1
        # own_rows is fetched with require_confidence=False (operational/accuracy need
        # labeled-but-unconfident rows too — a failed/aborted run, a non-LLM executor) —
        # but a calibration sample needs BOTH a label and a predicted confidence, so a
        # labeled row with no effective_confidence must not become a sample here.
        if label is not None and predicted is not None:
            combo.samples.append((predicted, label, label_source))
        if executed_at is not None and (combo.last_seen_at is None or executed_at > combo.last_seen_at):
            combo.last_seen_at = executed_at

        if executed_at is not None:
            if a_version is not None and (bname not in latest_at or executed_at > latest_at[bname]):
                latest_at[bname] = executed_at
                se.latest_agent_version = a_version
            if p_hash is not None and (bname not in latest_prompt_at or executed_at > latest_prompt_at[bname]):
                latest_prompt_at[bname] = executed_at
                se.latest_prompt_hash = p_hash

    for (step_name, agent, model, provider, p_hash, a_version,
         predicted, det_passed, executed_at, run_id, status, pipeline_name,
         step_outcome, run_outcome) in prod_rows:
        bname = _bucket_name(step_name)
        if bname not in by_step:
            continue
        se = by_step[bname]

        resolved = resolve_label(step_outcome, det_passed, run_outcome)
        label, label_source = resolved if resolved is not None else (None, None)

        combo_key = (agent, model, provider, p_hash, a_version)
        combo = se.prod_combos.setdefault(combo_key, ComboEvidence(
            agent=agent, model=model, provider=provider, prompt_hash=p_hash, agent_version=a_version,
        ))
        combo.rows += 1
        # prod_rows is fetched with require_confidence=True today, so predicted is never
        # None here in practice — guarded anyway so this can't regress the same way as
        # own_combos above if that query's filter ever changes.
        if label is not None and predicted is not None:
            combo.samples.append((predicted, label, label_source))
        if executed_at is not None and (combo.last_seen_at is None or executed_at > combo.last_seen_at):
            combo.last_seen_at = executed_at
        if pipeline_name is not None:
            se.prod_pipelines.setdefault(combo_key, set()).add(pipeline_name)

    # distinct run counts per combo
    for se in by_step.values():
        combo_runs: dict[tuple, set] = {}
        for row in se.rows:
            key = (row["agent"], row["model"], row["provider"], row["prompt_hash"], row["agent_version"])
            combo_runs.setdefault(key, set()).add(row["run_id"])
        for key, combo in se.own_combos.items():
            combo.runs = len(combo_runs.get(key, ()))

    return ReadinessEvidence(
        pipeline_name=pipeline.name, evidence_stage=evidence_stage,
        gathered_at=datetime.utcnow(), by_step=by_step,
    )


# ---------------------------------------------------------------------------
# §7d — pure evaluation
# ---------------------------------------------------------------------------

def _current_config_filter(spec: StepSpec, se: StepEvidence):
    def _match(row: dict) -> bool:
        prompt_ok = spec.kind == "parallel" or row["prompt_hash"] == spec.config_prompt_hash
        version_ok = row["agent_version"] == se.latest_agent_version
        return prompt_ok and version_ok
    return _match


def _current_config_block(spec: StepSpec, se: StepEvidence) -> dict:
    if spec.kind == "parallel":
        prompt_hash_val = None
        prompt_hash_source = "not_applicable_parallel_group"
        prompt_matches = True
    else:
        prompt_hash_val = spec.config_prompt_hash
        prompt_hash_source = "config"
        if prompt_hash_val is None:
            prompt_matches = True   # non-LLM / empty template — filter is inert, not an error
        else:
            observed = {r["prompt_hash"] for r in se.rows}
            prompt_matches = (prompt_hash_val in observed) if observed else True
    return {
        "prompt_hash": prompt_hash_val,
        "agent_version": se.latest_agent_version,
        "agent_version_source": "latest_observed" if se.latest_agent_version is not None else "unknown",
        "prompt_hash_source": prompt_hash_source,
        "prompt_hash_matches_history": prompt_matches,
    }


def _eval_operational(cfg: ReadinessOperationalConfig | None, se: StepEvidence, now: datetime, cc_filter) -> dict:
    if cfg is None:
        return {"verdict": "not_configured"}

    rows = se.rows
    if cfg.max_age_days is not None:
        cutoff = now - timedelta(days=cfg.max_age_days)
        rows = [r for r in rows if r["executed_at"] is not None and r["executed_at"] >= cutoff]
    if cfg.require_current_config:
        rows = [r for r in rows if cc_filter(r)]

    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)

    status_counts = {s: 0 for s in _ALL_STEP_STATUSES}
    runs_acceptable = 0
    for run_rows in by_run.values():
        if all(r["status"] in cfg.acceptable_statuses for r in run_rows):
            runs_acceptable += 1
        for r in run_rows:
            status_counts[r["status"]] = status_counts.get(r["status"], 0) + 1

    verdict = "pass" if runs_acceptable >= cfg.min_runs else "insufficient_data"
    return {
        "verdict": verdict, "min_runs": cfg.min_runs,
        "runs_acceptable": runs_acceptable, "runs_total": len(by_run),
        "status_counts": status_counts,
        "filtered_to_current_config": cfg.require_current_config,
        "max_age_days": cfg.max_age_days,
    }


def _eval_confidence(cfg: ReadinessConfidenceConfig | None, se: StepEvidence, cc_filter) -> dict:
    if cfg is None:
        return {"verdict": "not_configured"}

    rows = [r for r in se.rows if r["predicted"] is not None]
    if cfg.require_current_config:
        rows = [r for r in rows if cc_filter(r)]

    n = len(rows)
    mean = (sum(r["predicted"] for r in rows) / n) if n else None
    if n == 0 or (cfg.min_runs is not None and n < cfg.min_runs):
        verdict = "insufficient_data"
    elif mean < cfg.min_confidence:
        verdict = "fail"
    else:
        verdict = "pass"

    return {
        "verdict": verdict, "min_confidence": cfg.min_confidence, "min_runs": cfg.min_runs,
        "n": n, "mean_confidence": mean,
        "filtered_to_current_config": cfg.require_current_config,
    }


def _eval_accuracy(cfg: ReadinessAccuracyConfig | None, se: StepEvidence, cc_filter) -> dict:
    if cfg is None:
        return {"verdict": "not_configured"}

    rows = [r for r in se.rows if r["label"] is not None]
    if cfg.require_current_config:
        rows = [r for r in rows if cc_filter(r)]

    marked = len(rows)
    marked_runs = len({r["run_id"] for r in rows})
    outcome_counts = {"correct": 0, "partial": 0, "incorrect": 0}
    provenance = {LABEL_HUMAN: 0, LABEL_DETERMINISTIC: 0, LABEL_RUN_FALLBACK: 0}
    for r in rows:
        provenance[r["label_source"]] = provenance.get(r["label_source"], 0) + 1
        outcome_counts[_LABEL_TO_OUTCOME[r["label"]]] += 1
    weighted = (sum(r["label"] for r in rows) / marked) if marked else None
    human_marked = provenance.get(LABEL_HUMAN, 0)

    warnings: list[str] = []
    if human_marked == 0 and provenance.get(LABEL_DETERMINISTIC, 0) > 0:
        warnings.append(
            "Every label here comes from a failed deterministic check, with zero human "
            "review — a PASSING check produces no label at all, so this population can be "
            "100% failures and still be an accurate picture of a mostly-fine step. "
            "Treat this accuracy figure with caution, or set min_human_marked."
        )
    if marked and provenance.get(LABEL_RUN_FALLBACK, 0) / marked > 0.5:
        warnings.append(
            "More than half of these labels are inherited from a run-level rating rather "
            "than feedback on this step specifically — one run-level rating labels every "
            "step in that run, which can inflate the marked count without reflecting this "
            "step's own quality."
        )

    if marked < cfg.min_marked or (cfg.min_human_marked is not None and human_marked < cfg.min_human_marked):
        verdict = "insufficient_data"
    elif weighted < cfg.min_accuracy:
        verdict = "fail"
    else:
        verdict = "pass"

    return {
        "verdict": verdict, "min_accuracy": cfg.min_accuracy, "min_marked": cfg.min_marked,
        "min_human_marked": cfg.min_human_marked,
        "accuracy": weighted, "marked": marked, "marked_runs": marked_runs,
        "outcome_counts": outcome_counts, "provenance": provenance,
        "filtered_to_current_config": cfg.require_current_config,
        "warnings": warnings,
    }


def _combo_is_current(combo_key: tuple, spec: StepSpec, se: StepEvidence) -> bool:
    _agent, _model, _provider, p_hash, a_version = combo_key
    prompt_ok = spec.kind == "parallel" or p_hash == spec.config_prompt_hash
    version_ok = a_version == se.latest_agent_version
    return prompt_ok and version_ok


def _combo_view(combo: ComboEvidence, bin_width: float, n_min: int, max_divergence: float) -> dict:
    pairs = [(p, l) for p, l, _s in combo.samples]
    bins = bins_from_samples(pairs, bin_width, n_min)
    bucket = CalibrationBucket(
        step_name="", agent=combo.agent, model=combo.model, provider=combo.provider,
        prompt_hash=combo.prompt_hash, agent_version=combo.agent_version,
        bins=bins, total_n=len(pairs), last_seen_at=combo.last_seen_at,
    )
    recommendation = calibration_recommendation(bucket, max_divergence=max_divergence, n_min=n_min)
    provenance: dict[str, int] = {}
    for _p, _l, s in combo.samples:
        provenance[s] = provenance.get(s, 0) + 1
    divergences = [abs(b.mean_label - (b.lo + b.hi) / 2) for b in bins if b.n > 0]
    return {
        "total_n": len(pairs),
        "validated": any(b.validated for b in bins),
        "recommendation": recommendation,
        "max_divergence_observed": max(divergences) if divergences else None,
        "bins": [
            {"lo": b.lo, "hi": b.hi, "n": b.n, "mean_label": b.mean_label, "validated": b.validated}
            for b in bins
        ],
        "provenance": provenance,
    }


def _eval_calibration(cfg: ReadinessCalibrationConfig | None, spec: StepSpec, se: StepEvidence) -> dict:
    if cfg is None:
        return {"verdict": "not_configured"}

    evidence_views = ["own"] if cfg.require_own_evidence else ["own", "production"]
    combo_keys = set(se.own_combos)
    if "production" in evidence_views:
        combo_keys |= set(se.prod_combos)

    combos_out = []
    any_flagged = False
    any_validated_clean = False
    for key in sorted(combo_keys, key=lambda k: tuple(x if x is not None else "" for x in k)):
        agent, model, provider, p_hash, a_version = key
        is_current = _combo_is_current(key, spec, se)

        own_view = (
            _combo_view(se.own_combos[key], cfg.bin_width, cfg.n_min, cfg.max_divergence)
            if key in se.own_combos else None
        )
        prod_view = (
            _combo_view(se.prod_combos[key], cfg.bin_width, cfg.n_min, cfg.max_divergence)
            if "production" in evidence_views and key in se.prod_combos else None
        )

        if not is_current:
            combo_verdict = "not_current_config"
        else:
            flagged = any(v["recommendation"] for v in (own_view, prod_view) if v)
            validated_clean = any(v["validated"] and not v["recommendation"] for v in (own_view, prod_view) if v)
            if flagged:
                combo_verdict = "fail"
                any_flagged = True
            elif validated_clean:
                combo_verdict = "pass"
                any_validated_clean = True
            else:
                combo_verdict = "insufficient_data"

        combos_out.append({
            "agent": agent, "model": model, "provider": provider,
            "prompt_hash": p_hash, "agent_version": a_version,
            "is_current_config": is_current, "verdict": combo_verdict,
            "own": own_view, "production": prod_view,
            "production_pipelines": sorted(se.prod_pipelines.get(key, ())) if "production" in evidence_views else [],
        })

    verdict = "fail" if any_flagged else "pass" if any_validated_clean else "insufficient_data"

    return {
        "verdict": verdict, "n_min": cfg.n_min, "bin_width": cfg.bin_width,
        "max_divergence": cfg.max_divergence, "require_own_evidence": cfg.require_own_evidence,
        "evidence_views": evidence_views, "combos": combos_out,
    }


def _observed_combos(
    spec: StepSpec, se: StepEvidence, default_bin_width: float, default_n_min: int,
) -> list[dict]:
    """Every combo this step has actually been exercised under (own-stage evidence),
    with a calibration snapshot at the SERVICE DEFAULTS (never owner-configurable —
    this is 'what the numbers look like', not a bar anyone chose). Keeps the §11
    backward-compat guarantee: a pipeline with no readiness: block still shows
    observed calibration bins and any divergence flag."""
    out = []
    for key, combo in se.own_combos.items():
        agent, model, provider, p_hash, a_version = key
        out.append({
            "agent": agent, "model": model, "provider": provider,
            "prompt_hash": p_hash, "agent_version": a_version,
            "rows": combo.rows, "runs": combo.runs,
            "is_current_config": _combo_is_current(key, spec, se),
            "observed": _combo_view(combo, default_bin_width, default_n_min, 0.15),
        })
    out.sort(key=lambda c: c["rows"], reverse=True)
    return out


def readiness_narrative(step_result: dict) -> list[str]:
    """Plain-language, numbers-first walkthrough of one step's verdict — the readiness
    sibling of ui._confidence_narrative. Lives here, not in ui.py, so the preview
    endpoint returns identical text without duplicating the logic."""
    lines: list[str] = []
    tiers = step_result["tiers"]
    criteria = step_result["criteria"]

    op = tiers.get("operational")
    if op and op["verdict"] != "not_configured":
        statuses = " or ".join(criteria["operational"]["acceptable_statuses"])
        verb = "cleared" if op["verdict"] == "pass" else "not yet cleared"
        lines.append(
            f"This step needs {op['min_runs']} runs that ended {statuses}. "
            f"{op['runs_acceptable']} of its {op['runs_total']} observed runs qualify — {verb}."
        )

    conf = tiers.get("confidence")
    if conf and conf["verdict"] != "not_configured":
        if conf["n"]:
            lines.append(
                f"It needs mean self-reported confidence of at least {conf['min_confidence']:.0%}. "
                f"{conf['n']} run(s) average {conf['mean_confidence']:.0%}, which is "
                f"{'at or above' if conf['verdict'] == 'pass' else 'below'} the bar."
            )
        else:
            lines.append(
                f"It needs mean self-reported confidence of at least "
                f"{conf['min_confidence']:.0%}; no qualifying runs yet."
            )

    acc = tiers.get("accuracy")
    if acc and acc["verdict"] != "not_configured":
        if acc["marked"]:
            verdict_word = (
                "at or above" if acc["verdict"] == "pass"
                else "below" if acc["verdict"] == "fail" else "not yet enough to judge against"
            )
            lines.append(
                f"It needs {acc['min_accuracy']:.0%} judged accuracy over at least "
                f"{acc['min_marked']} marked results. {acc['marked']} are marked so far and "
                f"they average {acc['accuracy']:.0%}, which is {verdict_word} the bar."
            )
            prov = acc["provenance"]
            lines.append(
                f"Of those {acc['marked']} labels, {prov.get(LABEL_HUMAN, 0)} came from a human "
                f"marking the step, {prov.get(LABEL_DETERMINISTIC, 0)} from a failed automated "
                f"check, and {prov.get(LABEL_RUN_FALLBACK, 0)} were inherited from a run-level rating."
            )
        else:
            lines.append(
                f"It needs {acc['min_accuracy']:.0%} judged accuracy over at least "
                f"{acc['min_marked']} marked results; none marked yet."
            )
        lines.extend(acc.get("warnings", []))

    cal = tiers.get("calibration")
    if cal and cal["verdict"] != "not_configured":
        fullest_bin = 0
        for c in cal["combos"]:
            for view in (c["own"], c["production"]):
                if view:
                    fullest_bin = max(fullest_bin, max((b["n"] for b in view["bins"]), default=0))
        scope = (
            "using only this pipeline's own runs" if cal["require_own_evidence"]
            else "using this pipeline's own runs or shared production evidence for the identical configuration"
        )
        there_yet = "there" if fullest_bin >= cal["n_min"] else "not there"
        lines.append(
            f"It needs {cal['n_min']} marked results AT THE SAME CONFIDENCE LEVEL before "
            f"calibration counts, {scope}. The fullest confidence band has {fullest_bin} — {there_yet} yet."
        )

    overall_word = {
        "not_ready": "not ready", "building": "building", "no_data": "no data yet",
        "ready": "ready", "not_configured": "no criteria configured",
    }[step_result["verdict"]]
    reason = ""
    if step_result["verdict"] == "not_ready":
        failing = [name for name, t in tiers.items() if t.get("verdict") == "fail"]
        if failing:
            plural = "s are" if len(failing) > 1 else " is"
            reason = f" — the {', '.join(failing)} tier{plural} below the configured bar"
    lines.append(f"Overall: {overall_word}{reason}.")
    return lines


def _criteria_block(spec: StepSpec) -> dict:
    out: dict = {}
    for tier in READINESS_TIERS:
        cfg = getattr(spec.readiness, tier) if spec.readiness is not None else None
        if cfg is None:
            out[tier] = None
        else:
            d = cfg.model_dump()
            d["source"] = spec.readiness_source.get(tier, "step")
            out[tier] = d
    return out


def evaluate_readiness(
    evidence: ReadinessEvidence,
    pipeline: PipelineConfig,
    override: ReadinessConfig | None = None,
    override_steps: list[str] | None = None,
    default_bin_width: float = 0.1,
    default_n_min: int = 20,
) -> dict:
    """Pure — no I/O. `override`/`override_steps` exist for the preview endpoint
    (§8b); the GET calls this with neither."""
    specs = step_specs(pipeline)

    if override is not None:
        all_names = {s.name for s in specs}
        target_names = set(override_steps) if override_steps else set(all_names)
        unknown = target_names - all_names
        if unknown:
            raise ValueError(f"apply_to names unknown step(s): {sorted(unknown)}")
        override_source = {t: "preview" for t in READINESS_TIERS if getattr(override, t) is not None}
        specs = [
            replace(s, readiness=override, readiness_source=override_source)
            if s.name in target_names else s
            for s in specs
        ]

    now = datetime.utcnow()
    steps_out = []
    counts = {"ready": 0, "not_ready": 0, "building": 0, "no_data": 0, "not_configured": 0}

    for spec in specs:
        se = evidence.by_step.get(spec.name, StepEvidence())
        cc_filter = _current_config_filter(spec, se)
        cc = _current_config_block(spec, se)

        op_cfg = spec.readiness.operational if spec.readiness else None
        conf_cfg = spec.readiness.confidence if spec.readiness else None
        acc_cfg = spec.readiness.accuracy if spec.readiness else None
        cal_cfg = spec.readiness.calibration if spec.readiness else None

        op_result = _eval_operational(op_cfg, se, now, cc_filter)
        conf_result = _eval_confidence(conf_cfg, se, cc_filter)
        acc_result = _eval_accuracy(acc_cfg, se, cc_filter)
        cal_result = _eval_calibration(cal_cfg, spec, se)
        tiers = {
            "operational": op_result, "confidence": conf_result,
            "accuracy": acc_result, "calibration": cal_result,
        }

        if any(t.get("verdict") == "fail" for t in tiers.values()):
            step_verdict = "not_ready"
        elif any(t.get("verdict") == "insufficient_data" for t in tiers.values()):
            step_verdict = "no_data" if not se.rows else "building"
        elif any(t.get("verdict") == "pass" for t in tiers.values()):
            step_verdict = "ready"
        else:
            step_verdict = "not_configured"
        counts[step_verdict] += 1

        notes: list[str] = []
        if not cc["prompt_hash_matches_history"]:
            any_requires_current = any(
                cfg is not None and cfg.require_current_config
                for cfg in (op_cfg, conf_cfg, acc_cfg, cal_cfg)
            )
            if any_requires_current:
                excluded = len([
                    r for r in se.rows
                    if r["label"] is not None and r["prompt_hash"] != cc["prompt_hash"]
                ])
                notes.append(
                    f"Prompt template changed since the last recorded run — {excluded} earlier "
                    f"marked result(s) are excluded because this tier requires current-config evidence."
                )

        marked_total = len([r for r in se.rows if r["label"] is not None])
        executed_ats = [r["executed_at"] for r in se.rows if r["executed_at"] is not None]

        step_result = {
            "step_name": spec.name, "kind": spec.kind, "executor": spec.executor,
            "when": spec.when, "verdict": step_verdict,
            "confidence_threshold": spec.confidence_threshold,
            "criteria": _criteria_block(spec),
            "current_config": cc,
            "evidence": {
                "runs_total": len(se.runs), "rows_total": len(se.rows),
                "marked_total": marked_total,
                "first_seen_at": min(executed_ats) if executed_ats else None,
                "last_seen_at": max(executed_ats) if executed_ats else None,
            },
            "tiers": tiers,
            "observed_combos": _observed_combos(spec, se, default_bin_width, default_n_min),
            "notes": notes,
        }
        step_result["narrative"] = readiness_narrative(step_result)
        steps_out.append(step_result)

    if counts["not_ready"]:
        pipeline_verdict = "not_ready"
    elif counts["building"] or counts["no_data"]:
        pipeline_verdict = "building"
    elif counts["ready"]:
        pipeline_verdict = "ready"
    else:
        pipeline_verdict = "not_configured"

    criteria_source = "configured" if any(s.readiness is not None for s in specs) else "none"

    return {
        "pipeline_name": pipeline.name,
        "pipeline_stage": pipeline.stage,
        "evidence_stage": evidence.evidence_stage,
        "criteria_source": criteria_source,
        "verdict": pipeline_verdict,
        "gathered_at": evidence.gathered_at,
        "summary": {**counts, "total": len(steps_out)},
        "steps": steps_out,
    }


# ---------------------------------------------------------------------------
# §7g — knob help text
# ---------------------------------------------------------------------------

READINESS_KNOB_HELP: dict[str, str] = {
    "operational.min_runs": (
        "How many runs of this step must have ended in an acceptable status before the "
        "operational bar is cleared. The cheapest, least judgmental bar there is — pure "
        "run-counting, no confidence or human review required. Counts DISTINCT runs, never "
        "rows, so a single 20-branch fan-out or parallel group does not satisfy min_runs: 20 "
        "on its own — see acceptable_statuses below for what 'acceptable' means for a group."
    ),
    "operational.acceptable_statuses": (
        "Which end-states count as an acceptable run. Adding a status makes the bar LAXER, not "
        "stricter — [completed, escalated] accepts runs where a human had to step in, which is "
        "a weaker claim than [completed] alone. For a parallel or fan-out group, EVERY branch's "
        "status in that run must be in this list for the run to count."
    ),
    "operational.max_age_days": (
        "Restricts the operational tier to runs from the last N days. The only readiness knob "
        "with a time window — the other three tiers deliberately keep a lifetime track record, "
        "because 'judged accuracy over the last 7 days' has murky semantics when humans mark "
        "sparsely and in bursts. Leave unset (the default) for a lifetime count."
    ),
    "operational.require_current_config": (
        "False (default): a prompt typo-fix doesn't wipe out 30 clean runs recorded under the "
        "old wording, since operational only cares that the step RAN cleanly, not what exact "
        "prompt or agent version produced that run. Set true to restrict the count to runs that "
        "match the pipeline's current prompt_template and the step's most recently observed "
        "agent_version."
    ),
    "confidence.min_confidence": (
        "The minimum MEAN self-reported effective_confidence this step must show across its "
        "qualifying runs. A weak signal on its own — the model can be confidently wrong — best "
        "used as an early checkpoint before anyone has marked anything for the accuracy tier."
    ),
    "confidence.min_runs": (
        "Minimum number of confidence-bearing runs before this tier can pass. Strongly "
        "recommended: without it, min_confidence: 0.9 passes on the strength of a single 0.95 run."
    ),
    "confidence.require_current_config": (
        "False (default) counts every qualifying run regardless of which prompt or agent "
        "version produced it. True restricts to runs matching the current prompt_template and "
        "the step's most recently observed agent_version."
    ),
    "accuracy.min_accuracy": (
        "The minimum WEIGHTED judged accuracy required: correct=1.0, partial=0.5, incorrect=0.0, "
        "averaged over every marked result, using the same label-precedence chain calibration "
        "uses (human feedback beats a failed deterministic check beats a run-level rating)."
    ),
    "accuracy.min_marked": (
        "Minimum number of labelled results required before this tier can pass or fail. Below "
        "this, the tier reads insufficient_data rather than asserting anything about accuracy "
        "it hasn't actually seen enough evidence for."
    ),
    "accuracy.min_human_marked": (
        "How many of the marked results must come from a human reviewing the step, as opposed "
        "to automation. This matters because only a FAILED deterministic check produces a label "
        "— passing checks produce nothing — so a step with checks and no human feedback can have "
        "a labelled population that is 100% failures, and read as 0% accurate while being fine."
    ),
    "accuracy.require_current_config": (
        "True by default: judged accuracy is about whether THIS configuration's output is good, "
        "so a marked result from a prompt version that has since been edited is excluded unless "
        "you explicitly set this false."
    ),
    "calibration.n_min": (
        "How many marked results are needed AT THE SAME CONFIDENCE LEVEL before calibration "
        "counts. This is per confidence band, not a total: a step with 100 marked results "
        "spread evenly across 10 bands has only 10 in each, and will not validate at n_min: 20. "
        "Look at the 'in top band' number on the chip, not the total."
    ),
    "calibration.bin_width": (
        "Width of each confidence band calibration is measured against — must evenly divide "
        "1.0 (e.g. 0.1 gives 10 bands, 0.2 gives 5). Wider bands quantise the predicted score "
        "more coarsely, which can make divergence look smaller than it is — keep max_divergence "
        "well above bin_width/2, or the bin's own quantisation error can trip the flag on a "
        "perfectly calibrated step."
    ),
    "calibration.max_divergence": (
        "How many percentage points a validated band's actual accuracy may diverge from its "
        "predicted midpoint before calibration flags it. The default (0.15) matches the "
        "service-wide recommendation threshold; set it stricter for a step whose confidence "
        "number authorises a side effect."
    ),
    "calibration.require_own_evidence": (
        "False (default) lets this step count a shared library step's production track record "
        "from another pipeline, when the agent, model, prompt and agent version all match "
        "exactly. True restricts it to this pipeline's own runs. Note that overriding "
        "prompt_template locally on a `use:` step changes the prompt hash and silently forfeits "
        "that inherited evidence."
    ),
    "calibration.require_current_config": (
        "Cannot be set to false — a calibration bucket is keyed by (prompt_hash, agent_version) "
        "by definition, so 'ignore the version' would mean merging buckets and destroying the "
        "reset semantics prompt/agent versioning exists to protect. Use the accuracy tier if you "
        "want version-independent judged accuracy."
    ),
}

_TIER_MODELS = {
    "operational": ReadinessOperationalConfig,
    "confidence": ReadinessConfidenceConfig,
    "accuracy": ReadinessAccuracyConfig,
    "calibration": ReadinessCalibrationConfig,
}


def _tier_defaults(model: type) -> dict:
    """Model-default value for every field, None for a required field with no default —
    the seed for a tier the owner has never configured."""
    defaults = {}
    for name, info in model.model_fields.items():
        default = info.get_default(call_default_factory=True)
        defaults[name] = None if info.is_required() else default
    return defaults


def _tier_state(effective: ReadinessConfig | None) -> dict:
    state = {}
    for tier, model in _TIER_MODELS.items():
        value = getattr(effective, tier, None) if effective is not None else None
        if value is not None:
            state[tier] = {"enabled": True, **value.model_dump()}
        else:
            state[tier] = {"enabled": False, **_tier_defaults(model)}
    return state


def builder_seed(pipeline: PipelineConfig) -> dict:
    """Initial client-side state for the builder, so opening it on an already-configured
    pipeline starts from what is LIVE rather than from schema defaults — otherwise the
    first preview would silently propose weakening the owner's existing bar.

    Returns, for the pipeline-level scope and for each step scope:
        {scope_key: {tier: {"enabled": bool, **knob_values}}}
    where scope_key is "__pipeline__" or the step name. Values come from the effective
    resolved criteria (readiness.resolve_step_readiness), knobs the owner never set are
    filled with the model defaults, and `enabled` reflects whether the tier is configured.
    """
    pipeline_level, _ = resolve_step_readiness(pipeline.readiness, None, False)
    seed = {"__pipeline__": _tier_state(pipeline_level)}
    for spec in step_specs(pipeline):
        seed[spec.name] = _tier_state(spec.readiness)
    return seed
