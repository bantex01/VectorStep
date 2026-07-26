"""Calibration loop (Phase 3, SPEC-calibration.md): empirical accuracy per
(step_name, agent, model, provider, prompt_hash, agent_version) bucket, computed
fresh from existing pipeline_steps/step_feedback/run_feedback rows — no persisted
table, no curve-fitting dependency. See CONFIDENCE-REDESIGN.md §4 for the design
rationale, and SPEC-prompt-versioning.md §1/§4g for why prompt_hash and
agent_version are part of the key: without them, editing a step's prompt template
or a Gateway agent's config silently keeps counting outcomes from the old
configuration as evidence for the new one."""
import time
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from ..db.models import PipelineRun, PipelineStep, RunFeedback, StepFeedback

_OUTCOME_TO_LABEL = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass
class CalibrationBin:
    lo: float
    hi: float
    n: int
    mean_label: float
    validated: bool


@dataclass
class CalibrationBucket:
    step_name: str
    agent: str | None
    model: str | None
    provider: str | None
    prompt_hash: str | None
    agent_version: str | None
    bins: list[CalibrationBin]   # always exactly len == round(1 / bin_width), in order
    total_n: int                 # total labelled step-executions across all bins
    last_seen_at: datetime | None  # max(executed_at) across this bucket's rows

    def lookup(self, predicted: float) -> CalibrationBin | None:
        """Which bin a predicted score falls into. predicted == 1.0 lands in the last
        bin (bins are half-open [lo, hi) except the final bin, which is closed)."""
        for b in self.bins:
            if b.lo <= predicted < b.hi:
                return b
        if self.bins and predicted >= self.bins[-1].lo:
            return self.bins[-1]
        return None


async def compute_calibration_buckets(
    session_factory: async_sessionmaker,
    bin_width: float = 0.1,
    n_min: int = 20,
) -> dict[
    tuple[str, str | None, str | None, str | None, str | None, str | None],
    "CalibrationBucket",
]:
    """Production-scoped, recomputed fresh on every call (no persisted table — see
    SPEC-calibration.md §2). Bucketed by (step_name, agent, model, provider,
    prompt_hash, agent_version) — see SPEC-prompt-versioning.md §4g for why the last
    two are part of the key. NULL is its own bucket dimension for both, never a
    wildcard: a row with prompt_hash IS NULL never matches a row with a real hash
    (SPEC-prompt-versioning.md §2). Fan-out branches (step_name containing '/')
    collapse into their group's bucket, per CONFIDENCE-REDESIGN.md §7 item 8. Label
    precedence per step-execution: StepFeedback (human) > deterministic_passed=False
    (automated) > RunFeedback fallback (via run_id) > excluded (no label at all —
    not counted as 0, not counted as unlabelled-N)."""
    n_bins = round(1.0 / bin_width)
    assert abs(n_bins * bin_width - 1.0) < 1e-9, f"bin_width {bin_width} must evenly divide 1.0"

    async with session_factory() as session:
        rows = (await session.execute(
            select(
                PipelineStep.step_name, PipelineStep.agent, PipelineStep.model,
                PipelineStep.provider, PipelineStep.prompt_hash, PipelineStep.agent_version,
                PipelineStep.effective_confidence, PipelineStep.deterministic_passed,
                PipelineStep.executed_at,
                StepFeedback.outcome, RunFeedback.outcome,
            )
            .join(PipelineRun, PipelineStep.run_id == PipelineRun.id)
            .outerjoin(StepFeedback, StepFeedback.step_id == PipelineStep.id)
            .outerjoin(RunFeedback, RunFeedback.run_id == PipelineStep.run_id)
            .where(
                PipelineRun.stage == "production",
                PipelineStep.effective_confidence.is_not(None),
            )
        )).all()

    # bucket_key -> list of (predicted, label)
    samples: dict[tuple, list[tuple[float, float]]] = {}
    last_seen: dict[tuple, datetime] = {}
    for (step_name, agent, model, provider, p_hash, a_version,
         predicted, det_passed, executed_at, step_outcome, run_outcome) in rows:
        label: float | None = None
        if step_outcome is not None:
            label = _OUTCOME_TO_LABEL[step_outcome]
        elif det_passed is False:
            label = 0.0
        elif run_outcome is not None:
            label = _OUTCOME_TO_LABEL[run_outcome]
        if label is None:
            continue

        bucket_step_name = step_name.split("/", 1)[0]  # collapse fan-out branches
        key = (bucket_step_name, agent, model, provider, p_hash, a_version)
        samples.setdefault(key, []).append((predicted, label))
        if executed_at is not None and (key not in last_seen or executed_at > last_seen[key]):
            last_seen[key] = executed_at

    buckets: dict[tuple, CalibrationBucket] = {}
    for (step_name, agent, model, provider, p_hash, a_version), pairs in samples.items():
        bin_edges = [round(i * bin_width, 10) for i in range(n_bins + 1)]
        bins: list[CalibrationBin] = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = [label for predicted, label in pairs if lo <= predicted < hi or (i == n_bins - 1 and predicted == hi)]
            n = len(in_bin)
            mean_label = sum(in_bin) / n if n else 0.0
            bins.append(CalibrationBin(lo=lo, hi=hi, n=n, mean_label=mean_label, validated=n >= n_min))
        key = (step_name, agent, model, provider, p_hash, a_version)
        buckets[key] = CalibrationBucket(
            step_name=step_name, agent=agent, model=model, provider=provider,
            prompt_hash=p_hash, agent_version=a_version,
            bins=bins, total_n=len(pairs), last_seen_at=last_seen.get(key),
        )
    return buckets


def calibration_recommendation(bucket: CalibrationBucket) -> str | None:
    """Flag the first validated bin whose predicted score and observed accuracy diverge
    by >= 15 points — the exact style of recommendation CONFIDENCE-REDESIGN.md §4.3 uses
    as its own worked example. Returns None if every validated bin looks fine (or there
    are no validated bins yet)."""
    for b in bucket.bins:
        if not b.validated:
            continue
        midpoint = (b.lo + b.hi) / 2
        if abs(b.mean_label - midpoint) >= 0.15:
            return (
                f"runs scoring ~{round(midpoint * 100)}% in this configuration are only "
                f"{round(b.mean_label * 100)}% correct ({b.n} marked) — consider raising "
                f"the threshold, changing model, or adding grounding/deterministic checks."
            )
    return None


class CalibrationCache:
    """Not a source of truth — a short-TTL in-memory cache over
    compute_calibration_buckets(), so an enforced step's gate doesn't re-scan the DB on
    every single execution. Holds no state across process restarts; the first lookup
    after startup (or after the TTL expires) always recomputes from the DB."""

    def __init__(
        self,
        session_factory: async_sessionmaker,
        bin_width: float = 0.1,
        n_min: int = 20,
        ttl_seconds: int = 300,
    ):
        self._session_factory = session_factory
        self._bin_width = bin_width
        self.n_min = n_min
        self._ttl = ttl_seconds
        self._buckets: dict[tuple, CalibrationBucket] = {}
        self._computed_at: float = 0.0

    async def _ensure_fresh(self) -> None:
        now = time.time()
        if now - self._computed_at > self._ttl:
            self._buckets = await compute_calibration_buckets(
                self._session_factory, bin_width=self._bin_width, n_min=self.n_min,
            )
            self._computed_at = now

    async def get(
        self, step_name: str, agent: str | None, model: str | None, provider: str | None,
        prompt_hash: str | None, agent_version: str | None,
    ) -> CalibrationBucket | None:
        await self._ensure_fresh()
        return self._buckets.get((step_name, agent, model, provider, prompt_hash, agent_version))

    def previous_versions_for(
        self, step_name: str, agent: str | None, model: str | None, provider: str | None,
    ) -> list[CalibrationBucket]:
        """Every already-loaded bucket matching the first four key components,
        regardless of prompt_hash/agent_version — i.e. every OTHER version of this
        (step, agent, model, provider) combination. Used by the bucket_reset
        explanation (SPEC-prompt-versioning.md §4h) to tell an operator "this isn't
        a step with no history, it's a step whose history reset" — reads the cache's
        already-loaded bucket dict, no extra query. Only meaningful right after
        `get()` has run in this same request, so the cache is warm."""
        return [
            bucket for (s, a, m, p, _ph, _av), bucket in self._buckets.items()
            if (s, a, m, p) == (step_name, agent, model, provider)
        ]
