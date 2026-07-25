"""Calibration loop (Phase 3, SPEC-calibration.md): empirical accuracy per
(step_name, agent, model, provider) bucket, computed fresh from existing
pipeline_steps/step_feedback/run_feedback rows — no persisted table, no curve-fitting
dependency. See CONFIDENCE-REDESIGN.md §4 for the design rationale."""
import time
from dataclasses import dataclass

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
    bins: list[CalibrationBin]   # always exactly len == round(1 / bin_width), in order
    total_n: int                 # total labelled step-executions across all bins

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
) -> dict[tuple[str, str | None, str | None, str | None], "CalibrationBucket"]:
    """Production-scoped, recomputed fresh on every call (no persisted table — see
    SPEC-calibration.md §2). Bucketed by (step_name, agent, model, provider); fan-out
    branches (step_name containing '/') collapse into their group's bucket, per
    CONFIDENCE-REDESIGN.md §7 item 8. Label precedence per step-execution: StepFeedback
    (human) > deterministic_passed=False (automated) > RunFeedback fallback (via run_id)
    > excluded (no label at all — not counted as 0, not counted as unlabelled-N)."""
    n_bins = round(1.0 / bin_width)
    assert abs(n_bins * bin_width - 1.0) < 1e-9, f"bin_width {bin_width} must evenly divide 1.0"

    async with session_factory() as session:
        rows = (await session.execute(
            select(
                PipelineStep.step_name, PipelineStep.agent, PipelineStep.model,
                PipelineStep.provider, PipelineStep.effective_confidence,
                PipelineStep.deterministic_passed,
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
    for step_name, agent, model, provider, predicted, det_passed, step_outcome, run_outcome in rows:
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
        key = (bucket_step_name, agent, model, provider)
        samples.setdefault(key, []).append((predicted, label))

    buckets: dict[tuple, CalibrationBucket] = {}
    for (step_name, agent, model, provider), pairs in samples.items():
        bin_edges = [round(i * bin_width, 10) for i in range(n_bins + 1)]
        bins: list[CalibrationBin] = []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            in_bin = [label for predicted, label in pairs if lo <= predicted < hi or (i == n_bins - 1 and predicted == hi)]
            n = len(in_bin)
            mean_label = sum(in_bin) / n if n else 0.0
            bins.append(CalibrationBin(lo=lo, hi=hi, n=n, mean_label=mean_label, validated=n >= n_min))
        buckets[(step_name, agent, model, provider)] = CalibrationBucket(
            step_name=step_name, agent=agent, model=model, provider=provider,
            bins=bins, total_n=len(pairs),
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

    async def get(
        self, step_name: str, agent: str | None, model: str | None, provider: str | None,
    ) -> CalibrationBucket | None:
        now = time.time()
        if now - self._computed_at > self._ttl:
            self._buckets = await compute_calibration_buckets(
                self._session_factory, bin_width=self._bin_width, n_min=self.n_min,
            )
            self._computed_at = now
        return self._buckets.get((step_name, agent, model, provider))
