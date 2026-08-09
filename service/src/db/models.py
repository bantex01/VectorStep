import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..utils import utc_now


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"
    __table_args__ = (
        # Closes the dedup TOCTOU race (README §3a "Known limitation"): the DB itself
        # refuses a second 'running' row for the same pipeline+fingerprint. NULLs are
        # never equal in a unique index, so fingerprint=None rows (sub-pipelines,
        # dedup opt-outs) are unaffected. Both dialect kwargs are required —
        # postgresql_where alone is silently ignored on SQLite.
        Index(
            "ix_pipeline_runs_running_fingerprint", "pipeline_name", "fingerprint",
            unique=True,
            postgresql_where=text("status = 'running'"),
            sqlite_where=text("status = 'running'"),
        ),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_name: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    triggered_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    normalised_context: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    raw_payload: Mapped[str] = mapped_column(Text, nullable=False)         # JSON
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    logs: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array of run events
    fingerprint: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # dedup key
    parent_run_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # set for sub-pipeline runs
    team: Mapped[str | None] = mapped_column(String, nullable=True, index=True)  # owning team, from auth token resolution
    stage: Mapped[str] = mapped_column(String, nullable=False, default="production", index=True)  # "testing" | "production" — see PipelineConfig.stage
    # SHA-256[:12] of the pipeline's step sequence (name/executor/agent/model per
    # StepConfig) at trigger time, populated for every run regardless of `durable`.
    # Compared against the current config's fingerprint at resume time — a mismatch
    # means the pipeline changed while the run was down, so it's marked `interrupted`
    # rather than resumed under different semantics (SPEC-durable-runs.md §2).
    config_fingerprint: Mapped[str | None] = mapped_column(String, nullable=True)
    # Set when this run was resumed at least once after a restart. Drives
    # vectorstep_runs_resumed_total (metrics.py) and is never cleared once set, even
    # across a second resume of the same run.
    resumed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # JSON descriptor {source: {...bucket selector...}, candidate: {...}, mode, k,
    # sample_run_ids: [...]} — set only for synthetic runs created by a replay batch
    # (SPEC-replay-shadow-eval.md §2). NULL for every ordinary run. A replay run is
    # always stage='testing', so it needs no separate exclusion logic anywhere that
    # already scopes to stage='production' — this column exists purely to let the
    # replay report reconstruct what was replayed, not to drive any filtering itself.
    replay_of: Mapped[str | None] = mapped_column(Text, nullable=True)

    steps: Mapped[list["PipelineStep"]] = relationship(
        "PipelineStep", back_populates="run", order_by="PipelineStep.step_index"
    )


class PipelineStep(Base):
    __tablename__ = "pipeline_steps"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, ForeignKey("pipeline_runs.id"), nullable=False)
    step_name: Mapped[str] = mapped_column(String, nullable=False)
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    executor: Mapped[str] = mapped_column(String, nullable=False)
    agent: Mapped[str | None] = mapped_column(String, nullable=True)   # executor_config.agent
    model: Mapped[str | None] = mapped_column(String, nullable=True)  # actual model used (from response metadata)
    provider: Mapped[str | None] = mapped_column(String, nullable=True)  # gateway provider key (gateway executor only)
    prompt_hash: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # sha256[:12] of the step's normalised prompt_template — scopes the calibration
    # bucket to the prompt that actually produced the outcome. NULL for pre-migration
    # rows and for non-LLM steps with no template. NULL is its own bucket dimension,
    # never a wildcard (SPEC-prompt-versioning.md §2).
    agent_version: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    # Gateway-reported content hash of the agent's full config incl. soul.md, from
    # meta.agentMeta.agentVersion. NULL for non-gateway executors, for a Gateway
    # older than SPEC-prompt-versioning.md, and for pre-migration rows.
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)          # JSON
    parsed_output: Mapped[str | None] = mapped_column(Text, nullable=True)       # JSON
    verifier_output: Mapped[str | None] = mapped_column(Text, nullable=True)     # JSON
    verifier_mode: Mapped[str | None] = mapped_column(String, nullable=True)    # "critic" | "independent" (new rows); "reviewer" | "challenger" (historical rows, pre-SPEC-verifier-semantics.md)
    verifier_agent: Mapped[str | None] = mapped_column(String, nullable=True)    # "executor:agent", mirrors `agent` above; NULL for runs before this column existed or steps with no verifier
    verifier_model: Mapped[str | None] = mapped_column(String, nullable=True)   # actual model used by the verifier call (from response metadata)
    verifier_provider: Mapped[str | None] = mapped_column(String, nullable=True)  # gateway provider key (gateway executor only)
    verifier_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)  # rendered prompt actually sent to the verifier (gateway executor only); NULL if no verifier ran or it didn't stash one
    status: Mapped[str] = mapped_column(String, nullable=False)
    primary_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    effective_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # G ∈ [0,1], NULL when not computed
    trust_report: Mapped[str | None] = mapped_column(Text, nullable=True)         # JSON TrustReport (shadow, per step)
    deterministic_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # NULL = no checks declared
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    artifacts: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: {key: reference}
    agent_trace: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: ordered trace events
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verifier_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    verifier_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grounding_model: Mapped[str | None] = mapped_column(String, nullable=True)  # actual model used by the grounding judge
    grounding_provider: Mapped[str | None] = mapped_column(String, nullable=True)
    grounding_input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    grounding_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Priced once at step-save time from pricing.currency's rate table (SPEC-cost-accounting.md).
    # NULL = no rate matched the step's provider/model (unpriced), never 0 — 0 means an
    # operator explicitly priced that provider/model at zero (e.g. local Ollama). Never
    # recomputed later: historical cost reflects the rate in force when the tokens were
    # bought, even after the pricing table changes.
    cost: Mapped[float | None] = mapped_column(Float, nullable=True)

    run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="steps")


class PendingApproval(Base):
    """Durable mirror of executors/human.py's in-memory `_pending_approvals`/
    `_pending_meta` dicts — written when a HITL approval request is sent and deleted
    when it resolves normally (approved/rejected/timed out). A row that survives to
    the next startup means the process died while a human decision was outstanding;
    resume re-arms the wait against the same token instead of re-sending the message
    (SPEC-durable-runs.md §2 "HITL steps resume as waiting, not re-executed").
    """
    __tablename__ = "pending_approvals"

    token: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(String, ForeignKey("pipeline_runs.id"), nullable=False, index=True)
    step_name: Mapped[str] = mapped_column(String, nullable=False)
    pipeline_name: Mapped[str | None] = mapped_column(String, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    team: Mapped[str | None] = mapped_column(String, nullable=True)
    stage: Mapped[str] = mapped_column(String, nullable=False, default="production")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


class RunFeedback(Base):
    __tablename__ = "run_feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    pipeline_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String, nullable=False)  # "correct" | "partial" | "incorrect"
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


class StepPromptVersion(Base):
    """Content-addressed registry of prompt templates actually used by a run.

    Write-on-miss at step-save time. Grows with prompt EDITS, not runs — tens of
    rows a year. Because it is content-addressed, reverting a prompt to an earlier
    version automatically re-joins that version's existing calibration history.
    That is intended behaviour, not an accident.
    """
    __tablename__ = "step_prompt_versions"

    prompt_hash: Mapped[str] = mapped_column(String, primary_key=True)
    step_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    template: Mapped[str] = mapped_column(Text, nullable=False)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


class AgentVersionSnapshot(Base):
    """Snapshot of a Gateway agent's definition at the version VectorStep observed.

    VectorStep cannot compute this hash itself — the Gateway owns the algorithm and the
    source files. On seeing an unknown agent_version, VectorStep asks the Gateway for the
    current definition and stores it ONLY if the Gateway's reported current version
    matches the one being resolved. If the operator changed the agent again in
    between, text stays NULL and `note` records why — an honest gap beats a
    mislabelled snapshot.
    """
    __tablename__ = "agent_version_snapshots"

    agent_version: Mapped[str] = mapped_column(String, primary_key=True)
    agent: Mapped[str] = mapped_column(String, nullable=False, index=True)  # "gateway:name"
    soul_md: Mapped[str | None] = mapped_column(Text, nullable=True)
    agent_yaml: Mapped[str | None] = mapped_column(Text, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


class StepFeedback(Base):
    __tablename__ = "step_feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    step_id: Mapped[str] = mapped_column(
        String, ForeignKey("pipeline_steps.id"), nullable=False, unique=True, index=True
    )  # the specific step EXECUTION this feedback is for — one row per step, upserted
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)      # denormalised for lookup
    pipeline_name: Mapped[str] = mapped_column(String, nullable=False, index=True)  # denormalised
    step_name: Mapped[str] = mapped_column(String, nullable=False)               # denormalised (may contain "/" for fan-out)
    outcome: Mapped[str] = mapped_column(String, nullable=False)  # "correct" | "partial" | "incorrect"
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
