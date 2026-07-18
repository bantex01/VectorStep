import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from ..utils import utc_now


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

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
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)          # JSON
    parsed_output: Mapped[str | None] = mapped_column(Text, nullable=True)       # JSON
    verifier_output: Mapped[str | None] = mapped_column(Text, nullable=True)     # JSON
    verifier_mode: Mapped[str | None] = mapped_column(String, nullable=True)    # "reviewer" | "challenger"
    status: Mapped[str] = mapped_column(String, nullable=False)
    primary_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    effective_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # G ∈ [0,1], NULL when not computed
    trust_report: Mapped[str | None] = mapped_column(Text, nullable=True)         # JSON TrustReport (shadow, per step)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)
    artifacts: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: {key: reference}
    agent_trace: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: ordered trace events
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="steps")


class RunFeedback(Base):
    __tablename__ = "run_feedback"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    pipeline_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    outcome: Mapped[str] = mapped_column(String, nullable=False)  # "correct" | "partial" | "incorrect"
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utc_now)


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
