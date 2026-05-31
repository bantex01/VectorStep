import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    pipeline_name: Mapped[str] = mapped_column(String, nullable=False)
    source: Mapped[str] = mapped_column(String, nullable=False)
    triggered_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    status: Mapped[str] = mapped_column(String, nullable=False, default="running")
    normalised_context: Mapped[str] = mapped_column(Text, nullable=False)  # JSON
    raw_payload: Mapped[str] = mapped_column(Text, nullable=False)         # JSON
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    logs: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON array of run events

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
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)          # JSON
    parsed_output: Mapped[str | None] = mapped_column(Text, nullable=True)       # JSON
    verifier_output: Mapped[str | None] = mapped_column(Text, nullable=True)     # JSON
    verifier_mode: Mapped[str | None] = mapped_column(String, nullable=True)    # "reviewer" | "challenger"
    status: Mapped[str] = mapped_column(String, nullable=False)
    primary_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    verifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    effective_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    executed_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)

    run: Mapped["PipelineRun"] = relationship("PipelineRun", back_populates="steps")
