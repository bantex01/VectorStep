from typing import Literal
from pydantic import BaseModel, Field, field_validator


class NotificationConfig(BaseModel):
    channel: str
    template: str
    config: dict = Field(default_factory=dict)  # channel-specific settings (e.g. url for webhook)


class ParallelStepConfig(BaseModel):
    name: str
    executor: str
    executor_config: dict = Field(default_factory=dict)
    prompt_template: str
    timeout_seconds: int | None = None
    weight: float = 1.0  # used by weighted_average join strategy
    verifier: "VerifierConfig | None" = None


class VerifierTriggerConfig(BaseModel):
    always: bool = False             # if True, verify regardless of primary confidence
    confidence_below: float = 1.0   # only verify if primary confidence < this (ignored when always=True)
    confidence_above: float = 0.0   # skip if primary confidence < this (ignored when always=True)


class VerifierConfig(BaseModel):
    executor: str
    executor_config: dict = Field(default_factory=dict)
    mode: Literal["reviewer", "challenger"] = "reviewer"
    combination_strategy: Literal["minimum", "veto"] = "minimum"
    veto_floor: float = 0.60        # only used when combination_strategy is "veto"
    trigger: VerifierTriggerConfig = Field(default_factory=VerifierTriggerConfig)


ParallelStepConfig.model_rebuild()


class RetryConfig(BaseModel):
    attempts: int = 3
    backoff: Literal["fixed", "exponential"] = "exponential"
    delay_seconds: float = 1.0


class StepConfig(BaseModel):
    name: str
    executor: str
    executor_config: dict = Field(default_factory=dict)
    confidence_threshold: float = 0.75
    on_low_confidence: Literal["escalate", "abort", "proceed"] = "escalate"
    on_abort: str = "notify"
    prompt_template: str = ""
    timeout_seconds: int | None = None
    when: str | None = None
    verifier: VerifierConfig | None = None
    retry: RetryConfig | None = None


class ParallelGroupInner(BaseModel):
    name: str
    join: Literal["all_must_pass", "any_must_pass", "weighted_average"] = "all_must_pass"
    confidence_threshold: float = 0.75
    on_low_confidence: Literal["escalate", "abort", "proceed"] = "escalate"
    on_abort: str = "notify"
    timeout_seconds: int | None = None
    when: str | None = None
    steps: list[ParallelStepConfig]


class ParallelGroupConfig(BaseModel):
    parallel: ParallelGroupInner


class TriggerConfig(BaseModel):
    match: dict[str, str] = Field(default_factory=dict)


class ContextTemplateConfig(BaseModel):
    include: list[str] = Field(default_factory=list)


class ScheduleConfig(BaseModel):
    cron: str                              # standard 5-field cron expression
    summary: str = ""                     # injected as context.summary
    severity: str = "info"                # injected as context.severity
    labels: dict[str, str] = Field(default_factory=dict)  # extra labels for prompt context


class PipelineConfig(BaseModel):
    name: str
    description: str = ""
    version: int = 1
    trigger: TriggerConfig
    vars: dict[str, str] = Field(default_factory=dict)
    context_template: ContextTemplateConfig = Field(default_factory=ContextTemplateConfig)
    steps: list[StepConfig | ParallelGroupConfig]
    notifications: dict[str, list[NotificationConfig]] = Field(default_factory=dict)
    schedule: ScheduleConfig | None = None

    @field_validator("notifications", mode="before")
    @classmethod
    def _coerce_notifications(cls, v: dict) -> dict:
        """Allow a single notification block or a list per action.

        Single block (existing format, unchanged):
            escalate:
              channel: telegram
              template: ...

        Multiple blocks (new format):
            escalate:
              - channel: telegram
                template: ...
              - channel: webhook
                template: ...
                config:
                  url: https://...
        """
        if not isinstance(v, dict):
            return v
        return {
            action: [item] if isinstance(item, dict) else item
            for action, item in v.items()
        }
