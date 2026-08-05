from typing import Annotated, Any, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


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
    mode: Literal["critic", "independent"] = "critic"
    combination_strategy: Literal["minimum", "veto"] = "minimum"
    veto_floor: float = 0.60        # only used when combination_strategy is "veto"
    trigger: VerifierTriggerConfig = Field(default_factory=VerifierTriggerConfig)
    max_trace_chars: int = 1500     # only used in "critic" mode — see GroundingConfig.max_trace_chars,
                                     # same truncation caveat applies (per tool_result/text event)

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_legacy_mode_names(cls, v: object) -> object:
        """Permanent aliases (Phase 2, SPEC-verifier-semantics.md): 'reviewer' and
        'challenger' were the original names. They are renamed to 'critic' and
        'independent' for clarity, but existing pipeline YAML using the old names must
        keep parsing identically, forever — never remove this or warn that it's
        deprecated."""
        if v == "reviewer":
            return "critic"
        if v == "challenger":
            return "independent"
        return v


class GroundingConfig(BaseModel):
    """Shadow-mode grounding: after the step runs, a blind judge scores how well the
    agent's load-bearing claims are supported by evidence in its own execution trace.
    Phase 0 — recorded only, never gates. (Gating knobs like require_grounding arrive
    in Phase 1; do not add them here.)"""
    agent: str = "grounding-judge"   # gateway agent that performs the constrained cross-reference
    executor: str = "gateway"        # only gateway steps produce a trace to ground against
    executor_config: dict = Field(default_factory=dict)  # extra keys merged into the grounding call (e.g. model)
    timeout_seconds: int = 120       # keep the shadow pass cheap; never block the run for long
    enforce: bool = False   # NEW (Phase 1). False (default) = Phase 0 shadow behaviour,
                             # byte-identical to before this spec. True = G participates
                             # in the gate as a ceiling on combined_trust (§4). Opt-in,
                             # per step — existing grounding: blocks are unaffected.
    max_trace_chars: int = 1500   # per tool_result/text event, before truncation with "…".
                             # A claim whose supporting evidence lands past this cutoff is
                             # invisible to the judge — raise this for steps with long tool
                             # outputs (e.g. full document reads) if grounding is producing
                             # false "unsupported" verdicts because of truncation rather than
                             # an actual hallucination.


class CalibrationConfig(BaseModel):
    """Opt-in per-step calibrated gating (Phase 3, SPEC-calibration.md). Advisory
    calibration reporting requires no config at all — see the Steps Insights UI. This
    block only matters for a step that wants its *gate* to use the empirically-calibrated
    trust instead of the raw self-report/verifier number."""
    enforce: bool = False
    on_uncalibrated: Literal["proceed", "escalate"] = "proceed"
    # "proceed": bucket has < n_min marked outcomes → gate uses raw effective_confidence,
    #            unchanged, this run (advisory-only for this run; still recorded in the
    #            TrustReport as "not yet validated").
    # "escalate": same situation forces combined_trust=0.0, driving the step's EXISTING
    #             on_low_confidence action — the opt-in "no track record → human checks"
    #             policy from CONFIDENCE-REDESIGN.md §4.5. Not the default.


ReadinessStepStatus = Literal["completed", "failed", "aborted", "escalated", "stopped"]
# Deliberately duplicated from analytics.ALL_STEP_STATUSES rather than imported:
# analytics.py imports models.pipeline, so the reverse import would be circular.
# If a new step status is ever added, update both.


class ReadinessOperationalConfig(BaseModel):
    """Cheapest bar there is: pure PipelineStep.status counting. No judgment, no
    confidence, no labels — and the only tier a non-LLM step (notify/webhook/human)
    can ever satisfy, since those never write effective_confidence."""
    min_runs: int = Field(gt=0)   # required — `operational: {}` is a config error
    acceptable_statuses: list[ReadinessStepStatus] = Field(
        default_factory=lambda: ["completed"]
    )
    max_age_days: int | None = Field(default=None, gt=0)
    # Only tier with a time window. None = lifetime track record (the default).
    require_current_config: bool = False
    # False by default: fixing a typo in a prompt shouldn't wipe 30 clean runs.


class ReadinessConfidenceConfig(BaseModel):
    """Mean self-reported effective_confidence. A weak signal alone — the model can be
    confidently wrong — but a useful first checkpoint before anyone has marked anything."""
    min_confidence: float = Field(ge=0.0, le=1.0)   # required
    min_runs: int | None = Field(default=None, gt=0)
    # Strongly recommended. Without it, `min_confidence: 0.9` passes on a single 0.95 run.
    require_current_config: bool = False


class ReadinessAccuracyConfig(BaseModel):
    """Judged accuracy over marked outcomes: correct=1.0, partial=0.5, incorrect=0.0,
    using the SAME label-precedence chain calibration uses (see calibration.resolve_label).
    Proves the step's work is good, without additionally demanding that its confidence
    NUMBER be trustworthy — that stricter claim is the calibration tier."""
    min_accuracy: float = Field(ge=0.0, le=1.0)   # required
    min_marked: int = Field(gt=0)                  # required
    min_human_marked: int | None = Field(default=None, ge=0)
    # Require N labels from a human specifically. Guards the trap in §12.2: only
    # deterministic-check FAILURES produce a free label, so a step with checks and no
    # human feedback has a labelled population that is 100% failures.
    require_current_config: bool = True


class ReadinessCalibrationConfig(BaseModel):
    """The strongest bar: not just 'outcomes are good' but 'the confidence number can be
    trusted'. Uses the existing Phase 3 bucket machinery, with every previously-hardcoded
    constant now owner-settable."""
    n_min: int = Field(default=20, gt=0)
    # PER BIN, not total. See §12.5 — this is the single most misread knob here.
    bin_width: float = Field(default=0.1, gt=0.0, le=1.0)
    max_divergence: float = Field(default=0.15, ge=0.0, le=1.0)
    # Replaces the hardcoded 0.15 in calibration_recommendation().
    require_own_evidence: bool = False
    # False: a shared library step's PRODUCTION evidence from a different pipeline counts.
    # True: only this pipeline's own evidence counts.
    require_current_config: bool = True
    # Must stay True — see the validator below.

    @field_validator("bin_width")
    @classmethod
    def _bin_width_divides_one(cls, v: float) -> float:
        n = round(1.0 / v)
        if abs(n * v - 1.0) >= 1e-9:
            raise ValueError(f"bin_width {v} must evenly divide 1.0")
        return v

    @field_validator("require_current_config")
    @classmethod
    def _must_require_current_config(cls, v: bool) -> bool:
        if v is False:
            raise ValueError(
                "calibration.require_current_config cannot be false — a calibration "
                "bucket is keyed by (prompt_hash, agent_version) by definition, so "
                "'ignore the version' would mean merging buckets and destroying the "
                "reset semantics SPEC-prompt-versioning.md exists to protect. Use the "
                "accuracy tier if you want version-independent judged accuracy."
            )
        return v


class ReadinessConfig(BaseModel):
    """Owner-defined promotion bar (SPEC-readiness-criteria.md). Every CONFIGURED tier
    must pass for the step to be 'ready'; an unconfigured tier is not a failure, it is
    simply not asked. Strictly advisory — nothing here gates, blocks, or writes.

    Settable at pipeline level (house default for every step) and at step level
    (override). Tiers merge, tier contents replace — see readiness.resolve_step_readiness.
    """
    operational: ReadinessOperationalConfig | None = None
    confidence: ReadinessConfidenceConfig | None = None
    accuracy: ReadinessAccuracyConfig | None = None
    calibration: ReadinessCalibrationConfig | None = None


class ShellCheckConfig(BaseModel):
    """Run a shell command; evaluate its output. `run` is executed via the shell (so
    pipes/redirects work, e.g. `curl ... | jq ...`), inheriting the VectorStep process's
    environment and permissions — deliberately unsandboxed, see README §8."""
    type: Literal["shell"] = "shell"
    name: str
    run: str                                    # shell command string
    expect: str = "exit_code == 0"              # bare Jinja2 bool expr, same convention as
                                                 # step.when (NOT wrapped in {{ }} — evaluated
                                                 # via the same _eval_when as when:). Sees
                                                 # `result` (stdout, stripped) and `exit_code`,
                                                 # plus the normal step context (steps.*, vars, etc.)
    timeout_seconds: int = 30


class WebhookCheckConfig(BaseModel):
    """Call a URL; evaluate the response. Same shape as StepFailureWebhookConfig
    (url/method/headers/payload) — deliberately does not raise_for_status, since a
    check might legitimately expect a non-2xx status (e.g. 404 = "does not exist")."""
    type: Literal["webhook"] = "webhook"
    name: str
    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    payload: dict = Field(default_factory=dict)
    expect: str = "response.status_code < 400"   # bare Jinja2 bool expr, same convention as
                                                  # step.when. Sees `response` = {status_code, body}
    timeout_seconds: int = 30


class HumanCheckConfig(BaseModel):
    """Ask a human to approve/reject via the existing human-approval subsystem
    (executors/human.py) — same channels (Telegram/Slack/Teams), same per-team routing,
    same testing-stage behaviour as `executor: human`. Approved = pass, rejected OR
    timed out (in production) = fail."""
    type: Literal["human"] = "human"
    name: str
    message: str                    # Jinja2-rendered against the normal step context
    timeout_seconds: int = 300


DeterministicCheckConfig = Annotated[
    ShellCheckConfig | WebhookCheckConfig | HumanCheckConfig,
    Field(discriminator="type"),
]


ParallelStepConfig.model_rebuild()


class RetryConfig(BaseModel):
    attempts: int = 3
    backoff: Literal["fixed", "exponential"] = "exponential"
    delay_seconds: float = 1.0


class LoopConfig(BaseModel):
    """Refinement loop: re-run the step until effective confidence reaches the target.

    Each iteration receives {{loop.iteration}}, {{loop.prior_confidence}}, and
    {{loop.prior_output}} in its Jinja2 context so the prompt can ask the agent
    to self-correct. The final iteration's output is what gets saved to the DB.
    """
    confidence: float
    max_iterations: int = 3


class StepFailureWebhookConfig(BaseModel):
    url: str
    method: str = "POST"
    headers: dict[str, str] = Field(default_factory=dict)
    payload: dict = Field(default_factory=dict)
    timeout_seconds: int = 30


class StepFailureConfig(BaseModel):
    """Controls what happens when a step's executor raises an error or times out.

    policy:
        "abort"    — (default) stop the pipeline and mark it failed
        "continue" — log the failure and move on to the next step

    webhook: optional outbound call fired on failure regardless of policy
    """
    policy: Literal["abort", "continue"] = "abort"
    webhook: StepFailureWebhookConfig | None = None


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
    on_failure: StepFailureConfig = Field(default_factory=StepFailureConfig)
    verifier: VerifierConfig | None = None
    retry: RetryConfig | None = None
    loop_until: LoopConfig | None = None
    grounding: GroundingConfig | None = None
    deterministic_checks: list[DeterministicCheckConfig] = Field(default_factory=list)
    calibration: CalibrationConfig | None = None
    readiness: ReadinessConfig | None = None

    @field_validator("on_failure", mode="before")
    @classmethod
    def _coerce_on_failure(cls, v: object) -> object:
        if isinstance(v, str):
            return {"policy": v}
        return v


class ParallelGroupInner(BaseModel):
    name: str
    join: Literal["all_must_pass", "any_must_pass", "weighted_average"] = "all_must_pass"
    confidence_threshold: float = 0.75
    on_low_confidence: Literal["escalate", "abort", "proceed"] = "escalate"
    on_abort: str = "notify"
    timeout_seconds: int | None = None
    when: str | None = None
    readiness: ReadinessConfig | None = None
    steps: list[ParallelStepConfig]


class ParallelGroupConfig(BaseModel):
    parallel: ParallelGroupInner


class FanOutConfig(BaseModel):
    """Fan-out: resolve a Jinja2 expression to a list at runtime and spawn one branch per item."""
    name: str
    over: str                  # Jinja2 expression that resolves to a list
    as_var: str = Field(default="item", alias="as")  # variable injected into each branch context
    executor: str
    executor_config: dict = Field(default_factory=dict)
    prompt_template: str = ""
    timeout_seconds: int | None = None
    join: Literal["all_must_pass", "any_must_pass", "weighted_average"] = "all_must_pass"
    confidence_threshold: float = 0.75
    on_low_confidence: Literal["escalate", "abort", "proceed"] = "escalate"
    on_abort: str = "notify"
    max_items: int = 20
    on_empty: Literal["complete", "skip", "abort"] = "complete"
    when: str | None = None
    verifier: VerifierConfig | None = None
    readiness: ReadinessConfig | None = None

    model_config = ConfigDict(populate_by_name=True)


class FanOutGroupConfig(BaseModel):
    fan_out: FanOutConfig


class DedupConfig(BaseModel):
    """Per-pipeline override for webhook deduplication — see README §3a.

    Both fields default to None, meaning "fall back to the service-level
    dedup.enabled / dedup.window_seconds in config.yaml".
    """
    enabled: bool | None = None
    window_seconds: int | None = None


class TriggerConfig(BaseModel):
    # Value is either a plain string (exact match) or a single-key operator dict,
    # e.g. {"in": ["prod", "staging"]} or {"regex": "^api-.*"} — see resolver._matches.
    match: dict[str, Any] = Field(default_factory=dict)
    dedup: DedupConfig | None = None


class BudgetConfig(BaseModel):
    max_tokens: int | None = None  # abort run if accumulated tokens across all steps exceeds this


class ContextTemplateConfig(BaseModel):
    include: list[str] = Field(default_factory=list)


class ScheduleConfig(BaseModel):
    cron: str                              # standard 5-field cron expression
    summary: str = ""                     # injected as context.summary
    severity: str = "info"                # injected as context.severity
    labels: dict[str, str] = Field(default_factory=dict)  # extra labels for prompt context
    team: str | None = None               # owning team — no caller/token to derive it from, declared directly


class LibraryStepConfig(BaseModel):
    """A named, reusable step definition stored in the step library directory.

    Identical to StepConfig but adds description and tags for UI display.
    Used only for validation at load time; the raw dict is what gets merged
    into pipelines so description/tags are naturally stripped by StepConfig.
    """
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    executor: str
    executor_config: dict = Field(default_factory=dict)
    confidence_threshold: float = 0.75
    on_low_confidence: Literal["escalate", "abort", "proceed"] = "escalate"
    on_abort: str = "notify"
    prompt_template: str = ""
    timeout_seconds: int | None = None
    on_failure: StepFailureConfig = Field(default_factory=StepFailureConfig)
    verifier: VerifierConfig | None = None
    retry: RetryConfig | None = None
    loop_until: LoopConfig | None = None
    grounding: GroundingConfig | None = None
    deterministic_checks: list[DeterministicCheckConfig] = Field(default_factory=list)
    calibration: CalibrationConfig | None = None
    readiness: ReadinessConfig | None = None

    @field_validator("on_failure", mode="before")
    @classmethod
    def _coerce_on_failure(cls, v: object) -> object:
        if isinstance(v, str):
            return {"policy": v}
        return v


class PipelineConfig(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = Field(default_factory=list)
    version: int = 1
    stage: Literal["testing", "production"] = "testing"
    trigger: TriggerConfig
    vars: dict[str, str] = Field(default_factory=dict)
    context_template: ContextTemplateConfig = Field(default_factory=ContextTemplateConfig)
    steps: list[StepConfig | ParallelGroupConfig | FanOutGroupConfig]
    notifications: dict[str, list[NotificationConfig]] = Field(default_factory=dict)
    schedule: ScheduleConfig | None = None
    budget: BudgetConfig | None = None
    readiness: ReadinessConfig | None = None

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
