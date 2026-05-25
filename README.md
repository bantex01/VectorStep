# P-Ork Orchestration Service

## Project Overview

This is a **webhook-triggered, YAML-configured AI pipeline orchestration service** built in Python with FastAPI. It receives webhooks from any source (Alertmanager, Grafana, Atlassian, etc.), normalises the payload, resolves a named pipeline config, and executes a multi-step AI pipeline using pluggable agent backends (initially OpenClaw, extensible to others).

The service is designed to be:
- **Source agnostic** — any webhook source is supported via pluggable parsers
- **Tool agnostic** — AI executor backends are adapters behind a common interface
- **Config driven** — all pipeline logic lives in YAML files, not code
- **Modular** — adding a new source parser or executor adapter should not require changes to core logic

Primary use case is observability automation (alert triage, Grafana investigation, bounded remediation) but the design is intentionally general purpose.

---

## Tech Stack

- **Python 3.11+**
- **FastAPI** — webhook endpoint and optional status/runs API
- **Pydantic v2** — normalised context schema, pipeline config models, LLM output validation
- **SQLAlchemy + aiosqlite** — async SQLite for pipeline run storage
- **httpx** — async HTTP client for calling OpenClaw and other executors
- **Jinja2** — prompt template rendering (`{{variable}}` syntax in YAML configs)
- **PyYAML** — pipeline config loading
- **APScheduler 3.x** — in-process cron scheduler for time-triggered pipeline runs
- **uvicorn** — ASGI server for local development

---

## Project Structure

```
service/
├── agents/                     # SOUL.md drafts for OpenClaw agents (drop into workspace)
│   ├── order-intake/
│   │   └── SOUL.md
│   └── customer-comms/
│       └── SOUL.md
├── pipelines/                  # YAML pipeline configs (git controlled)
│   ├── alert-triage-critical.yaml
│   ├── new-order.yaml          # Generic source test pipeline (e-commerce order)
├── src/
│   ├── main.py                 # FastAPI app entry point, webhook endpoint
│   ├── normaliser/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseParser abstract class
│   │   ├── alertmanager.py     # Alertmanager-specific parser
│   │   └── generic.py          # Generic source parser (standardised JSON schema)
│   ├── executors/
│   │   ├── __init__.py
│   │   ├── base.py             # BaseExecutor abstract class
│   │   └── openclaw.py         # OpenClaw executor adapter
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── loader.py           # Loads and validates YAML pipeline configs
│   │   ├── resolver.py         # Matches normalised context to a pipeline
│   │   ├── runner.py           # Executes pipeline steps, manages flow control
│   │   └── context.py          # Builds and passes context between steps
│   ├── models/
│   │   ├── __init__.py
│   │   ├── context.py          # NormalisedContext Pydantic model
│   │   ├── pipeline.py         # PipelineConfig, StepConfig Pydantic models
│   │   └── llm.py              # LLMOutput Pydantic model (structured step output)
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py         # SQLAlchemy engine, session management
│   │   └── models.py           # PipelineRun, PipelineStep ORM models
│   └── notifications/
│       ├── __init__.py
│       └── telegram.py         # Telegram notification handler
├── tests/
├── Dockerfile
├── config.yaml                 # Service-level config (port, dirs, executor settings)
├── requirements.txt
└── CLAUDE.md                   # This file
```

---

## Core Concepts

### 1. Webhook Intake & Source Detection

Single endpoint: `POST /webhook`

Source is identified via query parameter: `/webhook?source=alertmanager`

Header fallback also supported: `X-Pipeline-Source: alertmanager`

The source value maps to a registered parser class. If no source param/header is present, the service attempts content-based detection as a last resort.

Registered sources: `alertmanager`, `generic`.

### 2. Normalisation Layer

Each source parser implements `BaseParser` and produces a `NormalisedContext` object. This is the universal data model that all downstream pipeline logic operates on. Source-specific details are preserved in `raw` and `metadata` fields but pipeline configs never reference them directly.

```python
class NormalisedContext(BaseModel):
    source: str                    # e.g. "alertmanager" — for audit only
    pipeline: str                  # pipeline config name to resolve
    severity: str | None           # critical / warning / info
    labels: dict[str, str]         # service, environment, etc.
    summary: str | None            # human readable description of the event
    raw: dict                      # original unmodified payload
    metadata: dict                 # source-specific extras
    received_at: datetime
```

### 2a. Generic Source

Any tool that can send HTTP can target `POST /webhook?source=generic` using a standardised schema — no bespoke parser needed. The generic source always requires an explicit `pipeline` name, which bypasses the resolver's trigger matching entirely.

**Generic payload schema:**
```json
{
  "pipeline": "my-pipeline",      // required — names the pipeline explicitly
  "event": "order.placed",        // optional — stored in labels["event"] for audit
  "source": "shopify",            // optional — defaults to "generic"
  "summary": "Human readable...", // optional
  "data": { ... }                 // optional — free-form dict, lands in metadata
}
```

**Mapping to NormalisedContext:**
- `pipeline` → `pipeline` (resolver uses this directly, skips trigger matching)
- `event` → `labels["event"]`
- `source` → `source`
- `summary` → `summary`
- `data` → `metadata` (accessible in prompts as `{{metadata.field_name}}` or just `{{field_name}}` via leaf flattening)

**Agent SOUL.md files** live under `service/agents/<agent-name>/SOUL.md` as drafts — copy into the OpenClaw agent workspace (`~/.openclaw/workspace-<agent-name>/SOUL.md`) when creating a new agent.

### 3. Pipeline Resolution

The `resolver` loads all YAML configs from `PIPELINE_CONFIG_DIR` and matches the incoming `NormalisedContext` against each config's `trigger.match` block. First match wins. Configs should be ordered by specificity (more specific matches first).

Pipeline name can also be explicitly set by the source parser if the webhook payload contains a pipeline attribute (e.g. an Alertmanager label `pipeline: alert-triage-critical`).

### 4. Pipeline Config Schema (YAML)

`model` is not a top-level step field. By default, model selection is handled by the named OpenClaw agent's own config. To override the model for a specific step, set `executor_config.model` — this is passed directly to the Gateway API and takes precedence over the agent's configured model.

Steps support an optional `verifier` block for independent confidence verification by a second agent. The verifier trigger can be set to `always: true` (fires unconditionally), or scoped to a confidence band (`confidence_above` to `confidence_below`) to save calls when the outcome is obvious. Combination strategies: `minimum` (both must clear `confidence_threshold`) or `veto` (verifier below `veto_floor` blocks regardless of primary score). See §5 for full trigger configuration examples.

The `steps` list is heterogeneous — each entry is either a sequential step (has a `name:` key) or a parallel group (has a `parallel:` key). See §6 for parallel group configuration.

```yaml
name: alert-triage-critical
description: Full triage pipeline for critical alerts
version: 1

trigger:
  match:
    severity: critical          # matched against NormalisedContext fields/labels
    environment: prod           # all conditions must match (AND logic)

context_template:
  include:                      # fields auto-injected into every step prompt
    - severity
    - labels.service
    - labels.environment
    - summary

steps:
  - name: initial-triage
    executor: openclaw
    executor_config:
      agent: sre-triage-sonnet          # named agent in OpenClaw
      session_key: "agent:sre-triage-sonnet:{{pipeline_run_id}}:triage"
      model: anthropic/claude-sonnet-4-6   # optional — overrides the agent's configured model
      thinking_level: low                  # optional — off|minimal|low|medium|high|xhigh
    confidence_threshold: 0.75
    on_low_confidence: escalate  # escalate | abort | proceed
    on_abort: notify
    timeout_seconds: 120              # optional — cancels hung executor calls; step status becomes "failed"
    prompt_template: |
      You are an SRE triaging a {{severity}} alert for {{labels.service}}
      in {{labels.environment}}.

      Alert summary: {{summary}}

      Return JSON only, no other text:
      {
        "confidence": 0.0,
        "summary": "...",
        "next_step_context": "...",
        "reasoning": {
          "supports": "...",
          "contradicts": "...",
          "assumptions": "..."
        }
      }

  - name: remediation
    executor: openclaw
    executor_config:
      agent: sre-remediation-sonnet
      session_key: "agent:sre-remediation-sonnet:{{pipeline_run_id}}:remediation"
    confidence_threshold: 0.85
    on_low_confidence: escalate
    on_abort: notify
    prompt_template: |
      ...
    verifier:
      executor: openclaw
      executor_config:
        agent: sre-verifier-opus        # stronger model for verification
      combination_strategy: veto        # or: minimum
      veto_floor: 0.60                  # only for veto strategy
      trigger:
        confidence_below: 0.95          # skip if primary is clearly passing
        confidence_above: 0.50          # skip if primary is clearly failing

notifications:
  # Single notifier per action (original format — still valid):
  escalate:
    channel: telegram
    template: |
      Escalated: {{pipeline_name}}
      Service: {{labels.service}}
      Step: {{current_step}}
      Reason: {{summary}}

  # Multiple notifiers per action (list format):
  escalate:
    - channel: telegram
      template: |
        Escalated: {{pipeline_name}}
        Service: {{labels.service}}
    - channel: webhook
      template: |
        {"text": "Escalated: {{pipeline_name}} — {{labels.service}}"}
      config:
        url: https://hooks.slack.com/services/...
        headers:
          Authorization: ${SLACK_TOKEN}

  notify:
    channel: telegram
    template: |
      Aborted: {{pipeline_name}}
      Service: {{labels.service}}
      Step: {{current_step}}

schedule:                        # optional — omit for webhook-only pipelines
  cron: "*/5 * * * *"           # standard 5-field cron expression
  summary: "Scheduled health check for my-service"
  severity: warning              # injected as context.severity
  labels:
    service: my-service
    environment: prod
```

### 5. Verifier Trigger Configuration

The `trigger` block on a `verifier` controls when the verifier agent fires. Three patterns are supported.

Two verifier **modes** are available:

| Mode | Behaviour |
|---|---|
| `reviewer` (default) | Verifier receives the primary agent's prompt and full response — critiques the reasoning |
| `challenger` | Verifier receives only the original task prompt — executes the same task independently without seeing the primary output |

Challenger mode is useful when you want a genuinely independent data point rather than a review. Typically the stronger model goes in the challenger role — it stands on its own assessment rather than reviewing someone else's work. Both modes produce an `LLMOutput` and feed into the same combination strategies.

**Combination strategies by mode:**

| Strategy | Reviewer interpretation | Challenger interpretation |
|---|---|---|
| `minimum` | Primary passes only if reviewer also agrees the reasoning is sound | Both agents independently attempted the task and must both be confident — strict consensus |
| `veto` | Reviewer can block if it spots poor reasoning in the primary's response | Challenger (typically the stronger model) gets a veto — if it independently came back less confident, that overrides the primary's score |

`minimum` is the natural default for challenger mode. `veto` is appropriate when you want to explicitly trust the challenger model more — it can override an overconfident primary even when both attempted the task independently.

---

**Always verify — run the verifier on every execution of this step regardless of primary confidence.**

Use this when the step is high-stakes and you want a second opinion unconditionally.

```yaml
verifier:
  executor: openclaw
  executor_config:
    agent: sre-verifier-opus
  mode: reviewer          # or: challenger
  combination_strategy: minimum
  trigger:
    always: true
```

---

**Band-based — only verify when primary confidence is uncertain (the interesting middle ground).**

Skips the verifier when the outcome is obvious (very high or very low confidence), saving LLM calls.

```yaml
verifier:
  executor: openclaw
  executor_config:
    agent: sre-verifier-opus
  combination_strategy: veto
  veto_floor: 0.60
  trigger:
    confidence_below: 0.95   # skip if primary is clearly confident
    confidence_above: 0.50   # skip if primary is clearly failing
```

The verifier fires only when: `confidence_above < primary_confidence < confidence_below`.

---

**Default (no trigger block) — equivalent to the full band with defaults.**

`confidence_below` defaults to `1.0` and `confidence_above` defaults to `0.0`, so the verifier fires for any primary confidence value. Functionally the same as `always: true` but slightly less explicit.

```yaml
verifier:
  executor: openclaw
  executor_config:
    agent: sre-verifier-opus
  combination_strategy: minimum
  # no trigger block — fires for all confidence values
```

---

**Combination strategies**

| Strategy | Behaviour |
|---|---|
| `minimum` | `effective = min(primary, verifier)` — both must be confident |
| `veto` | Primary passes through unless verifier < `veto_floor`, in which case verifier score is used (forces a threshold breach) |

Verifier failures (executor errors) are non-fatal — the runner logs a warning and falls back to primary confidence only.

---

### 6. Parallel Groups

A `parallel:` entry in the step list runs multiple branches concurrently via `asyncio.gather` and joins their confidence scores before applying standard flow control.

**YAML structure:**

```yaml
steps:
  - name: initial-triage          # sequential step — unchanged
    executor: openclaw
    ...

  - parallel:                     # parallel group — branches run concurrently
      name: context-gathering
      join: all_must_pass         # join strategy: all_must_pass | any_must_pass | weighted_average
      confidence_threshold: 0.70  # applied after join
      on_low_confidence: escalate
      on_abort: notify
      timeout_seconds: 90         # optional — cancels the whole group if exceeded
      steps:
        - name: check-runbook
          executor: openclaw
          executor_config:
            agent: runbook-lookup
          prompt_template: |
            Look up the runbook for {{labels.service}}...
        - name: check-grafana
          executor: openclaw
          executor_config:
            agent: grafana-analyst
          weight: 2.0             # optional — only used by weighted_average strategy
          prompt_template: |
            Check Grafana for {{labels.service}}...

  - name: remediation             # sequential — references parallel branch outputs directly
    executor: openclaw
    prompt_template: |
      Runbook: {{steps.check_runbook.summary}}
      Grafana: {{steps.check_grafana.summary}}
```

**Join strategies:**

| Strategy | Behaviour |
|---|---|
| `all_must_pass` | `effective = min(all confidences)` — any weak branch drags down the group |
| `any_must_pass` | `effective = max(all confidences)` — useful when one source finding is enough |
| `weighted_average` | Weighted mean across branches; each branch has an optional `weight:` (default 1.0) |

**Branch outputs in context:** Each branch's output is registered in `step_outputs` by its individual name, so downstream steps reference them as `{{steps.check_runbook.summary}}` — identical to sequential step references.

**Per-branch behaviour:**
- Sub-steps have no `confidence_threshold` / `on_low_confidence` — gating happens at the group level only.
- Each branch may have its own `timeout_seconds` (independent of the group timeout) and optional `verifier` block. The verifier adjusts that branch's confidence before the join.
- A branch that errors or times out contributes a confidence of `0.0` to the join.
- The `proceed` signal is ignored for branch outputs — flow control decisions belong to sequential steps only.

**DB storage:** Each branch is stored as a separate `pipeline_steps` row with `step_name = "<group_name>/<branch_name>"`. Branches share the group's `step_index` prefix so they sort together in run detail responses.

---

### 7. Cron Scheduler

Any pipeline can declare an optional `schedule:` block to run on a cron schedule in addition to (or instead of) webhook triggers. The scheduler is an in-process `AsyncIOScheduler` (APScheduler 3.x) that fires alongside the webhook listener — no separate process or infrastructure required.

**`ScheduleConfig` model:**

```python
class ScheduleConfig(BaseModel):
    cron: str                              # standard 5-field cron expression
    summary: str = ""                     # injected as context.summary
    severity: str = "info"                # injected as context.severity
    labels: dict[str, str] = {}           # extra labels for prompt context
```

**How it works:**

On startup (and after every `/reload` or SIGHUP), `_register_schedules()` removes all existing `pipeline:*` jobs and re-registers from loaded configs. When a job fires, `_run_scheduled_pipeline()` synthesises a `NormalisedContext` with `source="scheduler"` and the values from the `schedule:` block, then calls `_run_pipeline()` directly — identical code path to a webhook trigger.

**YAML example:**

```yaml
schedule:
  cron: "0 9 * * 1-5"     # 09:00 Mon–Fri
  summary: "Daily morning service health sweep"
  severity: info
  labels:
    service: my-service
    environment: prod
```

**Scheduling endpoint:**

```bash
GET /schedules
# → {"schedules": [{"pipeline": "my-pipeline", "cron": "0 9 * * 1-5", "next_run": "2026-04-28T09:00:00+01:00"}]}
```

**Notes:**
- A pipeline with no `schedule:` block is webhook-only — no job is registered.
- Schedules re-register atomically on reload; in-flight runs are never interrupted.
- The scheduler shuts down cleanly on process exit (`wait=False` so it doesn't block).
- Time zone follows the local system clock (APScheduler default); use `CronTrigger(timezone=...)` if you need explicit tz control in code.

---

### 8. Executor Adapter Pattern

All executors implement `BaseExecutor`:

```python
class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        pass
```

The `LLMOutput` model is the contract between steps:

```python
class LLMOutput(BaseModel):
    confidence: float
    proceed: bool = True          # false = pipeline closes cleanly; no further steps run
    proceed_reason: str | None = None  # why proceed was set; required for steps that may stop the pipeline
    summary: str
    next_step_context: str
    reasoning: dict | None = None
    model: str | None = None  # populated from executor metadata
    raw_response: dict        # full unparsed response for audit
    # extra fields allowed (e.g. jira_ticket, doc_found) — passed to next step context
```

Agents report findings and score their own confidence. They do not decide pipeline flow.
`proceed` is a pipeline signal only — it carries no domain meaning. Default is true; set false only when the agent is confident no further steps are warranted and this should (generally) be determined by the agents SOUL. Example, the agent should be scoped well enough for a particular job and have instructions that match the fact that the agent is just a atep in a process or a decision maker.

Currently implemented executors:
- `OpenClawWSExecutor` (`executor: openclaw`) — invokes OpenClaw agents via the Gateway WebSocket API (`ws://localhost:18789/rpc`). Authenticates using Ed25519 device identity from `~/.openclaw/identity/device.json`. Fires an `agent` call and waits for the final result frame, which contains the response text, model used, and duration. Session isolation is handled server-side by the Gateway — no file deletion needed. Scans payloads in reverse for the last valid JSON block.

`executor_config` keys:

| Key | Required | Description |
|---|---|---|
| `agent` | yes | OpenClaw agent name |
| `session_key` | no | Jinja2 template; must start with `agent:{agent-name}:`. Default: `agent:{{agent}}:pipeline:{{pipeline_run_id}}:{{current_step}}` |
| `model` | no | Model override, e.g. `anthropic/claude-opus-4-7`. Overrides the agent's configured model for this step only. |
| `thinking_level` | no | `off\|minimal\|low\|medium\|high\|xhigh` — controls the model's thinking budget |

- `HumanExecutor` (`executor: human`) — sends a Telegram inline keyboard message and pauses the pipeline until the operator clicks Approve or Reject, or `timeout_seconds` elapses.

| Outcome | confidence | proceed |
|---|---|---|
| Approved | 1.0 | true |
| Rejected | 0.0 | true — triggers `on_low_confidence` action |
| Timeout | — | step marked `failed` |

The `prompt_template` renders to the message text shown in Telegram — use it to give the operator enough context to make a decision. No `executor_config` keys are required. Default timeout is 300s if `timeout_seconds` is not set on the step. Credentials come from `config.yaml notifications.telegram` (`TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` env vars are a fallback). The HITL polling loop starts automatically when Telegram credentials are configured — no extra setup needed.

**Bot requirement:** must use a separate Telegram bot from OpenClaw. Telegram only allows one simultaneous `getUpdates` poller per bot token, and the HITL poller shares the token with the `/run` command handler.

```yaml
- name: approve-remediation
  executor: human
  timeout_seconds: 600        # default 300s
  confidence_threshold: 0.5
  on_low_confidence: abort
  on_abort: notify
  prompt_template: |
    <b>Approve remediation for {{labels.service}}?</b>

    Proposed action: {{steps.investigation.next_step_context}}

    Approve to proceed, Reject to abort.
```

- `WebhookExecutor` (`executor: webhook`) — POSTs the rendered `prompt_template` as the request body to a URL. Returns `confidence=1.0` on any HTTP 2xx; non-2xx raises and triggers the step's retry/fail flow. The response body (up to 500 chars) is stored in `next_step_context` so downstream steps can reference it. Works with `when:`, `retry:`, and verifiers like any other step.

```yaml
- name: notify-slack
  executor: webhook
  executor_config:
    url: https://hooks.slack.com/services/...
    method: POST                   # default
    content_type: application/json # default
    timeout_seconds: 10            # executor-level timeout (default 30)
    headers:
      Authorization: ${SLACK_TOKEN}  # ${ENV_VAR} substitution supported
  confidence_threshold: 0.0
  on_low_confidence: proceed
  prompt_template: |
    {"text": "Alert resolved: {{labels.service}} — {{steps.triage.summary}}"}
```

### 9. Flow Control

The `runner` controls all step execution and flow decisions. Agents never decide what happens next — that is the runner's job. For each step:

0. Evaluate optional `when:` condition — if false, skip the step entirely and move to the next one
1. Parse and validate the LLM response as `LLMOutput` via Pydantic
2. Check `confidence` against step's `confidence_threshold`
3. If confidence below threshold, apply `on_low_confidence` action (`escalate | abort | proceed`)
4. If confidence passes, check `proceed`:
   - `proceed: false` → pipeline stops with status `stopped` (triggers `stopped` notification if configured)
   - `proceed: true` → continue to next step
5. Build context for next step using `{{steps.step_name.field}}` references
6. Record step result to SQLite before proceeding

### Step and Run Status Reference

**Step statuses** — set on each `pipeline_steps` row:

| Status | Set when | Counts as failure? |
|---|---|---|
| `completed` | Step ran successfully and `proceed: true` — pipeline continues | No |
| `stopped` | Step ran successfully and returned `proceed: false` — agent signalled no further steps warranted | No |
| `escalated` | `effective_confidence < threshold` and `on_low_confidence: escalate` — pipeline escalates to a human | No |
| `aborted` | `effective_confidence < threshold` and `on_low_confidence: abort` — pipeline halts quietly | No |
| `failed` | Executor exception, timeout, bad JSON, or schema validation error — the step did not produce usable output | **Yes** |

`escalated` and `aborted` both fire because confidence was below the configured threshold — a pipeline policy decision, not a model failure. They differ only in which notification fires: `escalated` always routes to the `escalate` notification template (high-visibility, includes confidence detail); `aborted` routes to the step's `on_abort` notification (typically quieter).

For agent success rate calculations, `failed` is the only status that counts against a model. All other statuses represent the model returning usable output.

**Pipeline run statuses** — set on each `pipeline_runs` row:

| Status | Meaning |
|---|---|
| `completed` | All steps ran to completion |
| `stopped` | A step returned `proceed: false` — clean intentional stop |
| `escalated` | A step was escalated — run halted, human notified |
| `aborted` | A step aborted due to low confidence — run halted |
| `failed` | A step raised an unhandled error |
| `running` | Currently in progress |

If "escalation" is a desired business outcome (page someone, open a P1), it should be a pipeline *step* — not something an agent signals.

### 9a. Conditional Steps (`when:`)

Any sequential step or parallel group can have an optional `when:` field containing a Jinja2-compatible boolean expression. The runner evaluates it against the step context before calling any executor — a false result skips the step cleanly (no DB row, no executor call).

```yaml
steps:
  - name: triage
    executor: openclaw
    executor_config:
      agent: sre-triage
    prompt_template: |
      Analyse the alert and return JSON with an "action" field:
      "remediate" | "escalate_human" | "ignore"

  - name: auto-remediation
    when: "steps.triage.action == 'remediate'"
    executor: openclaw
    executor_config:
      agent: sre-remediator
    prompt_template: |
      Triage says: {{steps.triage.summary}}. Attempt remediation.

  - name: page-oncall
    when: "steps.triage.action == 'escalate_human'"
    executor: openclaw
    executor_config:
      agent: pagerduty-caller
    prompt_template: |
      Page oncall with context: {{steps.triage.summary}}
```

**Expression syntax** — evaluated via Jinja2 so dot-notation on dicts works identically to prompt templates. Anything available in a prompt is available in `when:`:

```yaml
when: "steps.triage.action == 'remediate'"      # branch on agent output field
when: "severity == 'critical'"                   # branch on alert severity
when: "steps.triage.confidence > 0.9"           # branch on confidence score
when: "labels.environment == 'prod'"             # branch on context label
```

**`when:` vs `proceed: false`** — they are complementary, not alternatives:
- `when:` — the pipeline *author* decides in advance which steps are relevant given what prior steps found
- `proceed: false` — the *agent* signals the pipeline is complete and nothing further is warranted

A skipped step (false `when:`) is invisible to subsequent steps — its name does not appear in `step_outputs` and cannot be referenced in downstream `{{steps.name.field}}` expressions.

### 10. Prompt Construction

Jinja2 renders prompt templates with a context dict containing:
- All fields from `context_template.include` resolved from `NormalisedContext`
- `{{pipeline_run_id}}` — unique ID for this run
- `{{pipeline_name}}` — name of the current pipeline
- `{{current_step}}` — name of the current step
- `{{steps.step_name.field}}` — output fields from any previously completed step (hyphens in step names must be written as underscores: `first-line-triage` → `steps.first_line_triage`)

### 11. Session Keys

Each pipeline step gets an isolated OpenClaw session key scoped to the run:
`pipeline:{run_id}:{step_name}`

This ensures:
- Concurrent pipeline runs don't share session state
- Each step starts clean without prior step noise
- On escalation, a human-readable summary can be injected into a new interactive session for conversational follow-up

### 12. Run Storage (SQLite)

Two tables:

**pipeline_runs**
- `id` (uuid, pk)
- `pipeline_name` (str)
- `source` (str)
- `triggered_at` (datetime)
- `status` (running / completed / aborted / escalated)
- `normalised_context` (json)
- `raw_payload` (json)
- `completed_at` (datetime, nullable)

**pipeline_steps**
- `id` (uuid, pk)
- `run_id` (fk → pipeline_runs.id)
- `step_name` (str)
- `step_index` (int)
- `executor` (str)
- `model` (str)
- `prompt` (text)
- `raw_output` (json)
- `parsed_output` (json)
- `status` (str)
- `confidence` (float, nullable)
- `duration_ms` (int)
- `executed_at` (datetime)

### 13. Management Endpoints

**Pipeline reload** — re-read all YAML configs from disk without restarting the process. Running pipeline runs are unaffected (they hold their resolved config already).

```bash
POST /reload
# → {"status": "reloaded", "pipelines_loaded": 3}

# Or via SIGHUP (same effect, logged to stdout):
kill -HUP <uvicorn-pid>
```

**Schedules** — list active cron jobs and their next fire times.

```bash
GET /schedules
# → {"schedules": [{"pipeline": "feature-test", "cron": "*/5 * * * *", "next_run": "2026-04-27T20:00:00+01:00"}]}
```

**Runs API** — query the SQLite run history over HTTP.

```bash
# List runs — newest first. Optional filters: ?status=escalated, ?pipeline=alert-triage-critical
# Pagination: ?limit=50&offset=0 (max limit 200)
GET /runs
# → {"runs": [{id, pipeline_name, source, status, triggered_at, completed_at}, ...], "limit": 50, "offset": 0}

# Full run detail — includes normalised_context and all steps with confidence scores
GET /runs/{run_id}
# → {id, pipeline_name, source, status, triggered_at, completed_at, normalised_context, steps: [...]}

# Step fields in detail response:
# name, index, executor, agent, model, status,
# primary_confidence, verifier_confidence, effective_confidence,
# duration_ms, executed_at, parsed_output
```

---

## OpenClaw Integration

OpenClaw is the primary executor backend. It is an autonomous AI agent gateway that:
- Routes requests to multiple model backends (Anthropic Claude, OpenRouter free tier)
- Manages MCP tool access (Grafana, Atlassian, filesystem, Tavily web search)
- Exposes a WebSocket Gateway API at `ws://localhost:18789/rpc`

### Gateway WebSocket API

P-Ork communicates with OpenClaw via the Gateway WebSocket API (`OpenClawWSExecutor`). The protocol uses JSON frames:

```
Request:  {"type": "req", "id": "<uuid>", "method": "<method>", "params": {...}}
Response: {"type": "res", "id": "<uuid>", "ok": true/false, "payload": {...}}
Event:    {"type": "event", "event": "<name>", "payload": {...}}
```

**Auth:** Ed25519 device-signature challenge/response on connect. Credentials read automatically from `~/.openclaw/identity/device.json` and `~/.openclaw/identity/device-auth.json` — no manual configuration needed.

**Agent call flow:**
1. Send `agent` request → gateway immediately responds with `status: "accepted"` and a `runId`
2. Events stream in (`lifecycle`, `assistant`, `chat`) while the agent runs
3. Gateway sends a second `res` for the same request ID once complete, with `status: "ok"` and the full result including `payloads`, `agentMeta.model`, and `durationMs`

**Gateway result format** (second `agent` response):
```json
{
  "runId": "...",
  "status": "ok",
  "result": {
    "payloads": [{ "text": "<agent response text>", "mediaUrl": null }],
    "meta": {
      "durationMs": 8503,
      "agentMeta": { "provider": "...", "model": "...", "usage": {...} },
      "aborted": false
    }
  }
}
```

The agent's response text (`result.payloads[-1].text`) should be JSON matching `LLMOutput`. The executor scans payloads in reverse for the last valid JSON block (models sometimes narrate before outputting JSON) and strips markdown code fences.

### Session Keys

Session keys **must** start with `agent:{agent-name}:` — the Gateway validates this and rejects calls where the session key's agent name doesn't match the `agentId`. Any suffix after the agent name prefix is free-form.

```yaml
# Correct
session_key: "agent:sre-triage:{{pipeline_run_id}}:triage"

# Wrong — gateway will reject this
session_key: "pipeline:{{pipeline_run_id}}:triage"
```

If `session_key` is omitted, the executor generates `agent:{agent}:pipeline:{pipeline_run_id}:{current_step}` automatically.

### Model Selection per Step

By default, models are bound to named agents in OpenClaw config. To override for a specific step, set `executor_config.model`:

```yaml
executor_config:
  agent: sre-triage
  model: anthropic/claude-opus-4-7   # overrides agent's configured model for this step
```

Typical agent tiers (model bound in OpenClaw agent config):
- Critical/remediation steps → agent backed by `anthropic/claude-sonnet-4-6`
- Verification steps → agent backed by a stronger model (e.g. Opus)
- Warning/routine steps → agent backed by `anthropic/claude-haiku-4-5-20251001` or free OpenRouter model

### Mac Setup (current machine)

- OpenClaw installed via nvm node: `~/.nvm/versions/node/v22.22.2/bin/openclaw`
- Config: `~/.openclaw/openclaw.json`
- Gateway runs on `localhost:18789` — start with: `openclaw gateway --port 18789`
- Telegram bot `@alexdclaw_bot` connected and paired (OpenClaw notifications)
- Separate Telegram bot created for P-Ork human approval steps — bot token and chat ID set in `config.yaml` under `notifications.telegram`; must be a different bot from the OpenClaw one as both long-poll and Telegram only allows one simultaneous poller per bot
- MCP servers configured and verified: `filesystem`, `grafana`, `grafana_google`, `tavily`, `atlassian`
- Providers: Anthropic (Claude Haiku 4.5, Sonnet 4.6) + OpenRouter (free tier models)
- No local Ollama — removed from config

### Known OpenClaw Notes
- Config is rewritten by the gateway on startup — do not rely on manual edits persisting without a restart
- `gateway.controlUi.allowInsecureAuth=true` is set intentionally for local dev (security warning is expected)
- Telegram pairing is per-instance — re-pair required when moving to a new machine (`openclaw pairing approve`)

---

## Service Configuration

`config.yaml` at service root:

```yaml
server:
  host: 0.0.0.0
  port: 8000

pipeline_config_dir: ./pipelines

database:
  url: sqlite+aiosqlite:///./runs.db

notifications:
  telegram:
    bot_token: ${TELEGRAM_BOT_TOKEN}   # or a literal value — env var placeholder is optional
    chat_id: ${TELEGRAM_CHAT_ID}

logging:
  level: INFO
```

`config.yaml` values are used directly; `${ENV_VAR}` placeholders are resolved at startup and replaced with the environment variable value (empty string if unset). Literal values in `config.yaml` take precedence over env vars for the human executor — env vars are a fallback only.

---

## Development Setup (MacBook Air)

```bash
# Clone repo
git clone <repo>
cd service

# Create virtualenv
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn src.main:app --reload --port 8000

# Test webhook (alertmanager)
curl -X POST "http://localhost:8000/webhook?source=alertmanager" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/alertmanager_critical.json

# Test webhook (generic source)
curl -X POST "http://localhost:8000/webhook?source=generic" \
  -H "Content-Type: application/json" \
  -d @tests/fixtures/generic_new_order.json
```

---

## Kubernetes Deployment

Target: ARM64 home lab cluster (Ubuntu snap k8s).

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY config.yaml .
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build for ARM64:
```bash
docker buildx build --platform linux/arm64 -t orchestration-service:latest .
```

Pipeline configs delivered via Kubernetes ConfigMap mounted at `/app/pipelines/`.
SQLite database on a PersistentVolumeClaim.
Secrets (tokens) via Kubernetes Secrets as environment variables.

---

## Adding a New Source Parser

1. Create `src/normaliser/<source>.py`
2. Implement `BaseParser` — produce a `NormalisedContext`
3. Register in `src/normaliser/__init__.py` source map
4. Add test fixture in `tests/fixtures/<source>_payload.json`

No other changes required.

## Adding a New Executor

1. Create `src/executors/<name>.py`
2. Implement `BaseExecutor` — accept `StepConfig` + context dict, return `LLMOutput`
3. Register in `src/executors/__init__.py` executor map
4. Reference by name in pipeline YAML step `executor:` field

No other changes required.

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|---|---|
| Single `/webhook` endpoint | Source agnostic — parsers handle differences, pipelines don't care |
| Generic source requires explicit `pipeline` | Callers targeting `?source=generic` control the sending tool, so they can always name the pipeline — avoids trigger matching complexity |
| Source via query param | Easiest for operators configuring webhook URLs in third party tools |
| YAML pipeline configs | Git-controlled, human readable, no UI needed, operator chooses delivery method |
| Structured JSON output from LLM | Makes flow control deterministic — runner reads `status`/`confidence`, not prose |
| Isolated session key per step | Prevents context bleed between concurrent runs and between steps |
| SQLite for run storage | Zero infrastructure dependency, file-based, easy backup, queryable |
| Adapter pattern for executors | Swap OpenClaw for Hermes or direct Anthropic API with config change only |
| Runner owns flow decisions | LLM recommends, service decides — never blindly chain prompts |
| Jinja2 for prompt templates | Standard, well understood, handles `{{steps.x.field}}` references cleanly |
| Confidence via forced reasoning | More reliable than asking for a bare float — model reasons first, scores after |
| `timeout_seconds` per step | Hung OpenClaw calls block the runner indefinitely without it; step-level granularity matches the fact that different steps have different cost/latency profiles |
| `POST /reload` + SIGHUP for pipeline reload | Config-driven system should never need a restart for a YAML edit; SIGHUP is the Unix convention for graceful reload |
| Runs API over direct SQLite access | Operators and tooling shouldn't need filesystem access to query run history; HTTP keeps the interface consistent |
| Parallel groups as a step-list concept | Keeps the overall pipeline flow readable at a glance; branches are sub-entries not separate top-level configs |
| No per-branch confidence gating | Flow control belongs to the runner at the group level; per-branch thresholds would create nested decision trees that are hard to reason about |
| Branch outputs keyed by branch name | Downstream steps reference `{{steps.branch_name.field}}` — identical to sequential step references, no special syntax required |
| In-process APScheduler for cron | Zero extra infrastructure — same process, same DB, same runner code path as webhooks; reloads atomically with pipeline configs |
| `schedule:` block per pipeline | Each pipeline owns its own schedule definition; reload is a single operation that replaces all jobs atomically |
| Scheduled runs synthesise `NormalisedContext` | Keeps the runner code path identical for webhook and scheduled triggers — no special-casing downstream |

---

## Current Status

Implementation in progress. All code lives under `service/`.

### Completed
1. **Project scaffolding** — full directory structure, `config.yaml`, `requirements.txt`
2. **Pydantic models** — `NormalisedContext`, `PipelineConfig`/`StepConfig`, `LLMOutput` (`src/models/`)
   - `StepConfig` uses `executor_config: dict` (not `model`) — model is bound to agent in OpenClaw
   - `StepConfig` has optional `verifier: VerifierConfig` with `combination_strategy`, `veto_floor`, and `trigger` band
3. **Webhook endpoint** — `POST /webhook` with `?source=` param and `X-Pipeline-Source` header fallback (`src/main.py`)
4. **Alertmanager parser** — two strategies: `?strategy=most_severe` (default) or `?strategy=common_labels` (`src/normaliser/alertmanager.py`)
5. **Base classes** — `BaseParser`, `BaseExecutor` abstract classes
6. **Test fixture** — `tests/fixtures/alertmanager_critical.json`
7. **Pipeline config loader** — `src/pipeline/loader.py` — globs `*.yaml` from pipelines dir, validates via Pydantic, alphabetical load order
8. **Pipeline resolver** — `src/pipeline/resolver.py` — explicit pipeline name takes priority; otherwise AND-logic match against `trigger.match` (top-level fields + labels); first match wins
9. **Example pipeline config** — `service/pipelines/alert-triage-critical.yaml`
10. **OpenClaw executor** — `src/executors/openclaw_ws.py` — invokes OpenClaw agents via Gateway WebSocket API; Ed25519 device-signature auth from `~/.openclaw/identity/`; waits for the final `agent` result frame which carries response text, actual model used, and duration; server-side session isolation (no file deletion); strips markdown fences; scans payloads in reverse for last valid JSON block
11. **OpenClaw Mac setup** — gateway running, Telegram paired, all MCP servers verified
12. **Context builder** — `src/pipeline/context.py` — resolves `context_template.include` dotted paths from `NormalisedContext`; injects `pipeline_run_id`, `pipeline_name`, `current_step`; `labels` dict always present; prior step outputs available as `steps.<name>.<field>`; leaf key flattening means `labels.service` is accessible as both `{{service}}` and `{{labels.service}}`
13. **Pipeline runner** — `src/pipeline/runner.py` — `PipelineRunner` class; sequential step execution; flow control on abort/escalate/proceed; confidence gate with `on_low_confidence` action; optional verifier with built-in internal prompt (user never writes verifier prompts); veto/minimum combination strategies; verifier failure is non-fatal (falls back to primary confidence); executor instances cached per runner; accepts optional `session_factory` for DB (no-op if not provided)
14. **SQLite DB layer** — `src/db/models.py`, `src/db/database.py`; `PipelineRun` and `PipelineStep` ORM models; stores `primary_confidence`, `verifier_confidence`, `effective_confidence` separately for audit; `init_db(url)` at startup, `create_tables()` safe to call on every boot; `get_session_factory()` for runner injection
15. **Telegram notifier** — `src/notifications/telegram.py`; renders notification template via Jinja2; posts to Telegram bot API; errors are logged but non-fatal
15a. **Webhook notifier** — `src/notifications/webhook.py`; POSTs rendered template as request body to a URL; supports `method`, `content_type`, `headers` (`${ENV_VAR}` substitution), `timeout_seconds`; errors are logged but non-fatal; registered automatically at startup (no credentials needed — URL is per-notification in `config:`)
15b. **Multiple notifiers per action** — `notifications:` block accepts either a single config dict (original format, unchanged) or a list of configs; the Pydantic validator coerces single dicts to lists so all existing YAML files keep working; runner iterates the list and dispatches each notifier in order
16. **`main.py` fully wired** — lifespan handler initialises DB, loads pipelines, configures notifiers, builds `PipelineRunner`; webhook resolves pipeline and fires run as background task (returns 202 immediately); `/pipelines` status endpoint added; `${ENV_VAR}` placeholders resolved in `config.yaml` (unresolved placeholders become `""` so misconfigured notifiers fail cleanly at startup rather than at send time)
17. **Generic source parser** — `src/normaliser/generic.py`; `GenericPayload` Pydantic model enforces `pipeline` as required; `event` stored in `labels`; `data` dict lands in `metadata`; registered as `"generic"` in parser map; tested end-to-end with `new-order` pipeline and `order-intake` / `customer-comms` OpenClaw agents
18. **Step timeouts** — `timeout_seconds: int | None` on `StepConfig`; runner wraps executor call in `asyncio.wait_for()`; timeout is logged distinctly and returns `status: failed` (same downstream handling as any executor failure)
19. **Hot pipeline reload** — `POST /reload` endpoint and `SIGHUP` handler both call `_do_reload()` which re-globs the YAML dir and replaces `_pipelines` in place; errors return 500 with reason; in-flight runs are unaffected
20. **Runs API** — `GET /runs` (list, newest first, filterable by `status`/`pipeline`, paginated) and `GET /runs/{run_id}` (full detail with steps, confidence scores, parsed output); backed by existing SQLite store via SQLAlchemy async queries with `selectinload` for steps
21. **Parallel groups** — `parallel:` entries in the step list run branches concurrently via `asyncio.gather`; join strategies: `all_must_pass` (min), `any_must_pass` (max), `weighted_average`; group-level `confidence_threshold` / `on_low_confidence` / `timeout_seconds`; each branch may have its own `timeout_seconds` and `verifier`; branch outputs registered individually in `step_outputs` so downstream steps reference them as `{{steps.branch_name.field}}`; branches stored as separate `pipeline_steps` rows with `step_name = "<group>/<branch>"`
22. **Conditional steps (`when:`)** — optional `when: <expr>` on any `StepConfig` or `ParallelGroupInner`; evaluated via Jinja2 before any executor call so dot-notation on step outputs works; false → step silently skipped (no DB row, no executor call); supports branching on agent output fields, severity, labels, confidence scores
23. **Cron scheduler** — optional `schedule:` block on `PipelineConfig` (`cron`, `summary`, `severity`, `labels`); in-process `AsyncIOScheduler` registers jobs at startup and re-registers atomically on every reload/SIGHUP; scheduled runs synthesise a `NormalisedContext` with `source="scheduler"` and fire through the standard runner code path; `GET /schedules` endpoint lists active jobs and next fire times
24. **Verifier challenger mode** — `mode: challenger` on `VerifierConfig`; verifier receives the original task prompt only (no primary output) and executes the same task independently; `mode: reviewer` (default) retains existing behaviour where verifier critiques the primary response; combination strategies (`minimum`/`veto`) work identically in both modes
25. **Run ID in webhook response** — `POST /webhook` now returns `run_id` in the 202 body; ID is generated before the background task fires so callers can immediately poll `GET /runs/{run_id}`
26. **Retry logic with backoff** — optional `retry:` block on `StepConfig` (`attempts`, `backoff: fixed|exponential`, `delay_seconds`); wraps executor call only — low-confidence results are valid outputs and never retried; exponential default doubles delay each attempt
27. **Human-in-the-loop approval** — `executor: human` step type sends a Telegram inline keyboard message (Approve/Reject buttons) and pauses the pipeline until a button is clicked or `timeout_seconds` elapses; approve → `confidence=1.0, proceed=true`; reject → `confidence=0.0, proceed=true` (triggers `on_low_confidence` action); timeout → step fails; a background Telegram long-poll task resolves approval futures; started automatically when Telegram credentials are configured; poller calls `deleteWebhook` at startup to avoid 409 conflicts with registered webhooks; **must use a separate bot from OpenClaw** — Telegram only allows one simultaneous `getUpdates` poller per bot; credentials sourced from `config.yaml` `notifications.telegram` (env vars are fallback)
28. **Webhook output executor** — `executor: webhook` posts the rendered `prompt_template` as the HTTP body to `executor_config.url`; supports `method`, `headers` (with `${ENV_VAR}` substitution), `content_type`, `timeout_seconds`; returns `confidence=1.0` on HTTP 2xx; non-2xx raises and triggers retry/fail flow; works as any regular step so `when:`, `retry:`, and verifiers apply
29. **Gateway WebSocket executor** — replaced CLI subprocess with direct Gateway WebSocket API calls; Ed25519 device-signature auth; model override via `executor_config.model`; server-side session isolation (no more session file deletion or concurrent-run caveats); actual model used surfaced in `LLMOutput.model` from `agentMeta` in the final result frame
30. **Model override per step** — `executor_config.model` on any `openclaw` step overrides the agent's configured model for that call only; passed directly to the Gateway `agent` params
31. **Thinking level per step** — `executor_config.thinking_level` (`off|minimal|low|medium|high|xhigh`) controls the model's thinking budget; passed as `thinking` in the Gateway `agent` params

### Up Next
1. **Dockerfile and Kubernetes manifests** — deferred until after testing

### Notes
- Run with: `cd service && uvicorn src.main:app --reload --port 8000`
- Alertmanager `strategy` param is designed to be encoded in the webhook URL in Alertmanager config, not set per-request
- OpenClaw gateway must be running before any pipeline execution: `openclaw gateway --port 18789`
- Named agents (e.g. `sre-triage-sonnet`) must be created in OpenClaw before pipeline steps can execute
