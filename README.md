# P-Ork Orchestration Service

## Project Overview

A **webhook-triggered, YAML-configured AI pipeline orchestration service** built in Python with FastAPI. It receives webhooks from any source (Alertmanager, Grafana, Atlassian, etc.), normalises the payload, resolves a named pipeline config, and executes a multi-step AI pipeline using pluggable agent executor backends.

The service is designed to be:
- **Source agnostic** — any webhook source is supported via pluggable parsers
- **Executor agnostic** — AI backends are adapters behind a common interface; steps in the same pipeline can mix executors freely
- **Config driven** — all pipeline logic lives in YAML files, not code
- **Modular** — adding a new source parser or executor adapter requires no changes to core logic

Primary use case is observability automation (alert triage, Grafana investigation, bounded remediation) but the design is intentionally general purpose.

---

## Tech Stack

- **Python 3.11+**
- **FastAPI** — webhook endpoint, status/runs API, and UI routes
- **Pydantic v2** — normalised context schema, pipeline config models, LLM output validation
- **SQLAlchemy + aiosqlite** — async SQLite for pipeline run storage
- **httpx** — async HTTP client for webhook executor and notification delivery
- **websockets** — async WebSocket client for OpenClaw and P-Ork Gateway executors
- **Jinja2** — prompt template rendering (`{{variable}}` syntax in YAML configs) and HTML UI templates
- **PyYAML** — pipeline config loading
- **APScheduler 3.x** — in-process cron scheduler for time-triggered pipeline runs
- **uvicorn** — ASGI server for local development

---

## Project Structure

```
samples/                      # Copy-and-adapt templates for new deployments (git controlled)
├── config.yaml.example         # Annotated service config template
├── pipelines/                  # Pipeline YAML templates — copy to service/pipelines/
│   ├── alert-triage-investigation-using-steps.yaml
│   ├── generic-webhook-new-order.yaml
│   ├── human-approval-test.yaml
│   └── otel-triage-verified.yaml
└── steps/                      # Step definition templates — copy to service/steps/
    ├── first-line-triage.yaml
    ├── sre-investigation.yaml
    ├── sre-investigation-verified.yaml
    ├── order-intake.yaml
    └── customer-comms.yaml

service/
├── agents/                     # Agent SOUL.md drafts (copy into executor workspace)
│   ├── order-intake/SOUL.md
│   └── customer-comms/SOUL.md
├── pipelines/                  # Active pipeline configs (git controlled)
│   └── *.yaml
├── steps/                      # Reusable step library — gitignored, personal to deployment
│   └── *.yaml                  # Copy from samples/steps/ and adapt (see §4a)
├── src/
│   ├── main.py                 # FastAPI app entry point, lifespan, webhook endpoint
│   ├── gateway.py              # Lightweight helper for calling OpenClaw Gateway WS API
│   ├── ui.py                   # UI routes (pipeline/agent/step library, run history)
│   ├── normaliser/
│   │   ├── base.py             # BaseParser abstract class
│   │   ├── alertmanager.py     # Alertmanager-specific parser
│   │   └── generic.py          # Generic source parser (standardised JSON schema)
│   ├── executors/
│   │   ├── base.py             # BaseExecutor abstract class
│   │   ├── openclaw_ws.py      # OpenClaw executor — Gateway WebSocket API (Ed25519 auth)
│   │   ├── openclaw.py         # OpenClaw executor — CLI subprocess (legacy, not registered)
│   │   ├── gateway.py          # P-Ork Gateway executor — WebSocket API (token auth)
│   │   ├── human.py            # Human-in-the-loop executor (Telegram inline keyboard)
│   │   └── webhook.py          # Webhook output executor (HTTP POST)
│   ├── pipeline/
│   │   ├── loader.py           # Loads pipelines and step library; resolves use: references
│   │   ├── resolver.py         # Matches normalised context to a pipeline
│   │   ├── runner.py           # Executes pipeline steps, manages flow control, emits run log
│   │   └── context.py          # Builds and passes Jinja2 template context between steps
│   ├── models/
│   │   ├── context.py          # NormalisedContext Pydantic model
│   │   ├── pipeline.py         # PipelineConfig, StepConfig, LibraryStepConfig Pydantic models
│   │   └── llm.py              # LLMOutput Pydantic model (step output contract)
│   ├── db/
│   │   ├── database.py         # SQLAlchemy engine, session management
│   │   └── models.py           # PipelineRun, PipelineStep ORM models
│   └── notifications/
│       ├── telegram.py         # Telegram notification handler
│       ├── telegram_poller.py  # Long-poll loop for HITL Telegram button callbacks
│       └── webhook.py          # Webhook notification handler
├── templates/                  # Jinja2 HTML templates for the UI
├── logs/                       # Rotating log files (auto-created, gitignored)
│   ├── service.log             # Application logs (10 MB × 5 files)
│   └── access.log              # HTTP access logs, separated from service logs
├── tests/
│   └── fixtures/               # Test webhook payloads (alertmanager, generic, etc.)
├── Dockerfile
├── config.yaml                 # Service-level config — gitignored, copy from samples/config.yaml.example
└── requirements.txt
```

---

## Core Concepts

### 1. Webhook Intake & Source Detection

Single endpoint: `POST /webhook`

Source is identified via query parameter: `/webhook?source=alertmanager`

Header fallback also supported: `X-Pipeline-Source: alertmanager`

The source value maps to a registered parser class. Registered sources: `alertmanager`, `generic`.

### 2. Normalisation Layer

Each source parser implements `BaseParser` and produces a `NormalisedContext` object. This is the universal data model that all downstream pipeline logic operates on.

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

### 3. Pipeline Resolution

The `resolver` loads all YAML configs from `PIPELINE_CONFIG_DIR` and matches the incoming `NormalisedContext` against each config's `trigger.match` block. First match wins. Configs should be ordered by specificity (more specific matches first).

Pipeline name can also be explicitly set by the source parser if the webhook payload contains a pipeline attribute (e.g. an Alertmanager label `pipeline: alert-triage-critical`).

### 4. Pipeline Config Schema (YAML)

Model selection is handled by the named agent's own config in the executor backend. To override the model for a specific step, set `executor_config.model` — this is passed directly to the executor and takes precedence over the agent's configured model.

Steps support an optional `verifier` block for independent confidence verification by a second agent. The verifier trigger can be set to `always: true` (fires unconditionally), or scoped to a confidence band. See §5 for full trigger configuration examples.

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

vars:                           # pipeline-level variables, available in all prompts
  jira_project: MYPROJECT
  confluence_space: MYSPACE

steps:
  - name: initial-triage
    executor: openclaw           # or: gateway | human | webhook
    executor_config:
      agent: sre-triage-sonnet          # named agent in the executor backend
      session_key: "agent:sre-triage-sonnet:{{pipeline_run_id}}:triage"
      model: anthropic/claude-sonnet-4-6   # optional — overrides agent's configured model
      thinking_level: low                  # optional — off|minimal|low|medium|high|xhigh
    confidence_threshold: 0.75
    on_low_confidence: escalate  # escalate | abort | proceed
    on_abort: notify
    timeout_seconds: 120
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
      Previous triage: {{steps.initial_triage.next_step_context}}
      ...
    verifier:
      executor: openclaw
      executor_config:
        agent: sre-verifier-opus
      combination_strategy: veto
      veto_floor: 0.60
      trigger:
        confidence_below: 0.95
        confidence_above: 0.50

notifications:
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
  cron: "*/5 * * * *"
  summary: "Scheduled health check for my-service"
  severity: warning
  labels:
    service: my-service
    environment: prod
```

---

### 4a. Step Library — Reusable Step Definitions

Named step definitions live in `step_library_dir` (default `./steps`). Each file defines a reusable step config that pipelines can reference by name using a `use:` key. The loader resolves references before Pydantic validation, so the runner is completely unaware of the library mechanism.

**Library step file (`steps/sre-investigation.yaml`):**
```yaml
name: sre-investigation
description: Grafana RED metrics investigation — updates Jira with findings
tags: [investigation, grafana, openclaw]

executor: openclaw
executor_config:
  agent: sre-investigation
  session_key: "agent:sre-investigation:{{pipeline_run_id}}:{{current_step}}"
confidence_threshold: 0.60
on_low_confidence: escalate
timeout_seconds: 1200
prompt_template: |
  ... default prompt ...
```

**Referencing a library step in a pipeline:**
```yaml
steps:
  - use: first-line-triage          # fully inherits the library step

  - use: sre-investigation          # inherit config, override just the threshold
    confidence_threshold: 0.80

  - use: sre-investigation          # add a model override — executor_config is deep-merged
    executor_config:                # so agent/session_key are still inherited
      model: anthropic/claude-opus-4-8

  - use: sre-investigation          # custom prompt for this pipeline
    prompt_template: |
      Pipeline-specific prompt referencing {{steps.first_line_triage.summary}} ...
```

**Merge rules:**
- All top-level fields: local value wins if present, library value is the default.
- `executor_config` only: **deep-merged** — local keys add to or override library keys, rather than replacing the whole block. This lets you add `model` or `thinking_level` without repeating `agent` and `session_key`.
- `description` and `tags` are library-only metadata and are stripped before the step is passed to the runner.

**Step library UI:** the `/ui/steps` page shows all loaded library steps with their executor/agent, confidence threshold, tags, which pipelines reference each step, and a copy button for the `- use: step-name` snippet.

**Hot reload:** `POST /reload` and SIGHUP reload the step library first, then re-resolve all pipeline references against the updated library.

**The `steps/` directory is gitignored.** Step definitions reference your specific agents, session key patterns, and confidence thresholds — they are personal to your deployment, like `config.yaml`. Copy the starter definitions from `samples/steps/` into `service/steps/` and adapt them to your agents.

**All templates** live in `samples/` at the repo root — `samples/pipelines/` for pipeline YAMLs and `samples/steps/` for step definitions. These are committed reference files. Copy them into the appropriate `service/` subdirectory and fill in your details.

---

### 5. LLMOutput — The Step Contract

Every executor backend must return an `LLMOutput`. This is the contract between pipeline steps — the runner reads it for flow decisions, and downstream steps reference its fields in prompt templates.

```python
class LLMOutput(BaseModel):
    model_config = ConfigDict(extra="allow")   # extra fields allowed and propagated

    # --- Mandatory ---
    confidence: float             # 0.0–1.0. Compared against confidence_threshold.
    summary: str                  # One-sentence human-readable outcome. Used in
                                  # notifications and downstream {{steps.name.summary}}.
    next_step_context: str        # Focused brief for the next step. Available as
                                  # {{steps.name.next_step_context}}. May be "" for
                                  # terminal steps.

    # --- Flow control ---
    proceed: bool = True          # false = pipeline stops cleanly here (status=stopped).
                                  # No further steps run. Use when the agent is confident
                                  # no further action is warranted.
    proceed_reason: str | None = None  # Required when proceed=false. Explain why.

    # --- Optional enrichment ---
    reasoning: dict | None = None  # Free-form audit dict. Conventional keys: supports,
                                   # contradicts, assumptions. Available downstream as
                                   # {{steps.name.reasoning.supports}} etc.

    # --- Artifacts (optional) ---
    artifacts: dict | None = None  # {name: content} — runner writes to disk, replaces
                                   # content with references. See §5a.

    # --- Set by the executor (agents must NOT include these) ---
    model: str | None = None      # Populated from API metadata by the executor.
    raw_response: dict = {}       # Full unparsed response for audit. Set by executor.
```

**Extra fields are allowed and fully propagated.** Any field returned by an agent beyond the schema above (e.g. `jira_ticket`, `doc_found`, `action`, `dashboard_uid`) is stored in the DB and available in all downstream prompt templates as `{{steps.step_name.field_name}}`. This is the primary mechanism for passing structured data between steps.

**Mandatory vs optional at a glance:**

| Field | Required? | Notes |
|---|---|---|
| `confidence` | **Yes** | Must be 0.0–1.0. No default — validation fails if missing. |
| `summary` | **Yes** | No default — validation fails if missing. |
| `next_step_context` | **Yes** | Empty string `""` is valid for terminal steps. |
| `proceed` | No | Defaults to `true`. Only set `false` when the pipeline should stop cleanly. |
| `proceed_reason` | No | Include whenever `proceed: false` to make the stop auditable. |
| `reasoning` | No | Recommended for triage/analysis steps; improves verifier quality. |
| `artifacts` | No | `{name: content}` dict. Runner writes each value to disk; content is not stored in the database. See §5a. |
| `model` | No | Do not include — the executor sets this from API metadata. |
| `raw_response` | No | Do not include — set by the executor. |
| Any extra field | No | Freely add domain fields (`jira_ticket`, `action`, etc.). All are stored and accessible downstream. |

**Accessing prior step output in prompts:**

Hyphens in step names must be written as underscores in template references:

```yaml
# Step named "first-line-triage" is referenced as:
{{steps.first_line_triage.summary}}
{{steps.first_line_triage.next_step_context}}
{{steps.first_line_triage.jira_ticket}}   # extra field
{{steps.first_line_triage.reasoning.contradicts}}
```

**Minimal valid agent response:**
```json
{
  "confidence": 0.85,
  "summary": "CPU spike on api-gateway traced to upstream timeout storm — self-resolving.",
  "next_step_context": "Check upstream service latency before closing ticket."
}
```

**Full response with optional fields:**
```json
{
  "confidence": 0.90,
  "proceed": true,
  "summary": "OTEL Collector scrape duration elevated — CPU pressure from upstream.",
  "next_step_context": "Dashboard uid=abc123. Query scrape_duration_seconds p99 from 06:30–07:05.",
  "jira_ticket": "OC-87",
  "doc_found": true,
  "reasoning": {
    "supports": "SLO breach aligns with known fragility pattern in service doc.",
    "contradicts": "No downstream impact observed yet.",
    "assumptions": "Alert timing and Confluence doc are accurate."
  }
}
```

---

### 5a. Artifact Storage

Steps can produce large artifacts (research reports, scraped data, compiled documents) that would be unwieldy to pass inline through `next_step_context`. The artifact store writes these to disk, keeps them out of the database, and makes them available in downstream prompt templates by content — not by reference.

#### Producing an artifact

An agent returns an `artifacts` dict alongside its normal `LLMOutput` fields. Each key is a name chosen by the agent; each value is the full text content:

```json
{
  "confidence": 0.9,
  "summary": "Research complete — 3 sources compiled",
  "next_step_context": "Coverage spans Q1–Q4 2025",
  "artifacts": {
    "research_report": "# Research Report\n\n## Source 1\n..."
  }
}
```

The runner intercepts the `artifacts` field before anything is stored in the database. The content is written to `{artifacts_dir}/{run_id}/{step_name}/{key}` and replaced with an opaque reference string (`local://...`). SQLite only stores the reference; the blob lives on disk.

#### Consuming an artifact

Downstream steps reference artifact content in their prompt templates using `{{artifacts.step_name.key}}`. The runner loads the content from disk at render time — only for the steps that actually reference it.

```yaml
steps:
  - name: research
    executor: openclaw
    executor_config:
      agent: web-researcher
    prompt_template: |
      Research the topic and compile a full report.
      Return JSON with the usual fields plus an "artifacts" key:
      {"confidence": ..., "summary": ..., "next_step_context": ...,
       "artifacts": {"research_report": "..."}}

  - name: proofread
    executor: openclaw
    executor_config:
      agent: editor
    prompt_template: |
      Proofread and improve the following document:

      {{artifacts.research.research_report}}

      Return the corrected document as an artifact named "final_report".
```

Hyphens in step names follow the same rule as `steps.*` references — use underscores in template expressions:

```
# Step named "web-research" is referenced as:
{{artifacts.web_research.report}}
```

#### Lifecycle and cleanup

Artifact directories are scoped to a run (`{artifacts_dir}/{run_id}/`). A daily APScheduler job (02:00) removes directories for runs older than `retention_days`. Failed runs retain their artifacts for the same period, which is useful for debugging.

To disable artifact storage entirely, omit the `artifacts:` block from `config.yaml`. Steps that return an `artifacts` field will have it passed through as a regular extra field rather than being written to disk.

---

### 6. Verifier Trigger Configuration

The `trigger` block on a `verifier` controls when the verifier agent fires.

Two verifier **modes** are available:

| Mode | Behaviour |
|---|---|
| `reviewer` (default) | Verifier receives the primary agent's prompt and full response — critiques the reasoning |
| `challenger` | Verifier receives only the original task prompt — executes the same task independently |

**Always verify:**
```yaml
verifier:
  executor: openclaw
  executor_config:
    agent: sre-verifier-opus
  mode: reviewer
  combination_strategy: minimum
  trigger:
    always: true
```

**Band-based — only verify in the uncertain middle ground:**
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

**Combination strategies:**

| Strategy | Behaviour |
|---|---|
| `minimum` | `effective = min(primary, verifier)` — both must be confident |
| `veto` | Primary passes through unless verifier < `veto_floor`, in which case verifier score overrides |

Verifier failures (executor errors) are non-fatal — the runner logs a warning and falls back to primary confidence only.

---

### 7. Parallel Groups

A `parallel:` entry in the step list runs multiple branches concurrently via `asyncio.gather` and joins their confidence scores before applying standard flow control.

```yaml
steps:
  - name: initial-triage
    executor: openclaw
    ...

  - parallel:
      name: context-gathering
      join: all_must_pass         # join strategy: all_must_pass | any_must_pass | weighted_average
      confidence_threshold: 0.70
      on_low_confidence: escalate
      on_abort: notify
      timeout_seconds: 90
      steps:
        - name: check-runbook
          executor: openclaw
          executor_config:
            agent: runbook-lookup
          prompt_template: |
            Look up the runbook for {{labels.service}}...
        - name: check-grafana
          executor: gateway
          executor_config:
            agent: grafana-analyst
          weight: 2.0             # optional — only used by weighted_average strategy
          prompt_template: |
            Check Grafana for {{labels.service}}...

  - name: remediation
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
| `weighted_average` | Weighted mean; each branch has an optional `weight:` (default 1.0) |

Branch outputs are registered individually so downstream steps reference them as `{{steps.check_runbook.summary}}` — identical to sequential step references.

---

### 8. Cron Scheduler

Any pipeline can declare an optional `schedule:` block to run on a cron schedule in addition to (or instead of) webhook triggers.

```yaml
schedule:
  cron: "0 9 * * 1-5"
  summary: "Daily morning service health sweep"
  severity: info
  labels:
    service: my-service
    environment: prod
```

Schedules register at startup and re-register atomically on every `/reload` or `SIGHUP`. Scheduled runs synthesise a `NormalisedContext` with `source="scheduler"` and fire through the standard runner — identical code path to webhook triggers.

```bash
GET /schedules
# → {"schedules": [{"pipeline": "my-pipeline", "cron": "0 9 * * 1-5", "next_run": "..."}]}
```

---

### 9. Executor Adapter Pattern

All executors implement `BaseExecutor`:

```python
class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        pass
```

Executors are registered by name in `src/executors/__init__.py` and referenced by name in pipeline YAML step `executor:` fields. Steps within the same pipeline can freely mix executors.

---

#### `openclaw` — OpenClaw Gateway WebSocket

**`executor: openclaw`** — Invokes OpenClaw agents via the OpenClaw Gateway WebSocket API (`ws://127.0.0.1:18789/rpc`). Uses Ed25519 device-signature auth from `~/.openclaw/identity/`. Fires an `agent` call and waits for the final result frame. Session isolation is server-side — no file deletion needed. Scans payloads in reverse for the last valid JSON block.

`executor_config` keys:

| Key | Required | Description |
|---|---|---|
| `agent` | **Yes** | OpenClaw agent name |
| `session_key` | No | Jinja2 template; must start with `agent:{agent-name}:`. Default: `agent:{{agent}}:pipeline:{{pipeline_run_id}}:{{current_step}}` |
| `model` | No | Model override, e.g. `anthropic/claude-sonnet-4-6`. Overrides the agent's configured model. |
| `thinking_level` | No | `off\|minimal\|low\|medium\|high\|xhigh` — controls model thinking budget |

**Service-level config** (under `executors.openclaw` in `config.yaml`, not per-step):

| Key | Default | Description |
|---|---|---|
| `url` | `ws://127.0.0.1:18789/rpc` | OpenClaw Gateway WebSocket URL |
| `identity_dir` | `~/.openclaw/identity` | Path to the directory containing `device.json` and `device-auth.json`. Override when OpenClaw is on a different machine and you have copied the identity files to a custom path. |

---

#### `gateway` — P-Ork Gateway WebSocket

**`executor: gateway`** — Invokes agents via the P-Ork Gateway WebSocket API. Token-based auth (no device identity required). The P-Ork Gateway is a separate service that can run different model backends (Anthropic, OpenRouter, Ollama, Google) and MCP tool configurations independent of OpenClaw.

`executor_config` keys:

| Key | Required | Description |
|---|---|---|
| `agent` | **Yes** | P-Ork Gateway agent name |
| `session_key` | No | Jinja2 template; must start with `agent:{agent-name}:`. Default: `agent:{{agent}}:pipeline:{{pipeline_run_id}}:{{step_name}}` |
| `model` | No | Model override string, e.g. `anthropic/claude-sonnet-4-6`, `openrouter/...`, `ollama-cloud/...` |
| `thinking_level` | No | Thinking level override: `low\|medium\|high` etc. |
| `timeout_seconds` | No | Per-request timeout override (default: 1200) |

Requires `executors.gateway.url` (WebSocket) and `executors.gateway.rest_url` (REST) in `config.yaml`.

---

#### `human` — Human-in-the-Loop (Telegram)

**`executor: human`** — Sends a Telegram inline keyboard message and pauses the pipeline until the operator clicks Approve or Reject, or `timeout_seconds` elapses.

| Outcome | confidence | proceed |
|---|---|---|
| Approved | 1.0 | true |
| Rejected | 0.0 | true — triggers `on_low_confidence` action |
| Timeout | — | step marked `failed` |

The `prompt_template` renders to the Telegram message text. Default timeout is 300s. Requires a **separate** Telegram bot from OpenClaw (Telegram only allows one simultaneous `getUpdates` poller per bot token).

```yaml
- name: approve-remediation
  executor: human
  timeout_seconds: 600
  confidence_threshold: 0.5
  on_low_confidence: abort
  on_abort: notify
  prompt_template: |
    <b>Approve remediation for {{labels.service}}?</b>

    Proposed action: {{steps.investigation.next_step_context}}
```

---

#### `webhook` — HTTP POST Output

**`executor: webhook`** — POSTs the rendered `prompt_template` as the request body to a URL. Returns `confidence=1.0` on any HTTP 2xx. Non-2xx raises and triggers the step's retry/fail flow. The response body (up to 500 chars) is stored in `next_step_context` so downstream steps can reference it.

`executor_config` keys:

| Key | Required | Description |
|---|---|---|
| `url` | **Yes** | Target URL |
| `method` | No | HTTP method (default: `POST`) |
| `content_type` | No | Content-Type header (default: `application/json`) |
| `headers` | No | Extra headers; `${ENV_VAR}` substitution supported |
| `timeout_seconds` | No | Per-request timeout (default: 30) |

```yaml
- name: notify-slack
  executor: webhook
  executor_config:
    url: https://hooks.slack.com/services/...
    headers:
      Authorization: ${SLACK_TOKEN}
  confidence_threshold: 0.0
  on_low_confidence: proceed
  prompt_template: |
    {"text": "Alert resolved: {{labels.service}} — {{steps.triage.summary}}"}
```

---

### 10. Flow Control

The `runner` controls all step execution and flow decisions. Agents never decide what happens next — they only report findings and score confidence. For each step:

0. Evaluate optional `when:` condition — if false, skip the step entirely
1. Parse and validate the LLM response as `LLMOutput` via Pydantic
2. Run verifier if configured and trigger fires
3. Check `effective_confidence` against `confidence_threshold`
4. If below threshold, apply `on_low_confidence` action (`escalate | abort | proceed`)
5. If confidence passes, check `proceed`:
   - `proceed: false` → pipeline stops with status `stopped`
   - `proceed: true` → continue to next step
6. Record step result to SQLite before proceeding

### Step and Run Status Reference

**Step statuses:**

| Status | Set when | Counts as failure? |
|---|---|---|
| `completed` | Step ran successfully and `proceed: true` | No |
| `stopped` | Step ran successfully and returned `proceed: false` | No |
| `escalated` | Confidence below threshold and `on_low_confidence: escalate` | No |
| `aborted` | Confidence below threshold and `on_low_confidence: abort` | No |
| `failed` | Executor exception, timeout, bad JSON, or schema validation error | **Yes** |

For agent success rate calculations, `failed` is the only status that counts against a model.

**Pipeline run statuses:**

| Status | Meaning |
|---|---|
| `completed` | All steps ran to completion |
| `stopped` | A step returned `proceed: false` — clean intentional stop |
| `escalated` | A step was escalated — run halted, human notified |
| `aborted` | A step aborted due to low confidence |
| `failed` | A step raised an unhandled error |
| `running` | Currently in progress |

---

### 10a. Conditional Steps (`when:`)

Any sequential step or parallel group can have an optional `when:` field containing a Jinja2-compatible boolean expression. A false result skips the step cleanly — no DB row, no executor call, invisible to subsequent steps.

```yaml
steps:
  - name: triage
    executor: openclaw
    executor_config:
      agent: sre-triage
    prompt_template: |
      Analyse the alert. Return JSON with an "action" field: "remediate" | "escalate_human" | "ignore"

  - name: auto-remediation
    when: "steps.triage.action == 'remediate'"
    executor: openclaw
    executor_config:
      agent: sre-remediator
    prompt_template: |
      Triage says: {{steps.triage.summary}}. Attempt remediation.

  - name: page-oncall
    when: "steps.triage.action == 'escalate_human'"
    executor: human
    prompt_template: |
      <b>Page oncall?</b> Triage: {{steps.triage.summary}}
```

**`when:` vs `proceed: false`:**
- `when:` — pipeline author decides in advance which steps are relevant given prior step outputs
- `proceed: false` — the agent signals the pipeline is complete and no further steps are warranted

---

### 11. Prompt Construction

Jinja2 renders prompt templates with a context dict containing:
- All fields from `context_template.include` resolved from `NormalisedContext`
- All fields from the pipeline `vars:` block
- `{{pipeline_run_id}}` — unique ID for this run
- `{{pipeline_name}}` — name of the current pipeline
- `{{current_step}}` — name of the current step
- `{{steps.step_name.field}}` — output fields from any previously completed step (hyphens → underscores: `first-line-triage` → `steps.first_line_triage`)
- `{{artifacts.step_name.key}}` — full text content of an artifact produced by a prior step (requires `artifacts:` config block; hyphens → underscores same as above). See §5a.
- `{{labels.service}}`, `{{labels.environment}}` etc. — `labels` dict is always present

### 12. Session Keys

Each pipeline step gets an isolated session key scoped to the run. Session keys for the `openclaw` and `gateway` executors must start with `agent:{agent-name}:` — the respective gateway validates this.

```yaml
session_key: "agent:sre-triage:{{pipeline_run_id}}:triage"
```

If `session_key` is omitted, the executor generates a default automatically.

### 13. Retry Logic

Optional `retry:` block on any step. Retries wrap the executor call only — low-confidence results are valid outputs and are never retried.

```yaml
retry:
  attempts: 3
  backoff: exponential   # fixed | exponential
  delay_seconds: 2.0
```

### 14. Run Storage (SQLite)

**pipeline_runs**

| Column | Type | Description |
|---|---|---|
| `id` | uuid, pk | Run identifier (returned in `/webhook` 202 response) |
| `pipeline_name` | str | |
| `source` | str | Webhook source or `"scheduler"` |
| `triggered_at` | datetime | |
| `status` | str | running / completed / stopped / aborted / escalated / failed |
| `normalised_context` | json | Full NormalisedContext at trigger time |
| `raw_payload` | json | Original unmodified webhook payload |
| `completed_at` | datetime, nullable | |
| `logs` | json, nullable | Structured run event log — array of `{ts, level, event, msg}` objects. Populated at run completion. Events cover step start/complete/fail/skip/escalate/abort, verifier results, parallel group outcomes, and notifications sent. |

**pipeline_steps**

| Column | Type | Description |
|---|---|---|
| `id` | uuid, pk | |
| `run_id` | fk | → pipeline_runs.id |
| `step_name` | str | `"step-name"` for sequential; `"group-name/branch-name"` for parallel branches |
| `step_index` | int | Sort order within run |
| `executor` | str | `openclaw` / `gateway` / `human` / `webhook` |
| `agent` | str | `executor:agent-name` (e.g. `openclaw:sre-triage`) |
| `model` | str | Actual model used, from executor metadata |
| `prompt` | text | Rendered prompt sent to the agent |
| `raw_output` | json | Full unparsed executor response |
| `parsed_output` | json | Validated LLMOutput (excluding raw_response) |
| `status` | str | completed / stopped / escalated / aborted / failed |
| `primary_confidence` | float | Raw confidence from the primary agent |
| `verifier_confidence` | float, nullable | Verifier agent confidence (if verifier ran) |
| `effective_confidence` | float | Confidence used for threshold gate (post-combination) |
| `duration_ms` | int | |
| `executed_at` | datetime | |
| `artifacts` | json, nullable | `{key: reference}` map — references are opaque strings pointing to artifact files on disk. Content is not stored in the DB. |

### 15. Management Endpoints

```bash
# Trigger a run (returns immediately — pipeline runs in background)
POST /webhook?source=<source>
# → {"status": "accepted", "run_id": "<uuid>"}

# Reload step library and all pipeline YAMLs from disk without restarting
POST /reload
# → {"status": "reloaded", "pipelines_loaded": 3}

# SIGHUP also triggers reload
kill -HUP <uvicorn-pid>

# List active cron schedules
GET /schedules
# → {"schedules": [{"pipeline": "...", "cron": "...", "next_run": "..."}]}

# List runs — newest first. Filters: ?status=escalated, ?pipeline=alert-triage-critical
# Pagination: ?limit=50&offset=0 (max 200)
GET /runs
# → {"runs": [{id, pipeline_name, source, status, triggered_at, completed_at}, ...]}

# Full run detail — includes all steps with confidence scores and parsed output
GET /runs/{run_id}

# List loaded pipelines
GET /pipelines
```

---

## UI

The web UI is served under `/ui` and provides the following pages:

| Page | Route | Description |
|---|---|---|
| Dashboard | `/ui/` | 24h run counts by status, success rate, pipeline activity, recent runs |
| Runs | `/ui/runs` | Filterable run history with status and pipeline filters |
| Run detail | `/ui/runs/{id}` | Full step breakdown with confidence bars, parsed output, verifier results, and collapsible run log |
| Pipelines | `/ui/pipelines` | All loaded pipelines with last-run status and run counts |
| Pipeline detail | `/ui/pipelines/{name}` | Config summary, recent runs, YAML viewer, and **Run now** button |
| Steps | `/ui/steps` | Step library — all named steps with executor/agent, tags, pipeline usage, and copy-ref button |
| Agents | `/ui/agents` | Unified agent library across all executor backends |
| Schedules | `/ui/schedules` | Active cron schedules with next-run times |

### Running a pipeline manually

Every pipeline detail page has a **Run now** button. This opens a modal where you can optionally set a summary and paste a full generic webhook payload (JSON). On submit it POSTs to `POST /webhook?source=generic` with `pipeline` forced to the current pipeline name. A banner appears with a link to the new run.

### Run log

Each completed run stores a structured event timeline in `pipeline_runs.logs`. The run detail page shows this as a collapsible log section with timestamped, colour-coded entries (info / warn / error) covering every step start, confidence score, verifier result, skip, escalation, notification, and final status.

### Agent Library

The `/ui/agents` page provides a unified library of agents across all configured executor backends. Agents are fetched live from each backend and merged into a single list with executor badges.

Agents are uniquely identified by `executor:name` — e.g. `openclaw:sre-investigation` and `gateway:sre-investigation` are treated as distinct agents. This prefix is stored in `pipeline_steps.agent` so run history, success rates, and model usage are attributed correctly per backend.

| Executor | Agent list | Soul / Tools files |
|---|---|---|
| `openclaw` | OpenClaw Gateway WS — `agents.list` RPC | OpenClaw Gateway WS — `agents.files.get` RPC |
| `gateway` | P-Ork Gateway REST — `GET /agents` | P-Ork Gateway REST — `GET /agents/{name}/soul` |

Both backends are queried concurrently. If one is unreachable, the other's agents still show with a warning banner. If both fail, stub entries from DB run history are surfaced.

---

## Service Configuration

`config.yaml` at service root:

```yaml
server:
  host: 0.0.0.0
  port: 8000

pipeline_config_dir: ./pipelines
step_library_dir: ./steps            # reusable step definitions; omit to disable library

database:
  url: sqlite+aiosqlite:///./runs.db

notifications:
  telegram:
    bot_token: ${TELEGRAM_BOT_TOKEN}
    chat_id: ${TELEGRAM_CHAT_ID}

executors:
  openclaw:
    url: ws://127.0.0.1:18789/rpc      # OpenClaw Gateway WebSocket URL
  gateway:
    url: ws://localhost:18780/ws        # P-Ork Gateway WebSocket URL
    token: ${PORK_GATEWAY_TOKEN}        # Bearer token; empty string for local dev
    rest_url: http://localhost:18780    # P-Ork Gateway REST base URL (used by Agents UI)

logging:
  level: INFO
  dir: ./logs                          # omit to disable file logging (stdout only)
                                       # creates service.log and access.log (rotating, 10 MB × 5)

artifacts:
  dir: ./artifacts                     # omit this block entirely to disable artifact storage
  retention_days: 7                    # artifact directories older than this are removed daily at 02:00
```

`${ENV_VAR}` placeholders are resolved at startup. Unresolved placeholders become `""`.

---

## Mac Setup (current machine)

- OpenClaw installed via nvm: `~/.nvm/versions/node/v22.22.2/bin/openclaw`
- Config: `~/.openclaw/openclaw.json`
- OpenClaw Gateway: `openclaw gateway --port 18789`
- P-Ork Gateway: separate service in `../P-Ork-Gateway`, runs on port 18780
- Telegram bot `@alexdclaw_bot` — OpenClaw notifications
- Separate Telegram bot for P-Ork HITL approval steps (different bot required)
- MCP servers configured: `filesystem`, `grafana`, `grafana_google`, `tavily`, `atlassian`
- Providers: Anthropic, OpenRouter (free tier), Ollama Cloud (`ollama-cloud/` prefix)

### OpenClaw Identity Files

The `openclaw` executor authenticates to the OpenClaw Gateway using **Ed25519 device-signature auth**. The required files are created automatically by OpenClaw — they are not something P-Ork creates:

| File | Created by | Purpose |
|---|---|---|
| `~/.openclaw/identity/device.json` | OpenClaw on first run / `openclaw configure` | Device ID + Ed25519 private key |
| `~/.openclaw/identity/device-auth.json` | OpenClaw device authorisation flow | Operator token + scopes |

**Co-located setup (P-Ork and OpenClaw on the same machine):** the files are already present and everything works automatically.

**Remote OpenClaw (gateway on a different machine):**
1. Copy `~/.openclaw/identity/` from the OpenClaw machine to the P-Ork machine (or mount it as a Kubernetes Secret)
2. Set `executors.openclaw.identity_dir` in `config.yaml` to the path where you copied the files
3. On the OpenClaw machine, confirm P-Ork's device is approved: `openclaw devices list`

**If the files are missing:** P-Ork logs a warning at startup and the `openclaw` executor is still registered, but any pipeline step using `executor: openclaw` will fail with a clear `FileNotFoundError` until the files are in place. The service continues to run normally — only openclaw steps are affected.

---

## Development Setup

```bash
cd service
python -m venv .venv
source .venv/bin/activate
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

## Adding a New Source Parser

1. Create `src/normaliser/<source>.py`
2. Implement `BaseParser` — produce a `NormalisedContext`
3. Register in `src/normaliser/__init__.py` source map

## Adding a New Executor

1. Create `src/executors/<name>.py`
2. Implement `BaseExecutor` — accept `StepConfig` + context dict, return `LLMOutput`
3. Register in `src/executors/__init__.py` executor map
4. Reference by name in pipeline YAML step `executor:` field

No other changes required in either case.

## Adding a Library Step

1. Create `steps/<your-step-name>.yaml` with at minimum `name`, `executor`, and `executor_config.agent`
2. Run `POST /reload` (or send SIGHUP) — the step will appear in `/ui/steps` immediately
3. Reference it in any pipeline with `- use: <your-step-name>`

The `steps/` directory is gitignored — steps are personal to your deployment. Copy starter definitions from `samples/steps/` and adapt them, or write your own. See `samples/pipelines/alert-triage-investigation-using-steps.yaml` for a worked example of a pipeline that uses library steps.

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Single `/webhook` endpoint | Source agnostic — parsers handle differences, pipelines don't care |
| Generic source requires explicit `pipeline` | Callers control the sending tool, so they can always name the pipeline |
| YAML pipeline configs | Git-controlled, human readable, no UI needed |
| Structured JSON output from LLM | Makes flow control deterministic — runner reads `confidence`/`proceed`, not prose |
| Extra fields allowed on LLMOutput | Domain fields (`jira_ticket`, `action`, etc.) pass between steps without schema changes |
| Isolated session key per step | Prevents context bleed between concurrent runs and between steps |
| SQLite for run storage | Zero infrastructure dependency, file-based, easy backup, queryable |
| Adapter pattern for executors | Swap or mix backends with config changes only; steps in the same pipeline can use different executors |
| Runner owns flow decisions | LLM recommends, service decides — never blindly chain prompts |
| `executor:name` agent identity in DB | Disambiguates same agent name across different backends in run history and success rates |
| Artifact content on disk, not in DB | SQLite is not a blob store; large documents stay in the filesystem. DB row holds only the reference. |
| `{{artifacts.step.key}}` explicit namespace | Template authors know they are pulling a potentially large blob. Keeps `when:` conditions and `steps.*` references unambiguous. |
| `LocalArtifactStore` behind ABC | Swapping in S3 or another backend requires only a new class implementing four methods — no runner or config changes. |
| In-process APScheduler for cron | Zero extra infrastructure; same DB and runner code path as webhooks |
| `POST /reload` + SIGHUP | Config-driven system should never need a restart for a YAML edit |
| Step library with `use:` references | Eliminates step config duplication across pipelines; resolved at load time so runner is unaffected |
| `executor_config` deep-merge on library steps | Lets pipelines add `model` or `thinking_level` without repeating the full agent/session_key block |
| Structured run event log in DB | Per-run timeline queryable from the UI without grepping stdout; survives process restarts |
| `uvicorn.access` separated from service logs | HTTP request noise no longer pollutes run event output on stdout or in service.log |

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

Pipeline configs (from `samples/pipelines/` or your own) delivered via Kubernetes ConfigMap mounted at `/app/pipelines/`.
Step library (from `samples/steps/` or your own) delivered via a separate ConfigMap mounted at `/app/steps/`.
SQLite database on a PersistentVolumeClaim.
Secrets (tokens) via Kubernetes Secrets as environment variables.
Log files written to a PersistentVolumeClaim or redirected to stdout by omitting `logging.dir`.
