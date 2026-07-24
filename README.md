# P-Ork Orchestration Service

## Project Overview

A **webhook-triggered, YAML-configured AI pipeline orchestration service** built in Python with FastAPI. It receives webhooks from any source (Alertmanager, Grafana, Atlassian, etc.), normalises the payload, resolves a named pipeline config, and executes a multi-step AI pipeline using pluggable agent executor backends.

The service is designed to be:
- **Source agnostic** — any webhook source is supported via pluggable parsers
- **Executor agnostic** — AI backends are adapters behind a common interface; steps in the same pipeline can mix executors freely
- **Config driven** — all pipeline logic lives in YAML files, not code
- **Modular** — adding a new source parser or executor adapter requires no changes to core logic

Primary use case is observability automation (alert triage, Grafana investigation, bounded remediation) but the design is intentionally general purpose.

> **New to the trust vector (S/V/G/D) and calibration?** See [`CONFIDENCE-EXPLAINED.md`](CONFIDENCE-EXPLAINED.md) for a plain-language walkthrough of how confidence is derived and every knob that affects it, before diving into the technical reference below ("Verifier Trigger Configuration", "Grounding (shadow mode)", "Deterministic checks & enforced grounding", "Calibration").

---

## Tech Stack

- **Python 3.11+**
- **FastAPI** — webhook endpoint, status/runs API, and UI routes
- **Pydantic v2** — normalised context schema, pipeline config models, LLM output validation
- **SQLAlchemy (async)** — pipeline run storage; SQLite (`aiosqlite`) for zero-infra local dev, PostgreSQL (`asyncpg`) recommended for production — see §Database below
- **httpx** — async HTTP client for webhook executor and notification delivery
- **websockets** — async WebSocket client for OpenClaw and P-Ork Gateway executors
- **Jinja2** — prompt template rendering (`{{variable}}` syntax in YAML configs) and HTML UI templates
- **PyYAML** — pipeline config loading
- **APScheduler 3.x** — in-process cron scheduler for time-triggered pipeline runs
- **uvicorn** — ASGI server for local development
- **OpenTelemetry** — optional per-run distributed tracing (disabled by default) — see §15b

---

## Project Structure

```
samples/                      # Copy-and-adapt templates for new deployments (git controlled)
├── config.yaml.example         # Annotated service config template
├── pipelines/                  # Pipeline YAML templates — copy to service/pipelines/
│   ├── alert-triage-investigation-using-steps.yaml
│   ├── fan-out-multi-service-triage.yaml
│   ├── sub-pipeline-example.yaml
│   ├── generic-webhook-new-order.yaml
│   ├── human-approval-test.yaml
│   ├── otel-triage-verified.yaml
│   ├── tagged-example.yaml
│   └── stage-testing-example.yaml
└── steps/                      # Step definition templates — copy to service/steps/
    ├── first-line-triage.yaml
    ├── sre-investigation.yaml
    ├── sre-investigation-verified.yaml
    ├── order-intake.yaml
    ├── customer-comms.yaml
    └── service-health-check.yaml

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
│   ├── tracing.py              # OpenTelemetry tracing setup + span helpers (see §15b)
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
│   │   │   ├── gateway.py          # P-Ork Gateway executor — WebSocket API (token auth)
│   │   ├── human.py            # Human-in-the-loop executor (Telegram/Slack/Teams, per-team routing)
│   │   ├── pipeline.py         # Sub-pipeline executor — calls another pipeline by name
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
│       ├── slack_poller.py     # Slack Socket Mode listener for HITL Slack button callbacks
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
    fingerprint: str | None        # dedup key — see §3a
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
  "idempotency_key": "order-12345", // optional — dedup key, see §3a. Omit to disable dedup.
  "data": { ... }                 // optional — free-form dict, lands in metadata
}
```

**Mapping to NormalisedContext:**
- `pipeline` → `pipeline` (resolver uses this directly, skips trigger matching)
- `event` → `labels["event"]`
- `source` → `source`
- `summary` → `summary`
- `idempotency_key` → `fingerprint`
- `data` → `metadata` (accessible in prompts as `{{metadata.field_name}}` or just `{{field_name}}` via leaf flattening)

### 3. Pipeline Resolution

The `resolver` loads all YAML configs from `PIPELINE_CONFIG_DIR` and matches the incoming `NormalisedContext` against each config's `trigger.match` block. First match wins. Configs should be ordered by specificity (more specific matches first).

Pipeline name can also be explicitly set by the source parser if the webhook payload contains a pipeline attribute (e.g. an Alertmanager label `pipeline: alert-triage-critical`).

**Match operators:** a `trigger.match` value can be a plain scalar (exact equality, the original behaviour) or a single-key operator dict for richer matching:

```yaml
trigger:
  match:
    severity: critical                    # exact match (unchanged)
    environment:
      in: [prod, staging]                 # membership
    service:
      not_in: [test-runner]                # exclusion
    summary:
      regex: "(?i)timeout"                 # regex search (re.search, not full match)
    error_rate:
      gt: "5"                              # numeric comparison — gt | gte | lt | lte
    severity:
      ne: info                             # not-equal
```

| Operator | Behaviour |
|---|---|
| `eq` | Same as a plain scalar — exact equality |
| `ne` | Not equal |
| `in` | Value is a member of the given list |
| `not_in` | Value is not a member of the given list |
| `regex` | `re.search(pattern, str(actual))` — `None` actual never matches |
| `gt` / `gte` / `lt` / `lte` | Numeric comparison — both sides are cast with `float()`; a non-numeric actual or expected value never matches |

A match value dict must have exactly one key; an unknown operator or a multi-key dict logs a warning and never matches (fails closed).

---

### 3a. Idempotency & Deduplication

Alertmanager (and similar sources) re-send the same alert repeatedly — every evaluation
interval while it's firing, and again on resolve. Without dedup, each resend spawns a
fresh pipeline run: redundant LLM spend, and worse, **overlapping remediation runs for
the same alert**.

#### Fingerprint

Each parser populates `NormalisedContext.fingerprint` — the dedup key:

| Source | Fingerprint source |
|---|---|
| `alertmanager` | The matched alert's `fingerprint` field (Alertmanager's own label-hash), or the group's `groupKey` for the `common_labels` strategy. Falls back to a hash of the relevant labels if neither is present. The alert `status` (`firing`/`resolved`) is appended, so a resolve notification is never suppressed as a duplicate of the firing run. |
| `generic` | The optional `idempotency_key` field (see §2a). If omitted, `fingerprint` is `None` and dedup is skipped for that webhook — generic triggers (orders, etc.) are opt-in. |

#### Dedup check

On `POST /webhook`, after the pipeline is resolved and before a run is started, P-Ork
looks for an existing `pipeline_runs` row with the same `pipeline_name` + `fingerprint`:

- **In-flight** (`status="running"`) — **always** suppressed, regardless of config. This
  is the race-prevention case: two overlapping triage/remediation runs for the same alert
  never run concurrently.
- **Recent** (`triggered_at` within `window_seconds` of now) — suppressed even if the
  prior run has completed. This absorbs Alertmanager's repeat-fire on a flapping alert.

If either matches, no new run is created. The webhook still gets a `202`, but with
`status: "deduplicated"` and the matching run's `run_id`:

```json
{
  "status": "deduplicated",
  "run_id": "<existing-run-id>",
  "source": "alertmanager",
  "pipeline": "alert-triage-critical",
  "severity": "critical",
  "summary": "...",
  "reason": "Duplicate of run <existing-run-id> (status=running)"
}
```

#### Configuration

Service-wide defaults in `config.yaml`:

```yaml
dedup:
  enabled: true
  window_seconds: 300
```

Per-pipeline override via `trigger.dedup` (both fields optional — `None` falls back to
the service default):

```yaml
trigger:
  match:
    severity: critical
  dedup:
    window_seconds: 600   # this pipeline's triage takes a while — widen the window
    # enabled: false       # or opt this pipeline out of dedup entirely
```

**Race safety:** the application-level check above narrows the window but two webhooks
with the same fingerprint arriving within milliseconds of each other can both pass it
before either's run row is inserted. The actual guarantee comes from a partial unique
index — `UNIQUE (pipeline_name, fingerprint) WHERE status = 'running'` (migration in
`service/src/db/database.py`). If both requests' inserts race, the database accepts
exactly one; the loser's insert raises `IntegrityError`, which `PipelineRunner` catches
in `_db_create_run()` and turns into an early `status="deduplicated"` result — no second
pipeline ever executes. NULL fingerprints are never equal in a unique index, so
fingerprint-less sources (sub-pipelines, re-runs) are unaffected. The one rough edge: the
loser's HTTP response was already sent as `"status": "accepted"` with its own `run_id`
before the conflict was discovered (responses are returned before the background task
runs), so that particular `run_id` 404s on `GET /runs/{run_id}` — the work itself is
never duplicated, only that one run_id is left unrealized.

---

### 3b. Team Attribution

To show LLM token spend broken down by owning team/department, every run is
tagged with a `team` — used for the `pork_pipeline_tokens_total` metric (§15a)
and the `GET /runs?team=` filter (§15).

**Team comes from the Bearer token that authenticated the webhook, not from a
field in the payload.** A self-reported `team` in a JSON body is spoofable and
easy to get wrong; tying team to the auth credential makes attribution
authoritative, and "onboarding a team" becomes synonymous with "issuing them a
token" — a natural gate.

**Configuration** — `auth.teams` replaces the single `auth.token`:

```yaml
auth:
  teams:
    - name: payments
      token: ${PORK_WEBHOOK_TOKEN_PAYMENTS}
    - name: platform
      token: ${PORK_WEBHOOK_TOKEN_PLATFORM}
  # token: ${PORK_WEBHOOK_TOKEN}   # legacy single-token form, still supported
```

Generate each team's token with `openssl rand -hex 24` (or any other source of
cryptographically random bytes) — a 48-character hex string. There's no token
issuance endpoint; this is a plain shared secret, handled the same way as
`executors.gateway.token` and the Telegram `bot_token` elsewhere in this file:
either resolved from an environment variable via `${ENV_VAR}` as shown above,
or written directly into `config.yaml` if you're not using env vars for
secrets — `config.yaml` is gitignored either way. Regenerate it yourself
locally rather than reusing a value that's appeared anywhere else (a chat
transcript, an issue tracker, etc.), since a real secret should only ever
exist in the one place it's actually used.

- If `auth.teams` is set, each entry's token is checked on `POST /webhook`;
  a recognized token resolves the run's `team`, an unrecognized or missing
  token still 401s exactly as before — no separate rejection path is needed
  for "no team supplied," since an unattributed/unauthenticated call already
  fails auth.
- If `auth.teams` is absent and the legacy `auth.token` is set, behaviour is
  unchanged — single shared token, every run's `team` is `None`
  (unattributed). If both are set, `auth.teams` wins silently.
- If neither is set, `POST /webhook` is unauthenticated, same as today.

**Non-webhook runs** don't have a caller/token to resolve team from:

- **Scheduled (cron) runs** declare `team:` directly on the pipeline's
  `schedule:` block (§8) — trusted because it's git-controlled config, not
  external input.
- **Sub-pipeline calls** (`executor: pipeline`, §9) inherit the parent run's
  `team` automatically, the same way they inherit `labels`/`metadata`, and it
  can be overridden per-call via `context: {team: "..."}` like any other
  field.

Out of scope for now: converting tokens to a dollar figure (no per-model
pricing table exists yet), and fixing the `openclaw` executor's lack of token
reporting (see §4's token budget note) — a team running mostly `openclaw`
steps will undercount regardless of this feature.

---

### 3c. Pipeline Stages (testing vs production)

Every pipeline has a `stage: testing | production` field. `testing` is the
**default** — an unmarked or newly-authored pipeline is fully executable and
fully observable inside P-Ork's own UI, but inert to the outside world and
excluded from every aggregate metric. `production` is today's pre-existing
behaviour. Promotion is a one-line YAML diff, reviewed in git like any other
config change, applied with `POST /reload`/SIGHUP — **there is no UI toggle**,
consistent with `tags`/`version` staying git-controlled.

```yaml
name: my-pipeline
stage: testing        # testing (default) | production
...
```

`stage` is **pipeline-level only** — there is no step-level override. It is
persisted on the run row at trigger time (`pipeline_runs.stage`), not derived
by joining against the current pipeline config, so promoting a pipeline to
`production` never retroactively reclassifies its prior testing runs.

#### What `testing` mutes

Four independent outbound paths are gated — every one of them logs what *would*
have happened instead of silently doing nothing:

| Path | Testing behaviour |
|---|---|
| `notifications:` block (§9a) | Forced to the `log` channel regardless of configured channel; run log gets a `notification_suppressed_testing` event instead of `notification_sent`. |
| `executor: notify` (§10c) | The HTTP call is skipped; the rendered body is logged and the step returns a synthetic success (`confidence=1.0`, `raw_response.suppressed_testing=true`) so downstream steps still run. |
| Step-level `on_failure.webhook` (§10b) | Skipped entirely; a `step_failure_webhook_suppressed_testing` run-log event records the URL that would have been called. |
| `executor: human` (§9 "human") | The external channel (Telegram/Slack/Teams) is **not** sent — but the approval is still registered in P-Ork's own UI (`/ui/approvals` and the run-detail banner), so a real Approve/Reject decision can be made. A Reject still resolves to `confidence=0.0` and drives `on_low_confidence`/downstream `when:` exactly as in production. Unlike production, a timeout **auto-approves** (`confidence=1.0`) rather than failing the step, so a forgotten testing approval never wedges the pipeline. A testing pipeline with no `human_approval` config at all still works — the channel is never resolved/built when testing. |

All four gates key off a single `_testing` boolean the runner injects into every
step's Jinja2/executor context (`{{ _testing }}` is available in prompts, though
the muting itself is automatic — pipeline authors don't need to reference it).

#### Trigger gating

A `stage: testing` pipeline does not fire from real ingestion traffic:

```bash
POST /webhook?source=alertmanager
# → {"status": "skipped_testing", "pipeline": "...", "reason": "..."}

POST /webhook?source=alertmanager&allow_testing=true
# → {"status": "accepted", "run_id": "..."}   — deliberately opted in
```

The **Run now** button (`POST /pipelines/{name}/run`) and re-run
(`POST /runs/{run_id}/rerun`) always run a testing pipeline — both are
deliberate manual actions, not real ingestion traffic.

#### Metrics and UI exclusion

A `stage: testing` run contributes **zero** to every aggregate/rollup surface:
`GET /metrics` (all series, including `pork_human_approvals_pending`, which
excludes testing approvals from its in-memory gauge the same way the
DB-backed counters do), the dashboard's stat cards and top-agents/top-tools
cards, the runs-page stat cards, pipeline success/accuracy bars, the
config-fingerprint accuracy comparison on `/ui/pipelines/{name}/feedback`, and
every Insights sub-page (`/ui/insights/pipelines`, `/steps`, `/agents`, `/models`, `/providers`, `/mcp`, `/teams`).

**Browse surfaces are the exception** — the runs list, dashboard's recent-runs
table, a pipeline's recent-runs table, and the chronological "every marked
run" table on the feedback page all show testing runs too, marked with an
amber **TESTING** badge, so testing activity stays fully visible for
debugging. `/ui/runs` has a `?stage=testing|production` filter (mirroring the
existing `team` filter) for browsing one stage at a time; the stat cards atop
that page always reflect production only, independent of this filter.

#### Promotion workflow

1. Develop and exercise a pipeline with the default `stage: testing` (or set
   it explicitly) — run it via **Run now** or `?allow_testing=true`, watch it
   in the UI, confirm accuracy feedback looks right.
2. When ready, change one line: `stage: production`.
3. `POST /reload` (or SIGHUP). The pipeline now fires from real traffic, its
   outbound notifications/webhooks/approvals go out for real, and new runs
   count toward every metric and rollup. Prior testing runs are unaffected —
   the DB already recorded them as `stage=testing`.

See `samples/pipelines/stage-testing-example.yaml` for a complete worked example
covering all three testing-gated executor paths (`notify`, `on_failure.webhook`,
`human`) plus muted pipeline `notifications:`, with the promotion comment inline.

---

### 4. Pipeline Config Schema (YAML)

Model selection is handled by the named agent's own config in the executor backend. To override the model for a specific step, set `executor_config.model` — this is passed directly to the executor and takes precedence over the agent's configured model.

Steps support an optional `verifier` block for independent confidence verification by a second agent. The verifier trigger can be set to `always: true` (fires unconditionally), or scoped to a confidence band. See §5 for full trigger configuration examples.

The `steps` list is heterogeneous — each entry is a sequential step (has a `name:` key), a parallel group (has a `parallel:` key), or a fan-out (has a `fan_out:` key). See §7 for parallel groups, §7a for fan-out, and §9 (`executor: pipeline`) for sub-pipeline calls.

```yaml
name: alert-triage-critical
description: Full triage pipeline for critical alerts
tags: [critical, sre, grafana]  # optional — free-form labels, searchable on /ui/pipelines
version: 1
stage: production                # testing (default) | production — see §3c

trigger:
  match:
    severity: critical          # matched against NormalisedContext fields/labels
    environment: prod           # all conditions must match (AND logic)
  dedup:                        # optional — overrides config.yaml dedup.* for this pipeline
    window_seconds: 600         # see §3a

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
    - channel: log                   # always available — zero config required
      template: |
        ESCALATED: {{pipeline_name}} — {{step_summary}}
      config:
        level: error                 # debug | info | warning | error | critical
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
    - channel: log
      template: "ABORTED: {{pipeline_name}} — {{step_summary}}"
      config:
        level: warning
    - channel: telegram
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

budget:                          # optional — omit to run with no token limit
  max_tokens: 50000              # abort run if accumulated tokens across all steps exceeds this
```

**Token budget guardrail:** if `budget.max_tokens` is set, the runner accumulates `input_tokens + output_tokens` from each completed step (including all branches of parallel/fan-out groups) and aborts the run with `status=aborted` if the total exceeds the ceiling. The check runs after each successful step — a step that's already failed or escalated won't trigger a second abort. A `budget_exceeded` event is appended to the run log.

Token counts come from `meta.agentMeta.usage` in the P-Ork Gateway response. Steps using other executors (`openclaw`, `human`, `webhook`) contribute 0 tokens to the accumulator — set `max_tokens` conservatively if your pipeline mixes executor types.

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

**Step library UI:** the `/ui/steps` page shows all loaded library steps with their executor/agent, confidence threshold, tags, which pipelines reference each step, and a copy button for the `- use: step-name` snippet. Each step with run history also gets a **per-pipeline/agent/model breakdown table** — runs, success rate, and avg tokens (in/out) for every distinct (pipeline, agent, model) combination that's executed this step, since the same library step can be wired to a different agent or model in different pipelines. Scoped to `stage=production` runs, same as every other rollup surface (§3c).

**Hot reload:** `POST /reload` and SIGHUP reload the step library first, then re-resolve all pipeline references against the updated library. A **Reload config** button on the `/ui/pipelines` page calls this endpoint directly from the browser.

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
    provider: str | None = None   # Gateway provider key (gateway executor only) — see §Providers.
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
| `provider` | No | Do not include — the `gateway` executor sets this from `agentMeta.provider` (the P-Ork Gateway provider that served the call, e.g. `anthropic`/`openrouter`/`azure`). `None` for other executors. |
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
| `critic` (default) | Verifier receives the primary agent's prompt, full response, **and a formatted transcript of the primary's own tool calls** (same trace grounding uses) — critiques the reasoning *and* can check specific claims ("a ticket was created", "a document was read") against actual evidence rather than just judging plausibility. Its *agreement* correlates with the primary's own errors and carries little signal; its *disagreement* is what's informative. |
| `independent` | Verifier receives only the original task prompt — executes the same task blind, with no sight of the primary's answer or its trace. Its agreement is uncorrelated with the primary's errors, so it's the stronger corroboration signal — prefer it for steps that authorise a side effect. |

> **Renamed from `reviewer`/`challenger`.** Those names still work — parsed as permanent
> aliases for `critic`/`independent` respectively — so no existing pipeline needs to
> change. New pipelines should prefer the new names; they describe the *role*
> (correlated critique vs. blind corroboration) rather than an adversarial framing.

**`verifier.max_trace_chars`** (default 1500, `critic` mode only) caps the transcript the same way `grounding.max_trace_chars` does — same truncation caveat applies: a claim whose evidence lands past the cutoff is invisible to the critic, and if the Gateway itself already truncated that tool result before P-Ork received it (`executor_config.trace_max_chars`, §8), no amount of raising this setting recovers it. See the "two independent truncation points" note under Grounding (§16) — it applies identically here.

See `samples/pipelines/trust-vector-remediation.yaml` for `critic` and `independent`
used side by side — cheap corroboration on a step that only informs, versus blind
corroboration on a step that authorises a side effect.

**Always verify:**
```yaml
verifier:
  executor: openclaw
  executor_config:
    agent: sre-verifier-opus
  mode: critic
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

Verifier failures (executor errors) are non-fatal — the runner logs a warning and falls
back to primary confidence only. **The verifier can only ever lower or hold the primary's
effective confidence — never raise it** (`minimum` takes the lower of the two; `veto`
only overrides when the verifier scores *below* `veto_floor`). This is a permanent
invariant, not an emergent property of the current code — see `_combine_confidence` and
its regression test in `tests/unit/test_confidence.py`.

**Which agent/model ran the verifier is persisted per-run** — `pipeline_steps.verifier_agent`
(`executor:agent`, mirroring the primary's `agent` column), `verifier_model`, and
`verifier_provider` (§14). This is deliberately a real column, not something read back out of
the *current* pipeline config at display time — a pipeline's `verifier.executor_config.agent`
can change between when a run executed and when someone looks at it later, and the audit
trail should reflect what actually ran, not what the config happens to say today.

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

### 7a. Fan-Out — Dynamic Parallelism

Parallel groups (§7) require branches to be listed in YAML at authoring time. Fan-out makes parallelism dynamic: a step emits a list at runtime and the runner spawns one branch per item, joining the results with the same confidence strategies.

**Example use cases:**
- A "list affected services" step returns `["api", "worker", "db"]` → fan out one triage branch per service
- An alert normaliser returns a list of firing alerts → fan out one remediation branch per alert

```yaml
steps:
  - name: identify-services
    executor: gateway
    executor_config:
      agent: service-lister
    prompt_template: |
      List affected services as a JSON array under the key "services".
      Alert: {{ summary }}
      Return JSON: {"confidence": 0.0, "summary": "...", "next_step_context": "...", "services": ["svc-a", "svc-b"]}

  - fan_out:
      name: triage-services
      over: "{{ steps.identify_services.services }}"
      as: service                # variable injected into each branch's Jinja2 context
      executor: gateway
      executor_config:
        agent: sre-triage-agent
      prompt_template: |
        Triage service "{{ service }}" (branch {{ fan_out_index + 1 }} of {{ fan_out_total }}).
        Context: {{ steps.identify_services.next_step_context }}
      join: all_must_pass        # same strategies as parallel groups
      confidence_threshold: 0.75
      on_low_confidence: escalate
      on_abort: notify
      max_items: 20              # hard cap — step fails if list exceeds this
      on_empty: skip             # complete | skip | abort
      timeout_seconds: 90        # per-branch timeout
      when: "steps.identify_services.proceed == true"
      verifier:
        executor: gateway
        executor_config:
          agent: reviewer
        trigger:
          always: true

  - name: consolidate
    executor: gateway
    executor_config:
      agent: decision-agent
    prompt_template: |
      Branch 0 verdict: {{ steps['triage-services/0'].summary }}
      Branch 1 verdict: {{ steps['triage-services/1'].summary }}
```

**`over` resolution:**

The `over` value is a Jinja2 template rendered against the current step context (the same context available in `prompt_template`). The rendered result is interpreted as a Python list — if the agent returned a Python-repr list (e.g. `"['a', 'b']"`), `ast.literal_eval` parses it; if it's a JSON array, `json.loads` is the fallback. A non-list result or parse failure marks the step as `failed`.

**Naming gotcha — avoid Python dict method names as extra field names.** Jinja2 attribute access (`dict.key`) tries `getattr` before `getitem`, so field names that shadow Python dict built-ins (`items`, `keys`, `values`, `get`, `pop`, `update`, etc.) resolve to the method object, not your data. Use descriptive names like `services`, `alerts`, `affected_hosts` rather than `items`.

**Per-branch context additions:**

| Variable | Value |
|---|---|
| `{{ service }}` (or whichever `as:` name you chose) | The current item value |
| `{{ fan_out_index }}` | 0-based position of this branch |
| `{{ fan_out_total }}` | Total number of branches spawned |

**Branch output references:**

Branch outputs are registered as `"{fan_out_name}/{index}"` — e.g. `triage-services/0`, `triage-services/1`. Reference them downstream using bracket notation (dot-notation breaks on `/`):

```yaml
{{ steps['triage-services/0'].summary }}
{{ steps['triage-services/1'].action }}
```

**Fan-out options:**

| Field | Default | Description |
|---|---|---|
| `over` | required | Jinja2 expression that resolves to a list |
| `as` | `item` | Variable name injected into each branch context |
| `join` | `all_must_pass` | `all_must_pass` \| `any_must_pass` \| `weighted_average` |
| `confidence_threshold` | `0.75` | Applied to the joined effective confidence |
| `on_low_confidence` | `escalate` | `escalate` \| `abort` \| `proceed` |
| `max_items` | `20` | Hard cap — step fails if the list is longer |
| `on_empty` | `complete` | `complete` (effective_confidence=1.0) \| `skip` (step skipped, no branch outputs) \| `abort` (step fails) |
| `timeout_seconds` | `null` | Per-branch timeout |
| `when` | `null` | Same conditional as sequential steps |
| `verifier` | `null` | Same verifier block as sequential steps — applied per branch |

See `samples/pipelines/fan-out-multi-service-triage.yaml` for a complete worked example.

---

### 8. Cron Scheduler

Any pipeline can declare an optional `schedule:` block to run on a cron schedule in addition to (or instead of) webhook triggers.

```yaml
schedule:
  cron: "0 9 * * 1-5"
  summary: "Daily morning service health sweep"
  severity: info
  team: platform           # owning team for token attribution — see §3b
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

Executors are registered by name in `src/executors/__init__.py` and referenced by name in pipeline YAML step `executor:` fields. Steps within the same pipeline can freely mix executors. Registered executors: `openclaw`, `gateway`, `human`, `webhook`, `notify`, `pipeline`.

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
| `trace_max_chars` | No | Overrides the Gateway's `limits.trace_tool_result_max_chars` (default 3000) for this step's `tool_result` trace events only — sent as `traceToolResultMax` on the agent request. Only affects the trace copy recorded/streamed for observability (and what `grounding.max_trace_chars` has available to hand the judge, see §16) — the agent's own conversation always sees the full, untruncated tool output regardless of this setting. Raise it on steps whose tools return long content if grounding or a human reviewing the trace is drawing false conclusions from evidence that was cut off before the Gateway ever sent it to P-Ork. |

Requires `executors.gateway.url` (WebSocket) and `executors.gateway.rest_url` (REST) in `config.yaml`.

The P-Ork Gateway exposes three REST endpoints consumed by P-Ork:

| Endpoint | Purpose |
|---|---|
| `GET /agents` | Agent list (name, model, model_fallbacks, tools) |
| `GET /agents/{name}/soul` | `soul.md` content — shown in the Soul tab of the agent detail page |
| `GET /agents/{name}/agent` | Raw `agent.yaml` content — shown in the Config tab of the agent detail page |

---

#### `human` — Human-in-the-Loop (Telegram, Slack, Microsoft Teams)

**`executor: human`** — Sends an approval request and pauses the pipeline until the operator approves or rejects, or `timeout_seconds` elapses.

| Outcome | confidence | proceed |
|---|---|---|
| Approved | 1.0 | true |
| Rejected | 0.0 | true — triggers `on_low_confidence` action |
| Timeout | — | step marked `failed` |

The `prompt_template` renders to the approval message text. Default timeout is 300s.

**Which channel a run uses is resolved per-team, not per-pipeline.** The same `executor: human` step works unchanged for every team — P-Ork looks up the run's `team` (resolved from the webhook auth token, §3b) against `human_approval.teams` in `config.yaml`, falling back to `human_approval.default`, falling back to the legacy Telegram-only config if `human_approval` is omitted entirely. This keeps team onboarding a config-only change (like issuing a token, §3b) rather than requiring a new executor or a pipeline fork per team.

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

**Config (`config.yaml`):**

```yaml
human_approval:
  ui_base_url: https://pork.internal.example.com   # required for the msteams channel — see below
  default:
    channel: telegram
  teams:
    team-a:
      channel: slack
      slack:
        bot_token: ${SLACK_BOT_TOKEN_TEAMA}
        app_token: ${SLACK_APP_TOKEN_TEAMA}
        channel_id: C0123456
    team-b:
      channel: msteams
      msteams:
        webhook_url: ${TEAMS_WEBHOOK_URL_TEAMB}
```

| Channel | How the human responds | Requires |
|---|---|---|
| `telegram` | Inline-keyboard Approve/Reject buttons, resolved by the existing Telegram long-poll (`notifications/telegram_poller.py`). Requires a **separate** Telegram bot from OpenClaw (Telegram only allows one simultaneous `getUpdates` poller per bot token). | `human_approval.*.telegram.{bot_token,chat_id}`, or falls back to `notifications.telegram` |
| `slack` | Interactive Approve/Reject buttons via a Slack app's Socket Mode connection (`notifications/slack_poller.py`) — no public HTTPS endpoint needed, free on any Slack plan. | `human_approval.*.slack.{bot_token,app_token,channel_id}` |
| `msteams` | One-way notification (via a Power Automate webhook flow) linking to a P-Ork web page (`GET /ui/approvals/{token}`) where the human clicks Approve/Reject. Real interactive Adaptive Card buttons in Teams need a registered Azure Bot with a public callback endpoint — this deployment doesn't expose one, so Teams gets a notify-and-click-through flow instead. | `human_approval.*.msteams.webhook_url`, `human_approval.ui_base_url` |

If `human_approval` is omitted entirely, every `human` step behaves exactly as before this feature existed — the single global `notifications.telegram` bot/chat, no team awareness required.

**The `notifications.telegram` fallback only applies to the `telegram` channel.** It's a special case: a `human_approval.*` entry with `channel: telegram` and no nested `telegram:` block reads `notifications.telegram` instead. `slack` and `msteams` have no equivalent fallback — a `default:` or `teams.<name>:` entry using either of those channels must include its own nested `slack:`/`msteams:` credentials, exactly like any other channel entry. Omitting it raises `RuntimeError: Slack/Teams approval channel missing ...` the first time a run resolves to that entry.

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

#### `pipeline` — Sub-Pipeline Call

**`executor: pipeline`** — Calls another named pipeline as a sub-pipeline. The sub-pipeline runs through the standard runner — full step execution, DB row, tracing — and its final step's `LLMOutput` becomes the current step's output. This turns pipelines into composable building blocks.

`executor_config` keys:

| Key | Required | Description |
|---|---|---|
| `pipeline` | **Yes** | Name of the pipeline to call. Must be loaded and present in the pipeline registry at call time. |
| `context` | No | `NormalisedContext` field overrides. Scalar fields are Jinja2-rendered strings. `labels` and `metadata` are dicts — rendered keys are **merged** with the parent values (parent keys preserved; overrides add or replace individual keys). |

The sub-pipeline inherits the parent's `NormalisedContext` by default. The `pipeline` and `source` fields are updated (`source` → `"sub-pipeline"`) and `fingerprint` is cleared (bypasses dedup). `team` (§3b) is inherited unchanged like `labels`/`metadata`, so a shared sub-pipeline's token spend rolls up to whichever team's call triggered it — overridable via `context: {team: "..."}` like any other field. Use `context:` to pass step-specific values:

```yaml
- name: triage
  executor: pipeline
  executor_config:
    pipeline: shared-triage
    context:
      # Scalar field override — Jinja2-rendered
      summary: "{{ steps.pre_filter.next_step_context }}"
      # Dict field override — merged with parent labels
      labels:
        routed_by: "{{ pipeline_name }}"
      metadata:
        focus: "Check database connection pool first"
  confidence_threshold: 0.75
  on_low_confidence: escalate
```

**Sub-pipeline DB linkage:** the sub-pipeline runs with its own `run_id`, stored in `pipeline_runs` with `parent_run_id` set to the parent run's ID. This makes sub-pipeline runs fully traceable — you can query `SELECT * FROM pipeline_runs WHERE parent_run_id = '<parent-run-id>'` to see all sub-pipeline invocations for a parent run.

**Extra fields on the parent step output:**

| Field | Description |
|---|---|
| `sub_run_id` | The `run_id` assigned to the sub-pipeline run |
| `sub_pipeline_status` | Terminal status of the sub-pipeline (`completed`, `failed`, `escalated`, etc.) |

These are available downstream as `{{ steps.triage.sub_run_id }}` and `{{ steps.triage.sub_pipeline_status }}`.

**Failure behaviour:** if the sub-pipeline has a `final_output` (its last step ran and produced output), that output is used as-is regardless of sub-pipeline status. If the sub-pipeline has no `final_output` (it was aborted/escalated before any step completed), the parent step synthesises an `LLMOutput` with `confidence=1.0` (completed) or `confidence=0.0` (any other status).

**Hot reload:** `POST /reload` and SIGHUP update the pipeline registry, so changes to a sub-pipeline YAML take effect immediately without restarting the service.

**Using `executor: pipeline` in a fan-out:** each fan-out branch can delegate to a sub-pipeline, passing the branch item via `context:`:

```yaml
- fan_out:
    name: per-service-triage
    over: "{{ steps.identify_services.services }}"
    as: service
    executor: pipeline
    executor_config:
      pipeline: shared-triage
      context:
        labels:
          service: "{{ service }}"
        metadata:
          focus: "Focus specifically on {{ service }}"
    join: all_must_pass
    confidence_threshold: 0.75
    on_low_confidence: escalate
```

See `samples/pipelines/sub-pipeline-example.yaml` for a complete worked example including conditional routing based on sub-pipeline output.

---

### 9a. Pipeline Notification Channels

The `notifications:` block in a pipeline YAML wires pipeline state transitions (escalate, abort, stop, notify) to outbound channels. Three channels are available:

| Channel | Config required | Description |
|---|---|---|
| `log` | None | Writes to the application logger — always available, zero dependencies |
| `telegram` | `notifications.telegram` in `config.yaml` | Sends a Telegram message via bot API |
| `webhook` | `url` in per-notification `config:` | POSTs the rendered template as an HTTP request body |

A single action can fan out to multiple channels by providing a list:

```yaml
notifications:
  escalate:
    - channel: log
      template: "ESCALATED: {{pipeline_name}} — {{step_summary}}"
      config:
        level: error
    - channel: telegram
      template: "🚨 {{pipeline_name}}: {{step_summary}}"
    - channel: webhook
      template: '{"text": "{{pipeline_name}} escalated: {{step_summary}}"}'
      config:
        url: https://hooks.slack.com/services/...
```

#### `log` channel

Always registered — no `config.yaml` entry needed to enable it. The rendered template is emitted via Python's standard `logging` module, landing in whatever log aggregation stack (stdout, rotating file, Loki, CloudWatch) the service ships to.

Per-notification `config:` keys:

| Key | Default | Description |
|---|---|---|
| `level` | `warning` | Log level: `debug` / `info` / `warning` / `error` / `critical`. Also accepts `warn` as an alias. |
| `logger` | `pork.notifications` | Logger name. Override to route to a specific logger hierarchy. |

```yaml
notifications:
  escalate:
    - channel: log
      template: |
        ESCALATED: {{pipeline_name}} — {{step_summary}}
        Alert: {{context.summary}}  Confidence: {{confidence}}
      config:
        level: error

  notify:
    - channel: log
      template: "PIPELINE ABORTED: {{pipeline_name}} — {{step_summary}}"
      config:
        level: warning
        logger: pork.ops            # route to a separate logger for ops tooling
```

The `log` channel is the recommended zero-setup choice for development and for production environments that already aggregate application logs centrally. Add `telegram` or `webhook` alongside it for real-time alerting.

#### `telegram` channel

Requires `notifications.telegram.bot_token` and `notifications.telegram.chat_id` in `config.yaml`. Messages use Telegram's HTML parse mode — `<b>`, `<code>`, and `<a href>` tags work in templates. Messages longer than 4096 characters are truncated with a `[truncated]` suffix.

#### `webhook` channel

POSTs the rendered `template` string as the raw request body. Per-notification `config:` keys match those of `executor: webhook` (url, method, content_type, headers, timeout_seconds). For structured JSON payloads it is cleaner to use an `executor: notify` step (see §10c) which renders a `payload:` dict rather than requiring inline JSON in a template string.

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
6. If the executor raised an error (step status = `failed`), check `on_failure`:
   - `on_failure: abort` (default) → abort the pipeline
   - `on_failure: continue` → log and continue to next step
   - If `on_failure.webhook` is set, fire that callback regardless of policy
7. Record step result to SQLite before proceeding

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
| `interrupted` | Service restarted/crashed mid-run — set by the startup sweep, not the runner |

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

### 10b. Per-Step Failure Policy (`on_failure`)

By default a failed step (executor exception, timeout) aborts the pipeline. For non-critical steps — enrichment lookups, external API calls, notifications — you can allow failures to pass through:

```yaml
- name: enrich-from-cmdb
  executor: webhook
  on_failure: continue       # pipeline keeps running if this step fails
  executor_config:
    url: "https://cmdb.internal/api/enrich"
```

The step is still recorded in run history with `status: failed` and the error message is available in `step_outputs[name].summary` for downstream `when:` conditions or prompt templates.

`on_failure` only applies to executor errors. Low-confidence results that trigger `on_low_confidence: abort` are LLM-quality decisions and are always pipeline-stopping regardless of this setting.

#### Step-level failure webhook callback

Attach a webhook to any step that fires when that step fails, without adding a separate notify step to the pipeline. The callback fires before the pipeline decides whether to abort or continue, so it always goes out regardless of the policy.

```yaml
- name: triage
  executor: gateway
  executor_config:
    agent: sre-triage
  on_failure:
    policy: continue        # pipeline continues even if this step fails
    webhook:
      url: "${PAGERDUTY_URL}"
      headers:
        Authorization: "Token ${PAGERDUTY_TOKEN}"
      payload:
        summary: "Triage step failed: {{step_failure.summary}}"
        severity: critical
```

`on_failure` as a block:

| Field | Default | Description |
|---|---|---|
| `policy` | `abort` | `abort` or `continue` — what the pipeline does after the failure |
| `webhook.url` | required | Outbound URL (`${ENV_VAR}` expansion supported) |
| `webhook.method` | `POST` | HTTP method |
| `webhook.headers` | `{}` | Header dict; values support `${ENV_VAR}` expansion |
| `webhook.payload` | `{}` | JSON body dict; string values are Jinja2 templates |
| `webhook.timeout_seconds` | `30` | Request timeout |

String shorthand (`on_failure: continue` or `on_failure: abort`) is equivalent to setting `policy` only with no webhook.

The Jinja2 context for the webhook payload includes all standard step context variables plus `step_failure.step`, `step_failure.summary`, and `step_failure.status`. Webhook delivery failures are logged as a `step_failure_webhook_failed` run-log event and never abort the pipeline.

---

### 10c. Outbound Notification Steps (`executor: notify`)

`executor: notify` is a first-class pipeline step that sends an outbound HTTP webhook with a structured payload. Unlike `executor: webhook` (which uses `prompt_template` as the raw request body), `notify` takes a `payload:` dict in `executor_config` and recursively renders every string value as a Jinja2 template before JSON-encoding the body.

This makes it ergonomic for services that expect structured JSON payloads — Slack blocks, PagerDuty events, Teams Adaptive Cards, Jira tickets, etc. — without the author having to inline raw JSON inside a YAML string.

```yaml
steps:
  - name: alert-pagerduty
    executor: notify
    when: "context.severity == 'critical'"
    on_failure: continue           # notification failure should not abort the run
    executor_config:
      url: "${PAGERDUTY_EVENTS_URL}"
      headers:
        Authorization: "Token token=${PAGERDUTY_TOKEN}"
      payload:
        routing_key: "${PAGERDUTY_ROUTING_KEY}"
        event_action: trigger
        payload:
          summary: "{{context.summary}}"
          severity: "{{context.severity}}"
          source: pork
          custom_details:
            triage_summary: "{{steps.triage.output.summary}}"
            confidence: "{{steps.triage.output.confidence}}"

  - name: notify-slack
    executor: notify
    on_failure: continue
    executor_config:
      url: "${SLACK_WEBHOOK_URL}"
      payload:
        blocks:
          - type: section
            text:
              type: mrkdwn
              text: "*Alert:* {{context.summary}}\n*Triage:* {{steps.triage.output.summary}}"
```

`executor_config` keys:

| Key | Default | Description |
|---|---|---|
| `url` | required | Target URL (`${ENV_VAR}` expansion supported) |
| `method` | `POST` | HTTP method |
| `headers` | `{}` | Header dict; values support `${ENV_VAR}` expansion |
| `payload` | `{}` | Body dict; all string values are recursively rendered as Jinja2 templates |
| `content_type` | `application/json` | `Content-Type` header shorthand |
| `timeout_seconds` | `30` | Request timeout |

The step returns `confidence: 1.0` on success so it never triggers low-confidence escalation. HTTP errors (non-2xx) propagate as executor exceptions; combine with `on_failure: continue` so notification failures never abort a pipeline run.

**When to use `executor: notify` vs `executor: webhook`:**
- Use `notify` when the target expects a structured JSON payload that you want to compose in YAML (Slack, PagerDuty, Teams, Jira, etc.)
- Use `webhook` when you need full control over the raw body and prefer rendering it as a `prompt_template` string

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

### 14. Run Storage

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
| `logs` | json, nullable | Structured run event log — array of `{ts, level, event, msg}` objects. Populated at run completion. Events cover step start/complete/fail/skip/escalate/abort, verifier results, parallel group outcomes, notifications sent, and (for `interrupted` runs) the startup interruption sweep. |
| `parent_run_id` | uuid, nullable, indexed | Set for sub-pipeline runs (`executor: pipeline`). Links back to the parent run. NULL for top-level runs. |
| `team` | str, nullable, indexed | Owning team, resolved from the Bearer token that authenticated the webhook (see §3b). NULL for unattributed/legacy-token runs. |
| `stage` | str, indexed | `testing` or `production`, copied from `PipelineConfig.stage` (see §3c) at trigger time — persisted per-run so promoting a pipeline never reclassifies its prior runs. Defaults to `production` at the DB layer for rows that predate this column. |

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
| `prompt` | text | Rendered prompt sent to the agent — the actual, fully-substituted text, not the `{{ }}` template. Populated for `executor: gateway` steps only; other executors don't yet stash their rendered prompt back out, so their rows fall back to a JSON dump of `executor_config` (recognisable by starting with `{`) — the UI's Prompt disclosure hides that fallback rather than showing it as if it were a real prompt. |
| `raw_output` | json | Full unparsed executor response |
| `parsed_output` | json | Validated LLMOutput (excluding raw_response) |
| `status` | str | completed / stopped / escalated / aborted / failed |
| `primary_confidence` | float | Raw confidence from the primary agent |
| `verifier_confidence` | float, nullable | Verifier agent confidence (if verifier ran) |
| `effective_confidence` | float | Confidence used for threshold gate (post-combination) |
| `grounding_score` | float, nullable | Shadow-mode grounding score **G** ∈ [0,1] — the fraction of the step's load-bearing claims a blind grounding judge found supported by evidence in the step's own execution trace. NULL when grounding wasn't configured for the step, or when it had no trace to check against. Never gates — see [§16 Grounding (shadow mode)](#grounding-shadow-mode). |
| `trust_report` | json, nullable | Per-step TrustReport: the trust vector `{S, S_after_V, V, V_mode, V_combination_strategy, V_veto_floor, G, C, D}`, `combined_trust`, `gate` (`{policy, confidence_threshold, on_low_confidence}` — `policy` is `legacy_confidence` / `trust_vector`), and — for grounding — the per-claim support breakdown, and — for deterministic checks — the full per-check detail. Populated for steps with a **verifier**, a `grounding:` block, `deterministic_checks:`, and/or `calibration: {enforce: true}` — i.e. any mechanism beyond the plain single-confidence gate, not just the trust-vector ones; `mode` is `"shadow"` when recorded-only or `"enforced"` when grounding/deterministic/calibration actually participated in the gate (a verifier-only step is always `"shadow"`, since the verifier's downward-only combine has always been part of the legacy gate). See [Deterministic checks & enforced grounding (Phase 1)](#deterministic-checks--enforced-grounding-phase-1). A `calibration` sub-key (bucket/bin/n/n_min/validated/raw/calibrated/on_uncalibrated) is present only for a step with `calibration: {enforce: true}` — see [Calibration (Phase 3)](#calibration-phase-3). |
| `deterministic_passed` | bool, nullable | Whole-step pass/fail across all declared deterministic checks — `True` only if every check passed. NULL when no `deterministic_checks:` were declared. Full per-check detail lives in `trust_report.deterministic_checks`. |
| `duration_ms` | int | |
| `executed_at` | datetime | |
| `artifacts` | json, nullable | `{key: reference}` map — references are opaque strings pointing to artifact files on disk. Content is not stored in the DB. |
| `agent_trace` | json, nullable | Ordered execution trace from the P-Ork Gateway executor — array of `{type, ...}` objects. `type` is one of: `llm_call` (iteration marker), `thinking` (extended thinking block), `text` (response text), `tool_call` (MCP tool invoked with arguments), `tool_result` (MCP tool response). **Not truncated** — the full content the Gateway returns on its final `ok` frame is stored and rendered as-is; only the ephemeral *live* SSE tail (§Live tail) truncates content (at 200 chars) for a fast in-progress preview, never the persisted record. If a `tool_result`'s content looks cut off on a *completed* run, that truncation happened upstream (the Gateway server or the MCP tool itself), not in this column. NULL for all other executors (`openclaw`, `human`, `webhook`). |
| `verifier_agent` | str, nullable | `executor:agent-name` for the verifier call (mirrors `agent` above), e.g. `gateway:principal-sre`. NULL if no verifier ran, or for rows persisted before this column existed. |
| `verifier_model` | str, nullable | Actual model used by the verifier call, from executor metadata. NULL if no verifier ran. |
| `verifier_provider` | str, nullable | Gateway provider key for the verifier call (gateway executor only). NULL if no verifier ran or the verifier used a non-gateway executor. |
| `verifier_prompt` | text, nullable | Rendered prompt actually sent to the verifier — for `critic` mode this is the meta-prompt with the primary's own prompt+response embedded; for `independent` mode it's a verbatim copy of the primary's prompt. Gateway executor only; NULL if no verifier ran, the verifier used a non-gateway executor, or the row predates this column. |
| `input_tokens` | int, nullable | Input tokens consumed by this step's primary executor call. Populated for `gateway` steps; NULL for others. For parallel/fan-out branches, each branch row has its own token count. |
| `output_tokens` | int, nullable | Output tokens produced by this step's primary executor call. |

**run_feedback**

| Column | Type | Description |
|---|---|---|
| `id` | uuid, pk | |
| `run_id` | str, unique, indexed | The run this feedback is for. One row per run — submitting again upserts. |
| `pipeline_name` | str, indexed | Denormalised from the run for efficient pipeline-level queries. |
| `outcome` | str | `correct` / `partial` / `incorrect` — human judgement of the run's result. |
| `notes` | text, nullable | Optional free-text context. |
| `submitted_at` | datetime | Created or last updated. |

**step_feedback**

| Column | Type | Description |
|---|---|---|
| `id` | uuid, pk | |
| `step_id` | str, unique, indexed | The specific step execution (`pipeline_steps.id`) this feedback is for. One row per step, upserted. |
| `run_id` | str, indexed | Denormalised for lookup. |
| `pipeline_name` | str, indexed | Denormalised. |
| `step_name` | str | Denormalised — may contain `/` for fan-out branches (e.g. `triage/0`). |
| `outcome` | str | `correct` / `partial` / `incorrect` — human judgement of that step's result. |
| `notes` | text, nullable | Optional free-text context. |
| `submitted_at` | datetime | Created or last updated. |

### 15. Management Endpoints

```bash
# Trigger a run (returns immediately — pipeline runs in background)
POST /webhook?source=<source>
# → {"status": "accepted", "run_id": "<uuid>"}
# → {"status": "deduplicated", "run_id": "<uuid>", "reason": "..."}  — see §3a
# → {"status": "skipped_testing", "pipeline": "...", "reason": "..."}  — see §3c;
#   pass ?allow_testing=true to run a stage=testing pipeline from this source

# Reload step library and all pipeline YAMLs from disk without restarting
POST /reload
# → {"status": "reloaded", "pipelines_loaded": 3}

# SIGHUP also triggers reload
kill -HUP <uvicorn-pid>

# List active cron schedules
GET /schedules
# → {"schedules": [{"pipeline": "...", "cron": "...", "next_run": "..."}]}

# List runs — newest first. Filters: ?status=escalated, ?pipeline=alert-triage-critical, ?team=payments
# Pagination: ?limit=50&offset=0 (max 200)
GET /runs
# → {"runs": [{id, pipeline_name, source, status, team, stage, triggered_at, completed_at}, ...]}
# stage is "testing" or "production" (see §3c), included on both the list and the
# full detail response below — persisted per-run, so it reflects the pipeline's
# stage at the time the run was triggered rather than its current config.

# Full run detail — includes all steps with confidence scores and parsed output
GET /runs/{run_id}

# Submit or update human accuracy feedback for a run (outcome: correct|partial|incorrect)
POST /runs/{run_id}/feedback
# body: {"outcome": "correct", "notes": "..."}
# → {"run_id": "...", "outcome": "correct", "notes": "...", "submitted_at": "..."}
# Upserts — submitting again overwrites the previous outcome and notes.

# Get current feedback for a run
GET /runs/{run_id}/feedback
# → {"feedback": {run_id, outcome, notes, submitted_at}} or {"feedback": null}

# Submit or update human accuracy feedback for a single step execution
# (outcome: correct|partial|incorrect). step_name may contain "/" for fan-out
# branches (e.g. triage/0) — the route uses a path converter to match it.
POST /runs/{run_id}/steps/{step_name}/feedback
# body: {"outcome": "correct", "notes": "..."}
# → {"run_id": "...", "step_name": "...", "outcome": "correct", "notes": "...", "submitted_at": "..."}
# Upserts — submitting again overwrites the previous outcome and notes.

# Get current feedback for a single step execution
GET /runs/{run_id}/steps/{step_name}/feedback
# → {"feedback": {run_id, step_name, outcome, notes, submitted_at}} or {"feedback": null}

# Re-run a pipeline from a specific step
POST /runs/{run_id}/rerun
# body: {"from_step": "step-name"}  — see §Re-run from a step

# Manually trigger a pipeline by name — powers the UI's Run now button.
# Not behind webhook Bearer auth (internal/management action, same trust
# boundary as /reload — not the public ingestion path auth.teams gates).
# Runs triggered here are unattributed (team=None).
POST /pipelines/{name}/run
# body: same shape as a generic webhook payload (see §2a), pipeline forced from the path
# → {"status": "accepted", "run_id": "<uuid>", ...}

# Server-Sent Events stream for live run tailing (see §UI / Live tail)
GET /ui/runs/{run_id}/stream

# List loaded pipelines
GET /pipelines

# Prometheus metrics — runs/steps by status, step duration histograms, verifier veto rate
GET /metrics
```

### 15a. Prometheus Metrics

`GET /metrics` exposes Prometheus text-format metrics, computed from `pipeline_runs` /
`pipeline_steps` at scrape time. All counters are cumulative all-time totals — use
`rate()`/ratios in PromQL for escalation rate, per-agent success rate, and verifier veto
frequency rather than relying on pre-baked percentages.

Every metric below is scoped to `stage=production` — a `stage=testing` pipeline (the
default, see §3c) contributes to none of them, including the metrics that query
`pipeline_steps` without otherwise touching `pipeline_runs`.

| Metric | Type | Labels | Description |
|---|---|---|---|
| `pork_pipeline_runs_total` | counter | `pipeline`, `status` | Total runs by pipeline and terminal status |
| `pork_pipeline_runs_in_progress` | gauge | — | Runs currently in `status=running` |
| `pork_pipeline_steps_total` | counter | `pipeline`, `step_name`, `executor`, `agent`, `model`, `provider`, `status` | Total steps by pipeline, step, executor, agent, model, provider, and status. `pipeline`/`step_name`/`model`/`provider` are what let a Grafana dashboard reconstruct the per-step and per-model success-rate breakdowns the Pipelines/Steps/Agents Insights UI pages (below) compute directly from the DB — the UI is for a quick look, this metric is for a real dashboard or alert. NULL agent/model/provider (non-gateway executors, or a gateway build predating the `provider` field) are bucketed as `""`. |
| `pork_pipeline_step_duration_seconds` | histogram | `pipeline`, `step_name`, `executor`, `agent`, `model`, `provider` | Step execution duration (buckets: 1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, +Inf seconds) |
| `pork_verifier_runs_total` | counter | `agent` | Steps where a verifier ran, by primary agent |
| `pork_verifier_overrides_total` | counter | `agent` | Verifier runs where the verifier lowered the primary's effective confidence |
| `pork_pipeline_tokens_total` | counter | `team`, `pipeline`, `step_name`, `executor`, `agent`, `model`, `provider`, `direction` | Cumulative input/output tokens consumed, broken down by owning team (§3b) for cost attribution. `direction` is `input`/`output`. NULL team/model/provider are bucketed as `""` rather than dropped, so unattributed spend stays visible. Steps from executors that don't report tokens (`openclaw`, `human`, `webhook`) are excluded rather than padded as zero. |
| `pork_human_approval_decisions_total` | counter | `team`, `pipeline`, `decision` | Cumulative `human` step (§9 `executor: human`) approve/reject decisions. `decision` is `approved`/`rejected`, derived from `primary_confidence` (1.0/0.0 — see the executor's contract). Timeouts leave `primary_confidence` NULL and are excluded rather than miscounted as either outcome. NULL team is bucketed as `""`. |
| `pork_pipeline_feedback_total` | counter | `pipeline`, `outcome` | Cumulative human accuracy feedback (§Accuracy feedback — the same data backing `/ui/pipelines/{name}/feedback`). `outcome` is `correct`/`partial`/`incorrect`. |
| `pork_step_feedback_total` | counter | `pipeline`, `step_name`, `agent`, `model`, `provider`, `outcome` | Cumulative per-step human accuracy feedback (§Accuracy feedback), production-scoped. `outcome` is `correct`/`partial`/`incorrect`. |
| `pork_step_grounding_score` | histogram | `pipeline`, `step_name`, `agent`, `model`, `provider` | Shadow-mode grounding score (G) distribution per step (§Grounding (shadow mode)), production-scoped (buckets: 0.1, 0.2, ..., 1.0, +Inf). Only steps with a `grounding:` block that produced a score contribute — NULL/not-computed steps are excluded, not padded as zero. |
| `pork_step_deterministic_check_total` | counter | `pipeline`, `step_name`, `outcome` | Cumulative whole-step deterministic-check outcomes (§Deterministic checks & enforced grounding (Phase 1)), production-scoped. `outcome` is `passed`/`failed` — `passed` only when every declared check for that step run passed. Steps with no `deterministic_checks:` declared are excluded. |
| `pork_human_approvals_pending` | gauge | `team` | Currently pending `human` step approvals, awaiting a response on whichever channel (Telegram/Slack/Teams) that team is routed to (§ "human — Human-in-the-Loop"). Unlike every other metric here this isn't derived from the database — pending approvals are in-memory only — so it reflects only this process's current state, not a historical/cumulative total. NULL team is bucketed as `""`. Always emits at least a zero-valued series so the metric doesn't disappear from dashboards when nothing's pending. Excludes `stage=testing` pending approvals, same as every other series on this page — see §3c. |

Dollar-cost conversion is intentionally not provided — there's no per-model
pricing table yet, so this metric is raw token counts only.

Standard `python_*` / `process_*` / `python_gc_*` process-health metrics are included
automatically via `prometheus_client`'s default collectors.

### 15b. OpenTelemetry Tracing

While `/metrics` gives aggregate, all-time counters, **distributed tracing gives a
per-run drill-down** — one trace per pipeline run, with a span for every step, branch,
verifier, and underlying LLM call. Use `/metrics` to spot trends ("escalation rate is
up this week"); use traces to answer "why was *this* run slow / why did *this* step
escalate".

Tracing is **disabled by default** and adds zero overhead until enabled — the OTel
SDK's default `TracerProvider` is a no-op.

**Enabling tracing** — add to `config.yaml`:

```yaml
observability:
  otel:
    enabled: true
    exporter: otlp          # otlp | console
    endpoint: http://localhost:4318/v1/traces   # OTLP/HTTP collector endpoint
    service_name: pork-service
```

`exporter: console` prints spans to stdout — useful for local verification without a
collector. `exporter: otlp` (default) sends spans via OTLP/HTTP to a collector (e.g.
Grafana Alloy/Tempo).

**Span hierarchy** — each pipeline run is its own trace root:

```
pipeline.run: <team>/<pipeline>       (pork.pipeline.name, pork.run.id, pork.source, pork.team, pork.run.status)
├── <step name>                       (pork.span.kind=step, pork.executor, pork.agent, confidences, pork.model, pork.provider)
│   ├── gen_ai.<executor>             (pork.span.kind=gen_ai — the LLM call itself)
│   └── <step>:verifier|:independent   (pork.span.kind=verifier, pork.verifier.mode, pork.confidence)
│       └── gen_ai.<executor>
├── <group name>                      (pork.span.kind=parallel_group, pork.join_strategy, pork.branch_count)
│   ├── <branch name>                 (pork.span.kind=branch, pork.executor, pork.agent, pork.confidence, pork.model, pork.provider)
│   │   └── gen_ai.<executor>
│   └── ... (concurrent siblings)
├── <fan_out name>                    (pork.span.kind=fan_out, pork.join_strategy)
│   ├── <fan_out name>/0              (pork.span.kind=branch, pork.executor, pork.agent, pork.confidence)
│   │   └── gen_ai.<executor>
│   └── ... (one branch per runtime list item)
└── ...
```

The root span name is `pipeline.run: <team>/<pipeline-name>` when a team is attributed (e.g. `pipeline.run: payments/alert-triage-critical`), or `pipeline.run: <pipeline-name>` for unattributed runs. This makes the Name column in Grafana Tempo immediately useful without needing to expand attributes. `pork.team` is also set as a span attribute when present, so traces can be filtered/grouped by team in PromQL or Tempo query expressions.

Each `pipeline_runs.logs` event (`step_started`, `verifier_ran`, `branch_completed`,
etc. — see §14) is also recorded as a **span event** on the current span, so the
structured run log and the trace timeline tell the same story from two angles.

**`gen_ai.*` span attributes:**

| Attribute | Description |
|---|---|
| `gen_ai.system` | Executor name (`openclaw`, `openclaw_ws`, `gateway`) |
| `gen_ai.request.model` | Model override requested, if any |
| `gen_ai.response.model` | Actual model used, from executor metadata |
| `pork.gateway.duration_ms` | Backend-reported call duration |
| `pork.agent` | Agent name |

**Forward compatibility with the P-Ork Gateway:** every `gen_ai.*` span injects the
current `traceparent`/`tracestate` (W3C Trace Context) into the outbound `agent`
request sent to the OpenClaw/P-Ork Gateway WebSocket APIs. Today's gateways ignore
these extra params. Once the P-Ork Gateway adds its own OTel instrumentation, its
LLM/tool-call spans will automatically nest under the corresponding `gen_ai.*` span —
giving one connected trace from webhook → pipeline → step → individual LLM/tool calls,
with no further changes needed on the P-Ork side.

---

## UI

The web UI is served under `/ui` and provides the following pages:

| Page | Route | Description |
|---|---|---|
| Dashboard | `/ui/` | 24h run counts by status, success rate, pipeline activity, recent runs (production only, see §3c) — recent runs shows testing runs too, badged |
| Runs | `/ui/runs` | Run history with status/pipeline/team/**stage** (`?stage=testing\|production`, §3c) filters and a selectable time range (24h/7d/30d/all-time); stat cards (run count, success rate, escalated/failed, accuracy marked, accuracy %, tokens) are always scoped to production, independent of the stage filter; the list itself shows testing runs too, badged |
| Run detail | `/ui/runs/{id}` | Full step breakdown with confidence bars, parsed output, verifier results, collapsible agent trace (gateway steps), collapsible run log, live tail for in-progress runs, **accuracy feedback widget**, and a **TESTING** badge when `stage=testing` |
| Approvals | `/ui/approvals` | Every pending `executor: human` approval (§ "human — Human-in-the-Loop"), regardless of channel — a universal fallback so a team isn't stuck if their primary chat channel (Slack/Telegram) is unreachable. No standalone sidebar entry; reached via a pending-count badge next to **Runs** (only shown when the count is non-zero) |
| Approval decision | `/ui/approvals/{token}` | Standalone page (no sidebar) reached via a direct token link — used by the Teams approval channel, which posts this link instead of an in-chat button since Teams interactive cards need a public Bot Framework callback endpoint this deployment doesn't expose (see `executors/human.py` `TeamsApprovalChannel`). Approve/Reject decision buttons post back to this same route |
| Pipelines | `/ui/pipelines` | All loaded pipelines with last-run status, run counts, per-pipeline agent badges (read from config), all-time success rate, avg tokens in/out per run, a **TESTING** badge per pipeline (§3c), and **tag** (`?tag=`) / **agent** (`?agent=`) filters; header stat cards and all per-pipeline rollups are scoped to production |
| Pipeline detail | `/ui/pipelines/{name}` | Config summary, tags, stage badge, **Agents card** (every agent used by the pipeline — including verifier agents in critic or independent mode — with its role(s), the step(s) it's used in, and its live-configured model + fallback models fetched from the backend), accuracy feedback summary bar (production only), recent runs (badged, all stages), YAML viewer, and **Run now** button (always runs regardless of stage) |
| Pipeline accuracy | `/ui/pipelines/{name}/feedback` | Accuracy breakdown by pipeline configuration (see §Accuracy feedback) — summary cards and the config-fingerprint comparison are production only; the chronological "every marked run" table shows all stages, badged |
| Steps | `/ui/steps` | Step library — all named steps with executor/agent, tags, pipeline usage, copy-ref button, a **tag filter** (`?tag=`), and a per-pipeline/agent/model breakdown table (runs, success rate, avg tokens) for steps with run history |
| Agents | `/ui/agents` | Unified agent library across all executor backends, with per-agent step success rate, avg duration, avg tokens in/out per step, configured model + fallback models (gateway agents), which pipelines use each agent, and **executor**/**model** filters (`?executor=`/`?model=`, the latter matching either the primary or a fallback model) |
| MCP Tools | `/ui/mcp` | Live MCP tool/server registry — every tool's schema, and each server's running/pid/restart_count, fetched from the P-Ork Gateway's `GET /mcp/tools` and `GET /mcp/servers`. Config-and-schema browsing only; see Insights — MCP for call-usage analytics |
| Schedules | `/ui/schedules` | Active cron schedules with next-run times |
| Insights — Overview | `/ui/insights` | Run/failure/token/accuracy totals, runs by team, and MCP tool-use counts, over a selectable time range (24h/7d/30d/all-time) — production only |
| Insights — Pipelines | `/ui/insights/pipelines` | Per-pipeline run/failure/duration/token totals, top-pipelines table, and a per-pipeline drilldown (status/accuracy breakdown, timeseries, recent runs, and a step/agent/model breakdown table) — production only |
| Insights — Steps | `/ui/insights/steps` | Per-step run/failure/duration/token totals, top-steps table, and a per-step drilldown (status breakdown, timeseries, recent executions, and a pipeline/agent/model breakdown table) — production only |
| Insights — Agents | `/ui/insights/agents` | Per-agent step/success-rate/duration/token totals, top-agents table, and a per-agent drilldown (status breakdown, timeseries, recent executions, and a pipeline/step/model breakdown table) — production only |
| Insights — Models | `/ui/insights/models` | Per-model (provider-qualified — see §Agent Library "Model display") success-rate/duration/token totals, top-models table, and a per-model drilldown (status breakdown, timeseries, recent calls, and a pipeline/step/agent breakdown table) — production only, `executor: gateway` steps only |
| Insights — Providers | `/ui/insights/providers` | Calls/success-rate/duration/token totals grouped by LLM provider (`anthropic`, `openrouter`, `azure`, etc.), top-providers table, and a per-provider drilldown — same drilldown shape as the other Insights pages. Folds in what used to be the standalone `/ui/providers` page (old links redirect here); unlike every other Insights page, this one falls back to a best-effort provider guess from the model string for pre-migration rows with no `provider` value, since the whole point of this page is provider bucketing — production only, `executor: gateway` steps only |
| Insights — MCP | `/ui/insights/mcp` | Tool call usage extracted from the agent trace on `executor: gateway` steps (OpenClaw steps don't expose intermediate events) — calls/errors by tool and by server, a per-tool drilldown showing which pipelines/steps/agents call it, over a selectable time range — production only |
| Insights — Teams | `/ui/insights/teams` | Per-team run/success-rate/duration/token totals, top-teams table, and a per-team drilldown giving a complete picture of what a team uses and where (pipelines used, and a pipeline/step/agent/model breakdown table) plus its token spend, for informed cost decisions. NULL team is bucketed as "Unattributed" — production only |

### Running a pipeline manually

Every pipeline detail page has a **Run now** button. This opens a modal where you can optionally set a summary and paste a full generic webhook payload (JSON). On submit it POSTs to `POST /pipelines/{name}/run` — a separate internal endpoint from the public `/webhook`, so it keeps working regardless of `auth.teams` configuration (no Bearer token required, and the run is unattributed). A banner appears with a link to the new run.

### Run log

Each completed run stores a structured event timeline in `pipeline_runs.logs`. The run detail page shows this as a collapsible log section with timestamped, colour-coded entries (info / warn / error) covering every step start, confidence score, verifier result, skip, escalation, notification, and final status.

### Live tail

While a run is in progress the run detail page shows a **Live tail** panel. It connects via Server-Sent Events (`GET /ui/runs/{id}/stream`) and streams two categories of events in real-time:

**Pipeline log events** — fire for all executor types:
`step_started`, `step_completed`, `step_failed`, `step_skipped`, `step_escalated`, `step_aborted`, `verifier_ran`, `parallel_group_started`, `parallel_group_completed`, `notification_sent`, `run_started`, `run_finished`.

**Agent trace events** — `executor: gateway` steps only. Each LLM call, thinking block, text response, tool call (with arguments), and tool result streams into the live tail as it happens — not as a batch when the step finishes. These are rendered with compact colour-coded formatting: violet for thinking, cyan for tool calls, green/red for tool results. Content is truncated at 200 chars in the live tail; the full content is in the step detail panel's Agent trace section.

Late-connecting clients receive a full history replay of everything that happened before they connected, then transition into the live stream — no events are missed.

When the run finishes the page reloads automatically to show the final state. A 5-second polling fallback (`GET /runs/{id}`) reloads the page if the SSE connection was lost.

### Agent trace

Each step's expanded detail panel includes a collapsible **Agent trace** section showing the complete internal execution trace: LLM call markers, extended thinking blocks, response text, every tool call with arguments, and every tool result. This is available for **`executor: gateway` steps only** — the gateway streams each event back to P-Ork as it fires, which stores the full trace in `pipeline_steps.agent_trace`.

The trace toggle label shows a count of LLM calls and tool calls at a glance (e.g. `3 LLM calls, 12 tool calls`). Tool result content is truncated at 3 000 chars in the stored trace; full content is always available in the gateway's own logs at `DEBUG` level.

The same events that populate this panel also appear in the **live tail** during the step's execution — the detail panel is the persistent post-run record; the live tail is the real-time view.

For `openclaw` steps, `agent_trace` is NULL — OpenClaw does not expose intermediate events to P-Ork.

### Accuracy feedback

After a run completes, any user can mark it with a human judgement of whether the pipeline's outcome was correct. The feedback widget appears at the bottom of every finished run's detail page (hidden for `running` and `interrupted` runs).

**Outcomes:**

| Outcome | When to use |
|---|---|
| `Correct` | The pipeline did what it was supposed to do |
| `Partial` | The pipeline did useful work but didn't fully achieve the goal |
| `Incorrect` | The outcome was wrong or misleading |

An optional notes field lets you record why — useful context when reviewing patterns later. Submitting again overwrites the previous outcome (upsert).

**Where accuracy data surfaces:**

- **Run detail** — the feedback widget; shows current outcome if already marked.
- **Pipeline detail** — a colour-coded correct/partial/incorrect bar with counts, and a "View breakdown →" link.
- **Insights — Overview** — an "Accuracy" stat card showing the % correct of all marked runs in the selected time range.
- **Pipeline accuracy page** (`/ui/pipelines/{name}/feedback`) — the full breakdown:
  - Summary cards (total marked, correct, partial, incorrect with percentages)
  - Overall accuracy distribution bar
  - **Accuracy by configuration table** — runs are grouped by a fingerprint of the exact (step sequence × agents × models) combination. When you change a model, add a step, or swap an agent, the new runs fall into a new group automatically, so you can directly compare accuracy before and after any pipeline change without manually tagging versions.
  - Chronological table of every marked run with its outcome, run status, config fingerprint, and notes.

**Per-step feedback.** In addition to run-level feedback, you can mark an individual step *execution* correct/partial/incorrect. The control appears inside each finished step's expanded detail panel (same collapsible body as the parsed output and agent trace), so marking is optional and sparse — mark only the step(s) you have an opinion on. Fan-out branches (`triage/0`, `triage/1`, ...) are marked independently, since each is its own step execution.

- **Steps Insights** (`/ui/insights/steps`) — the pipeline/agent/model breakdown table has an **Accuracy** column, and the per-step drilldown has an **Accuracy** mini-card, both computed as `correct / total_marked` over the selected time range, production-scoped.
- `pork_step_feedback_total` (see §Metrics) exposes the same counts for Grafana/alerting.

Per-step feedback is currently pure data collection — it does not affect gating or flow control. It's a building block for future work on calibrating trust scores against real outcomes.

### Grounding (shadow mode)

**What it is.** After a step runs, an optional second call — a "grounding judge" — checks how many of the step's *load-bearing claims* (a stated root cause, a metric value, a causal link, a referenced ticket/dashboard id) are actually supported by evidence in that step's own execution trace, rather than just asserted. The result is a **grounding score G ∈ [0,1]** (the fraction of claims that are supported) plus a per-claim breakdown, both persisted as `pipeline_steps.grounding_score` and `pipeline_steps.trust_report`.

**Phase 0 — shadow only.** This is pure observation: G is recorded, never enforced. It never touches `effective_confidence`, the `confidence_threshold` gate, `on_low_confidence`, or any abort/escalate/stop path. The point is to accumulate, on real traffic, a record of how far an agent's self-reported confidence (S) and its actual grounding (G) diverge — near-identical self-reported confidence can hide a well-evidenced conclusion or a confidently-asserted guess, and shadow mode is how you find out which. A later phase may let grounding gate side-effecting steps; that isn't wired up yet.

**Opt-in per step, gateway-only.** Grounding only runs for steps that declare a `grounding:` block, and only for `executor: gateway` steps — only the gateway executor emits the ordered tool-call trace (`agent_trace`) grounding cross-references against. Steps on other executors, or a gateway step whose trace has no tool activity, record `grounding_score = NULL` ("no evidence trail to check"), never `0` ("claims are unsupported"). Grounding is not yet wired into parallel/fan-out branches — sequential steps only.

```yaml
steps:
  - name: investigate
    executor: gateway
    executor_config: { agent: sre-investigation }
    grounding:
      agent: grounding-judge      # gateway agent; must be configured on the gateway
      max_trace_chars: 4000       # optional — see below
```

`grounding.agent` (default `grounding-judge`), `grounding.executor` (default `gateway`), `grounding.executor_config`, `grounding.timeout_seconds` (default 120), and `grounding.max_trace_chars` (default 1500) are the only knobs — there's no threshold or cap here; that's Phase 1.

**`max_trace_chars` — truncation, not a hallucination.** The transcript handed to the judge truncates each `tool_result`/agent-text event at `max_trace_chars`, with a trailing `…`. A claim whose supporting evidence lands past that cutoff is genuinely invisible to the judge — that shows up looking exactly like "the primary agent is making things up," when actually the trace transcript just didn't include the relevant part. Steps whose tools return long content (a full document read, a large query result) should raise this; the default (1500) is tuned for cheap, short evidence, not long reads.

**There are two independent truncation points, and raising this one alone may not be enough.** `grounding.max_trace_chars` only controls how much of the trace P-Ork *already has* it hands to the judge. The Gateway itself caps each `tool_result` event's content **before P-Ork ever receives it** (`limits.trace_tool_result_max_chars`, default 3000, in the Gateway's own config) — if a tool result was truncated there, no `grounding.max_trace_chars` setting on the P-Ork side can recover the missing part, because it was never sent. Use the primary step's `executor_config.trace_max_chars` (§8, `gateway` executor) to raise the Gateway-side cap for that step, *then* raise `grounding.max_trace_chars` here to match — otherwise you've just moved the same cutoff from the judge's transcript-formatting step to the Gateway's own capture step. Either cutoff being too low produces the identical symptom (a claim that looks unsupported but genuinely wasn't), so if grounding keeps flagging claims you believe are backed by real evidence, check both.

**Soft failure.** Like the verifier, a grounding call that errors or times out logs a warning and records `grounding_score = NULL` with the error captured in the report — it never breaks or delays the step. When the failure is specifically the judge's own output not parsing as JSON — most commonly because a step with many load-bearing claims produced a per-claim evidence list long enough to hit the judge agent's output-length ceiling mid-generation — the report's `raw_output` field carries the judge's full, untruncated text (not just the 500-character snippet in the log/exception message), so a reviewer can tell "genuinely truncated" apart from "malformed from the start."

**The `grounding-judge` agent contract.** Grounding calls a gateway agent (configured on the **P-Ork Gateway**, not in this repo) whose only job is a constrained cross-reference — it cannot browse or add outside knowledge. It receives:

1. the original task given to the primary agent (its rendered `prompt_template` — the same thing a `critic`-mode verifier sees, see §6),
2. the primary agent's structured output (`summary`, `next_step_context`, `reasoning`), and
3. a formatted transcript of the primary agent's tool calls and results.

The judge's prompt is explicit that (1) is *given, trusted input* — a claim that merely restates a fact already present in the original task (alert severity, service name, environment, summary, ...) needs no trace evidence, because the agent was told it, not asked to discover it. Only claims that go beyond the given input (a root cause, a specific metric value, a causal link, a ticket/dashboard id it created or looked up) need a supporting tool result. Without (1), the judge has no way to tell "restates the input" apart from "claims something it needed to discover," and will mark plain input facts as unsupported — a false "unsupported" verdict, not a real evidence gap.

...and must return an `LLMOutput`-shaped JSON object where:

- **`confidence`** carries **G** itself — the fraction of load-bearing claims supported by trace evidence, in `[0,1]`. (This reuses the existing `confidence` field as the transport so the ordinary `GatewayExecutor` parse path works unchanged; it is not the judge's confidence in itself.)
- **`summary`** — one sentence, e.g. *"3 of 4 load-bearing claims are supported by tool results; the root-cause claim is not."*
- **`reasoning.claims`** — a list of `{ "claim": str, "supported": bool, "evidence": str }`, one per load-bearing claim identified.
- **`next_step_context`** — unused, `""`.

The persisted `trust_report.grounding` also records **which** agent/model actually judged this run — `agent` (from `grounding.agent` config), plus `model`/`provider` read straight from the judge's own response metadata (both `null` when grounding didn't compute, e.g. an error or no trace) — and, gateway executor only, `prompt`: the judge's own fully-rendered prompt (the `_GROUNDING_PROMPT_TEMPLATE` with the original task, primary's response, and trace all substituted in), so a reviewer can see exactly what the judge was shown, not just what it concluded.

**Where it surfaces.** Each grounded step's expanded detail panel shows a **"Trust (shadow)"** widget: self-report (S) vs. verifier (V, if any, with its own agent/model shown alongside) vs. grounding (G, with its judging agent/model shown alongside), a divergence flag when `|G − S| ≥ 0.2`, the judge's own one-sentence verdict (`trust_report.grounding.summary`, e.g. *"3 of 4 load-bearing claims are supported by tool results; the root-cause claim is not."*), and the per-claim ✓/✗ breakdown with evidence. `pork_step_grounding_score` (see §Metrics) exposes the score distribution for Grafana.

**The Trust panel isn't just for grounded steps.** Any step with a verifier — even with no grounding, deterministic checks, or calibration configured — gets a "Trust (shadow)" panel too, so how S and V combined is never invisible. A **"How was this calculated?"** button reveals a plain-language, numbers-first walkthrough of that specific run: self-report → verifier combine → calibration (if enforced) → grounding (if configured) → deterministic checks (if declared) → the final figure and what it decided. No config keys, just what actually happened on this run — built from the same `trust_report` data, not a re-derivation from the pipeline's current config.

Two things worth knowing about how honest this walkthrough can be:
- **`V_veto_floor`** is persisted in `trust_report.signals` (alongside `V_mode`) specifically so the narrative can say *why* a verifier's lower score didn't change anything ("this step only lowers confidence below X%") instead of just asserting it did nothing. Rows from before this field existed fall back to vaguer wording rather than inventing a number.
- **`grounding.enforce`** is persisted per-run for the same reason — grounding computes and reports a score even in pure shadow mode, so its presence alone can't tell you whether it actually gated a given historical run. Rows predating this field say so explicitly ("isn't recorded for this older run") rather than guessing either way.

A **Prompt** disclosure (collapsed by default) now sits above each gateway step's parsed output, showing the actual rendered prompt the agent received — necessary for marking step accuracy honestly, since a grounding claim like "the agent didn't check X" might mean the prompt never asked it to. The verifier pane and the grounding claims section each get their own matching disclosure (`verifier_prompt`, `trust_report.grounding.prompt`) — all three (primary, verifier, grounding judge) are computed from the same executor-level stash (`GatewayExecutor.execute` writes it onto `raw_response["prompt"]` for every call it makes), so seeing one doesn't mean the others are guaranteed present — each is independently `null` if that particular call used a non-gateway executor or predates this fix.

A **Step configuration** disclosure sits alongside "How was this calculated?" in the Trust panel — a plain-language summary of what this step is *set up* to do: confidence threshold and `on_low_confidence`; verifier mode and combination strategy, naming a `veto` floor by its actual number rather than leaving "why didn't this change anything" unexplained; grounding's enforce state; declared deterministic checks by name; calibration's `n_min`/`on_uncalibrated`. Built from the same `trust_report` data as the narrative (`_step_config_summary` in `ui.py`), not a live read of the pipeline's current config.

### Deterministic checks & enforced grounding (Phase 1)

Phase 0 (above) only ever records G — it never gates. Phase 1 adds two **opt-in per
step** mechanisms that can actually change a step's outcome: deterministic checks (D)
and grounding-as-a-gate (`grounding.enforce: true`). A step that declares neither
behaves byte-for-byte as it did before this feature existed — the legacy
`effective_confidence < confidence_threshold` comparison is a permanent, first-class
gate policy, not a deprecated code path.

**The gate formula.**

```
combined_trust = effective_confidence                    # today's S after the verifier's veto, unchanged
if grounding.enforce and G is not None:
    combined_trust = min(combined_trust, G)               # G can only ever pull trust down
if deterministic_checks declared and not all_passed:
    combined_trust = 0.0                                  # a failed check is dispositive

# the SAME comparison, SAME threshold, SAME on_low_confidence action as today:
if combined_trust < step.confidence_threshold:
    <on_low_confidence action>
```

There is deliberately no second, separately-tuned threshold for grounding (no
`require_grounding: 0.7` or similar) — `grounding.enforce` reuses the step's *existing*
`confidence_threshold`. A null G (grounding wasn't computed this run — no trace, or the
grounding call itself soft-failed) never triggers the cap; `combined_trust` is simply
left as whatever it already was, consistent with Phase 0's "no evidence trail to check"
≠ "unsupported" rule.

**Deterministic checks (D).** A step can declare a list of `deterministic_checks:`,
each a pass/fail assertion the *runner* evaluates directly — no LLM involved. Three
check types:

- **`shell`** — run a command; evaluate its output.
  ```yaml
  deterministic_checks:
    - type: shell
      name: still_breaching
      run: "curl -s 'http://prometheus/api/v1/query?query=rate(http_5xx[5m])' | jq '.data.result[0].value[1]'"
      expect: "result | float > 0.02"     # bare Jinja2 bool expr — same convention as `when:`,
                                          # NOT wrapped in {{ }}. Sees `result` (stdout, stripped)
                                          # and `exit_code`, plus the normal step context.
      timeout_seconds: 30                # default
  ```
- **`webhook`** — call a URL; evaluate the response. Same shape as the existing
  `on_failure.webhook` config (`url`/`method`/`headers`/`payload`) — deliberately does
  **not** call `raise_for_status`, since a check might legitimately expect a non-2xx
  status (e.g. 404 = "does not exist").
  ```yaml
  deterministic_checks:
    - type: webhook
      name: dashboard_resolves
      url: "https://grafana.example.com/api/dashboards/uid/{{ steps.investigate.dashboard_uid }}"
      method: GET
      expect: "response.status_code == 200"   # sees `response` = {status_code, body}
  ```
- **`human`** — ask a person to approve/reject, reusing the *existing* human-approval
  subsystem (same channels, same per-team routing, same testing-stage behaviour as
  `executor: human`).
  ```yaml
  deterministic_checks:
    - type: human
      name: sre_signoff
      message: "Auto-remediate {{ steps.investigate.summary }}? Approve to proceed."
      timeout_seconds: 300               # default
  ```

**Fail-closed, universally.** A check that cannot be evaluated — a shell command
errors or times out, a webhook is unreachable, a human approval times out — is
recorded as **failed**, never silently skipped. D is meant to be the strongest,
most trustworthy signal in the trust vector, so an unanswerable check must not
quietly vanish from the computation. (This is a deliberate divergence from
grounding's soft-fail philosophy above — grounding failing soft just means "less
signal"; a deterministic check failing soft would mean "we lost the ability to catch
a real problem.")

**Stage behaviour differs by check type.** `shell` and `webhook` checks are semantically
*queries*, not outbound notifications, so — unlike `on_failure.webhook` — they are
**not** muted by `stage: testing`; muting them would make it impossible to test check
logic in a testing pipeline. `human`-type checks **do** inherit `executor: human`'s
existing testing-stage behaviour (external channel not sent; the decision is made via
P-Ork's own `/ui/approvals`; a timeout auto-approves).

**Unsandboxed by design.** A `shell` check runs with the full environment and
permissions of the P-Ork process — there is no sandboxing or resource-limiting. This is
a deliberate choice for a single-operator, self-hosted deployment where the operator
already fully controls pipeline YAML and already has executors capable of far more (MCP
tools, OpenClaw). Revisit this if the deployment model ever becomes multi-tenant.

**Grounding as a gate.** Add `enforce: true` to an existing `grounding:` block:

```yaml
steps:
  - name: investigate
    executor: gateway
    executor_config: { agent: sre-investigation }
    confidence_threshold: 0.75
    grounding:
      agent: grounding-judge
      enforce: true                      # G now participates as a ceiling on combined_trust
    deterministic_checks:
      - type: shell
        name: still_breaching
        run: "curl -s '...' | jq '.data.result[0].value[1]'"
        expect: "result | float > 0.02"
```

**Where it surfaces.** The run-detail Trust panel header now reads **"Trust
(enforced)"** instead of "(shadow)" for any step where grounding or a deterministic
check actually participated in the gate, with a `Combined trust` figure alongside S/V/G,
a `Checks (D)` PASS/FAIL chip, and a per-check ✓/✗ list (name, type, detail).
`pork_step_deterministic_check_total` (see §Metrics) exposes check outcomes for
Grafana.

See `samples/pipelines/trust-vector-remediation.yaml` for a complete worked example of
enforced grounding and a deterministic check gating a side-effecting step.

### Calibration (Phase 3)

Every step execution's `effective_confidence` has been persisted since Phase 0, and
per-step/per-run human feedback (`step_feedback`/`run_feedback`) plus deterministic-check
failures (Phase 1) have been accumulating. Phase 3 finally checks whether the *number*
means what it claims: does a step that reports 0.75 confidence in a specific
`(step × agent × model × provider)` configuration actually turn out correct roughly 75%
of the time?

**Bucketing.** Marked step-executions are grouped by `(step_name, agent, model,
provider)` — a `library step` (see §Adding a Library Step) used across five pipelines
feeds **one** bucket instead of five, and changing one step's model resets only that
step's bucket. Fan-out branches (`step_name` like `triage/0`, `triage/1`) collapse into
their parent step's bucket rather than one bucket per branch index.

**Label precedence, per step-execution:**
1. **Human** — a resolved `StepFeedback` row for that step execution (`correct → 1.0`,
   `partial → 0.5`, `incorrect → 0.0`) — authoritative when present.
2. **Deterministic (D)** — `pipeline_steps.deterministic_passed == False` labels the
   step `0.0`, for free, at scale. A *passing* check is **not** used as a positive
   label on its own — only failure is a strong-enough automated signal.
3. **Run-level fallback** — the enclosing run's `RunFeedback.outcome`, used only when
   neither of the above exists for that step execution.
4. Otherwise the step-execution is **excluded** entirely — not counted as a 0, not
   counted toward `N`.

**Binning, not curve-fitting.** Rather than isotonic/Platt regression (which would pull
in `scipy`/`sklearn`, a dependency this service otherwise has zero of), calibration uses
simple fixed-width bins — default width 0.1 (10 bins across `[0, 1]`) — and reports each
bin's sample count and mean label. This is directly interpretable and matches the exact
language calibration recommendations use: *"runs scoring ~70% in this configuration are
only 50% correct (40 runs)."* A bin needs `n_min` (default 20) marked outcomes before
it's considered **validated**; nothing computed from an unvalidated bin is used to gate
anything.

**Advisory by default, no opt-in required.** `/ui/insights/steps` shows every bucket's
calibration bins and, for any validated bin whose predicted score and observed accuracy
diverge by 15 points or more, a recommendation string — with no `calibration:` config on
any step. Nothing here changes a run's outcome; it's a report the human can inspect and act
on, exactly the same "tool informs, human governs" posture as the accuracy pages already
have.

**Enforcing (opt-in per step, never silent).** A step can opt its *gate* into using the
bucket's empirical accuracy instead of the raw self-report/verifier number:

```yaml
- name: investigate
  executor: gateway
  executor_config: { agent: sre-investigation }
  confidence_threshold: 0.75
  calibration:
    enforce: true
    on_uncalibrated: proceed   # or "escalate" — see below
```

When enforced and the step's bucket/bin is validated, `combined_trust` is **replaced**
with the bin's `mean_label` before grounding's `min()` and deterministic checks' force-zero
apply on top — the *same* `confidence_threshold` then decides `on_low_confidence`, no new
threshold config. The `TrustReport`'s `calibration` block always shows the arithmetic:
raw score, calibrated score, bin, `n`/`n_min`, so a calibrated escalation is never a
mysterious abort.

When the bucket/bin has **not** yet accumulated `n_min` marked outcomes,
`on_uncalibrated` decides the posture:
- `proceed` (default) — `combined_trust` is left as the raw `effective_confidence`,
  unchanged; the run behaves exactly as it would with no `calibration:` block. The
  `TrustReport` still records "not yet validated, N=x/N_min" for transparency.
- `escalate` — forces `combined_trust = 0.0`, driving the step's *existing*
  `on_low_confidence` action. An explicit "no track record → a human checks" policy for
  high-blast-radius steps; not imposed as a universal default.

A step with **no** `calibration:` block is unaffected by any of this — same posture as
Phase 1's core invariant.

**No new column, no new metric.** Calibration is computed fresh from the existing
`pipeline_steps`/`step_feedback`/`run_feedback` tables on every request, the same way the
Insights pages already recompute their rollups — there is no persisted calibration curve
to migrate or invalidate.

See `samples/pipelines/trust-vector-remediation.yaml` for a complete worked example
combining `critic`/`independent` verifier modes, enforced grounding, deterministic
checks, and calibration into a single trust-vector gate on a side-effecting step.

### Agent Library

The `/ui/agents` page provides a unified library of agents across all configured executor backends. Agents are fetched live from each backend and merged into a single list with executor badges.

Agents are uniquely identified by `executor:name` — e.g. `openclaw:sre-investigation` and `gateway:sre-investigation` are treated as distinct agents. This prefix is stored in `pipeline_steps.agent` so run history, success rates, and model usage are attributed correctly per backend.

| Executor | Agent list | Agent files |
|---|---|---|
| `openclaw` | OpenClaw Gateway WS — `agents.list` RPC | `agents.files.get` RPC — `SOUL.md`, `TOOLS.md`, `IDENTITY.md` tabs |
| `gateway` | P-Ork Gateway REST — `GET /agents` | `GET /agents/{name}/soul` (Soul tab) · `GET /agents/{name}/agent` (Config tab — raw `agent.yaml`) |

Both backends are queried concurrently. If one is unreachable, the other's agents still show with a warning banner. If both fail, stub entries from DB run history are surfaced.

The **Config** tab on a gateway agent detail page shows the raw `agent.yaml` — model, `max_tokens`, and the list of MCP tool names the agent has access to.

**Overview tab** — a per-model breakdown table (runs, succeeded, failed, success rate, avg duration, avg tokens in/out, last run), two "usage over time" line charts (runs and tokens, both split by model), and a **recent activity** list of the last 15 steps this agent ran across any pipeline — each row links to its pipeline and its run detail page.

**Steps tab** — which pipeline steps this agent executes, broken down by pipeline and model (runs, success rate, avg tokens, last run). The same step name can be wired to a different model in different pipelines, so pipeline is a first-class column here rather than folded away.

All of the above is scoped to `stage=production` runs (§3c), same as every other rollup surface.

**Model display and the `provider` column:** wherever a model name is shown alongside run history (this page, `/ui/steps`, and the Insights pages), it's prefixed with its provider when the DB actually recorded one — e.g. `anthropic/claude-sonnet-5`, `openrouter/deepseek/deepseek-v4-pro`. `pipeline_steps.provider` is only populated for `executor: gateway` steps (from the Gateway's `agentMeta.provider`); other executors, or steps run on an older Gateway build that predates this field, leave it NULL and the bare model name is shown as-is — the UI does not guess a provider it has no evidence for, since a wrong guess is worse than no answer.

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
  # url: postgresql+asyncpg://user:password@localhost:5432/pork   # production — see §Database

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

dedup:
  enabled: true                        # omit this block (or set false) to disable dedup entirely
  window_seconds: 300                  # see §3a — overridable per-pipeline via trigger.dedup

concurrency:
  max_runs: 10                         # maximum simultaneous pipeline executions (default: 10).
                                       # POST /webhook returns 429 when at capacity.
                                       # GET /health exposes active_runs / max_concurrent_runs.

auth:
  teams:                               # per-team tokens — see §3b. Each team's token resolves
                                       # the `team` attribution on every run it authenticates.
    - name: payments
      token: ${PORK_WEBHOOK_TOKEN_PAYMENTS}
    - name: platform
      token: ${PORK_WEBHOOK_TOKEN_PLATFORM}
  # token: ${PORK_WEBHOOK_TOKEN}       # legacy single-token form — still supported if `teams`
                                       # is omitted; every run's team is then unattributed (None).
                                       # If both `teams` and `token` are set, `teams` wins.
                                       # Omit this whole block (or leave empty) to run unauthenticated.
                                       # Alertmanager sends its token via http_config.authorization.credentials —
                                       # route different teams' alerts to different receivers with different tokens.

observability:
  otel:
    enabled: false                     # omit this block (or set false) to disable tracing entirely
    exporter: otlp                     # otlp | console — see §15b
    endpoint: http://localhost:4318/v1/traces
    service_name: pork-service

calibration:                           # omit this block entirely for the defaults shown below
  n_min: 20                            # marked outcomes required before a bucket is "validated"
  bin_width: 0.1                       # must evenly divide 1.0 — see §Calibration (Phase 3)
  cache_ttl_seconds: 300                # how long the in-process bucket cache is reused before refetching
```

`${ENV_VAR}` placeholders are resolved at startup. Unresolved placeholders become `""`.

---

## Database

The ORM layer (SQLAlchemy async) is dialect-agnostic — switching backends is a
`database.url` change only, no code changes. Two supported backends:

| Backend | URL | When to use |
|---|---|---|
| SQLite (`aiosqlite`) | `sqlite+aiosqlite:///./runs.db` | Local dev, zero infrastructure, single process |
| PostgreSQL (`asyncpg`) | `postgresql+asyncpg://user:pass@host:5432/dbname` | Production — concurrent writers, real backup/replication story |

**Setup (Postgres):**
```bash
createdb pork
# config.yaml:
database:
  url: postgresql+asyncpg://user:password@localhost:5432/pork
```

Tables and migrations run automatically on startup (`create_tables()` in
`service/src/db/database.py`) — same as SQLite, no Alembic or manual migration step.

**Migration mechanism:** new columns are added via a small `_COLUMN_MIGRATIONS` list run
on every boot. Postgres uses native `ADD COLUMN IF NOT EXISTS`; SQLite has no such syntax
(confirmed unsupported as of SQLite 3.51), so it attempts the plain `ADD COLUMN` and
ignores `OperationalError` (logged at `DEBUG`) when the column is already there. Index
creation (`_INDEX_MIGRATIONS`, including `CREATE UNIQUE INDEX IF NOT EXISTS`) is portable
across both dialects as-is.

**Dedup race hardening:** a partial unique index —
`UNIQUE (pipeline_name, fingerprint) WHERE status = 'running'` — closes the TOCTOU race
described in §3a at the database layer, not just the application-level pre-check. See
§3a "Race safety" for the full mechanism.

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

### Running Tests

Unit tests cover the pure-function logic: pipeline resolver matching, verifier
confidence-combination and parallel join strategies, step-library `use:`
deep-merge, and webhook dedup fingerprinting/settings.

```bash
# from repo root — installs requirements.txt plus pytest/pytest-asyncio
pip install -r requirements-dev.txt

cd service
pytest
```

By default the suite runs against SQLite — a fresh per-test temp-file DB, no setup
required. To run the same tests against Postgres instead (exercising the
Postgres-only migration branch in `create_tables()`), point `PORK_TEST_DATABASE_URL`
at a throwaway database:

```bash
createdb pork_test
PORK_TEST_DATABASE_URL=postgresql+asyncpg://localhost:5432/pork_test pytest
```

Isolation on Postgres is via a `DROP SCHEMA public CASCADE; CREATE SCHEMA public;`
reset around each test (see `tests/conftest.py`'s `db` fixture) rather than a fresh
file, since there's no per-test temp file on a shared server. CI
(`.github/workflows/tests.yml`) runs both backends on every push.

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
| SQLAlchemy async ORM, dialect swap via config only | SQLite for zero-infra local dev, Postgres for production — same code path, no Alembic |
| DB-level partial unique index for in-flight dedup | The application-level pre-check (§3a) narrows but cannot close a TOCTOU race on its own — the DB constraint is the actual correctness guarantee, the pre-check just avoids the round-trip in the common case |
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
| In-flight dedup always wins regardless of `window_seconds` | Prevents two overlapping triage/remediation runs for the same alert — the dangerous case — independent of how the recency window is tuned |
| Alert `status` (firing/resolved) folded into the Alertmanager fingerprint | A resolve notification must never be suppressed as a duplicate of the firing run it's closing out |
| `trigger.dedup` as a sibling of `trigger.match`, not inside it | Keeps `match` purely about resolver conditions (`_matches()` iterates every key as a field/label comparison) — dedup is an execution-policy concern, not a matching condition |
| `stage` defaults to `testing`, not `production` | New/WIP pipelines are safe by default — nothing pages a real human or counts toward metrics until someone deliberately promotes it (§3c) |
| `pipeline_runs.stage` persisted per-run, not joined from the live config | Captures stage-at-run-time — promoting a pipeline to `production` never retroactively moves prior testing runs into production metrics |
| `stage` gates four outbound paths individually rather than one flag | `notifications:`, `executor: notify`, `on_failure.webhook`, and `executor: human` are genuinely independent side-effect sources — muting the pipeline as a whole would still need per-path logic, so it's implemented where each one fires |
| No UI toggle for `stage` | Consistent with `tags`/`version` — pipeline behaviour stays entirely git-controlled config, reviewable in a diff |

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
Database: SQLite on a PersistentVolumeClaim for a single-replica deployment, or PostgreSQL
(in-cluster or managed) for multi-replica — see §Database. PostgreSQL is the only option
once you run more than one replica, since SQLite has no concept of a network connection.
Secrets (tokens) via Kubernetes Secrets as environment variables.
Log files written to a PersistentVolumeClaim or redirected to stdout by omitting `logging.dir`.
