# Implementation Plan — Pipeline `stage: testing | production` gating

> Drop-in task spec. Everything needed to implement the feature is here, with file
> anchors against the current tree. Work top-to-bottom; each numbered section is a
> commit. Run `cd service && pytest` after each. Default to **testing** everywhere so
> nothing can page a real human before it's ready.

## Goal

Add a single-valued `stage` field to a pipeline. `stage: testing` (the default) makes a
pipeline fully executable and fully observable **inside P-Ork's own UI**, but inert to the
outside world and excluded from production metrics. `stage: production` is today's behaviour.

Promotion is a one-line YAML diff, reviewed in git and applied with `POST /reload`/SIGHUP.
There is **no UI toggle** — config stays git-controlled, consistent with `tags`/`version`.

## Locked design decisions

- **Pipeline-level only.** No step-level `stage`. Every gated behaviour is pipeline-scoped.
- **Default `testing`.** New/WIP pipelines are safe by default.
- **Persist `stage` on the run row** (not a registry join at query time). Captures
  stage-at-run-time, so promoting a pipeline never retroactively moves old runs between
  metric buckets, and the exclusion is one `WHERE stage='production'` predicate everywhere,
  including `/metrics`.
- **`testing` mutes every outbound side-effect, not just `notifications:` blocks.** There
  are four independent outbound paths and all four must be gated (see §3).
- **Human-in-the-loop in testing:** skip the external channel send, but keep the approval
  in P-Ork's own UI (`/ui/approvals` + run banner) so a real Approve/Reject decision can
  still be made; timeout defaults to **approved** instead of failing. No new
  `on_approve`/`on_reject` fields — rejection already flows through `confidence=0.0` →
  `on_low_confidence` + downstream `when:` conditions.
- **UI:** no separate pages. Aggregate/rollup surfaces exclude testing by default; browse
  surfaces show testing rows with a badge, plus one new `stage` filter on the runs list.

## Central mechanism

Two things carry `stage` through the system:

1. `PipelineConfig.stage` — read by the runner and `main.py` (config is always in hand there).
2. A persisted `pipeline_runs.stage` column — read by every UI/metrics rollup query.

The runner injects a `_testing` boolean into each step's Jinja2/executor context so
executors (`notify`, `human`) can honour it without the runner special-casing executor names.

---

## 1. Model field — `service/src/models/pipeline.py`

Add to `PipelineConfig` (near `version`/`tags`, ~line 203-214):

```python
stage: Literal["testing", "production"] = "testing"
```

`Literal` is already imported. No validator needed — pydantic rejects any other value.

**Test:** `service/tests/unit/test_stage.py` — a pipeline dict with no `stage` loads as
`"testing"`; explicit `production`/`testing` load correctly; a bogus value raises
`ValidationError`.

---

## 2. Persist stage on the run row

### `service/src/db/models.py` — `PipelineRun` (after `team`, ~line 28)

```python
stage: Mapped[str] = mapped_column(String, nullable=False, default="production", index=True)
```

Default `production` at the DB layer is deliberate: pre-existing rows created before this
column migrate to `production` so historical metrics are unchanged. The *application*
default (unmarked pipeline == testing) is enforced by the model field in §1; the runner
always writes an explicit value (§3.1), so the column default only ever applies to old rows.

### `service/src/db/database.py` — migration (`_COLUMN_MIGRATIONS`, ~line 55)

Add:

```python
("pipeline_runs", "stage", "TEXT DEFAULT 'production'"),
```

Add an index entry to `_INDEX_MIGRATIONS` (~line 68) for the filter/rollup predicate:

```python
"CREATE INDEX IF NOT EXISTS ix_pipeline_runs_stage ON pipeline_runs (stage)",
```

(The `index=True` on the column covers ORM-created tables; the explicit statement covers
already-existing DBs going through migration — both portable across SQLite/Postgres.)

---

## 3. Runner — `service/src/pipeline/runner.py`

### 3.1 Write stage on run creation (`_db_create_run`, ~line 1386)

Add to the `PipelineRun(...)` insert:

```python
stage=pipeline.stage,
```

`pipeline` is already a parameter of `_db_create_run`.

### 3.2 Inject `_testing` into every step context

Cleanest single point: `build_context()` in `service/src/pipeline/context.py` (it already
receives `pipeline` as its first arg, and is called for sequential steps, parallel branches,
fan-out branches, and verifiers — so one edit covers all executor paths). Add:

```python
ctx["_testing"] = pipeline.stage == "testing"
```

(Key is underscore-prefixed so it never collides with a user template var. If `build_context`
builds its dict incrementally, set it alongside the other framework keys.)

### 3.3 Gate pipeline notifications (`_dispatch_notification`, ~line 1607)

When testing, force every notification to the `log` channel and record a distinct run-log
event so the run timeline shows exactly what *would* have gone out:

```python
async def _dispatch_notification(self, pipeline, action, context, run_log):
    notifications = pipeline.notifications.get(action)
    if not notifications:
        logger.debug("No notification config for action '%s' — skipping", action)
        return
    testing = pipeline.stage == "testing"
    for notification in notifications:
        channel = "log" if testing else notification.channel
        notifier = self._notifiers.get(channel)
        if not notifier:
            logger.warning("No notifier registered for channel '%s' — skipping", channel)
            continue
        await notifier.send(notification if not testing else _as_log(notification), context)
        _log_event(
            run_log, "info",
            "notification_suppressed_testing" if testing else "notification_sent",
            (f"[testing] Notification routed to log: {action} → would have been "
             f"{notification.channel}") if testing else
            f"Notification sent: {action} → {notification.channel}",
            action=action, channel=notification.channel,
        )
```

`_as_log(notification)` is a tiny helper returning a copy with `channel="log"` so
`LogNotifier` renders the same template. (If `LogNotifier.send` ignores `notification.channel`
and only uses `template`, you can pass `notification` unchanged and drop the helper — verify
against `service/src/notifications/log.py`.)

### 3.4 Gate the step-failure webhook (`_fire_step_failure_webhook`, ~line 1548; call site ~line 349)

At the top of `_fire_step_failure_webhook`, short-circuit in testing:

```python
if pipeline.stage == "testing":
    _log_event(run_log, "info", "step_failure_webhook_suppressed_testing",
               f"[testing] Step-failure webhook suppressed: {step.name} → {step.on_failure.webhook.url}",
               step=step.name)
    return
```

`pipeline` is already passed to this method.

### 3.5 `executor: notify` — see §4 (gated in the executor via `_testing`).
### 3.6 `executor: human` — see §5.

---

## 4. Notify executor — `service/src/executors/notify.py`

Honour `_testing` from context — skip the real `httpx` call, log the rendered body instead,
return a success `LLMOutput` so the step still "completes" and downstream steps run:

At the top of `execute()` (after resolving `url`/`body`, before the `httpx.AsyncClient` block ~line 67):

```python
if context.get("_testing"):
    logger.info("[testing] NotifyExecutor suppressed: step=%s url=%s body=%s",
                step.name, url, body)
    return LLMOutput(
        confidence=1.0,
        summary=f"[testing] notify suppressed (would POST to {url})",
        next_step_context="Notification suppressed in testing stage.",
        raw_response={"suppressed_testing": True, "url": url},
    )
```

**Test** (`test_notify_executor.py`, extend existing): with `context={"_testing": True}` no
HTTP call is made (mock/assert `httpx` unused) and the returned output carries
`suppressed_testing`; with `_testing` absent/False the existing behaviour is unchanged.

---

## 5. Human executor — `service/src/executors/human.py`

Two changes inside `execute()`.

**5.1 Skip external send but keep the in-UI approval.** The pending-approval registration
(`_pending_approvals`/`_pending_meta`, ~line 329-337) must still happen so `/ui/approvals`
and the run-detail banner show it. Only the `channel.send()` call (~line 340) is gated:

```python
testing = context.get("_testing", False)
...
# (register _pending_approvals / _pending_meta exactly as today)
try:
    if not testing:
        await channel.send(message_text, token)
    else:
        logger.info("[testing] Human approval NOT sent externally; awaiting UI decision: "
                    "step=%s token=%s", step.name, token)
    approved = await asyncio.wait_for(asyncio.shield(future), timeout=timeout)
except asyncio.TimeoutError:
    if testing:
        logger.info("[testing] Human approval timed out → auto-approving: step=%s token=%s",
                    step.name, token)
        approved = True          # testing default: assume acceptance
    else:
        logger.warning("Human approval timed out: step=%s token=%s", step.name, token)
        raise RuntimeError(f"Human approval timed out after {timeout}s")
finally:
    _pending_approvals.pop(token, None)
    _pending_meta.pop(token, None)
```

Note `channel = _build_channel(channel_cfg)` currently runs before the send and can raise if
no channel is configured (~line 318-325). In testing we don't need a channel — move channel
resolution/build inside `if not testing:` so a testing pipeline with no approval channel
configured still works (it's UI-only). Keep the existing `RuntimeError` for the production
no-channel case.

The rest is unchanged: a real Reject in the UI still resolves the future to `False` →
`confidence=0.0` → `on_low_confidence` + downstream `when:` fire for real. This is why no
`on_approve`/`on_reject` fields are needed.

**Test** (`test_human_executor.py`): with `_testing=True` and no channel configured, no send
happens and the approval is registered in `_pending_meta`; resolving the token `False` returns
`confidence=0.0`; a timeout returns `confidence=1.0` (auto-approve). With `_testing=False`,
existing behaviour (send + timeout raises) is unchanged.

---

## 6. Trigger gating — `service/src/main.py`

A testing pipeline should not fire from real ingestion traffic, but must still be runnable
manually. Thread an `allow_testing` flag through `_trigger_run` (~line 646).

**6.1 `_trigger_run`** — add param and check right after `resolve_pipeline` (~line 651):

```python
async def _trigger_run(normalised: NormalisedContext, allow_testing: bool = False) -> JSONResponse:
    pipeline = resolve_pipeline(normalised, _pipelines)
    if not pipeline:
        raise HTTPException(status_code=422, detail=...)  # unchanged

    if pipeline.stage == "testing" and not allow_testing:
        logger.info("Testing pipeline '%s' not triggered from %s (allow_testing not set)",
                    pipeline.name, normalised.source)
        return JSONResponse(status_code=202, content={
            "status": "skipped_testing",
            "pipeline": pipeline.name,
            "reason": "Pipeline is stage=testing; pass allow_testing=true to run it from this source.",
        })
    # ...dedup / overload / create_task unchanged
```

**6.2 `POST /webhook`** (~line 579) — accept an opt-in override and pass it through. Add a
query param `allow_testing: bool = False` to the handler signature and forward it:
`return await _trigger_run(normalised, allow_testing=allow_testing)`. (Alertmanager can set
`?allow_testing=true` on the receiver URL when deliberately exercising a testing pipeline.)

**6.3 `POST /pipelines/{name}/run`** (the UI "Run now" button, ~line 620) and
**`POST /runs/{run_id}/rerun`** (~line 888): these are deliberate manual actions — pass
`allow_testing=True` so a testing pipeline is always runnable from the UI. For rerun, the run
is created directly via `_run_pipeline` (not `_trigger_run`), so no gating applies there
already — just confirm it isn't routed through the new check.

**Test** (`tests/test_stage_gating.py`, integration-style): a `stage=testing` pipeline
returns `skipped_testing` from `/webhook` without the flag, `accepted` with
`?allow_testing=true`, and `accepted` from `/pipelines/{name}/run`. A `stage=production`
pipeline is unaffected.

---

## 7. Metrics — `service/src/metrics.py`

Exclude testing runs from the all-time aggregates in `fetch_metrics_data` (~line 44). Every
query that starts from `PipelineRun` (or joins it) gets `.where(PipelineRun.stage == "production")`:

- `run_counts` (~line 47)
- `in_progress` (~line 53)
- the step/executor joins that go through `PipelineRun` (~line 86-117)
- `RunFeedback` rollup (~line 122) — join `PipelineRun` and filter, or filter on a persisted
  stage if you also denormalise it onto feedback (not required; the join is fine).

Step-only aggregates that never touch `PipelineRun` (pure `PipelineStep` scans, ~line 57-78)
can stay as-is, or join+filter for exactness — recommend join+filter so a testing pipeline's
step durations don't skew `pork_pipeline_step_duration_seconds`.

> Alternative considered: add a `stage` **label** to the Prometheus metrics instead of
> filtering. Rejected for the default because the feature's contract is "testing doesn't
> count." If you later want testing visibility in Grafana, add the label as a follow-up.

**Test:** extend metrics tests — a testing run contributes nothing to `run_counts` /
`in_progress`.

---

## 8. UI — `service/src/ui.py` + templates

### 8.1 Exclude testing from aggregate/rollup surfaces

Add `.where(PipelineRun.stage == "production")` to the **rollup** queries only. Leave
per-run detail and browse lists showing everything.

- `ui_runs` (~line 543) — the **stat-card** aggregates (`status_counts`, `feedback_counts`,
  `token_total`; ~line 583-618) get the predicate. **Do not** filter the actual run list
  query (~line 566) — the list is a browse surface (see 8.2).
- `ui_dashboard` (~line 357) — `counts_24h`, `counts_all`, `pipeline_activity`,
  `teams_by_pipeline`, `tokens_by_pipeline`, `source_counts`, and the feedback aggregates get
  the predicate. The `recent_runs` top-10 (~line 415) stays unfiltered (browse) but rows get
  a badge (8.3).
- `ui_insights_pipelines` (~line 1237) — every query in the function gets the predicate.
- `insights_overview` accuracy card and `ui_pipelines` success-rate/accuracy bars
  (~line 942-967) — filter the run/feedback rollups to production.

Suggested helper to keep it DRY:

```python
def _production_only(q):
    return q.where(PipelineRun.stage == "production")
```

### 8.2 Add a `stage` filter to the runs browse list (`ui_runs`)

Mirror the existing `team` filter exactly (~line 543-560):

- Add `stage: str | None = None` to the signature.
- In `_filtered()`: `if stage: q = q.where(PipelineRun.stage == stage)`.
- Build a `stage_values = ["testing", "production"]` (or `SELECT DISTINCT stage`) for the
  dropdown; pass `selected_stage` to the template.

The list defaults to **all** runs (badged) so it stays a debugging surface; the user can
filter to `production` or `testing` on demand.

### 8.3 Badge + template changes

- `service/templates/_macros.html` — add a `stage_badge(stage)` macro rendering an amber
  "TESTING" chip when `stage == "testing"` and nothing (or a quiet marker) otherwise. Reuse
  the existing chip styling used by tags/feedback badges.
- `runs.html` — render `stage_badge(run.stage)` on each row; add the `stage` `<select>` next
  to the existing status/team filters.
- `dashboard.html` — `stage_badge` on each recent-runs row.
- `run_detail.html` — `stage_badge` in the run header.
- `pipelines.html` / `pipeline_detail.html` — `stage_badge(pipeline.stage)` on the card and
  detail header (reads from the in-memory registry, `p.stage`, not the DB). Optional: a
  `?stage=` filter on `/ui/pipelines` mirroring the existing tag filter.

`PipelineRun.stage` is now on the ORM object, so templates can read `run.stage` directly with
no extra query.

---

## 9. Docs + samples

- **README** — new "§ Pipeline stages (testing vs production)" section: the field, the safe
  default, the four muted side-effects, the human-in-the-loop testing behaviour, the metrics
  exclusion, the trigger override (`?allow_testing=true`), and the git promotion workflow.
  Add `stage` to the pipeline schema table (§4) and `pipeline_runs.stage` to the DB schema
  table (§14).
- **Sample** — `samples/pipelines/stage-testing-example.yaml`: a pipeline with
  `stage: testing`, a `notify` step, and a `human` approval step, so the muting is
  demonstrable end-to-end. Add a one-line comment showing the `stage: production` promotion.

---

## 10. Sequencing (one commit per section)

1. §1 model field (+ test)
2. §2 DB column + migration
3. §3.1 write stage on run row
4. §3.2 `_testing` context injection
5. §3.3–3.4 gate pipeline notifications + step-failure webhook
6. §4 notify executor gating (+ test)
7. §5 human executor testing behaviour (+ test)
8. §6 trigger gating (+ test)
9. §7 metrics exclusion (+ test)
10. §8 UI rollup exclusion + runs filter + badges
11. §9 README + sample

Run `cd service && pytest` after each. Steps 1–4 are prerequisites for everything else; 5–9
are independent of each other and can be reordered.

## 11. Acceptance criteria

- A pipeline with no `stage:` loads as `testing`.
- A `stage: testing` run: sends nothing to Telegram/Slack/PagerDuty via any of the four
  paths (`notifications:`, `executor: notify`, `on_failure.webhook`, `executor: human`);
  every suppressed side-effect appears in the run-log timeline; a human step is decidable in
  P-Ork's UI and a Reject drives downstream rejection logic; the run is fully visible on its
  detail page.
- A `stage: testing` run contributes **zero** to Insights, dashboard aggregate cards,
  runs-page stat cards, pipeline success/accuracy bars, and `/metrics`.
- The runs list still shows testing runs, badged, with a working `stage` filter.
- `stage: testing` pipelines don't fire from `/webhook` without `?allow_testing=true`, but
  always run from the "Run now" button.
- Promoting to `stage: production` (YAML edit + `POST /reload`) flips all of the above, and
  **does not** retroactively reclassify prior testing runs (persisted stage).
- All existing tests still pass; new tests for §1, §4, §5, §6, §7 added.

## 12. Decisions to confirm before starting

- **Trigger-gating default:** plan blocks testing pipelines on `/webhook` unless
  `?allow_testing=true`. If you'd rather it be off by default (never block, purely
  advisory), drop the §6.1 early-return and keep only the badge/metrics behaviour.
- **Human timeout in testing:** plan auto-approves on timeout. If you'd rather it stay
  pending indefinitely (or fail), change §5.1.
- **`/metrics`:** plan excludes testing. Switch to a `stage` label if you want testing
  visible in Grafana instead.
