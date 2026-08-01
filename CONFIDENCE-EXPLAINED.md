# How confidence and calibration actually work

This is a plain-language explainer for how P-Ork decides how much to trust an agent's
output, and what every knob along the way actually does. If you want the precise
technical reference (field names, config schemas, DB columns), see `README.md` — this
document is for understanding the *why* and the *effect*, not the exact syntax.

---

## 1. The problem this solves

An agent finishes a task and says "I'm 95% confident." Should you believe it?

LLMs are not naturally good at this. A model can be completely wrong and still say
"95% confident" in a calm, well-structured sentence — confidence, as an agent reports
it, is a *style* of writing as much as it is a real measurement. If P-Ork just trusted
that number at face value, an overconfident wrong answer would sail straight through
any gate you set.

So instead of trusting one number, P-Ork builds up a **trust vector** — several
independent signals, combined conservatively — and only lets a step act automatically
when the *weakest* one still clears the bar. Not an average. The floor.

There are four signals:

| Signal | Question it answers | Nickname |
|---|---|---|
| **S** | What did the agent say about itself? | Self-report |
| **V** | What does a second opinion say? | Verifier |
| **G** | Is the agent's story actually backed by evidence it collected? | Grounding |
| **D** | Does an external, computer-checkable fact confirm or deny it? | Deterministic check |

Plus **calibration**, which sits on top of all of this and asks a different question
entirely: *does this agent's confidence number actually mean what it claims to mean,
based on its real track record?*

Below, each signal in turn — what it is, how it's produced, and every knob that changes
its behaviour.

---

## 2. S — the self-report

This is just the `confidence` field the primary agent returns in its own JSON output,
e.g. `0.95`. It costs nothing extra — every step already produces this. It's the
starting point, and by itself it's the least trustworthy number in the whole system,
because it's the agent grading its own work with no outside check at all.

Nothing to configure here — it's just whatever the agent says.

---

## 3. V — the verifier (a second opinion)

A verifier is a *second* agent call whose whole job is to sanity-check the first one.
It only fires if you configure a `verifier:` block on a step.

### Two ways to ask for a second opinion

**`mode: critic`** (the default) — the second agent sees the primary's original
instructions *and* its finished answer, and is asked "does this reasoning hold up?" It's
like a colleague reviewing your written report: they can catch a logical gap, a claim
that contradicts itself, or (as of a recent fix) check the primary's specific factual
claims — "was a ticket really created," "was a document really read" — against the
actual tool-call evidence, not just judge whether the prose sounds plausible.

**`mode: independent`** — the second agent gets *only* the original task, with zero
sight of what the primary said. It goes and does the whole task itself, blind, from
scratch. It's like handing the same assignment to two different people separately and
seeing if they land in the same place.

**Why the difference matters:** a critic's *agreement* with the primary tells you very
little, because if the primary made a mistake, the same mistake is often sitting right
there in the material the critic is reading — the critic's errors are *correlated* with
the primary's. An independent agent's agreement is worth much more, because it made its
own mistakes (or didn't) completely separately — its errors are *uncorrelated*. Use
`independent` for any step that's about to trigger a real side effect; `critic` is fine
(and cheaper) for steps that just inform the next step.

### How the two scores get combined

Two strategies, set via `combination_strategy`:

- **`minimum`** (the default) — the final score is simply whichever of the two is
  lower. Blunt, but simple: neither agent gets to inflate the other's score.
- **`veto`** — the primary's score is left alone *unless* the verifier scores below a
  set floor (`veto_floor`, default 0.60). Below that floor, the verifier's low score
  overrides. Above it, the verifier's opinion is noted but doesn't change anything.
  Think of it as "I'll take your word for it, unless a second opinion actively says
  this is bad."

**The one rule that never breaks, regardless of strategy: a verifier can only ever
lower the primary's score, or leave it unchanged — never raise it.** This is
deliberate and enforced by code, with a permanent regression test locking it in. If a
correlated second opinion were allowed to *boost* your confidence, you'd get exactly
the "two guesses agree, so it must be right" false comfort that makes AI consensus
untrustworthy in high-stakes settings.

### The trace-visibility knob

`max_trace_chars` (default 1500, `critic` mode only) controls how much of the primary's
actual tool-call transcript the critic gets to see. If a claim's supporting evidence
lives past this cutoff, the critic genuinely cannot see it — it isn't being harsh, it's
blind past that point. Raise this for steps whose tools return long content (a full
document read, a large query result).

---

## 4. G — grounding (fact-checking the story against the evidence)

Where the verifier asks "does the *reasoning* hold up," grounding asks a narrower,
harder question: **for every specific claim the agent made, is there an actual tool
call or tool result in its own trace that backs it up?** It's not judging whether the
conclusion is *right* — only whether it's *anchored to real evidence* rather than
plausible-sounding assertion.

A separate "judge" agent gets the original task, the primary's response, and a
formatted transcript of the primary's tool calls, and returns a score — the fraction of
load-bearing claims (a root cause, a specific number, a ticket ID) that a tool result
actually supports.

### Shadow vs enforced

By default, grounding is **shadow mode**: it computes and records the score, but never
changes anything. This lets you watch, for weeks if you want, how often an agent's
confidence and its actual grounding diverge, with zero risk to production behaviour.

Set `grounding.enforce: true` and it starts to matter: the grounding score becomes a
**ceiling**. `combined_trust = min(combined_trust, G)` — if the evidence is weak, no
amount of confident-sounding prose from anywhere else in the chain can push the trust
above that ceiling.

### The truncation gotcha (a real, common source of false alarms)

Both the trace-formatting step in P-Ork (`grounding.max_trace_chars`, default 1500) and
the Gateway that actually runs the tool calls (`limits.trace_tool_result_max_chars`,
default 3000, overridable per-step via `executor_config.trace_max_chars`) cap how much
of a tool result gets kept for the record. If a claim's supporting evidence sits past
either cutoff, grounding will flag it as "unsupported" — and that looks *exactly* like
a hallucination, even though the agent genuinely did the work and the evidence genuinely
exists, just not within reach of what got captured. If grounding keeps flagging things
you're confident are real, raise both cutoffs together — raising only one just moves
the same wall to a different spot in the pipe.

Reassuringly: this cutoff only affects what gets *recorded for review*. The agent
itself always sees the full, untruncated tool output when it's actually doing the work
— truncation never makes the agent itself less capable, only makes grounding (and a
human reviewer) less able to check its work after the fact.

### The other blind spot this fixes: knowing what was actually asked

Grounding (and critic-mode verifiers) are shown the primary's *original instructions*,
not just its answer. Without this, a claim that simply restates something the agent was
*given* — the alert's severity, which service it's about — would look exactly like an
unverified claim, because there'd be no tool call backing it up (it never needed one;
it was handed the fact as input). The judge is told explicitly: facts already present
in the original task need no evidence, only claims that go beyond the given input do.

---

## 5. D — deterministic checks (hard, computer-verified facts)

Everything above is still an LLM making a judgement call. Deterministic checks are the
one signal that isn't: a `shell` command, a `webhook` call, or a `human` approval,
evaluated directly by the runner — no model involved in the pass/fail decision at all.

Examples: "is this metric still actually breaching, right now, via a real Prometheus
query" or "does this ticket ID actually resolve." These aren't opinions — they're facts
you can check.

**A failed deterministic check is dispositive.** If any declared check fails,
`combined_trust` is forced straight to **zero**, no matter how confident everything
else in the chain was. This is the strongest, most trustworthy signal in the whole
system, and it's treated that way: no averaging, no partial credit, nothing else gets a
vote once a hard check fails.

**Fail-closed, not fail-soft.** If a check errors, times out, or can't be evaluated for
any reason, that counts as **failed** — the opposite of grounding's philosophy, where a
failed grounding call just means "no extra signal, no penalty." The reasoning: D is
meant to be the one thing you can absolutely rely on, so an unanswerable check must not
silently vanish and let everything proceed as if nothing was checked.

---

## 6. Putting it together — the actual formula

For a step with everything turned on, in the exact order the runner applies it:

```
S              — the primary's self-report
  ↓ (verifier combine — minimum or veto)
S_after_V      — S adjusted by the verifier; can only be ≤ S
  ↓ (calibration, IF enforce: true and the bucket is validated)
calibrated     — S_after_V REPLACED by this exact agent/model's real track record
                 (can go up OR down — see §7, this is the one exception to "never raises")
  ↓ (grounding, IF enforce: true — a ceiling, min())
capped         — pulled down further if the evidence doesn't back up the claims
  ↓ (deterministic checks — force to zero on any failure)
combined_trust — the final number
  ↓
compare to the step's confidence_threshold
  ↓
proceed  or  escalate / abort / proceed-anyway (per on_low_confidence)
```

**Weakest link, not average.** At every stage, a bad signal can only pull the number
down (calibration is the sole exception, and even then it's replacing the seed with a
*measured* fact, not "boosting" it for optimism's sake). A control decision — "should
this step be allowed to actually do something" — needs the floor, because a
comfortable-looking average can hide the one thing that's actually wrong.

**A step with none of these configured is completely unaffected.** No verifier, no
grounding, no deterministic checks, no calibration — the step behaves exactly as it
always has: raw self-report versus threshold, nothing more. Every mechanism above is
strictly additive and opt-in.

---

## 7. Calibration — does the number mean what it claims?

Everything above assumes S, V, and G are each individually meaningful. Calibration is
the mechanism that actually checks that assumption, per agent, per model, empirically —
using this exact system's own history rather than trusting the number on its own
terms.

### The core idea

Suppose a particular agent, running on a particular model, has reported "90–100%
confidence" on 30 past occasions. If a human later marked those runs, and it turns out
only 20 of the 30 were actually correct — that's **67% actual accuracy**, not the
~95% the confidence number implied. That agent, at that confidence level, is
overconfident. Calibration measures this directly and, if you opt in, uses the measured
67% instead of trusting the self-report.

### How it's measured

For every distinct `(step, agent, model, provider, prompt_hash, agent_version)`
combination, every past marked run is sorted into a **bin** based on what confidence it
reported — by default, ten bins,
each covering a 10-point range (0–10%, 10–20%, …, 90–100%). Within each bin, calibration
computes the **mean label** — the average of what actually happened, not what was
predicted:

- A human marked it `correct` → counts as 1.0
- A human marked it `partial` → counts as 0.5
- A human marked it `incorrect` → counts as 0.0
- No human mark, but a deterministic check failed → counts as 0.0 (a failed hard check
  is a strong enough automated signal to count as "wrong," for free, at scale)
- No step-level mark or check failure, but the whole *run* was marked → falls back to
  that run's outcome
- None of the above → excluded entirely (not counted as a zero, not counted at all)

That mean is the bin's real, measured accuracy — completely independent of what the
confidence score itself said, which is the whole point.

Fixed-width bins, not a fitted curve — deliberately. It's directly readable ("this
bucket, this many samples, this accuracy") the way a smooth regression line isn't, and
it needed no new numerical-computing dependency for a single-operator service.

### Why it needs enough data before you trust it

A bin with 2 samples telling you "0% accurate" is noise, not a fact. Each bin needs
`n_min` (default 20) marked outcomes before it's considered **validated**. Below that,
it's advisory-only regardless of what the raw number looks like.

### Editing a prompt or an agent resets the measurement — deliberately

The bucket key above includes two more components than you might expect:
`prompt_hash` (a content hash of the step's `prompt_template`) and `agent_version` (a
content hash the Gateway computes over an agent's entire config, including `soul.md`).
Here's why, and what you'll actually see when it happens.

**The bug this closes.** Before these existed, editing a step's prompt — or editing an
agent's `soul.md` on the Gateway, which P-Ork had no visibility into at all — silently
kept counting outcomes from the OLD configuration as evidence for the NEW one. For a
step with `calibration: {enforce: true}`, that meant the gate was making a real control
decision using a measured accuracy figure that described a prompt or agent that no
longer existed. Nothing told you this was happening.

**What you'll see instead.** Edit a prompt template, or edit an agent's `agent.yaml`/
`soul.md` on the Gateway, and the next run under that changed configuration starts a
*new* bucket at n=0 — even though the step/agent/model/provider combination is
identical to before. If that step had `calibration: {enforce: true}` and was passing
(a validated bucket), it falls back to `on_uncalibrated` behaviour (escalating, by
default proceeding) until the new bucket earns its own `n_min` marked results. The
run-detail Trust panel names this explicitly rather than leaving you to guess why a
step "suddenly" started behaving differently:

> Calibration was reset when this step's prompt changed on 3 Jul. The previous version
> had 47 marked results; this one has 4 of the 20 needed.

**This is deliberate, and reverting restores it.** A bucket that blends two different
prompts isn't a bigger sample, it's a wrong one — the reset is the correct behaviour,
not a side effect to route around. And because the registry is content-addressed (keyed
by the hash itself), reverting a prompt back to its exact previous text automatically
rejoins that version's original bucket and every label it earned — nothing is lost by
trying an edit and changing your mind. Two versions of the same step, side by side with
their own calibration bins, are visible from the Steps Insights page's **Prompt
history** disclosure (`GET /steps/{name}/versions`) — and the equivalent for an agent's
config history is `GET /agents/{name}/versions`, reachable from the `agent <hash>` chip
on any run that used it.

### Two postures: advisory (always on) vs enforcing (opt-in)

**Advisory — no configuration needed, ever on.** The Steps Insights page shows, for
every agent/model combination, its calibration bins — validated ones colour-coded by
whether the predicted score and the actual accuracy line up, with a plain-English flag
when a validated bin diverges by 15 points or more ("runs scoring ~90% here are only
75% correct"). Nothing about this changes any run's outcome. You look, you decide —
raise the threshold, swap the model, add grounding — the tool never acts on your
behalf.

**Enforcing — opt-in, per step, `calibration: {enforce: true}`.** Once you trust a
measurement enough to let it actually change behaviour, this step's gate stops trusting
the raw self-report/verifier number outright and instead asks: "for this specific
confidence level, what does this exact agent/model's real history say?" If the
relevant bin is validated, that measured number *replaces* the raw score before
grounding and deterministic checks apply on top of it.

If the bin **isn't** validated yet — a brand-new agent/model combination with no track
record — `on_uncalibrated` decides what happens:
- **`proceed`** (the default) — behave exactly as if calibration weren't configured at
  all. No penalty for being new and unproven.
- **`escalate`** — treat "no track record yet" as untrustworthy on purpose, forcing
  `combined_trust` to zero until the bucket earns enough history. This is the deliberate
  "a brand-new configuration should have a human check it until it's proven" policy —
  an opt-in choice for high-stakes steps, never the default anywhere.

### The one place calibration breaks the "only ever lowers" rule

Every other mechanism above is downward-only. Calibration is not — if an agent has been
systematically *underselling* itself (says 60%, is actually right 95% of the time), an
enforced, validated bucket will raise the number used for the gate above the raw
self-report. This is intentional: calibration isn't correcting for a second opinion's
correlated errors, it's substituting a measured fact for a guess, and that measured
fact can point either direction.

---

## 8. A worked example, start to finish

A step self-reports 95% confidence. It has a `critic` verifier, enforced grounding, one
deterministic check, and enforced calibration. `confidence_threshold: 0.75`.

1. **S = 95%.** The agent's own number.
2. **Verifier (critic, veto strategy, floor 60%)** comes back at 85%. 85% is above the
   60% floor, so nothing changes — **S_after_V stays at 95%**.
3. **Calibration** looks up this exact agent/model's history at the 90–100% confidence
   level: 31 past marked runs, only correct 62% of the time. The bucket is validated
   (31 ≥ 20), so **62% replaces the 95%**.
4. **Grounding (enforced)** checks the tool trace against the claims made, and finds
   only 50% of them actually backed by evidence. 50% is lower than 62%, so it becomes
   the new ceiling — **capped at 50%**.
5. **Deterministic check** ("is this still breaching, checked live against Prometheus")
   passes — no further effect.
6. **Final combined_trust = 50%.** Compared against the 75% threshold: 50% falls
   short, so the step **escalates** to a human instead of proceeding automatically —
   even though the agent itself was "95% confident" the whole time.

Every one of those five steps, with the exact numbers for that specific run, is visible
in the run-detail page's Trust panel, under **"How was this calculated?"**

---

## 9. Quick-reference: every knob and what it does

| Knob | Where | Default | Effect |
|---|---|---|---|
| `confidence_threshold` | step | `0.75` | The bar `combined_trust` is compared against. Existed before any of this; nothing above replaces it. |
| `on_low_confidence` | step | `escalate` | What happens when the bar isn't cleared: `escalate` / `abort` / `proceed`. |
| `verifier.mode` | step | `critic` | `critic` (sees primary's answer, reviews reasoning) or `independent` (blind, redoes the task). |
| `verifier.combination_strategy` | step | `minimum` | `minimum` (always take the lower score) or `veto` (only override below a floor). |
| `verifier.veto_floor` | step | `0.60` | Only used with `veto` — how low V has to score to override S. |
| `verifier.max_trace_chars` | step | `1500` | How much tool-call transcript a `critic` verifier gets to see. |
| `grounding.enforce` | step | `false` | `false` = record G only, never gate. `true` = G caps `combined_trust`. |
| `grounding.max_trace_chars` | step | `1500` | How much tool-call transcript the grounding judge gets to see. |
| `executor_config.trace_max_chars` | step (gateway executor) | `3000` (Gateway's own default) | Raises the Gateway's own cap on tool-result trace length for this step — the upstream half of the truncation gotcha in §4. |
| `deterministic_checks` | step | none | List of `shell` / `webhook` / `human` checks; any failure forces trust to zero. |
| `calibration.enforce` | step | `false` | `false` = advisory only (still visible on Insights). `true` = validated buckets replace the raw score. |
| `calibration.on_uncalibrated` | step | `proceed` | What an enforced-but-unvalidated bucket does: `proceed` (no penalty) or `escalate` (treat as untrustworthy until proven). |
| `calibration.n_min` | service config.yaml | `20` | Marked outcomes a bin needs before it's trusted. |
| `calibration.bin_width` | service config.yaml | `0.1` | Width of each confidence bucket. |
| `prompt_hash` | derived, not configured | — | Content hash of the step's `prompt_template`. Part of every calibration bucket's key — editing the prompt starts a new bucket. See §7's "Editing a prompt or an agent resets the measurement". |
| `agent_version` | derived, not configured (Gateway-owned) | — | Content hash of the Gateway agent's full config incl. `soul.md`. Also part of every bucket's key — editing `agent.yaml`/`soul.md` on the Gateway resets calibration in P-Ork, even though nothing in P-Ork's own YAML changed. |
| `readiness.operational.min_runs` | pipeline or step `readiness:` | required | Distinct runs (never rows) that must end in an acceptable status. Can only ever be `pass`/`insufficient_data` — never `fail`. |
| `readiness.operational.acceptable_statuses` | pipeline or step `readiness:` | `[completed]` | End-states that count. Adding a status makes the bar LAXER, not stricter — `[completed, escalated]` is weaker than `[completed]` alone. |
| `readiness.operational.max_age_days` | pipeline or step `readiness:` | `null` (lifetime) | Restricts `operational` to runs from the last N days. The only readiness tier with a time window. |
| `readiness.operational.require_current_config` | pipeline or step `readiness:` | `false` | `true` filters to runs matching the current prompt/agent version. |
| `readiness.confidence.min_confidence` | pipeline or step `readiness:` | required | Minimum mean self-reported `effective_confidence` over qualifying runs. |
| `readiness.confidence.min_runs` | pipeline or step `readiness:` | `null` | Minimum sample size before `min_confidence` is trusted — strongly recommended. |
| `readiness.accuracy.min_accuracy` | pipeline or step `readiness:` | required | Minimum weighted judged accuracy (correct=1.0, partial=0.5, incorrect=0.0). |
| `readiness.accuracy.min_marked` | pipeline or step `readiness:` | required | Minimum labelled results before `min_accuracy` is evaluated. |
| `readiness.accuracy.min_human_marked` | pipeline or step `readiness:` | `null` | Minimum labels from a HUMAN specifically — guards against a labelled population that's 100% failed deterministic checks reading as 0% accurate. |
| `readiness.calibration.n_min` | pipeline or step `readiness:` | `20` | Marked outcomes needed AT THE SAME CONFIDENCE BAND — per band, not a total. The single most misread readiness knob. |
| `readiness.calibration.bin_width` | pipeline or step `readiness:` | `0.1` | Must evenly divide 1.0 — rejected at config load, not at request time, if it doesn't. |
| `readiness.calibration.max_divergence` | pipeline or step `readiness:` | `0.15` | Owner-settable version of the hardcoded 15-point divergence flag. |
| `readiness.calibration.require_own_evidence` | pipeline or step `readiness:` | `false` | `true` restricts calibration to this pipeline's own runs, excluding a shared library step's production track record from elsewhere. |
| `readiness.calibration.require_current_config` | pipeline or step `readiness:` | `true`, cannot be `false` | A calibration bucket is keyed by `(prompt_hash, agent_version)` by definition — "ignore the version" would mean merging buckets. |

**Safety property:** adding a `readiness:` block never resets a calibration bucket — `prompt_hash` is computed from `prompt_template` **text only**, so `readiness:` (which lives alongside, not inside, the prompt) can't touch it.

---

## 10. Where to actually see all of this

- **Run detail page, per step:** the **Step configuration** disclosure (what this step
  is set up to do) and **"How was this calculated?"** (what actually happened, this run,
  in plain language with real numbers) both live in the Trust panel. **Prompt**
  disclosures on the primary, verifier, and grounding sections show exactly what each
  agent was actually asked, with `template <hash>` / `agent <hash>` chips linking to
  that prompt's or agent's own version history.
- **`/ui/insights/steps`:** calibration bins for every agent/model/prompt-version
  combination, with the divergence flag — this is where you watch calibration before
  ever turning on `enforce: true`. The **Prompt history** disclosure per step shows
  every recorded prompt version with its date range, run count, labelled count, and a
  diff against the version before it — "did that edit actually help?" made answerable
  with data.
- **`/ui/agents/gateway/<name>` → Versions tab:** every `agent_version` P-Ork has a
  snapshot for, with a diff of `soul.md` against the version before it and the list of
  steps that version actually affected.
- **Pipeline detail page, "Promotion readiness" card** (`stage: testing` pipelines):
  per-step tier chips for `operational`/`confidence`/`accuracy`/`calibration`, a "How is
  this judged?" disclosure with a plain-language narrative and label provenance, and an
  "Observed (service defaults)" fallback for steps with no `readiness:` criteria
  configured yet. See "Promotion readiness (owner-defined criteria)" in `README.md`.
- **Same card, "Build criteria" disclosure:** a guided, preview-only builder for the
  `readiness:` knobs above — turn one on, see within ~300ms what it would say about this
  pipeline's accumulated evidence, and copy a ready-to-paste YAML snippet. Writes nothing;
  you still commit the snippet yourself. See "Criteria builder (guided UI)" in `README.md`.
