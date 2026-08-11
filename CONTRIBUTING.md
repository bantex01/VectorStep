# Contributing to VectorStep

Contributions are welcome. VectorStep is Apache-2.0 licensed, maintained in the open,
and pull requests get read and merged.

If you're here because you hit a bug, wanted behaviour that doesn't exist, or
found the docs wrong — all three are worth your time to report, and a report is
never a waste of mine.

## Before a large change, open an issue

For anything small — a bug fix, a doc correction, a missing test, a tidy-up —
just open a pull request. No preamble needed.

For anything larger — a new executor adapter, a new source parser, a schema
change, a refactor across modules — please open an issue first and sketch what
you have in mind. This isn't gatekeeping: it's so you don't spend a weekend on
something that turns out to conflict with work already in flight, or with a
design decision that has a reason behind it that isn't obvious from the code.

## Getting set up

```bash
cd service
python -m venv .venv
source .venv/bin/activate
pip install -r ../requirements.txt
pip install -r ../requirements-dev.txt
```

Full setup — including Postgres, the Gateway, and running a first pipeline end
to end — is in [Installation](https://vectorstep.io/docs/getting-started/installation/).

Run the tests:

```bash
cd service && python -m pytest

# the same suite against Postgres rather than SQLite:
VECTORSTEP_TEST_DATABASE_URL=postgresql://... python -m pytest
```

Tests should pass before you open a pull request. If something fails on a clean
checkout, that's a bug — please report it rather than working around it.

## What makes a pull request easy to merge

- **One concern per pull request.** Two unrelated fixes are two pull requests.
- **A test that fails before your change and passes after it.** For a bug fix,
  this is the single most useful thing you can include.
- **Existing style.** Match the surrounding code rather than importing your own
  conventions. There's no separate style guide — the code is the style guide.
- **Say why, not just what.** The diff shows what changed. The description
  should say what problem it solves.
- **Docs updated if behaviour changed.** Detailed docs live in the
  [VectorStep-Website](https://github.com/bantex01/VectorStep-Website) repo, not
  here — a note in your pull request saying what needs changing there is enough,
  and it will get done.

## Licensing of contributions

By opening a pull request you're contributing under the project's Apache-2.0
licence — this is Apache-2.0 section 5, and it applies by default. There's no
CLA to sign and no copyright assignment. Your commits stay yours, in your name,
in the history.

## What else genuinely helps

- **Bug reports.** Steps to reproduce, what you expected, what happened, and the
  version you were on.
- **Use cases.** Especially the ones explaining what you were trying to do, not
  only what you want added. These shape the roadmap more than feature requests.
- **Questions about how something works.** If the docs didn't answer it, that's
  a documentation bug and reporting it is useful on its own.

## A realistic word on response times

This is a single-maintainer project run alongside a full-time job. Issues and
pull requests get read, but not always quickly, and not always the same week.
If something has gone quiet for a fortnight, a nudge on the thread is welcome
and won't annoy anyone.

## Forking

Forking is fine and always was. If you need something the project doesn't
provide and an upstream change isn't the right fit, maintaining a fork is a
legitimate answer. The one limit: don't use the VectorStep name or logo in a way
that implies your fork is endorsed by or affiliated with this project.

## Security issues

Please don't open a public issue for a suspected vulnerability. See
[SECURITY.md](SECURITY.md) for how to report one privately.

## Getting in touch

If you'd rather just talk to a person — about a use case, whether something is a
bug, or anything else — I'm at **alex@vectorstep.io**.
