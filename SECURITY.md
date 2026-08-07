# Security Policy

## Reporting a vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Use GitHub's private vulnerability reporting instead: go to the **Security** tab
of this repository and choose **Report a vulnerability**. That opens a private
channel visible only to the maintainer.

Please include:

- A description of the issue and why you believe it is a security problem
- Steps to reproduce, or a proof of concept
- The version or commit you were running
- Any deployment details that seem relevant (executor in use, auth
  configuration, whether the Gateway is exposed, and so on)

## What to expect

VectorStep is maintained by a single author as a best-effort open source
project. There is no SLA, and there is no bug bounty.

That said: reports will be acknowledged as promptly as is realistically
possible, you will be told whether the issue is accepted and what the fix
timeline looks like, and you will be credited in the release notes when a fix
ships — unless you would prefer not to be.

Please allow a reasonable period for a fix before disclosing publicly.

## Supported versions

Only the latest released version receives security fixes. There are no
long-term support branches.

## Scope

VectorStep executes AI pipelines that can call tools and take actions. Some
behaviour that looks alarming is intentional and configurable rather than a
vulnerability — for example, an agent taking an action that its `agent.yaml`
grants it, or a pipeline step running without a verifier because none was
configured. Reports about the *trust and gating machinery not behaving as
documented* are firmly in scope; reports that amount to "a permissive
configuration is permissive" generally are not. If you are unsure, report it
anyway and say so.
