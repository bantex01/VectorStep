"""Content hashing for step prompt templates — the substrate that scopes a
calibration bucket to the configuration it actually measured. See
SPEC-prompt-versioning.md."""

import hashlib
import logging

import httpx
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.models import AgentVersionSnapshot, StepPromptVersion
from ..utils import utc_now

logger = logging.getLogger(__name__)


def normalise_template(text: str) -> str:
    """Strip trailing whitespace per line, strip leading/trailing blank lines.

    Deliberately conservative — see SPEC-prompt-versioning.md §2. Reformatting
    a prompt IS a behavioural change; a false merge corrupts an enforcing gate,
    a false split only costs labels.
    """
    return "\n".join(line.rstrip() for line in text.splitlines()).strip("\n")


def prompt_hash(template: str | None) -> str | None:
    """sha256[:12] of the normalised template, or None for a step with no
    prompt template at all (webhook / notify / human / pipeline executors —
    non-LLM steps, where a prompt version is meaningless).
    """
    if not template:
        return None
    normalised = normalise_template(template)
    if not normalised:
        return None
    return hashlib.sha256(normalised.encode()).hexdigest()[:12]


async def _upsert(session: AsyncSession, row) -> None:
    """Insert `row` (primary key already set), refreshing an existing row's
    last_seen_at instead if another concurrent writer beat us to the same
    primary key. Uses a SAVEPOINT so a lost race only unwinds this insert,
    not the caller's outer transaction (the PipelineStep insert it shares a
    session with). A duplicate row is impossible — the hash/version IS the
    primary key — so IntegrityError here can only mean "someone else already
    wrote this key"."""
    model = type(row)
    pk = model.__mapper__.primary_key[0].name
    try:
        async with session.begin_nested():
            session.add(row)
            await session.flush()
    except IntegrityError:
        existing = await session.get(model, getattr(row, pk))
        if existing is not None:
            existing.last_seen_at = row.last_seen_at


async def record_prompt_version(
    session: AsyncSession, *, hash_: str | None, step_name: str, template: str
) -> None:
    """Write-on-miss upsert into step_prompt_versions, refreshing last_seen_at.

    Called inside the same session/transaction as the PipelineStep insert so a
    step row can never reference a version that isn't in the registry.
    """
    if hash_ is None:
        return
    now = utc_now()
    existing = await session.get(StepPromptVersion, hash_)
    if existing is not None:
        existing.last_seen_at = now
        return
    await _upsert(session, StepPromptVersion(
        prompt_hash=hash_, step_name=step_name, template=template,
        first_seen_at=now, last_seen_at=now,
    ))


async def record_agent_version(
    session: AsyncSession, rest_url: str, *, agent_version: str | None, agent: str
) -> None:
    """Snapshot an unseen Gateway agent version by asking the Gateway for its
    current definition, storing it only if the Gateway confirms that IS the
    current version. Best-effort: never raises, never blocks the run."""
    if agent_version is None:
        return
    now = utc_now()
    existing = await session.get(AgentVersionSnapshot, agent_version)
    if existing is not None:
        existing.last_seen_at = now
        return

    agent_name = agent.removeprefix("gateway:")
    soul_md: str | None = None
    agent_yaml: str | None = None
    note: str | None = None
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{rest_url}/agents/{agent_name}")
            response.raise_for_status()
            data = response.json()
    except Exception as exc:
        logger.warning(
            "Could not snapshot agent version %s for %s: %s", agent_version, agent, exc,
        )
        note = "gateway unreachable at snapshot time"
    else:
        if data.get("version") == agent_version:
            soul_md = data.get("soul_md")
            agent_yaml = data.get("agent_yaml")
        else:
            note = "agent config changed before P-Ork could snapshot this version"

    await _upsert(session, AgentVersionSnapshot(
        agent_version=agent_version, agent=agent, soul_md=soul_md, agent_yaml=agent_yaml,
        note=note, first_seen_at=now, last_seen_at=now,
    ))
