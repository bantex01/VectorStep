import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCAL_PREFIX = "local://"


class ArtifactStore(ABC):
    @abstractmethod
    async def write(self, run_id: str, step_name: str, key: str, content: str) -> str:
        """Write content and return an opaque reference string."""

    @abstractmethod
    async def read(self, reference: str) -> str:
        """Return content for a reference. Raises FileNotFoundError if missing."""

    @abstractmethod
    async def delete_run(self, run_id: str) -> None:
        """Delete all artifacts for a run."""

    @abstractmethod
    async def runs_older_than(self, cutoff: datetime) -> list[str]:
        """Return run IDs whose artifact directories were last modified before cutoff."""


class LocalArtifactStore(ArtifactStore):
    """Stores artifacts as files under {base_dir}/{run_id}/{step_name}/{key}."""

    def __init__(self, base_dir: str) -> None:
        self._base = Path(base_dir).expanduser().resolve()

    def _artifact_path(self, run_id: str, step_name: str, key: str) -> Path:
        safe_step = step_name.replace("/", "_")
        safe_key = key.replace("/", "_")
        return self._base / run_id / safe_step / safe_key

    def _ref_to_path(self, reference: str) -> Path:
        if not reference.startswith(_LOCAL_PREFIX):
            raise ValueError(f"Not a local artifact reference: {reference!r}")
        return self._base / reference[len(_LOCAL_PREFIX):]

    async def write(self, run_id: str, step_name: str, key: str, content: str) -> str:
        path = self._artifact_path(run_id, step_name, key)
        await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        safe_step = step_name.replace("/", "_")
        safe_key = key.replace("/", "_")
        return f"{_LOCAL_PREFIX}{run_id}/{safe_step}/{safe_key}"

    async def read(self, reference: str) -> str:
        path = self._ref_to_path(reference)
        return await asyncio.to_thread(path.read_text, encoding="utf-8")

    async def delete_run(self, run_id: str) -> None:
        run_dir = self._base / run_id
        if await asyncio.to_thread(run_dir.exists):
            await asyncio.to_thread(shutil.rmtree, run_dir, True)

    async def runs_older_than(self, cutoff: datetime) -> list[str]:
        if not await asyncio.to_thread(self._base.exists):
            return []
        cutoff_ts = cutoff.timestamp()
        entries = await asyncio.to_thread(list, self._base.iterdir())
        return [
            e.name for e in entries
            if e.is_dir() and e.stat().st_mtime < cutoff_ts
        ]
