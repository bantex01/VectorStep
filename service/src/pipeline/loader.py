import logging
from pathlib import Path

import yaml

from ..models.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


def load_pipelines(config_dir: str | Path) -> list[PipelineConfig]:
    """Load and validate all YAML pipeline configs from a directory.

    Files are returned in filesystem order (alphabetical). Callers that need
    priority ordering should name files accordingly (e.g. 10-critical.yaml,
    20-warning.yaml) or sort the returned list themselves.
    """
    config_dir = Path(config_dir)
    if not config_dir.is_dir():
        raise ValueError(f"Pipeline config directory not found: {config_dir}")

    pipelines: list[PipelineConfig] = []
    for path in sorted(config_dir.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text())
            pipeline = PipelineConfig.model_validate(raw)
            pipelines.append(pipeline)
            logger.info("Loaded pipeline config: %s (from %s)", pipeline.name, path.name)
        except Exception as exc:
            logger.error("Failed to load pipeline config %s: %s", path.name, exc)
            raise

    logger.info("Loaded %d pipeline config(s) from %s", len(pipelines), config_dir)
    return pipelines
