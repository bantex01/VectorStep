from abc import ABC, abstractmethod
from ..models.pipeline import StepConfig
from ..models.llm import LLMOutput


class LLMParseError(RuntimeError):
    """Raised when no payload from an executor call contained valid, schema-matching JSON.

    Carries the untruncated raw text so callers that want to persist it for debugging
    (e.g. the grounding pass, whose report would otherwise only capture the message's
    truncated snippet) don't have to re-parse it out of the exception string.
    """

    def __init__(self, message: str, raw_text: str):
        super().__init__(message)
        self.raw_text = raw_text


class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        pass
