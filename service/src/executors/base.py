from abc import ABC, abstractmethod
from ..models.pipeline import StepConfig
from ..models.llm import LLMOutput


class BaseExecutor(ABC):
    @abstractmethod
    async def execute(self, step: StepConfig, context: dict) -> LLMOutput:
        pass
