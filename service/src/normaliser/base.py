from abc import ABC, abstractmethod
from ..models.context import NormalisedContext


class BaseParser(ABC):
    @abstractmethod
    async def parse(self, payload: dict) -> NormalisedContext:
        pass
