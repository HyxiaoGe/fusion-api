from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


class EmbeddingError(RuntimeError):
    def __init__(self, code: str, summary: str, *, retryable: bool):
        self.code = code
        self.summary = summary
        self.retryable = retryable
        super().__init__(summary)


@dataclass(frozen=True)
class EmbeddingProfile:
    provider: str
    model: str
    dimension: int
    distance_metric: str = "COSINE"
    collection_name: str | None = None
    revision: str | None = None


class EmbeddingAdapter(ABC):
    @abstractmethod
    async def embed(self, texts: list[str], profile: EmbeddingProfile) -> list[list[float]]: ...
