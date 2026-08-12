from __future__ import annotations

import asyncio
import math
from typing import Any

import litellm

from app.ai.embeddings.base import EmbeddingAdapter, EmbeddingError, EmbeddingProfile
from app.core.config import settings


class LiteLLMEmbeddingAdapter(EmbeddingAdapter):
    """通过配置的 LiteLLM embedding alias 生成向量。"""

    async def embed(self, texts: list[str], profile: EmbeddingProfile) -> list[list[float]]:
        if not texts:
            return []
        try:
            route = settings.resolve_knowledge_embedding_route(profile.model, profile.revision)
        except ValueError as exc:
            raise EmbeddingError(
                "KNOWLEDGE_CONFIG_INVALID",
                "Embedding revision 路由配置无效",
                retryable=False,
            ) from exc
        try:
            response = await asyncio.wait_for(
                litellm.aembedding(
                    model=f"litellm_proxy/{route}",
                    input=texts,
                    api_base=settings.LITELLM_PROXY_URL,
                    api_key=settings.LITELLM_API_KEY,
                    timeout=settings.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS,
                ),
                timeout=settings.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            raise EmbeddingError(
                "KNOWLEDGE_EMBEDDING_UNAVAILABLE",
                "Embedding 服务暂时不可用",
                retryable=True,
            ) from exc
        raw_items = self._response_items(response)
        if len(raw_items) != len(texts):
            raise EmbeddingError(
                "KNOWLEDGE_EMBEDDING_COUNT_MISMATCH",
                "Embedding 返回数量与请求不一致",
                retryable=False,
            )
        missing_index = object()
        indexed_items: list[tuple[int, Any]] = []
        for item in raw_items:
            raw_index = self._read(item, "index", missing_index)
            if not isinstance(raw_index, int) or isinstance(raw_index, bool):
                raise EmbeddingError(
                    "KNOWLEDGE_EMBEDDING_COUNT_MISMATCH",
                    "Embedding 返回缺少有效索引",
                    retryable=False,
                )
            indexed_items.append((raw_index, item))
        indexed_items.sort(key=lambda pair: pair[0])
        indices = [index for index, _item in indexed_items]
        if indices != list(range(len(texts))):
            raise EmbeddingError(
                "KNOWLEDGE_EMBEDDING_COUNT_MISMATCH",
                "Embedding 返回索引不连续",
                retryable=False,
            )
        ordered = [item for _index, item in indexed_items]
        vectors = [list(self._read(item, "embedding", [])) for item in ordered]
        self.validate_vectors(vectors, expected_count=len(texts), dimension=profile.dimension)
        return vectors

    @staticmethod
    def validate_vectors(vectors: list[list[float]], *, expected_count: int, dimension: int) -> None:
        if len(vectors) != expected_count:
            raise EmbeddingError(
                "KNOWLEDGE_EMBEDDING_COUNT_MISMATCH",
                "Embedding 返回数量与请求不一致",
                retryable=False,
            )
        for vector in vectors:
            if len(vector) != dimension:
                raise EmbeddingError(
                    "KNOWLEDGE_EMBEDDING_DIMENSION_MISMATCH",
                    "Embedding 返回维度与知识库配置不一致",
                    retryable=False,
                )
            try:
                numeric = [float(value) for value in vector]
            except (TypeError, ValueError) as exc:
                raise EmbeddingError(
                    "KNOWLEDGE_EMBEDDING_INVALID",
                    "Embedding 返回了非数值向量",
                    retryable=False,
                ) from exc
            if any(not math.isfinite(value) for value in numeric) or not any(value != 0 for value in numeric):
                raise EmbeddingError(
                    "KNOWLEDGE_EMBEDDING_INVALID",
                    "Embedding 返回了空、零或非有限向量",
                    retryable=False,
                )

    @staticmethod
    def _response_items(response: Any) -> list[Any]:
        data = response.get("data") if isinstance(response, dict) else getattr(response, "data", None)
        return list(data) if data is not None else []

    @staticmethod
    def _read(item: Any, field: str, default: Any) -> Any:
        return item.get(field, default) if isinstance(item, dict) else getattr(item, field, default)
