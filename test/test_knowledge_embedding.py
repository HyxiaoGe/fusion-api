import asyncio
import json
import math
import unittest
from unittest.mock import AsyncMock, patch

from app.ai.embeddings.base import EmbeddingError, EmbeddingProfile
from app.ai.embeddings.litellm_embedding import LiteLLMEmbeddingAdapter


class KnowledgeEmbeddingValidationTests(unittest.TestCase):
    def test_valid_vectors_pass(self):
        LiteLLMEmbeddingAdapter.validate_vectors([[1.0, 2.0], [3.0, 4.0]], expected_count=2, dimension=2)

    def test_count_mismatch_is_rejected(self):
        with self.assertRaises(EmbeddingError) as raised:
            LiteLLMEmbeddingAdapter.validate_vectors([[1.0, 2.0]], expected_count=2, dimension=2)
        self.assertEqual(raised.exception.code, "KNOWLEDGE_EMBEDDING_COUNT_MISMATCH")

    def test_dimension_mismatch_is_rejected(self):
        with self.assertRaises(EmbeddingError) as raised:
            LiteLLMEmbeddingAdapter.validate_vectors([[1.0]], expected_count=1, dimension=2)
        self.assertEqual(raised.exception.code, "KNOWLEDGE_EMBEDDING_DIMENSION_MISMATCH")

    def test_zero_and_non_finite_vectors_are_rejected(self):
        for vector in ([0.0, 0.0], [1.0, math.nan], [1.0, math.inf]):
            with self.subTest(vector=vector), self.assertRaises(EmbeddingError) as raised:
                LiteLLMEmbeddingAdapter.validate_vectors([vector], expected_count=1, dimension=2)
            self.assertEqual(raised.exception.code, "KNOWLEDGE_EMBEDDING_INVALID")


class KnowledgeEmbeddingAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_embedding_uses_registered_alias_through_litellm_proxy(self):
        response = {"data": [{"index": 0, "embedding": [1.0, 0.5]}]}
        profile = EmbeddingProfile("litellm", "embedding-v1", 2, "COSINE", revision="embedding-r1")
        with (
            patch("app.ai.embeddings.litellm_embedding.settings.LITELLM_PROXY_URL", "http://proxy:4000"),
            patch("app.ai.embeddings.litellm_embedding.settings.LITELLM_API_KEY", "proxy-key"),
            patch("app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", 15),
            patch(
                "app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_REVISION_ROUTES",
                json.dumps({"embedding-v1@embedding-r1": "embedding-v1-r1-immutable"}),
            ),
            patch(
                "app.ai.embeddings.litellm_embedding.litellm.aembedding",
                new=AsyncMock(return_value=response),
            ) as embedding,
        ):
            vectors = await LiteLLMEmbeddingAdapter().embed(["正文"], profile)

        self.assertEqual(vectors, [[1.0, 0.5]])
        embedding.assert_awaited_once_with(
            model="litellm_proxy/embedding-v1-r1-immutable",
            input=["正文"],
            api_base="http://proxy:4000",
            api_key="proxy-key",
            timeout=15,
        )

    async def test_missing_historical_revision_mapping_fails_before_request(self):
        profile = EmbeddingProfile("litellm", "embedding-v0", 2, revision="embedding-r0")
        with (
            patch(
                "app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_REVISION_ROUTES",
                json.dumps({"embedding-v1@embedding-r1": "embedding-v1-r1-immutable"}),
            ),
            patch("app.ai.embeddings.litellm_embedding.litellm.aembedding", new=AsyncMock()) as embedding,
            self.assertRaises(EmbeddingError) as raised,
        ):
            await LiteLLMEmbeddingAdapter().embed(["正文"], profile)

        self.assertEqual(raised.exception.code, "KNOWLEDGE_CONFIG_INVALID")
        self.assertFalse(raised.exception.retryable)
        embedding.assert_not_awaited()

    async def test_embedding_timeout_is_retryable_unavailable(self):
        profile = EmbeddingProfile("litellm", "embedding-v1", 2, revision="embedding-r1")

        async def never_returns(**_kwargs):
            await asyncio.sleep(60)

        with (
            patch("app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", 0.001),
            patch(
                "app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_REVISION_ROUTES",
                json.dumps({"embedding-v1@embedding-r1": "embedding-v1-r1-immutable"}),
            ),
            patch("app.ai.embeddings.litellm_embedding.litellm.aembedding", new=AsyncMock(side_effect=never_returns)),
            self.assertRaises(EmbeddingError) as raised,
        ):
            await LiteLLMEmbeddingAdapter().embed(["正文"], profile)

        self.assertEqual(raised.exception.code, "KNOWLEDGE_EMBEDDING_UNAVAILABLE")
        self.assertTrue(raised.exception.retryable)

    async def test_batch_response_without_explicit_indices_is_rejected(self):
        response = {
            "data": [
                {"embedding": [1.0, 0.5]},
                {"embedding": [0.5, 1.0]},
            ]
        }
        profile = EmbeddingProfile("litellm", "embedding-v1", 2, revision="embedding-r1")
        with (
            patch("app.ai.embeddings.litellm_embedding.settings.LITELLM_PROXY_URL", "http://proxy:4000"),
            patch("app.ai.embeddings.litellm_embedding.settings.LITELLM_API_KEY", "proxy-key"),
            patch("app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS", 15),
            patch(
                "app.ai.embeddings.litellm_embedding.settings.KNOWLEDGE_EMBEDDING_REVISION_ROUTES",
                json.dumps({"embedding-v1@embedding-r1": "embedding-v1-r1-immutable"}),
            ),
            patch(
                "app.ai.embeddings.litellm_embedding.litellm.aembedding",
                new=AsyncMock(return_value=response),
            ),
            self.assertRaises(EmbeddingError) as raised,
        ):
            await LiteLLMEmbeddingAdapter().embed(["正文一", "正文二"], profile)

        self.assertEqual(raised.exception.code, "KNOWLEDGE_EMBEDDING_COUNT_MISMATCH")
        self.assertFalse(raised.exception.retryable)


if __name__ == "__main__":
    unittest.main()
