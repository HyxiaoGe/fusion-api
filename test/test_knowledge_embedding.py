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
        profile = EmbeddingProfile("litellm", "embedding-v1", 2, "COSINE")
        with (
            patch("app.ai.embeddings.litellm_embedding.settings.LITELLM_PROXY_URL", "http://proxy:4000"),
            patch("app.ai.embeddings.litellm_embedding.settings.LITELLM_API_KEY", "proxy-key"),
            patch(
                "app.ai.embeddings.litellm_embedding.litellm.aembedding",
                new=AsyncMock(return_value=response),
            ) as embedding,
        ):
            vectors = await LiteLLMEmbeddingAdapter().embed(["正文"], profile)

        self.assertEqual(vectors, [[1.0, 0.5]])
        embedding.assert_awaited_once_with(
            model="litellm_proxy/embedding-v1",
            input=["正文"],
            api_base="http://proxy:4000",
            api_key="proxy-key",
        )


if __name__ == "__main__":
    unittest.main()
