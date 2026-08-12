import unittest
from unittest.mock import AsyncMock, patch

from app.ai.embeddings.base import EmbeddingProfile
from app.services.knowledge.chunker import KnowledgeChunk
from app.services.knowledge.milvus import (
    KnowledgeVectorError,
    KnowledgeVectorRecord,
    MilvusKnowledgeStore,
)


class MilvusKnowledgeStoreTests(unittest.TestCase):
    def test_collection_name_is_dimension_bucketed_and_controlled(self):
        with (
            patch("app.services.knowledge.milvus.settings.MILVUS_COLLECTION_PREFIX", "fusion-knowledge"),
            patch(
                "app.services.knowledge.milvus.settings.KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS",
                "1024",
            ),
        ):
            self.assertEqual(MilvusKnowledgeStore.collection_name(1024), "fusion_knowledge_v1_d1024")
            with self.assertRaises(KnowledgeVectorError):
                MilvusKnowledgeStore.collection_name(768)

    def test_filter_includes_user_bases_and_versions(self):
        expression = MilvusKnowledgeStore.build_search_filter(
            "user-1",
            ["kb-2", "kb-1"],
            ["version-2", "version-1"],
        )

        self.assertIn('user_id == "user-1"', expression)
        self.assertIn('knowledge_base_id in ["kb-1", "kb-2"]', expression)
        self.assertIn('index_version in ["version-1", "version-2"]', expression)

    def test_root_credentials_are_rejected(self):
        with (
            patch("app.services.knowledge.milvus.settings.MILVUS_URI", "http://milvus:19530"),
            patch("app.services.knowledge.milvus.settings.MILVUS_USERNAME", "root"),
            patch("app.services.knowledge.milvus.settings.MILVUS_PASSWORD", "secret"),
            patch("app.services.knowledge.milvus.settings.MILVUS_DATABASE", "fusion"),
            self.assertRaises(KnowledgeVectorError) as raised,
        ):
            MilvusKnowledgeStore._build_client()

        self.assertEqual(raised.exception.code, "KNOWLEDGE_VECTOR_CONFIG_INVALID")

    def test_existing_collection_schema_mismatch_fails_closed(self):
        with self.assertRaises(KnowledgeVectorError) as raised:
            MilvusKnowledgeStore._validate_collection({"fields": []}, 1024)
        self.assertEqual(raised.exception.code, "KNOWLEDGE_VECTOR_SCHEMA_MISMATCH")


class MilvusKnowledgeStoreBatchTests(unittest.IsolatedAsyncioTestCase):
    class FakeClient:
        def __init__(self):
            self.upsert_sizes = []
            self.readback_sizes = []

        def upsert(self, *, data, **_kwargs):
            self.upsert_sizes.append(len(data))
            return {"upsert_count": len(data)}

        def get(self, *, ids, **_kwargs):
            self.readback_sizes.append(len(ids))
            return [{"chunk_id": chunk_id} for chunk_id in ids]

        def close(self):
            return None

    async def test_upsert_payload_and_strong_readback_are_bounded_by_embedding_batch_size(self):
        client = self.FakeClient()
        store = MilvusKnowledgeStore(client_factory=lambda: client)
        store.ensure_collection = AsyncMock(return_value="knowledge_v1_d2")
        profile = EmbeddingProfile("litellm", "embed-v1", 2, "COSINE", "knowledge_v1_d2", "r1")
        records = [self._record(index) for index in range(5)]

        with patch("app.services.knowledge.milvus.settings.KNOWLEDGE_EMBEDDING_BATCH_SIZE", 2):
            await store.upsert(profile, records)

        store.ensure_collection.assert_awaited_once_with(profile)
        self.assertEqual(client.upsert_sizes, [2, 2, 1])
        self.assertEqual(client.readback_sizes, [2, 2, 1])

    async def test_prepared_upsert_rejects_collection_from_another_profile(self):
        store = MilvusKnowledgeStore(client_factory=self.FakeClient)
        profile = EmbeddingProfile("litellm", "embed-v1", 2, "COSINE", "knowledge_v1_d2", "r1")

        with self.assertRaises(KnowledgeVectorError) as raised:
            await store.upsert_prepared(profile, "knowledge_v1_d3", [self._record(0)])

        self.assertEqual(raised.exception.code, "KNOWLEDGE_VECTOR_CONFIG_INVALID")
        self.assertFalse(raised.exception.retryable)

    @staticmethod
    def _record(index: int) -> KnowledgeVectorRecord:
        chunk = KnowledgeChunk(
            chunk_id=f"{index:064x}",
            ordinal=index,
            text=f"chunk-{index}",
            char_start=index * 10,
            char_end=index * 10 + 7,
            page=None,
            section=None,
        )
        return KnowledgeVectorRecord(
            chunk=chunk,
            vector=[1.0, 0.5],
            user_id="user-1",
            knowledge_base_id="kb-1",
            document_id="doc-1",
            index_version="version-1",
            filename="manual.txt",
        )


if __name__ == "__main__":
    unittest.main()
