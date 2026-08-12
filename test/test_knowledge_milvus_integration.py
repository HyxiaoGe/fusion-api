import os
import unittest
import uuid

from app.ai.embeddings.base import EmbeddingProfile
from app.core.config import settings
from app.services.knowledge.chunker import KnowledgeChunk
from app.services.knowledge.milvus import KnowledgeVectorRecord, MilvusKnowledgeStore


@unittest.skipUnless(
    os.getenv("RUN_KNOWLEDGE_MILVUS_INTEGRATION") == "1",
    "需要显式启用真实 Milvus 验收",
)
class KnowledgeMilvusIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_non_root_milvus_2_6_21_write_search_and_cleanup(self):
        self.assertNotEqual(settings.MILVUS_USERNAME.casefold(), "root")
        settings.validate_knowledge_base_configuration()
        store = MilvusKnowledgeStore()
        client = store._build_client()
        self.assertEqual(
            client.get_server_version(timeout=settings.MILVUS_TIMEOUT_SECONDS),
            os.getenv("KNOWLEDGE_MILVUS_EXPECTED_VERSION", "2.6.21"),
        )
        profile = EmbeddingProfile(
            settings.KNOWLEDGE_EMBEDDING_PROVIDER,
            settings.KNOWLEDGE_EMBEDDING_MODEL,
            settings.KNOWLEDGE_EMBEDDING_DIMENSION,
            settings.KNOWLEDGE_DISTANCE_METRIC,
        )
        suffix = uuid.uuid4().hex
        user_id = f"integration-user-{suffix}"
        knowledge_base_id = f"integration-base-{suffix}"
        other_user_id = f"integration-other-user-{suffix}"
        other_base_id = f"integration-other-base-{suffix}"
        document_id = f"integration-doc-{suffix}"
        index_version = str(uuid.uuid4())
        vector = [0.0] * profile.dimension
        vector[0] = 1.0
        chunk = KnowledgeChunk(
            chunk_id=uuid.uuid4().hex + uuid.uuid4().hex,
            ordinal=0,
            text="真实 Milvus 集成验收文本",
            char_start=0,
            char_end=14,
            page=1,
            section="integration",
        )
        record = KnowledgeVectorRecord(
            chunk=chunk,
            vector=vector,
            user_id=user_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            index_version=index_version,
            filename="integration.txt",
        )

        try:
            await store.upsert(profile, [record])
            hits = await store.search(
                profile=profile,
                query_vector=vector,
                user_id=user_id,
                knowledge_base_ids=[knowledge_base_id],
                index_versions=[index_version],
                limit=5,
            )
            self.assertEqual([hit.chunk_id for hit in hits], [chunk.chunk_id])
            self.assertEqual(hits[0].text, chunk.text)

            for denied_user, denied_base in (
                (other_user_id, knowledge_base_id),
                (user_id, other_base_id),
            ):
                denied = await store.search(
                    profile=profile,
                    query_vector=vector,
                    user_id=denied_user,
                    knowledge_base_ids=[denied_base],
                    index_versions=[index_version],
                    limit=5,
                )
                self.assertEqual(denied, [])
        finally:
            await store.delete_index_version(profile, index_version)

        cleaned = await store.search(
            profile=profile,
            query_vector=vector,
            user_id=user_id,
            knowledge_base_ids=[knowledge_base_id],
            index_versions=[index_version],
            limit=5,
        )
        self.assertEqual(cleaned, [])
        client.close()


if __name__ == "__main__":
    unittest.main()
