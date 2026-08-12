import unittest
from unittest.mock import patch

from app.services.knowledge.milvus import KnowledgeVectorError, MilvusKnowledgeStore


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


if __name__ == "__main__":
    unittest.main()
