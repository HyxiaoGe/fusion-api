import unittest

from app.core.config import Settings


def enabled_settings(**overrides):
    values = {
        "DATABASE_URL": "sqlite:///:memory:",
        "KNOWLEDGE_BASE_ENABLED": True,
        "KNOWLEDGE_EMBEDDING_PROVIDER": "litellm",
        "KNOWLEDGE_EMBEDDING_MODEL": "embedding-v1",
        "KNOWLEDGE_EMBEDDING_REVISION": "embedding-v1-r1",
        "LITELLM_PROXY_URL": "http://litellm-proxy:4000",
        "LITELLM_API_KEY": "proxy-key",
        "KNOWLEDGE_EMBEDDING_DIMENSION": 1024,
        "KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS": "1024,1536",
        "KNOWLEDGE_DISTANCE_METRIC": "COSINE",
        "KNOWLEDGE_CHUNK_SIZE": 1200,
        "KNOWLEDGE_CHUNK_OVERLAP": 200,
        "KNOWLEDGE_WORKER_LEASE_SECONDS": 180,
        "KNOWLEDGE_WORKER_HEARTBEAT_SECONDS": 30,
        "MILVUS_URI": "http://milvus:19530",
        "MILVUS_USERNAME": "fusion_knowledge",
        "MILVUS_PASSWORD": "secret",
        "MILVUS_DATABASE": "fusion_knowledge",
        "MILVUS_COLLECTION_PREFIX": "fusion_knowledge_chunks",
    }
    values.update(overrides)
    return Settings(**values)


class KnowledgeConfigTests(unittest.TestCase):
    def test_valid_non_root_application_configuration_is_accepted(self):
        enabled_settings().validate_knowledge_base_configuration()

    def test_root_account_and_invalid_heartbeat_fail_closed(self):
        configured = enabled_settings(
            MILVUS_USERNAME="root",
            KNOWLEDGE_WORKER_HEARTBEAT_SECONDS=100,
        )

        with self.assertRaises(ValueError) as raised:
            configured.validate_knowledge_base_configuration()

        self.assertIn("root", str(raised.exception))
        self.assertIn("HEARTBEAT", str(raised.exception))

    def test_disabled_feature_does_not_require_external_dependencies(self):
        configured = enabled_settings(
            KNOWLEDGE_BASE_ENABLED=False,
            MILVUS_URI="",
            MILVUS_USERNAME="",
            MILVUS_PASSWORD="",
            MILVUS_DATABASE="",
        )

        configured.validate_knowledge_base_configuration()


if __name__ == "__main__":
    unittest.main()
