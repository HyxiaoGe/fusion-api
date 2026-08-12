import json
import unittest

from app.core.config import Settings


def enabled_settings(**overrides):
    values = {
        "DATABASE_URL": "sqlite:///:memory:",
        "KNOWLEDGE_BASE_ENABLED": True,
        "KNOWLEDGE_EMBEDDING_PROVIDER": "litellm",
        "KNOWLEDGE_EMBEDDING_MODEL": "embedding-v1",
        "KNOWLEDGE_EMBEDDING_REVISION": "embedding-v1-r1",
        "KNOWLEDGE_EMBEDDING_REVISION_ROUTES": json.dumps(
            {"embedding-v1@embedding-v1-r1": "embedding-v1-r1-immutable"}
        ),
        "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS": 30,
        "LITELLM_PROXY_URL": "http://litellm-proxy:4000",
        "LITELLM_API_KEY": "proxy-key",
        "KNOWLEDGE_EMBEDDING_DIMENSION": 1024,
        "KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS": "1024,1536",
        "KNOWLEDGE_DISTANCE_METRIC": "COSINE",
        "KNOWLEDGE_CHUNK_SIZE": 1200,
        "KNOWLEDGE_CHUNK_OVERLAP": 200,
        "KNOWLEDGE_WORKER_LEASE_SECONDS": 180,
        "KNOWLEDGE_WORKER_HEARTBEAT_SECONDS": 30,
        "KNOWLEDGE_WORKER_RETRY_BASE_SECONDS": 5,
        "KNOWLEDGE_WORKER_RETRY_MAX_SECONDS": 300,
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

    def test_resource_and_chunk_progress_bounds_fail_closed(self):
        invalid_cases = (
            ("KNOWLEDGE_MAX_FILE_SIZE", 0, "MAX_FILE_SIZE"),
            ("KNOWLEDGE_MAX_FILE_SIZE", 50 * 1024 * 1024 + 1, "MAX_FILE_SIZE"),
            ("KNOWLEDGE_WORKER_POLL_SECONDS", 0, "POLL_SECONDS"),
            ("KNOWLEDGE_WORKER_POLL_SECONDS", 61, "POLL_SECONDS"),
            ("KNOWLEDGE_WORKER_RETRY_BASE_SECONDS", 0, "RETRY_BASE_SECONDS"),
            ("KNOWLEDGE_WORKER_RETRY_MAX_SECONDS", 3601, "RETRY_MAX_SECONDS"),
            ("KNOWLEDGE_CHUNK_OVERLAP", 601, "CHUNK_OVERLAP"),
        )
        for name, value, message in invalid_cases:
            with self.subTest(name=name, value=value), self.assertRaises(ValueError) as raised:
                enabled_settings(**{name: value}).validate_knowledge_base_configuration()
            self.assertIn(message, str(raised.exception))

        with self.assertRaises(ValueError) as raised:
            enabled_settings(
                KNOWLEDGE_CHUNK_SIZE=150, KNOWLEDGE_CHUNK_OVERLAP=60
            ).validate_knowledge_base_configuration()
        self.assertIn("最小步长", str(raised.exception))

        with self.assertRaises(ValueError) as raised:
            enabled_settings(
                KNOWLEDGE_WORKER_RETRY_BASE_SECONDS=301,
                KNOWLEDGE_WORKER_RETRY_MAX_SECONDS=300,
            ).validate_knowledge_base_configuration()
        self.assertIn("base <= max", str(raised.exception))

        enabled_settings(
            KNOWLEDGE_WORKER_RETRY_BASE_SECONDS=3600,
            KNOWLEDGE_WORKER_RETRY_MAX_SECONDS=3600,
        ).validate_knowledge_base_configuration()

    def test_embedding_timeout_and_revision_registry_fail_closed(self):
        invalid_cases = (
            ({"KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS": 0}, "TIMEOUT_SECONDS"),
            ({"KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS": 121}, "TIMEOUT_SECONDS"),
            ({"MILVUS_TIMEOUT_SECONDS": 0}, "MILVUS_TIMEOUT_SECONDS"),
            ({"MILVUS_TIMEOUT_SECONDS": 61}, "MILVUS_TIMEOUT_SECONDS"),
            ({"KNOWLEDGE_EMBEDDING_REVISION_ROUTES": "not-json"}, "REVISION_ROUTES"),
            (
                {
                    "KNOWLEDGE_EMBEDDING_REVISION_ROUTES": json.dumps(
                        {"other-model@embedding-v1-r1": "other-alias-immutable"}
                    )
                },
                "持久化 model@revision",
            ),
        )
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError) as raised:
                enabled_settings(**overrides).validate_knowledge_base_configuration()
            self.assertIn(message, str(raised.exception))

    def test_historical_embedding_revision_resolves_to_persisted_immutable_alias(self):
        configured = enabled_settings(
            KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(
                {
                    "embedding-v0@embedding-v0-r1": "embedding-v0-r1-immutable",
                    "embedding-v1@embedding-v1-r1": "embedding-v1-r1-immutable",
                }
            )
        )

        self.assertEqual(
            configured.resolve_knowledge_embedding_route("embedding-v0", "embedding-v0-r1"),
            "embedding-v0-r1-immutable",
        )
        with self.assertRaises(ValueError):
            configured.resolve_knowledge_embedding_route("embedding-v0", "deleted-revision")

    def test_profile_identity_rejects_surrounding_whitespace(self):
        configured = enabled_settings(
            KNOWLEDGE_EMBEDDING_MODEL=" embedding-v1",
            KNOWLEDGE_EMBEDDING_REVISION="embedding-v1-r1 ",
        )

        with self.assertRaises(ValueError) as raised:
            configured.validate_knowledge_base_configuration()

        self.assertIn("KNOWLEDGE_EMBEDDING_MODEL 前后不能包含空白", str(raised.exception))
        self.assertIn("KNOWLEDGE_EMBEDDING_REVISION 前后不能包含空白", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
