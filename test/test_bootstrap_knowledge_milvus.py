import os
import sys
import types
import unittest
from unittest.mock import patch

from scripts.bootstrap_knowledge_milvus import bootstrap


class _FakeMilvusClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls = []
        self.__class__.instances.append(self)

    def list_databases(self):
        return []

    def create_database(self, **kwargs):
        self.calls.append(("create_database", kwargs))

    def list_users(self):
        return []

    def create_user(self, **kwargs):
        self.calls.append(("create_user", kwargs))

    def list_roles(self):
        return []

    def create_role(self, **kwargs):
        self.calls.append(("create_role", kwargs))

    def describe_user(self, **kwargs):
        self.calls.append(("describe_user", kwargs))
        return {"roles": []}

    def grant_role(self, **kwargs):
        self.calls.append(("grant_role", kwargs))

    def grant_privilege_v2(self, **kwargs):
        self.calls.append(("grant_privilege_v2", kwargs))

    def list_collections(self):
        self.calls.append(("list_collections", {}))
        return []


class KnowledgeMilvusBootstrapTests(unittest.TestCase):
    def setUp(self):
        _FakeMilvusClient.instances = []

    def test_bootstrap_grants_database_and_collection_read_write(self):
        fake_module = types.SimpleNamespace(MilvusClient=_FakeMilvusClient)
        environment = {
            "MILVUS_URI": "http://milvus.example:19530",
            "MILVUS_BOOTSTRAP_PASSWORD": "root-secret",
            "MILVUS_USERNAME": "fusion_knowledge",
            "MILVUS_PASSWORD": "application-secret",
            "MILVUS_DATABASE": "fusion_knowledge",
        }

        with (
            patch.dict(os.environ, environment, clear=True),
            patch.dict(
                sys.modules,
                {"pymilvus": fake_module},
            ),
        ):
            bootstrap()

        self.assertEqual(len(_FakeMilvusClient.instances), 2)
        admin, application = _FakeMilvusClient.instances
        grants = [kwargs for name, kwargs in admin.calls if name == "grant_privilege_v2"]
        self.assertEqual(
            grants,
            [
                {
                    "role_name": "fusion_knowledge_role",
                    "privilege": "DatabaseReadWrite",
                    "collection_name": "*",
                    "db_name": "fusion_knowledge",
                },
                {
                    "role_name": "fusion_knowledge_role",
                    "privilege": "CollectionReadWrite",
                    "collection_name": "*",
                    "db_name": "fusion_knowledge",
                },
            ],
        )
        self.assertEqual(application.kwargs["user"], "fusion_knowledge")
        self.assertEqual(application.kwargs["db_name"], "fusion_knowledge")
        self.assertIn(("list_collections", {}), application.calls)


if __name__ == "__main__":
    unittest.main()
