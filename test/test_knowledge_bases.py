import importlib
import os
import sys
import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")


class FakeKnowledgeService:
    def __init__(self):
        self.calls = []

    def create_knowledge_base(self, user_id, payload):
        self.calls.append(("create", user_id, payload))
        return self._base(payload.name)

    def list_knowledge_bases(self, user_id, page, page_size):
        self.calls.append(("list", user_id, page, page_size))
        return {
            "items": [],
            "page": page,
            "page_size": page_size,
            "total": 0,
            "total_pages": 0,
            "has_next": False,
            "has_prev": page > 1,
        }

    def get_knowledge_base(self, user_id, knowledge_base_id):
        return self._base("手册", knowledge_base_id)

    def update_knowledge_base(self, user_id, knowledge_base_id, payload):
        return self._base(payload.name, knowledge_base_id)

    def delete_knowledge_base(self, user_id, knowledge_base_id):
        return self._task("task-delete-base", "delete_knowledge_base")

    async def upload_document(self, user_id, knowledge_base_id, file):
        self.calls.append(("upload", user_id, knowledge_base_id, file.filename))
        return {
            "document": self._document(knowledge_base_id),
            "task": self._task("task-1", "index_document"),
        }

    def list_documents(self, user_id, knowledge_base_id, page, page_size):
        return {
            "items": [],
            "page": page,
            "page_size": page_size,
            "total": 0,
            "total_pages": 0,
            "has_next": False,
            "has_prev": page > 1,
        }

    def get_document(self, user_id, knowledge_base_id, document_id):
        return self._document(knowledge_base_id, document_id)

    def delete_document(self, user_id, knowledge_base_id, document_id):
        return self._task("task-delete-document", "delete_document")

    def retry_document(self, user_id, knowledge_base_id, document_id):
        return self._task("task-retry", "index_document")

    def rebuild_document(self, user_id, knowledge_base_id, document_id):
        return self._task("task-rebuild", "index_document")

    def get_task(self, user_id, task_id):
        return self._task(task_id, "index_document")

    async def retrieve(self, user_id, payload):
        self.calls.append(("search", user_id, payload))
        return {"hits": [], "query": payload.query, "top_k": payload.top_k}

    @staticmethod
    def _base(name, knowledge_base_id="kb-1"):
        return {
            "id": knowledge_base_id,
            "name": name,
            "description": "",
            "business_type": "product",
            "status": "active",
            "document_stats": {"total": 0, "ready": 0, "processing": 0, "failed": 0},
            "embedding_provider": "litellm",
            "embedding_model": "embed-v1",
            "embedding_revision": "embed-r1",
            "embedding_dimension": 2,
            "distance_metric": "COSINE",
            "created_at": "2026-08-12T00:00:00Z",
            "updated_at": "2026-08-12T00:00:00Z",
            "deleted_at": None,
        }

    @staticmethod
    def _task(task_id, task_type):
        return {
            "id": task_id,
            "task_type": task_type,
            "status": "pending",
            "phase": "queued",
            "attempt_count": 0,
            "max_attempts": 5,
            "error_code": None,
            "error_summary": None,
            "created_at": "2026-08-12T00:00:00Z",
            "updated_at": "2026-08-12T00:00:00Z",
        }

    @staticmethod
    def _document(knowledge_base_id, document_id="doc-1"):
        return {
            "id": document_id,
            "knowledge_base_id": knowledge_base_id,
            "filename": "manual.txt",
            "mimetype": "text/plain",
            "size": 7,
            "checksum_sha256": "a" * 64,
            "status": "queued",
            "parser_version": "parser-v1",
            "chunker_version": "chunker-v1",
            "embedding_provider": "litellm",
            "embedding_model": "embed-v1",
            "embedding_revision": "embed-r1",
            "embedding_dimension": 2,
            "distance_metric": "COSINE",
            "desired_index_version": "version-1",
            "active_index_version": None,
            "chunk_count": 0,
            "error_code": None,
            "error_summary": None,
            "attempt_count": 0,
            "created_at": "2026-08-12T00:00:00Z",
            "updated_at": "2026-08-12T00:00:00Z",
            "ready_at": None,
            "deleted_at": None,
        }


class KnowledgeBasesApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        from app.api.deps import get_current_user, get_knowledge_service

        self.service = FakeKnowledgeService()
        self.main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1")
        self.main.app.dependency_overrides[get_knowledge_service] = lambda: self.service

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def test_crud_and_async_status_contract(self):
        created = self.client.post(
            "/api/knowledge-bases/",
            json={"name": "手册", "description": "", "business_type": "product"},
        )
        listed = self.client.get("/api/knowledge-bases/?page=1&page_size=20")
        deleted = self.client.delete("/api/knowledge-bases/kb-1")
        uploaded = self.client.post(
            "/api/knowledge-bases/kb-1/documents",
            files={"file": ("manual.txt", b"content", "text/plain")},
        )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(deleted.status_code, 202)
        self.assertEqual(uploaded.status_code, 202)
        self.assertIn(("upload", "user-1", "kb-1", "manual.txt"), self.service.calls)

    def test_search_route_is_not_shadowed_by_dynamic_detail_route(self):
        response = self.client.post(
            "/api/knowledge-bases/search",
            json={"knowledge_base_ids": ["kb-1"], "query": "配置方法", "top_k": 5},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["query"], "配置方法")

    def test_openapi_contains_complete_knowledge_surface(self):
        document = self.client.get("/openapi.json").json()
        paths = document["paths"]

        expected = {
            "/api/knowledge-bases/",
            "/api/knowledge-bases/{knowledge_base_id}",
            "/api/knowledge-bases/{knowledge_base_id}/documents",
            "/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}",
            "/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}/retry",
            "/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}/rebuild",
            "/api/knowledge-bases/tasks/{task_id}",
            "/api/knowledge-bases/search",
        }
        self.assertTrue(expected.issubset(paths))
        create_responses = paths["/api/knowledge-bases/"]["post"]["responses"]
        self.assertIn("201", create_responses)
        self.assertIn("409", create_responses)
        self.assertIn("503", create_responses)
        self.assertIn("content", create_responses["201"])

    def test_validation_errors_use_api_envelope(self):
        response = self.client.post(
            "/api/knowledge-bases/",
            json={"name": None, "description": "", "business_type": "product"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["code"], "INVALID_PARAM")
        self.assertIsNone(response.json()["data"])


if __name__ == "__main__":
    unittest.main()
