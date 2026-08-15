import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from app.schemas.knowledge import normalize_knowledge_base_name
from app.schemas.response import ApiException, ErrorCode

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
        self.calls.append(("update", user_id, knowledge_base_id, payload))
        return self._base(payload.name or "手册", knowledge_base_id)

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

    def list_document_chunks(self, user_id, knowledge_base_id, document_id, page, page_size):
        self.calls.append(("list_chunks", user_id, knowledge_base_id, document_id, page, page_size))
        return {
            "document_id": document_id,
            "active_index_version": "version-1",
            "chunker_version": "chunker-v2",
            "chunk_size": 1200,
            "chunk_overlap": 200,
            "items": [
                {
                    "chunk_id": "chunk-1",
                    "ordinal": 0,
                    "text": "正文",
                    "char_start": 0,
                    "char_end": 2,
                    "page": None,
                    "section": None,
                }
            ],
            "page": page,
            "page_size": page_size,
            "total": 1,
            "total_pages": 1,
            "has_next": False,
            "has_prev": page > 1,
        }

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
            files={"file": ("manual.TXT", b"content", "text/plain")},
        )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(deleted.status_code, 202)
        self.assertEqual(uploaded.status_code, 202)
        self.assertIn(("upload", "user-1", "kb-1", "manual.TXT"), self.service.calls)

    def test_document_chunks_contract_and_pagination_validation(self):
        response = self.client.get("/api/knowledge-bases/kb-1/documents/doc-1/chunks?page=1&page_size=20")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["active_index_version"], "version-1")
        self.assertEqual(response.json()["data"]["items"][0]["text"], "正文")
        self.assertIn(("list_chunks", "user-1", "kb-1", "doc-1", 1, 20), self.service.calls)

        invalid = self.client.get("/api/knowledge-bases/kb-1/documents/doc-1/chunks?page=1&page_size=51")
        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(invalid.json()["code"], "INVALID_PARAM")

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
            "/api/knowledge-bases/{knowledge_base_id}/documents/{document_id}/chunks",
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

    def test_expanded_normalized_name_is_rejected_for_create_and_patch(self):
        expanding_name = "ﬃ" * 120

        created = self.client.post(
            "/api/knowledge-bases/",
            json={"name": expanding_name, "description": "", "business_type": "product"},
        )
        updated = self.client.patch(
            "/api/knowledge-bases/kb-1",
            json={"name": expanding_name},
        )

        self.assertEqual(created.status_code, 422)
        self.assertEqual(updated.status_code, 422)
        self.assertEqual(created.json()["code"], "INVALID_PARAM")
        self.assertEqual(updated.json()["code"], "INVALID_PARAM")

    def test_knowledge_text_fields_reject_nul_with_validation_envelope(self):
        create_payload = {"name": "手册", "description": "说明", "business_type": "product"}
        for field in ("name", "description", "business_type"):
            payload = dict(create_payload)
            payload[field] += "\x00invalid"
            response = self.client.post("/api/knowledge-bases/", json=payload)

            self.assertEqual(response.status_code, 422, field)
            self.assertEqual(response.json()["code"], "INVALID_PARAM", field)
            self.assertIsNone(response.json()["data"], field)

        for field, value in (
            ("name", "手册\x00invalid"),
            ("description", "说明\x00invalid"),
            ("business_type", "product\x00invalid"),
        ):
            response = self.client.patch(
                "/api/knowledge-bases/kb-1",
                json={field: value},
            )

            self.assertEqual(response.status_code, 422, field)
            self.assertEqual(response.json()["code"], "INVALID_PARAM", field)
            self.assertIsNone(response.json()["data"], field)

        for payload in (
            {"knowledge_base_ids": ["kb-1"], "query": "配置\x00invalid", "top_k": 5},
            {"knowledge_base_ids": ["kb-1\x00invalid"], "query": "配置", "top_k": 5},
        ):
            response = self.client.post("/api/knowledge-bases/search", json=payload)

            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["code"], "INVALID_PARAM")
            self.assertIsNone(response.json()["data"])

        self.assertEqual(self.service.calls, [])

    def test_knowledge_path_ids_reject_encoded_nul_before_service_call(self):
        self.service.get_knowledge_base = MagicMock(side_effect=AssertionError("service must not run"))
        self.service.get_document = MagicMock(side_effect=AssertionError("service must not run"))
        self.service.get_task = MagicMock(side_effect=AssertionError("service must not run"))

        for path in (
            "/api/knowledge-bases/kb%00invalid",
            "/api/knowledge-bases/kb-1/documents/doc%00invalid",
            "/api/knowledge-bases/tasks/task%00invalid",
        ):
            response = self.client.get(path)

            self.assertEqual(response.status_code, 422, path)
            self.assertEqual(response.json()["code"], "INVALID_PARAM", path)
            self.assertIsNone(response.json()["data"], path)
        self.service.get_knowledge_base.assert_not_called()
        self.service.get_document.assert_not_called()
        self.service.get_task.assert_not_called()

    def test_knowledge_text_normalization_contract_is_preserved(self):
        created = self.client.post(
            "/api/knowledge-bases/",
            json={"name": "  ＡＢＣ  ", "description": "  说明  ", "business_type": "product"},
        )
        searched = self.client.post(
            "/api/knowledge-bases/search",
            json={"knowledge_base_ids": ["kb-1"], "query": "  配置方法  ", "top_k": 5},
        )

        self.assertEqual(created.status_code, 201)
        self.assertEqual(created.json()["data"]["knowledge_base"]["name"], "ＡＢＣ")
        self.assertEqual(searched.status_code, 200)
        self.assertEqual(searched.json()["data"]["query"], "配置方法")
        self.assertEqual(normalize_knowledge_base_name("  ＡＢＣ  "), "abc")
        create_call = next(call for call in self.service.calls if call[0] == "create")
        self.assertEqual(create_call[2].description, "说明")

    def test_upload_type_mismatch_uses_stable_api_error(self):
        async def reject_upload(*_args):
            raise ApiException(
                ErrorCode.KNOWLEDGE_DOCUMENT_TYPE_MISMATCH,
                "文档 MIME 与扩展名不一致",
                400,
            )

        self.service.upload_document = reject_upload

        for filename in ("README", "note.md"):
            response = self.client.post(
                "/api/knowledge-bases/kb-1/documents",
                files={"file": (filename, b"content", "text/plain")},
            )
            self.assertEqual(response.status_code, 400)
            self.assertEqual(response.json()["code"], "KNOWLEDGE_DOCUMENT_TYPE_MISMATCH")


if __name__ == "__main__":
    unittest.main()
