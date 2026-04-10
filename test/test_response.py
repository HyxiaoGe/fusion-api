import unittest

from app.schemas.response import ApiException, ApiResponse, ErrorCode, generate_request_id, success


class TestGenerateRequestId(unittest.TestCase):
    def test_format(self):
        rid = generate_request_id()
        self.assertTrue(rid.startswith("req_"))
        self.assertEqual(len(rid), 16)  # "req_" + 12 hex chars

    def test_unique(self):
        ids = {generate_request_id() for _ in range(100)}
        self.assertEqual(len(ids), 100)


class TestErrorCode(unittest.TestCase):
    def test_success_value(self):
        self.assertEqual(ErrorCode.SUCCESS, "SUCCESS")
        self.assertEqual(ErrorCode.NOT_FOUND, "NOT_FOUND")

    def test_is_string(self):
        self.assertIsInstance(ErrorCode.SUCCESS, str)


class TestApiResponse(unittest.TestCase):
    def test_success_with_data(self):
        resp = ApiResponse(data={"key": "value"}, request_id="req_abc123def456")
        self.assertEqual(resp.code, "SUCCESS")
        self.assertEqual(resp.message, "ok")
        self.assertEqual(resp.data, {"key": "value"})
        self.assertEqual(resp.request_id, "req_abc123def456")

    def test_success_null_data(self):
        resp = ApiResponse(request_id="req_abc123def456")
        self.assertIsNone(resp.data)

    def test_serialization(self):
        resp = ApiResponse(data=[1, 2, 3], request_id="req_abc123def456")
        d = resp.model_dump()
        self.assertEqual(d["code"], "SUCCESS")
        self.assertEqual(d["data"], [1, 2, 3])


class TestSuccessHelper(unittest.TestCase):
    def test_default(self):
        resp = success(request_id="req_abc123def456")
        self.assertEqual(resp.code, "SUCCESS")
        self.assertEqual(resp.message, "ok")
        self.assertIsNone(resp.data)

    def test_with_data_and_message(self):
        resp = success(data={"id": 1}, message="已创建", request_id="req_abc123def456")
        self.assertEqual(resp.message, "已创建")
        self.assertEqual(resp.data, {"id": 1})


class TestApiException(unittest.TestCase):
    def test_fields(self):
        exc = ApiException(ErrorCode.NOT_FOUND, "会话不存在", 404)
        self.assertEqual(exc.code, "NOT_FOUND")
        self.assertEqual(exc.message, "会话不存在")
        self.assertEqual(exc.status_code, 404)

    def test_default_status_code(self):
        exc = ApiException(ErrorCode.INVALID_PARAM, "参数错误")
        self.assertEqual(exc.status_code, 400)


if __name__ == "__main__":
    unittest.main()


import importlib
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")
os.environ.setdefault("SERVER_HOST", "http://dev.example:8002")
os.environ.setdefault("FRONTEND_URL", "http://dev.example:3004")
os.environ.setdefault("AUTH_SERVICE_BASE_URL", "http://auth.example:8100")
os.environ.setdefault("AUTH_SERVICE_CLIENT_ID", "fusion-client")
os.environ.setdefault("AUTH_SERVICE_JWKS_URL", "http://auth.example:8100/.well-known/jwks.json")

from fastapi.testclient import TestClient


class TestRequestIdMiddleware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        main.init_db = lambda: None
        cls.client = TestClient(main.app)

    def test_health_has_request_id_header(self):
        response = self.client.get("/health")
        self.assertIn("x-request-id", response.headers)
        self.assertTrue(response.headers["x-request-id"].startswith("req_"))

    def test_request_id_is_unique_per_request(self):
        r1 = self.client.get("/health")
        r2 = self.client.get("/health")
        self.assertNotEqual(r1.headers["x-request-id"], r2.headers["x-request-id"])


class TestGlobalExceptionHandlers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        main.init_db = lambda: None
        cls.main = main
        cls.client = TestClient(main.app)

        from app.api import chat as chat_api
        from app.api import files as files_api
        from app.core.security import get_current_user
        from app.db.database import get_db

        cls.chat_api = chat_api
        cls.files_api = files_api
        cls.get_current_user = get_current_user
        cls.get_db = get_db
        cls.fake_user = SimpleNamespace(id="user-123")

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def _auth(self):
        self.main.app.dependency_overrides[self.get_current_user] = lambda: self.fake_user
        self.main.app.dependency_overrides[self.chat_api.get_current_user] = lambda: self.fake_user
        self.main.app.dependency_overrides[self.get_db] = lambda: (yield object())
        self.main.app.dependency_overrides[self.chat_api.get_db] = lambda: (yield object())

    def test_http_exception_returns_unified_format(self):
        self._auth()
        with patch.object(self.chat_api, "ChatService") as cls:
            cls.return_value.get_conversation.return_value = None
            response = self.client.get("/api/chat/conversations/nonexistent")
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "NOT_FOUND")
        self.assertEqual(body["message"], "会话不存在或无权访问")
        self.assertIsNone(body["data"])
        self.assertTrue(body["request_id"].startswith("req_"))

    def test_401_returns_unauthorized_code(self):
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertEqual(body["code"], "UNAUTHORIZED")
        self.assertIsNone(body["data"])
        self.assertIn("request_id", body)
