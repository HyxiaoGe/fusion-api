import importlib
import os
import sys
import unittest
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")
os.environ.setdefault("SERVER_HOST", "http://dev.example:8002")
os.environ.setdefault("FRONTEND_URL", "http://dev.example:3004")
os.environ.setdefault("AUTH_SERVICE_BASE_URL", "http://auth.example:8100")
os.environ.setdefault("AUTH_SERVICE_CLIENT_ID", "fusion-client")
os.environ.setdefault("AUTH_SERVICE_JWKS_URL", "http://auth.example:8100/.well-known/jwks.json")

from fastapi.testclient import TestClient

from app.schemas.response import ApiException, ApiResponse, ErrorCode, generate_request_id, success


class TestGenerateRequestId(unittest.TestCase):
    def test_format(self):
        rid = generate_request_id()
        self.assertEqual(len(rid), 32)  # uuid4 hex, 32 chars
        int(rid, 16)  # 应为合法的十六进制字符串

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

    def test_bad_request_factory(self):
        exc = ApiException.bad_request("字段 X 无效")
        self.assertEqual(exc.code, "INVALID_PARAM")
        self.assertEqual(exc.status_code, 400)
        self.assertEqual(exc.message, "字段 X 无效")

    def test_not_found_factory(self):
        exc = ApiException.not_found("会话不存在")
        self.assertEqual(exc.code, "NOT_FOUND")
        self.assertEqual(exc.status_code, 404)

    def test_not_found_default_message(self):
        exc = ApiException.not_found()
        self.assertEqual(exc.message, "资源不存在")

    def test_conflict_factory(self):
        exc = ApiException.conflict("ID 已存在")
        self.assertEqual(exc.code, "CONFLICT")
        self.assertEqual(exc.status_code, 409)

    def test_internal_error_factory(self):
        exc = ApiException.internal_error()
        self.assertEqual(exc.code, "INTERNAL_ERROR")
        self.assertEqual(exc.status_code, 500)


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
        self.assertEqual(len(response.headers["x-request-id"]), 32)

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

        # 从路由的实际依赖中提取函数引用，避免模块重导入导致函数对象不一致
        cls._dep_overrides = {}
        for route in main.app.routes:
            if hasattr(route, "path") and route.path == "/api/chat/conversations/{conversation_id}":
                for dep in route.dependant.dependencies:
                    cls._dep_overrides[dep.call.__qualname__] = dep.call
                break

        cls.fake_user = SimpleNamespace(id="user-123")

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def _auth(self):
        gcu = self._dep_overrides["get_current_user"]
        self.main.app.dependency_overrides[gcu] = lambda: self.fake_user

    def test_http_exception_returns_unified_format(self):
        self._auth()
        gcs = self._dep_overrides["get_chat_service"]
        mock_svc = SimpleNamespace(get_conversation=lambda *a, **kw: None)
        self.main.app.dependency_overrides[gcs] = lambda: mock_svc

        response = self.client.get("/api/chat/conversations/nonexistent")
        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "NOT_FOUND")
        self.assertEqual(body["message"], "会话不存在或无权访问")
        self.assertIsNone(body["data"])
        self.assertEqual(len(body["request_id"]), 32)

    def test_401_returns_unauthorized_code(self):
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertEqual(body["code"], "UNAUTHORIZED")
        self.assertIsNone(body["data"])
        self.assertIn("request_id", body)


class TestValueErrorHandler(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        main.init_db = lambda: None
        cls.main = main
        cls.client = TestClient(main.app)

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def test_value_error_returns_400(self):
        """ValueError 应返回 400 而非 500"""
        from fastapi import APIRouter

        test_router = APIRouter()

        @test_router.get("/test-value-error")
        async def raise_value_error():
            raise ValueError("模型不存在")

        self.main.app.include_router(test_router)
        try:
            response = self.client.get("/test-value-error")
            self.assertEqual(response.status_code, 400)
            body = response.json()
            self.assertEqual(body["code"], "INVALID_PARAM")
            self.assertEqual(body["message"], "模型不存在")
            self.assertIn("request_id", body)
        finally:
            # 清理测试路由
            self.main.app.routes[:] = [r for r in self.main.app.routes if not getattr(r, 'path', '').endswith('/test-value-error')]
