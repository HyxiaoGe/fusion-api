import asyncio
import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.responses import Response, StreamingResponse
from fastapi.testclient import TestClient
from starlette.requests import Request

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["AUTH_SERVICE_BASE_URL"] = "http://auth.example:8100"
os.environ["AUTH_SERVICE_CLIENT_ID"] = "fusion-client"
os.environ["AUTH_SERVICE_JWKS_URL"] = "http://auth.example:8100/.well-known/jwks.json"


class ChatCoreSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")

        cls.main = main
        cls.client = TestClient(main.app)
        cls.fake_user = SimpleNamespace(id="user-123")

        # 从路由依赖中提取实际函数引用，避免模块重导入导致的函数对象不一致
        cls._route_deps = {}
        for route in main.app.routes:
            if hasattr(route, "dependant"):
                for dep in route.dependant.dependencies:
                    name = dep.call.__qualname__
                    if name not in cls._route_deps:
                        cls._route_deps[name] = dep.call

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def _enable_authenticated_overrides(self):
        gcu = self._route_deps.get("get_current_user")
        if gcu:
            self.main.app.dependency_overrides[gcu] = lambda: self.fake_user

        gdb = self._route_deps.get("get_db")
        if gdb:

            def override_db():
                yield object()

            self.main.app.dependency_overrides[gdb] = override_db

    def test_health_endpoint_stays_available(self):
        class HealthySession:
            def execute(self, statement):
                return None

            def close(self):
                return None

        original_session_local = self.main.SessionLocal
        original_get_redis_pool = self.main.get_redis_pool
        self.main.SessionLocal = HealthySession
        self.main.get_redis_pool = lambda: SimpleNamespace(ping=AsyncMock(return_value=True))
        try:
            response = self.client.get("/health")
        finally:
            self.main.SessionLocal = original_session_local
            self.main.get_redis_pool = original_get_redis_pool

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["service"], "fusion-api")
        self.assertEqual(payload["redis"], "connected")
        self.assertTrue(payload["timestamp"].endswith("+08:00"))

    def test_health_returns_503_when_redis_is_unavailable(self):
        class HealthySession:
            def execute(self, statement):
                return None

            def close(self):
                return None

        original_session_local = self.main.SessionLocal
        original_get_redis_pool = self.main.get_redis_pool
        self.main.SessionLocal = HealthySession
        self.main.get_redis_pool = lambda: None
        try:
            response = self.client.get("/health")
        finally:
            self.main.SessionLocal = original_session_local
            self.main.get_redis_pool = original_get_redis_pool

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["status"], "unhealthy")
        self.assertEqual(payload["database"], "connected")
        self.assertEqual(payload["redis"], "unavailable")

    def test_health_returns_503_when_database_is_unavailable(self):
        class BrokenSession:
            def execute(self, statement):
                raise RuntimeError("database down")

            def close(self):
                return None

        original_session_local = self.main.SessionLocal
        original_get_redis_pool = self.main.get_redis_pool
        self.main.SessionLocal = BrokenSession
        self.main.get_redis_pool = lambda: SimpleNamespace(ping=AsyncMock(return_value=True))
        try:
            response = self.client.get("/health")
        finally:
            self.main.SessionLocal = original_session_local
            self.main.get_redis_pool = original_get_redis_pool

        self.assertEqual(response.status_code, 503)
        payload = response.json()
        self.assertEqual(payload["database"], "unavailable")
        self.assertEqual(payload["redis"], "connected")

    def test_openapi_exposes_shared_auth_surface(self):
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]

        self.assertIn("/api/auth/me", paths)
        self.assertIn("/api/chat/conversations", paths)
        self.assertIn("/api/files/upload", paths)
        self.assertIn("/api/models/", paths)

        self.assertNotIn("/api/auth/login/{provider}", paths)
        self.assertNotIn("/api/auth/callback/{provider}", paths)
        self.assertNotIn("/api/users/profile", paths)
        self.assertNotIn("/api/credentials", paths)
        self.assertNotIn("/api/rss/sources", paths)
        self.assertNotIn("/api/digests", paths)
        self.assertNotIn("/api/web_search/search", paths)
        self.assertNotIn("/api/settings", paths)
        # BYOK / admin / user-credentials 三套 API 已随模型注册表迁移到 LiteLLM 彻底删除
        self.assertNotIn("/api/user/credentials", paths)
        self.assertNotIn("/api/admin/providers/{provider_id}/recover", paths)
        self.assertNotIn("/api/providers", paths)
        # /api/prompts 现已存在（动态示例问题端点 GET /api/prompts/examples）
        # 旧断言"prompts 不该出现"已过时

    def test_auth_me_requires_authentication(self):
        response = self.client.get("/api/auth/me")
        self.assertEqual(response.status_code, 401)
        body = response.json()
        self.assertEqual(body["code"], "UNAUTHORIZED")
        self.assertIsNone(body["data"])

    def test_send_message_routes_to_chat_service(self):
        self._enable_authenticated_overrides()
        payload = {
            "model_id": "gpt-4.1",
            "message": "hello",
            "conversation_id": "conv-1",
            "user_message_id": "11111111-1111-4111-8111-111111111111",
            "assistant_message_id": "22222222-2222-4222-8222-222222222222",
            "stream": False,
            "options": {"temperature": 0.3},
            "file_ids": ["file-1"],
        }

        service = SimpleNamespace()
        service.process_message = AsyncMock(
            return_value={
                "conversation_id": "conv-1",
                "message": {"content": "hi"},
            }
        )
        self.main.app.dependency_overrides[self._route_deps["get_chat_service"]] = lambda: service

        response = self.client.post("/api/chat/send", json=payload)

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["conversation_id"], "conv-1")
        # process_message 在 agent observability 上线后加了 trace_id 参数（每次请求唯一），
        # 用 ANY 兼容；其余参数显式断言。
        from unittest.mock import ANY

        service.process_message.assert_awaited_once_with(
            model_id="gpt-4.1",
            message="hello",
            user_id="user-123",
            conversation_id="conv-1",
            user_message_id="11111111-1111-4111-8111-111111111111",
            assistant_message_id="22222222-2222-4222-8222-222222222222",
            stream=False,
            options={"temperature": 0.3},
            file_ids=["file-1"],
            trace_id=ANY,
        )

    def test_send_message_can_return_streaming_response(self):
        self._enable_authenticated_overrides()

        async def event_stream():
            yield "data: hello\n\n"

        service = SimpleNamespace()
        service.process_message = AsyncMock(
            return_value=StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
            )
        )
        self.main.app.dependency_overrides[self._route_deps["get_chat_service"]] = lambda: service

        response = self.client.post(
            "/api/chat/send",
            json={
                "model_id": "gpt-4.1",
                "message": "stream please",
                "stream": True,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertIn("data: hello", response.text)
        service.process_message.assert_awaited_once()
        self.assertIsNone(service.process_message.await_args.kwargs["user_message_id"])
        self.assertIsNone(service.process_message.await_args.kwargs["assistant_message_id"])

    def test_get_conversations_uses_authenticated_user_id(self):
        self._enable_authenticated_overrides()

        service = SimpleNamespace()
        service.get_conversations_paginated = lambda *a, **kw: {
            "items": [],
            "total": 0,
            "page": 1,
            "page_size": 10,
        }
        self.main.app.dependency_overrides[self._route_deps["get_chat_service"]] = lambda: service

        response = self.client.get("/api/chat/conversations?page=1&page_size=10")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["total"], 0)

    def _mock_file_service(self, **methods):
        """创建文件服务 mock 并注入依赖覆盖"""
        from unittest.mock import MagicMock

        service = MagicMock()
        for name, val in methods.items():
            setattr(service, name, val)
        gfs = self._route_deps.get("get_file_service")
        if gfs:
            self.main.app.dependency_overrides[gfs] = lambda: service
        return service

    def test_file_upload_routes_to_file_service(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service(
            upload_files=AsyncMock(return_value=[{"file_id": "file-1"}, {"file_id": "file-2"}])
        )

        response = self.client.post(
            "/api/files/upload",
            data={
                "provider": "openai",
                "model": "gpt-4.1",
                "conversation_id": "conv-1",
            },
            files=[("files", ("note.txt", b"hello", "text/plain"))],
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["files"], [{"file_id": "file-1"}, {"file_id": "file-2"}])
        service.upload_files.assert_awaited_once()
        args = service.upload_files.await_args
        self.assertEqual(args.args[1:], ("user-123", "conv-1", "openai", "gpt-4.1"))
        self.assertEqual(len(args.args[0]), 1)
        self.assertEqual(args.args[0][0].filename, "note.txt")

    def test_file_upload_value_error_returns_invalid_param(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service(
            upload_files=AsyncMock(side_effect=ValueError("图片文件损坏或无法读取，请重新保存后再上传"))
        )

        response = self.client.post(
            "/api/files/upload",
            data={
                "provider": "openai",
                "model": "gpt-4.1",
                "conversation_id": "conv-1",
            },
            files=[("files", ("broken.png", b"not-an-image", "image/png"))],
        )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["code"], "INVALID_PARAM")
        self.assertEqual(body["message"], "图片文件损坏或无法读取，请重新保存后再上传")
        self.assertIsNone(body["data"])
        service.upload_files.assert_awaited_once()

    def test_file_direct_upload_init_routes_to_file_service(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service(
            create_direct_upload=AsyncMock(
                return_value={
                    "file_id": "file-1",
                    "upload_url": "https://oss.example.com/files/v1/users/user-123/conversations/conv-1/files/file-1/original",
                    "method": "PUT",
                    "headers": {"Content-Type": "image/png"},
                    "expires_in": 600,
                }
            )
        )

        response = self.client.post(
            "/api/files/upload/init",
            json={
                "provider": "qwen",
                "model": "qwen-vl-max",
                "conversation_id": "conv-1",
                "filename": "photo.png",
                "mimetype": "image/png",
                "size": 123,
            },
        )

        self.assertEqual(response.status_code, 201)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["upload"]["file_id"], "file-1")
        self.assertEqual(
            body["data"]["upload"]["upload_url"],
            "https://oss.example.com/files/v1/users/user-123/conversations/conv-1/files/file-1/original",
        )
        service.create_direct_upload.assert_awaited_once_with(
            user_id="user-123",
            conversation_id="conv-1",
            provider="qwen",
            model="qwen-vl-max",
            filename="photo.png",
            mimetype="image/png",
            size=123,
        )

    def test_file_direct_upload_init_disabled_returns_fallback_code(self):
        self._enable_authenticated_overrides()
        self._mock_file_service(create_direct_upload=AsyncMock(side_effect=NotImplementedError("disabled")))

        response = self.client.post(
            "/api/files/upload/init",
            json={
                "provider": "qwen",
                "model": "qwen-vl-max",
                "conversation_id": "conv-1",
                "filename": "photo.png",
                "mimetype": "image/png",
                "size": 123,
            },
        )

        self.assertEqual(response.status_code, 400)
        body = response.json()
        self.assertEqual(body["code"], "DIRECT_UPLOAD_DISABLED")
        self.assertEqual(body["message"], "当前存储后端未开启直传上传")
        self.assertIsNone(body["data"])

    def test_file_direct_upload_complete_routes_to_file_service(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service(
            complete_direct_upload=AsyncMock(
                return_value={
                    "file_id": "file-1",
                    "thumbnail_url": "https://oss.example.com/files/v1/users/user-123/conversations/conv-1/files/file-1/thumbnail.jpg",
                    "status": "processed",
                }
            )
        )

        response = self.client.post("/api/files/upload/complete", json={"file_id": "file-1"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["file"]["file_id"], "file-1")
        self.assertEqual(body["data"]["file"]["status"], "processed")
        service.complete_direct_upload.assert_awaited_once_with("file-1", "user-123")

    def test_timeout_middleware_uses_longer_budget_for_file_upload(self):
        middleware = self.main.TimeoutMiddleware(lambda scope, receive, send: None, timeout_seconds=10)
        captured = {}

        async def call_next(_request):
            return Response("ok")

        async def fake_wait_for(awaitable, timeout):
            captured["timeout"] = timeout
            return await awaitable

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/files/upload",
                "headers": [],
            }
        )

        with patch.object(self.main.asyncio, "wait_for", side_effect=fake_wait_for):
            response = asyncio.run(middleware.dispatch(request, call_next))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["timeout"], self.main.settings.FILE_UPLOAD_TIMEOUT_SECONDS)

    def test_timeout_middleware_uses_longer_budget_for_direct_upload_complete(self):
        middleware = self.main.TimeoutMiddleware(lambda scope, receive, send: None, timeout_seconds=10)
        captured = {}

        async def call_next(_request):
            return Response("ok")

        async def fake_wait_for(awaitable, timeout):
            captured["timeout"] = timeout
            return await awaitable

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/files/upload/complete",
                "headers": [],
            }
        )

        with patch.object(self.main.asyncio, "wait_for", side_effect=fake_wait_for):
            response = asyncio.run(middleware.dispatch(request, call_next))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["timeout"], self.main.settings.FILE_UPLOAD_TIMEOUT_SECONDS)

    def test_timeout_middleware_uses_coordinated_budget_for_mcp_admin_operation(self):
        middleware = self.main.TimeoutMiddleware(lambda scope, receive, send: None, timeout_seconds=10)
        captured = {}

        async def call_next(_request):
            return Response("ok")

        async def fake_wait_for(awaitable, timeout):
            captured["timeout"] = timeout
            return await awaitable

        request = Request(
            {
                "type": "http",
                "method": "POST",
                "path": "/api/admin/mcp/servers/server-1/tools/refresh",
                "headers": [],
            }
        )

        with patch.object(self.main.asyncio, "wait_for", side_effect=fake_wait_for):
            response = asyncio.run(middleware.dispatch(request, call_next))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["timeout"], 35)

    def test_timeout_middleware_keeps_default_budget_for_normal_routes(self):
        middleware = self.main.TimeoutMiddleware(lambda scope, receive, send: None, timeout_seconds=10)
        captured = {}

        async def call_next(_request):
            return Response("ok")

        async def fake_wait_for(awaitable, timeout):
            captured["timeout"] = timeout
            return await awaitable

        request = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/health",
                "headers": [],
            }
        )

        with patch.object(self.main.asyncio, "wait_for", side_effect=fake_wait_for):
            response = asyncio.run(middleware.dispatch(request, call_next))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(captured["timeout"], 10)

    def test_file_status_returns_not_found_when_service_has_no_record(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service()
        service.get_file_status.return_value = None

        response = self.client.get("/api/files/file-404/status")

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "NOT_FOUND")
        self.assertEqual(body["message"], "文件不存在或无权访问")
        service.get_file_status.assert_called_once_with("file-404", user_id="user-123")

    def test_user_files_use_authenticated_user_scope(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service()
        service.get_files_by_user = AsyncMock(return_value=[{"id": "file-1"}])

        response = self.client.get("/api/files/")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["files"], [{"id": "file-1"}])
        service.get_files_by_user.assert_awaited_once_with("user-123")

    def test_conversation_files_require_authorized_conversation(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service()
        service.get_conversation_files_for_user = AsyncMock(return_value=None)

        response = self.client.get("/api/files/conversation/conv-404")

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "NOT_FOUND")
        self.assertEqual(body["message"], "对话不存在或无权访问")
        service.get_conversation_files_for_user.assert_awaited_once_with("conv-404", "user-123")

    def test_conversation_files_use_authenticated_user_scope(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service()
        service.get_conversation_files_for_user = AsyncMock(return_value=[{"id": "file-1"}])

        response = self.client.get("/api/files/conversation/conv-1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["files"], [{"id": "file-1"}])
        service.get_conversation_files_for_user.assert_awaited_once_with("conv-1", "user-123")


if __name__ == "__main__":
    unittest.main()
