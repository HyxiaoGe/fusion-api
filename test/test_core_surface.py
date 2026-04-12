import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

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
        main.init_db = lambda: None

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
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["service"], "fusion-api")

    def test_openapi_exposes_shared_auth_surface(self):
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]

        self.assertIn("/api/auth/me", paths)
        self.assertIn("/api/chat/conversations", paths)
        self.assertIn("/api/files/upload", paths)
        self.assertIn("/api/models/", paths)
        self.assertIn("/api/models/{model_id}/credentials", paths)
        self.assertIn("/api/models/credentials/test", paths)

        self.assertNotIn("/api/auth/login/{provider}", paths)
        self.assertNotIn("/api/auth/callback/{provider}", paths)
        self.assertNotIn("/api/users/profile", paths)
        self.assertNotIn("/api/credentials", paths)
        self.assertNotIn("/api/rss/sources", paths)
        self.assertNotIn("/api/digests", paths)
        self.assertNotIn("/api/web_search/search", paths)
        self.assertNotIn("/api/settings", paths)
        self.assertNotIn("/api/prompts", paths)

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
        service.process_message.assert_awaited_once_with(
            model_id="gpt-4.1",
            message="hello",
            user_id="user-123",
            conversation_id="conv-1",
            stream=False,
            options={"temperature": 0.3},
            file_ids=["file-1"],
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

    def test_conversation_files_require_authorized_conversation(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service()
        service.get_conversation_files_for_user.return_value = None

        response = self.client.get("/api/files/conversation/conv-404")

        self.assertEqual(response.status_code, 404)
        body = response.json()
        self.assertEqual(body["code"], "NOT_FOUND")
        self.assertEqual(body["message"], "对话不存在或无权访问")
        service.get_conversation_files_for_user.assert_called_once_with("conv-404", "user-123")

    def test_conversation_files_use_authenticated_user_scope(self):
        self._enable_authenticated_overrides()
        service = self._mock_file_service()
        service.get_conversation_files_for_user.return_value = [{"id": "file-1"}]

        response = self.client.get("/api/files/conversation/conv-1")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["code"], "SUCCESS")
        self.assertEqual(body["data"]["files"], [{"id": "file-1"}])
        service.get_conversation_files_for_user.assert_called_once_with("conv-1", "user-123")


if __name__ == "__main__":
    unittest.main()
