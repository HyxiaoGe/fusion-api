import importlib
import os
import sys
import unittest
from urllib.parse import quote
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from fastapi.responses import StreamingResponse


os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["GITHUB_CLIENT_ID"] = "github-test-client"
os.environ["GITHUB_CLIENT_SECRET"] = "github-test-secret"
os.environ["GOOGLE_CLIENT_ID"] = "google-test-client"
os.environ["GOOGLE_CLIENT_SECRET"] = "google-test-secret"


class ChatCoreSurfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")

        # Keep startup cheap and deterministic for route-surface checks.
        main.init_db = lambda: None
        main.init_function_registry = lambda: None

        cls.main = main
        cls.client = TestClient(main.app)

        from app.api import chat as chat_api
        from app.api import files as files_api
        from app.api import auth as auth_api
        from app.core.config import settings
        from app.core.security import get_current_user
        from app.db.database import get_db

        cls.chat_api = chat_api
        cls.files_api = files_api
        cls.auth_api = auth_api
        cls.settings = settings
        cls.get_current_user = get_current_user
        cls.get_db = get_db
        cls.fake_user = SimpleNamespace(id="user-123")

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def _enable_authenticated_overrides(self):
        current_user_override = lambda: self.fake_user
        self.main.app.dependency_overrides[self.get_current_user] = current_user_override
        self.main.app.dependency_overrides[self.chat_api.get_current_user] = current_user_override
        self.main.app.dependency_overrides[self.files_api.get_current_user] = current_user_override

        def override_db():
            yield object()

        self.main.app.dependency_overrides[self.get_db] = override_db
        self.main.app.dependency_overrides[self.chat_api.get_db] = override_db
        self.main.app.dependency_overrides[self.files_api.get_db] = override_db

    def _override_db_dependency(self, db_obj):
        def override_db():
            yield db_obj

        self.main.app.dependency_overrides[self.get_db] = override_db
        self.main.app.dependency_overrides[self.auth_api.get_db] = override_db

    def test_health_endpoint_stays_available(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(payload["service"], "fusion-api")

    def test_openapi_exposes_chat_only_route_surface(self):
        response = self.client.get("/openapi.json")

        self.assertEqual(response.status_code, 200)
        paths = response.json()["paths"]

        self.assertIn("/api/auth/me", paths)
        self.assertIn("/api/auth/login/{provider}", paths)
        self.assertIn("/api/auth/callback/{provider}", paths)
        self.assertIn("/api/chat/conversations", paths)
        self.assertIn("/api/files/upload", paths)
        self.assertIn("/api/models/", paths)
        self.assertIn("/api/models/{model_id}/credentials", paths)
        self.assertIn("/api/models/credentials/test", paths)

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
        self.assertEqual(response.json()["detail"], "Not authenticated")

    def test_send_message_routes_to_chat_service(self):
        self._enable_authenticated_overrides()
        payload = {
            "provider": "openai",
            "model": "gpt-4.1",
            "message": "hello",
            "conversation_id": "conv-1",
            "stream": False,
            "options": {"temperature": 0.3},
            "file_ids": ["file-1"],
        }

        with patch.object(self.chat_api, "ChatService") as chat_service_cls:
            service = chat_service_cls.return_value
            service.process_message = AsyncMock(
                return_value={
                    "conversation_id": "conv-1",
                    "message": {"content": "hi"},
                }
            )

            response = self.client.post("/api/chat/send", json=payload)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["conversation_id"], "conv-1")
        service.process_message.assert_awaited_once_with(
            user_id="user-123",
            provider="openai",
            model="gpt-4.1",
            message="hello",
            conversation_id="conv-1",
            stream=False,
            options={"temperature": 0.3},
            file_ids=["file-1"],
        )

    def test_send_message_can_return_streaming_response(self):
        self._enable_authenticated_overrides()

        async def event_stream():
            yield "data: hello\n\n"

        with patch.object(self.chat_api, "ChatService") as chat_service_cls:
            service = chat_service_cls.return_value
            service.process_message = AsyncMock(
                return_value=StreamingResponse(
                    event_stream(),
                    media_type="text/event-stream",
                )
            )

            response = self.client.post(
                "/api/chat/send",
                json={
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "message": "stream please",
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith("text/event-stream")
        )
        self.assertIn("data: hello", response.text)
        service.process_message.assert_awaited_once()

    def test_get_conversations_uses_authenticated_user_id(self):
        self._enable_authenticated_overrides()

        with patch.object(self.chat_api, "ChatService") as chat_service_cls:
            service = chat_service_cls.return_value
            service.get_conversations_paginated.return_value = {
                "items": [],
                "total": 0,
                "page": 1,
                "page_size": 10,
            }

            response = self.client.get("/api/chat/conversations?page=1&page_size=10")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["total"], 0)
        service.get_conversations_paginated.assert_called_once_with(
            "user-123",
            1,
            10,
        )

    def test_file_upload_routes_to_file_service(self):
        self._enable_authenticated_overrides()

        with patch.object(self.files_api, "FileService") as file_service_cls:
            service = file_service_cls.return_value
            service.upload_files = AsyncMock(return_value=["file-1", "file-2"])

            response = self.client.post(
                "/api/files/upload",
                data={
                    "provider": "openai",
                    "model": "gpt-4.1",
                    "conversation_id": "conv-1",
                },
                files=[
                    ("files", ("note.txt", b"hello", "text/plain")),
                ],
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "success", "file_ids": ["file-1", "file-2"]})
        service.upload_files.assert_awaited_once()
        args = service.upload_files.await_args
        self.assertEqual(args.args[1:], ("user-123", "conv-1", "openai", "gpt-4.1"))
        self.assertEqual(len(args.args[0]), 1)
        self.assertEqual(args.args[0][0].filename, "note.txt")

    def test_file_status_returns_not_found_when_service_has_no_record(self):
        self._enable_authenticated_overrides()

        with patch.object(self.files_api, "FileService") as file_service_cls:
            service = file_service_cls.return_value
            service.get_file_status.return_value = None

            response = self.client.get("/api/files/file-404/status")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "文件不存在或无权访问")
        service.get_file_status.assert_called_once_with("file-404", user_id="user-123")

    def test_conversation_files_require_authorized_conversation(self):
        self._enable_authenticated_overrides()

        with patch.object(self.files_api, "FileService") as file_service_cls:
            service = file_service_cls.return_value
            service.get_conversation_files_for_user.return_value = None

            response = self.client.get("/api/files/conversation/conv-404")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "对话不存在或无权访问")
        service.get_conversation_files_for_user.assert_called_once_with("conv-404", "user-123")

    def test_conversation_files_use_authenticated_user_scope(self):
        self._enable_authenticated_overrides()

        with patch.object(self.files_api, "FileService") as file_service_cls:
            service = file_service_cls.return_value
            service.get_conversation_files_for_user.return_value = [{"id": "file-1"}]

            response = self.client.get("/api/files/conversation/conv-1")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"files": [{"id": "file-1"}]})
        service.get_conversation_files_for_user.assert_called_once_with("conv-1", "user-123")

    def test_github_login_redirect_uses_configured_server_host(self):
        response = self.client.get(
            "/api/auth/login/github",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        location = response.headers["location"]
        self.assertIn("github.com/login/oauth/authorize", location)
        expected_redirect_uri = quote(
            f"{self.settings.SERVER_HOST.rstrip('/')}/api/auth/callback/github",
            safe="",
        )
        self.assertIn(
            f"redirect_uri={expected_redirect_uri}",
            location,
        )
        self.assertNotIn("localhost", location)

    def test_google_login_redirect_uses_configured_server_host(self):
        response = self.client.get(
            "/api/auth/login/google",
            follow_redirects=False,
        )

        self.assertEqual(response.status_code, 302)
        location = response.headers["location"]
        self.assertIn("accounts.google.com/o/oauth2/auth", location)
        expected_redirect_uri = quote(
            f"{self.settings.SERVER_HOST.rstrip('/')}/api/auth/callback/google",
            safe="",
        )
        self.assertIn(
            f"redirect_uri={expected_redirect_uri}",
            location,
        )
        self.assertNotIn("localhost", location)

    def test_github_callback_redirects_to_frontend_with_token_for_existing_social_account(self):
        fake_db = MagicMock()
        fake_user = SimpleNamespace(
            id="user-1",
            email=None,
            nickname=None,
            avatar=None,
        )
        fake_social_account = SimpleNamespace(user=fake_user)
        fake_client = MagicMock()
        fake_client.authorize_access_token = AsyncMock(return_value={"access_token": "provider-token"})
        fake_client.get = AsyncMock(
            return_value=SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "id": 123,
                    "login": "fusion-user",
                    "email": "fusion@example.com",
                    "name": "Fusion User",
                    "avatar_url": "https://example.com/avatar.png",
                },
            )
        )
        self._override_db_dependency(fake_db)

        with patch.object(self.auth_api.oauth, "create_client", return_value=fake_client), \
             patch.object(self.auth_api, "UserRepository") as user_repo_cls, \
             patch.object(self.auth_api, "SocialAccountRepository") as social_repo_cls, \
             patch.object(self.auth_api.security, "create_access_token", return_value="jwt-token"):
            user_repo_cls.return_value = MagicMock()
            social_repo_cls.return_value.get_by_provider.return_value = fake_social_account

            response = self.client.get(
                "/api/auth/callback/github",
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 307)
        self.assertEqual(
            response.headers["location"],
            f"{self.settings.FRONTEND_AUTH_CALLBACK_URL}?token=jwt-token&token_type=bearer",
        )
        fake_client.authorize_access_token.assert_awaited_once()
        fake_client.get.assert_awaited_once_with("user", token={"access_token": "provider-token"})
        self.assertEqual(fake_user.email, "fusion@example.com")
        self.assertEqual(fake_user.nickname, "Fusion User")
        self.assertEqual(fake_user.avatar, "https://example.com/avatar.png")
        fake_db.commit.assert_called_once()
        fake_db.refresh.assert_called_once_with(fake_user)

    def test_google_callback_returns_500_when_provider_userinfo_fails(self):
        fake_db = MagicMock()
        fake_client = MagicMock()
        fake_client.authorize_access_token = AsyncMock(return_value={"access_token": "provider-token"})
        fake_client.get = AsyncMock(
            return_value=SimpleNamespace(
                status_code=500,
                json=lambda: {},
            )
        )
        self._override_db_dependency(fake_db)

        with patch.object(self.auth_api.oauth, "create_client", return_value=fake_client):
            response = self.client.get(
                "/api/auth/callback/google",
                follow_redirects=False,
            )

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "OAuth callback failed.")
        fake_client.authorize_access_token.assert_awaited_once()
        fake_client.get.assert_awaited_once_with("userinfo", token={"access_token": "provider-token"})


if __name__ == "__main__":
    unittest.main()
