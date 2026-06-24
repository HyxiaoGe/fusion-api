import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["AUTH_SERVICE_BASE_URL"] = "http://auth.example:8100"
os.environ["AUTH_SERVICE_CLIENT_ID"] = "fusion-client"
os.environ["AUTH_SERVICE_JWKS_URL"] = "http://auth.example:8100/.well-known/jwks.json"


class NetworkDiagnosticsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        cls.main = main
        cls.client = TestClient(main.app)

        cls._route_deps = {}
        for route in main.app.routes:
            if hasattr(route, "dependant"):
                for dep in route.dependant.dependencies:
                    cls._route_deps.setdefault(dep.call.__qualname__, dep.call)

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def _enable_overrides(self, *, is_superuser: bool = False, conversation=None, diagnostics=None):
        user = SimpleNamespace(id="user-123", is_superuser=is_superuser)
        self.main.app.dependency_overrides[self._route_deps["get_current_user"]] = lambda: user

        chat_service = SimpleNamespace(get_conversation=MagicMock(return_value=conversation))
        self.main.app.dependency_overrides[self._route_deps["get_chat_service"]] = lambda: chat_service

        diagnostics_service = diagnostics or SimpleNamespace(build_for_message=MagicMock())
        self.main.app.dependency_overrides[self._route_deps["get_network_diagnostics_service"]] = lambda: (
            diagnostics_service
        )
        return chat_service, diagnostics_service

    def test_diagnostics_rejects_user_message(self):
        conversation = SimpleNamespace(messages=[SimpleNamespace(id="user-1", role="user")])
        _, diagnostics_service = self._enable_overrides(conversation=conversation)

        response = self.client.get("/api/chat/conversations/conv-1/messages/user-1/diagnostics")

        self.assertEqual(response.status_code, 404)
        diagnostics_service.build_for_message.assert_not_called()

    def test_diagnostics_returns_empty_for_assistant_without_logs(self):
        from app.schemas.network_diagnostics import NetworkDiagnosticsResponse

        conversation = SimpleNamespace(messages=[SimpleNamespace(id="assistant-1", role="assistant")])
        diagnostics_service = SimpleNamespace(
            build_for_message=MagicMock(
                return_value=NetworkDiagnosticsResponse(
                    conversation_id="conv-1",
                    message_id="assistant-1",
                    is_empty=True,
                )
            )
        )
        self._enable_overrides(conversation=conversation, diagnostics=diagnostics_service)

        response = self.client.get("/api/chat/conversations/conv-1/messages/assistant-1/diagnostics")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertTrue(data["is_empty"])
        self.assertEqual(data["tools"], [])
        diagnostics_service.build_for_message.assert_called_once_with(
            conversation_id="conv-1",
            message_id="assistant-1",
            is_admin=False,
        )

    def test_diagnostics_admin_gets_admin_field(self):
        from app.schemas.network_diagnostics import NetworkDiagnosticsResponse, NetworkDiagnosticsToolItem

        conversation = SimpleNamespace(messages=[SimpleNamespace(id="assistant-1", role="assistant")])
        diagnostics_service = SimpleNamespace(
            build_for_message=MagicMock(
                return_value=NetworkDiagnosticsResponse(
                    conversation_id="conv-1",
                    message_id="assistant-1",
                    visibility="admin",
                    tools=[
                        NetworkDiagnosticsToolItem(
                            tool_call_log_id="log-1",
                            tool_name="web_search",
                            status="success",
                            target="redis",
                            requested_count=8,
                            actual_count=7,
                            context_count=6,
                            intent="comparison",
                            domains=["redis.io"],
                            recency_days=30,
                            budget_limited=True,
                            admin={"trace_id": "trace-1"},
                        )
                    ],
                )
            )
        )
        self._enable_overrides(is_superuser=True, conversation=conversation, diagnostics=diagnostics_service)

        response = self.client.get("/api/chat/conversations/conv-1/messages/assistant-1/diagnostics")

        self.assertEqual(response.status_code, 200)
        tool = response.json()["data"]["tools"][0]
        self.assertEqual(tool["admin"]["trace_id"], "trace-1")
        self.assertEqual(tool["requested_count"], 8)
        self.assertEqual(tool["actual_count"], 7)
        self.assertEqual(tool["context_count"], 6)
        self.assertEqual(tool["intent"], "comparison")
        self.assertEqual(tool["domains"], ["redis.io"])
        self.assertEqual(tool["recency_days"], 30)
        self.assertTrue(tool["budget_limited"])
        diagnostics_service.build_for_message.assert_called_once_with(
            conversation_id="conv-1",
            message_id="assistant-1",
            is_admin=True,
        )
