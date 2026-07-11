import importlib
import json
import os
import sys
import unittest
from datetime import datetime
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["AUTH_SERVICE_BASE_URL"] = "http://auth.example:8100"
os.environ["AUTH_SERVICE_CLIENT_ID"] = "fusion-client"
os.environ["AUTH_SERVICE_JWKS_URL"] = "http://auth.example:8100/.well-known/jwks.json"


class AdminAuditApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        from app.db.database import Base

        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self._seed()

        from app.api.deps import get_current_user, get_db

        self.current_user = SimpleNamespace(
            id="admin-1",
            username="root",
            email="root@example.com",
            is_superuser=True,
        )
        self.main.app.dependency_overrides[get_current_user] = lambda: self.current_user
        self.main.app.dependency_overrides[get_db] = lambda: self.db

    def tearDown(self):
        self.main.app.dependency_overrides.clear()
        self.db.close()
        self.engine.dispose()

    def _seed(self):
        from app.db.models import Conversation, Message, User

        created = datetime(2026, 7, 11, 12, 0, 0)
        self.db.add_all(
            [
                User(
                    id="admin-1",
                    username="root",
                    email="root@example.com",
                    is_superuser=True,
                    created_at=created,
                    updated_at=created,
                ),
                User(
                    id="user-1",
                    username="alice",
                    nickname="Alice",
                    email="alice@example.com",
                    system_prompt="请勿泄漏 Bearer secret-token",
                    created_at=created,
                    updated_at=created,
                ),
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="审计对话",
                    model_id="deepseek-chat",
                    created_at=created,
                    updated_at=created,
                ),
                Message(
                    id="msg-1",
                    conversation_id="conv-1",
                    role="user",
                    content=[{"type": "text", "text": "Bearer message-secret"}],
                    created_at=created,
                ),
            ]
        )
        self.db.commit()

    def test_rejects_non_auditor_and_sets_no_store_for_admin(self):
        self.current_user.is_superuser = False
        forbidden = self.client.get("/api/admin/audit/users")
        self.current_user.is_superuser = True
        allowed = self.client.get("/api/admin/audit/users")

        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(allowed.status_code, 200)
        self.assertEqual(allowed.headers["cache-control"], "private, no-store")
        self.assertEqual(allowed.json()["data"]["total"], 2)

    def test_sensitive_reads_are_redacted_and_write_audit_events(self):
        user_response = self.client.get(
            "/api/admin/audit/users/user-1",
            headers={"X-Admin-Audit-Reason": "support-investigation"},
        )
        messages_response = self.client.get("/api/admin/audit/conversations/conv-1/messages")
        events_response = self.client.get("/api/admin/audit/events")

        self.assertEqual(user_response.status_code, 200)
        self.assertNotIn("secret-token", user_response.text)
        self.assertEqual(messages_response.status_code, 200)
        self.assertNotIn("message-secret", messages_response.text)
        actions = [item["action"] for item in events_response.json()["data"]["items"]]
        self.assertIn("admin.audit.user.view", actions)
        self.assertIn("admin.audit.messages.list", actions)
        self.assertNotIn("message-secret", events_response.text)

    def test_audit_metadata_does_not_store_search_text_or_secret_reason(self):
        response = self.client.get(
            "/api/admin/audit/users?q=alice%40example.com",
            headers={"X-Admin-Audit-Reason": "Bearer audit-secret"},
        )

        self.assertEqual(response.status_code, 200)
        from app.db.models import AdminAuditEvent

        event = (
            self.db.query(AdminAuditEvent)
            .filter(AdminAuditEvent.action == "admin.audit.users.list")
            .order_by(AdminAuditEvent.created_at.desc())
            .first()
        )
        serialized = json.dumps(
            {"metadata": event.extra_metadata, "reason": event.reason, "snapshot": event.admin_snapshot}
        )
        self.assertNotIn("alice@example.com", serialized)
        self.assertNotIn("audit-secret", serialized)
        self.assertEqual(event.extra_metadata["query"], {"present": True, "length": 17})

    def test_audit_failure_is_fail_closed_for_sensitive_content(self):
        from unittest.mock import patch

        with patch(
            "app.db.admin_audit_repository.AdminAuditRepository.create_audit_event",
            side_effect=RuntimeError("audit unavailable"),
        ):
            response = self.client.get("/api/admin/audit/conversations/conv-1/messages")

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("message-secret", response.text)

    def test_performance_import_is_idempotent_and_never_returns_credentials(self):
        payload = {
            "run_id": "perf-20260711-safe",
            "environment": "production",
            "model_id": "deepseek-chat",
            "status": "completed",
            "schema_version": 1,
            "safe_summary": {
                "stages": [{"kind": "sse", "concurrency": 1, "successful": 1}],
            },
            "started_at": "2026-07-11T12:00:00+08:00",
            "finished_at": "2026-07-11T12:01:00+08:00",
        }

        first = self.client.post("/api/admin/audit/performance-runs/import", json=payload)
        second = self.client.post("/api/admin/audit/performance-runs/import", json=payload)
        detail = self.client.get("/api/admin/audit/performance-runs/perf-20260711-safe")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertTrue(first.json()["data"]["created"])
        self.assertFalse(second.json()["data"]["created"])
        self.assertEqual(detail.status_code, 200)
        self.assertNotIn("access_token", detail.text)
        from app.db.models import PerformanceRun

        self.assertEqual(self.db.query(PerformanceRun).count(), 1)

    def test_performance_import_rejects_unsafe_payload_shape(self):
        response = self.client.post(
            "/api/admin/audit/performance-runs/import",
            content=json.dumps({"run_id": "bad", "safe_summary": "raw"}),
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)

    def test_performance_import_rejects_identifiers_and_content_even_for_direct_api_call(self):
        response = self.client.post(
            "/api/admin/audit/performance-runs/import",
            json={
                "run_id": "perf-unsafe",
                "environment": "production",
                "schema_version": 1,
                "safe_summary": {
                    "stages": [],
                    "agent_run_ids": ["run-secret"],
                    "message": "user@example.com private prompt",
                },
            },
        )

        self.assertEqual(response.status_code, 422)
        from app.db.models import PerformanceRun

        self.assertEqual(self.db.query(PerformanceRun).count(), 0)

    def test_performance_import_rolls_back_when_audit_write_fails(self):
        from unittest.mock import patch

        with patch(
            "app.db.admin_audit_repository.AdminAuditRepository.create_audit_event",
            side_effect=RuntimeError("audit unavailable"),
        ):
            response = self.client.post(
                "/api/admin/audit/performance-runs/import",
                json={
                    "run_id": "perf-no-audit",
                    "environment": "production",
                    "schema_version": 1,
                    "safe_summary": {"p95_ms": 1200},
                },
            )

        self.assertEqual(response.status_code, 503)
        from app.db.models import PerformanceRun

        self.assertEqual(self.db.query(PerformanceRun).count(), 0)


if __name__ == "__main__":
    unittest.main()
