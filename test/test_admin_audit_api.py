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

    def test_message_route_serialization_never_returns_unapproved_block_or_credential_fields(self):
        from app.db.models import Message

        google_key = "AIza" + "C" * 35
        slack_token = "xox" + "b-123456789012-123456789012-route-secret-token"
        self.db.add(
            Message(
                id="msg-route-security",
                conversation_id="conv-1",
                role="assistant",
                content=[
                    {"type": "text", "id": "text-route", "text": f"google={google_key} slack={slack_token}"},
                    {
                        "type": "file",
                        "id": "file-route",
                        "file_id": "file-1",
                        "filename": "report.pdf",
                        "mime_type": "application/pdf",
                        "thumbnail_url": "https://storage.example/thumb?token=thumbnail-route-sentinel",
                        "storage_key": "storage-route-sentinel",
                        "path": "/path-route-sentinel",
                    },
                    {
                        "type": "search",
                        "id": "search-route",
                        "query": "Fusion",
                        "status": "success",
                        "sources": [
                            {
                                "title": "GCS 文件",
                                "url": (
                                    "https://storage.googleapis.com/private-bucket/report.pdf"
                                    "?X-Goog-Algorithm=GOOG4-RSA-SHA256"
                                    "&X-Goog-Credential=gcs-route-credential"
                                    "&X-Goog-Signature=gcs-route-signature"
                                    "&download=route-report.pdf"
                                ),
                            }
                        ],
                        "provider_payload": {"content": "provider-route-sentinel"},
                    },
                    {
                        "type": "future_private_block",
                        "id": "unknown-route",
                        "status": "streaming",
                        "payload": "unknown-route-sentinel",
                    },
                ],
                usage={"input_tokens": 1, "output_tokens": 2, "provider_payload": "usage-route-sentinel"},
                suggested_questions=[f"继续使用 {google_key}", f"继续使用 {slack_token}"],
                created_at=datetime(2026, 7, 11, 12, 1, 0),
            )
        )
        self.db.commit()

        response = self.client.get("/api/admin/audit/conversations/conv-1/messages")

        self.assertEqual(response.status_code, 200)
        item = next(row for row in response.json()["data"]["items"] if row["id"] == "msg-route-security")
        unknown = next(block for block in item["content"] if block["type"] == "future_private_block")
        search = next(block for block in item["content"] if block["type"] == "search")
        self.assertEqual(
            unknown,
            {
                "type": "future_private_block",
                "id": "unknown-route",
                "status": "streaming",
                "content_hidden": True,
            },
        )
        projected_gcs_url = search["sources"][0]["url"]
        self.assertIn("storage.googleapis.com/private-bucket/report.pdf", projected_gcs_url)
        self.assertIn("X-Goog-Algorithm", projected_gcs_url)
        self.assertIn("X-Goog-Credential", projected_gcs_url)
        self.assertIn("X-Goog-Signature", projected_gcs_url)
        self.assertIn("download", projected_gcs_url)
        serialized = json.dumps(item, ensure_ascii=False)
        for sentinel in (
            "thumbnail_url",
            "thumbnail-route-sentinel",
            "storage_key",
            "storage-route-sentinel",
            '"path"',
            "path-route-sentinel",
            "provider_payload",
            "provider-route-sentinel",
            "usage-route-sentinel",
            "unknown-route-sentinel",
            "GOOG4-RSA-SHA256",
            "gcs-route-credential",
            "gcs-route-signature",
            "route-report.pdf",
            google_key,
            slack_token,
        ):
            self.assertNotIn(sentinel, serialized)

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

    def test_performance_schema_v2_round_trip_preserves_all_safe_sections(self):
        payload = {
            "run_id": "perf-20260712-schema-v2",
            "environment": "production",
            "model_id": "deepseek-chat",
            "status": "stopped",
            "schema_version": 2,
            "safe_summary": {
                "stages": [
                    {
                        "scenario": "conversation_list",
                        "kind": "http",
                        "concurrency": 25,
                        "duration_seconds": 60,
                        "total": 100,
                        "successful": 99,
                        "failed": 1,
                        "requests_per_second": 12.5,
                        "p50_ms": 90,
                        "p95_ms": 180,
                        "p99_ms": 220,
                        "error_rate": 0.01,
                    },
                    {
                        "scenario": "sse_short",
                        "kind": "sse",
                        "concurrency": 8,
                        "flows": 8,
                        "flows_with_output": 8,
                        "output_chunks": 96,
                        "visible_chars": 2048,
                        "approx_tokens": 512,
                        "first_output_p95_ms": 450,
                        "chunk_interval_p95_ms": 80,
                        "output_window_p95_ms": 2500,
                        "tokens_per_second_p95": 42,
                    },
                    {
                        "scenario": "disconnect_reconnect",
                        "kind": "recovery",
                        "concurrency": 4,
                        "initial_events": 40,
                        "recovered_events": 40,
                        "duplicate_events": 0,
                        "lost_events": 0,
                        "ordering_errors": 0,
                        "recovery_latency_p95_ms": 320,
                    },
                    {
                        "scenario": "stop_stream",
                        "kind": "stop",
                        "concurrency": 3,
                        "stop_attempted": True,
                        "cancelled": True,
                        "persistence_verified": True,
                        "stop_attempts": 3,
                        "cancelled_count": 3,
                        "persistence_verified_count": 3,
                        "stop_latency_p95_ms": 210,
                    },
                    {
                        "scenario": "soak",
                        "kind": "soak",
                        "concurrency": 2,
                        "cadence_seconds": 30,
                        "window_seconds": 300,
                        "executed_ticks": 10,
                        "skipped_ticks": 0,
                        "window_count": 2,
                        "consecutive_failures": 0,
                    },
                ],
                "stopped": True,
                "stop_reasons": ["resource:api_memory"],
                "cleanup": {
                    "conversations_deleted": 8,
                    "tokens_revoked": 2,
                    "users_deleted": 1,
                    "agent_steps_deleted": 5,
                    "errors": ["token_revoke_failed"],
                },
                "resources": {
                    "api": {"cpu_percent": 88.5, "memory_mib": 920, "restarts": 0, "oom": False},
                    "postgres": {"connections": 24, "restarts": 0},
                    "redis": {"connections": 12, "rejected_connections": 0, "evicted_keys": 0},
                    "host": {"cpu_percent": 62, "memory_mib": 4096, "memory_percent": 64},
                    "nginx": {"connections": 16},
                    "litellm": {"cpu_percent": 35, "memory_mib": 512},
                },
                "rps": 12.5,
                "p50_ms": 90,
                "p90_ms": 150,
                "p95_ms": 180,
                "p99_ms": 220,
                "max_ms": 310,
                "ttft_ms": 450,
                "error_rate": 0.01,
            },
            "started_at": "2026-07-12T12:00:00+08:00",
            "finished_at": "2026-07-12T12:10:00+08:00",
        }

        imported = self.client.post("/api/admin/audit/performance-runs/import", json=payload)
        from app.db.admin_audit_repository import AdminAuditRepository

        rows, _ = AdminAuditRepository(self.db).list_performance_runs(page=1, page_size=25)
        listed = self.client.get("/api/admin/audit/performance-runs")
        detail = self.client.get("/api/admin/audit/performance-runs/perf-20260712-schema-v2")

        self.assertEqual(imported.status_code, 200)
        self.assertFalse(hasattr(rows[0], "safe_summary"))
        self.assertFalse(hasattr(rows[0], "imported_by_user_id"))
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(detail.status_code, 200)
        list_item = listed.json()["data"]["items"][0]
        self.assertNotIn("safe_summary", list_item)
        self.assertNotIn("imported_by_user_id", list_item)
        data = detail.json()["data"]
        self.assertEqual(data["schema_version"], 2)
        self.assertEqual(data["imported_by_user_id"], "admin-1")
        self.assertEqual(data["safe_summary"], payload["safe_summary"])
        self.assertEqual(
            [stage["kind"] for stage in data["safe_summary"]["stages"]],
            ["http", "sse", "recovery", "stop", "soak"],
        )
        from app.db.models import AdminAuditEvent

        actions = {
            event.action
            for event in self.db.query(AdminAuditEvent)
            .filter(AdminAuditEvent.resource_id == "perf-20260712-schema-v2")
            .all()
        }
        self.assertIn("admin.audit.performance_run.view", actions)

    def test_performance_reads_degrade_invalid_stored_summary_without_leaking_values(self):
        from app.db.models import PerformanceRun

        self.db.add(
            PerformanceRun(
                run_id="perf-invalid-stored-summary",
                environment="production",
                status="completed",
                schema_version=2,
                safe_summary={
                    "stages": [],
                    "conversation_id": "private-conversation-id",
                    "agent_run_ids": ["private-agent-run-id"],
                    "message": "private-message-body",
                },
                imported_by_user_id="admin-1",
            )
        )
        self.db.commit()

        detail = self.client.get("/api/admin/audit/performance-runs/perf-invalid-stored-summary")
        listed = self.client.get("/api/admin/audit/performance-runs")

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(listed.status_code, 200)
        expected_summary = {
            "stages": [],
            "stopped": True,
            "stop_reasons": ["invalid_safe_summary"],
            "cleanup": {"conversations_deleted": 0, "tokens_revoked": 0, "errors": []},
        }
        self.assertEqual(detail.json()["data"]["safe_summary"], expected_summary)
        list_item = listed.json()["data"]["items"][0]
        self.assertNotIn("safe_summary", list_item)
        self.assertNotIn("imported_by_user_id", list_item)
        serialized = detail.text + listed.text
        self.assertNotIn("private-conversation-id", serialized)
        self.assertNotIn("private-agent-run-id", serialized)
        self.assertNotIn("private-message-body", serialized)

    def test_performance_import_rejects_unsafe_payload_shape(self):
        response = self.client.post(
            "/api/admin/audit/performance-runs/import",
            content=json.dumps({"run_id": "bad", "safe_summary": "raw"}),
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)

    def test_performance_import_rejects_unsupported_schema_version(self):
        response = self.client.post(
            "/api/admin/audit/performance-runs/import",
            json={
                "run_id": "perf-unsupported-schema-import",
                "environment": "production",
                "schema_version": 99,
                "safe_summary": {"p95_ms": 120},
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_performance_detail_degrades_historical_unknown_schema_without_interpreting_summary(self):
        from app.db.models import PerformanceRun

        self.db.add(
            PerformanceRun(
                run_id="perf-historical-schema-99",
                environment="production",
                status="completed",
                schema_version=99,
                safe_summary={
                    "p95_ms": 120,
                    "stages": [{"kind": "http", "concurrency": 1, "successful": 1}],
                },
                imported_by_user_id="admin-1",
            )
        )
        self.db.commit()

        detail = self.client.get("/api/admin/audit/performance-runs/perf-historical-schema-99")

        self.assertEqual(detail.status_code, 200)
        self.assertEqual(
            detail.json()["data"]["safe_summary"],
            {
                "stages": [],
                "stopped": True,
                "stop_reasons": ["unsupported_schema_version"],
                "cleanup": {"conversations_deleted": 0, "tokens_revoked": 0, "errors": []},
            },
        )
        self.assertNotIn("p95_ms", detail.json()["data"]["safe_summary"])

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
