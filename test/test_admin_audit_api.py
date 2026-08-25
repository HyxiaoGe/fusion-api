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

    def _add_trajectory_run(
        self,
        run_id: str,
        *,
        conversation_id: str = "conv-1",
        user_id: str = "user-1",
    ) -> None:
        from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta, ToolCallLog

        created = datetime(2026, 7, 11, 12, 0, 0)
        self.db.add(
            AgentSession(
                id=run_id,
                conversation_id=conversation_id,
                message_id=f"msg-{run_id}",
                user_id=user_id,
                model_id="model-1",
                provider="provider-1",
                status="completed",
                total_steps=2,
                total_tool_calls=1,
                total_duration_ms=123,
                created_at=created,
                terminal_at=created,
            )
        )
        self.db.add(
            RunTrajectoryMeta(
                run_id=run_id,
                conversation_id=conversation_id,
                trajectory_status="complete",
                event_count=1,
                expected_last_sequence=0,
            )
        )
        self.db.add(
            AgentEvent(
                conversation_id=conversation_id,
                message_id=f"msg-{run_id}",
                run_id=run_id,
                sequence=0,
                event_type="step_completed",
                schema_version=1,
                event_ts=created,
                payload={"duration_ms": 10, "safe_summary": "event-0"},
            )
        )
        self.db.add(
            ToolCallLog(
                id=f"tool-{run_id}",
                conversation_id=conversation_id,
                message_id=f"msg-{run_id}",
                user_id=user_id,
                tool_name="private_tool",
                status="success",
                model_id="model-1",
                provider="provider-1",
                input_params={"secret": "input-secret"},
                output_data={"secret": "output-secret"},
                error_message="error-secret",
                trace_id=run_id,
                tool_call_id=f"call-{run_id}",
                step_number=1,
                created_at=created,
            )
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

    def test_itinerary_stability_returns_bounded_safe_aggregates(self):
        from app.db.models import AgentSession, Message, ToolCallLog

        created = datetime(2026, 7, 27, 10, 0, 0)
        message = Message(
            id="msg-trip",
            conversation_id="conv-1",
            role="assistant",
            content=[
                {
                    "type": "itinerary_results",
                    "status": "degraded",
                    "origin": "private-origin",
                    "destination": "private-destination",
                }
            ],
            model_id="deepseek-chat",
            created_at=created,
        )
        session = AgentSession(
            id="run-trip",
            conversation_id="conv-1",
            message_id="msg-trip",
            user_id="user-1",
            model_id="deepseek-chat",
            provider="deepseek",
            status="limit_reached",
            limit_reason="max_tool_calls",
            total_steps=2,
            total_tool_calls=3,
            total_duration_ms=4000,
            created_at=created,
        )
        logs = [
            ToolCallLog(
                id="tool-flight",
                conversation_id="conv-1",
                message_id="msg-trip",
                user_id="user-1",
                tool_name="search_flights",
                status="success",
                duration_ms=1000,
                model_id="deepseek-chat",
                provider="deepseek",
                output_data={"status": "success", "private_payload": "must-not-leak"},
                trace_id="run-trip",
                created_at=created,
            ),
            ToolCallLog(
                id="tool-weather",
                conversation_id="conv-1",
                message_id="msg-trip",
                user_id="user-1",
                tool_name="weather_forecast",
                status="failed",
                duration_ms=2000,
                model_id="deepseek-chat",
                provider="deepseek",
                output_data={"error_code": "upstream_error", "private_payload": "must-not-leak"},
                trace_id="run-trip",
                created_at=created,
            ),
            ToolCallLog(
                id="tool-flight-timeout",
                conversation_id="conv-1",
                message_id="msg-trip",
                user_id="user-1",
                tool_name="search_flights",
                status="failed",
                duration_ms=3000,
                model_id="deepseek-chat",
                provider="deepseek",
                output_data={"error_code": "call_timeout"},
                trace_id="run-trip",
                created_at=created,
            ),
        ]
        self.db.add(message)
        self.db.commit()
        self.db.add(session)
        self.db.commit()
        self.db.add_all(logs)
        self.db.commit()

        response = self.client.get(
            "/api/admin/audit/itinerary-stability",
            params={
                "created_from": "2026-07-27T00:00:00",
                "created_to": "2026-07-28T00:00:00",
                "model_id": "deepseek-chat",
            },
            headers={"X-Admin-Audit-Reason": "itinerary-stability-check"},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["scope"]["sample_definition"], "terminal_run_with_travel_tool")
        self.assertEqual(data["summary"]["itinerary"]["total"], 1)
        self.assertEqual(data["summary"]["itinerary"]["partial"], 1)
        self.assertEqual(data["summary"]["signals"]["agent_limit_reached"], 1)
        self.assertEqual(data["summary"]["signals"]["upstream_error"], 1)
        self.assertEqual(data["summary"]["run_latency_ms"]["p50_ms"], 4000)
        self.assertEqual(data["by_model"][0]["model_id"], "deepseek-chat")
        self.assertEqual(data["by_tool"][0]["tool_name"], "search_flights")
        self.assertEqual(data["by_tool"][0]["calls"]["success"], 1)
        self.assertEqual(data["by_tool"][0]["calls"]["failed"], 1)
        self.assertEqual(data["by_tool"][0]["calls"]["timeout"], 1)
        self.assertEqual(
            data["summary"]["product_tools"]["total"],
            data["summary"]["product_tools"]["success"]
            + data["summary"]["product_tools"]["degraded"]
            + data["summary"]["product_tools"]["failed"],
        )
        self.assertNotIn("private-origin", response.text)
        self.assertNotIn("private-destination", response.text)
        self.assertNotIn("must-not-leak", response.text)

        events = self.client.get("/api/admin/audit/events?action=admin.audit.itinerary_stability.view").json()["data"][
            "items"
        ]
        self.assertEqual(len(events), 1)

    def test_itinerary_stability_rejects_ranges_over_seven_days(self):
        response = self.client.get(
            "/api/admin/audit/itinerary-stability",
            params={
                "created_from": "2026-07-01T00:00:00",
                "created_to": "2026-07-09T00:00:00",
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_itinerary_stability_converts_shanghai_window_to_utc_storage_time(self):
        from app.db.models import AgentSession, Message, ToolCallLog

        stored_utc = datetime(2026, 7, 27, 7, 30, 0)
        self.db.add(
            Message(
                id="msg-timezone-trip",
                conversation_id="conv-1",
                role="assistant",
                content=[{"type": "itinerary_results", "status": "success"}],
                model_id="kimi-k2.5",
                created_at=stored_utc,
            )
        )
        self.db.commit()
        self.db.add(
            AgentSession(
                id="run-timezone-trip",
                conversation_id="conv-1",
                message_id="msg-timezone-trip",
                user_id="user-1",
                model_id="kimi-k2.5",
                provider="moonshot",
                status="completed",
                total_steps=1,
                total_tool_calls=1,
                total_duration_ms=2000,
                created_at=stored_utc,
            )
        )
        self.db.commit()
        self.db.add(
            ToolCallLog(
                id="tool-timezone-trip",
                conversation_id="conv-1",
                message_id="msg-timezone-trip",
                user_id="user-1",
                tool_name="search_trains",
                status="success",
                duration_ms=1000,
                model_id="kimi-k2.5",
                provider="moonshot",
                output_data={"status": "success"},
                trace_id="run-timezone-trip",
                created_at=stored_utc,
            )
        )
        self.db.commit()

        response = self.client.get(
            "/api/admin/audit/itinerary-stability",
            params={
                "created_from": "2026-07-27T15:00:00+08:00",
                "created_to": "2026-07-27T16:00:00+08:00",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["summary"]["itinerary"]["total"], 1)
        self.assertEqual(data["scope"]["created_from"], "2026-07-27T15:00:00+08:00")
        self.assertEqual(data["scope"]["created_to"], "2026-07-27T16:00:00+08:00")

    def test_audit_events_include_target_user_summary_and_keep_deleted_target_id(self):
        from app.db.models import AdminAuditEvent

        created = datetime(2026, 7, 11, 12, 0, 0)
        self.db.add_all(
            [
                AdminAuditEvent(
                    id="event-live-target",
                    admin_user_id="historical-admin-id",
                    admin_snapshot={
                        "id": "historical-admin-id",
                        "username": "historical-admin",
                        "email_masked": "h***@example.com",
                        "email": "historical-admin-private@example.com",
                        "unknown_pii": "historical-admin-id-card",
                    },
                    action="admin.audit.user.view",
                    resource_type="user",
                    resource_id="user-1",
                    target_user_id="user-1",
                    request_id="request-live-target",
                    extra_metadata={
                        "page": 1,
                        "query": {
                            "present": True,
                            "length": 17,
                            "raw": "alice-private@example.com",
                        },
                        "customer_email": "customer-private@example.com",
                        "future_secret": "future-private-value",
                    },
                    created_at=created,
                ),
                AdminAuditEvent(
                    id="event-deleted-target",
                    admin_user_id="historical-admin-id",
                    admin_snapshot={"id": "historical-admin-id", "username": "historical-admin"},
                    action="admin.audit.user.view",
                    resource_type="user",
                    resource_id="deleted-user",
                    target_user_id="deleted-user",
                    request_id="request-deleted-target",
                    extra_metadata={},
                    created_at=created,
                ),
            ]
        )
        self.db.commit()

        response = self.client.get("/api/admin/audit/events?page_size=100")

        self.assertEqual(response.status_code, 200)
        events = {item["id"]: item for item in response.json()["data"]["items"]}
        self.assertEqual(
            events["event-live-target"]["target_user"],
            {
                "id": "user-1",
                "username": "alice",
                "nickname": "Alice",
            },
        )
        self.assertEqual(events["event-live-target"]["admin_snapshot"]["username"], "historical-admin")
        self.assertEqual(
            events["event-live-target"]["admin_snapshot"],
            {
                "id": "historical-admin-id",
                "username": "historical-admin",
            },
        )
        self.assertEqual(
            events["event-live-target"]["metadata"],
            {"page": 1, "query": {"present": True, "length": 17}},
        )
        for sentinel in (
            "historical-admin-private@example.com",
            "historical-admin-id-card",
            "alice-private@example.com",
            "customer-private@example.com",
            "future-private-value",
        ):
            self.assertNotIn(sentinel, response.text)
        self.assertEqual(events["event-deleted-target"]["target_user_id"], "deleted-user")
        self.assertIsNone(events["event-deleted-target"]["target_user"])
        for event in events.values():
            self.assertNotIn("email", event["admin_snapshot"])
            self.assertNotIn("email_masked", event["admin_snapshot"])
            if event["target_user"] is not None:
                self.assertNotIn("email", event["target_user"])
                self.assertNotIn("email_masked", event["target_user"])

        persisted_list_event = (
            self.db.query(AdminAuditEvent)
            .filter(AdminAuditEvent.action == "admin.audit.events.list")
            .order_by(AdminAuditEvent.created_at.desc())
            .first()
        )
        self.assertIn("email_masked", persisted_list_event.admin_snapshot)

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

    def test_model_operations_list_and_detail_merge_catalog_with_real_history_safely(self):
        from unittest.mock import patch

        from app.db.models import AdminAuditEvent, AgentSession, Conversation, Message, PerformanceRun

        created = datetime(2026, 7, 12, 12, 0, 0)
        self.db.add_all(
            [
                Message(
                    id="msg-model-active",
                    conversation_id="conv-1",
                    role="assistant",
                    content=[{"type": "text", "text": "active"}],
                    model_id="deepseek-chat",
                    usage={"input_tokens": 1_500_000_000, "output_tokens": 20},
                    created_at=created,
                ),
                Conversation(
                    id="conv-retired",
                    user_id="user-1",
                    title="历史模型对话",
                    model_id="retired/model-v1",
                    created_at=created,
                    updated_at=created,
                ),
                Message(
                    id="msg-model-retired",
                    conversation_id="conv-retired",
                    role="assistant",
                    content=[{"type": "text", "text": "retired"}],
                    model_id="retired/model-v1",
                    usage={"input_tokens": 30, "output_tokens": 40},
                    created_at=created,
                ),
                PerformanceRun(
                    run_id="perf-model-retired",
                    environment="production",
                    model_id="retired/model-v1",
                    status="completed",
                    schema_version=2,
                    safe_summary={
                        "stages": [{"kind": "sse", "concurrency": 1, "p95_ttft_ms": 250}],
                    },
                    imported_by_user_id="admin-1",
                    created_at=created,
                ),
            ]
        )
        self.db.commit()
        self.db.add(
            AgentSession(
                id="run-model-retired",
                conversation_id="conv-retired",
                message_id="msg-model-retired",
                user_id="user-1",
                model_id="retired/model-v1",
                provider="legacy",
                status="error",
                error_message="provider-private-error",
                total_duration_ms=4321,
                created_at=created,
            )
        )
        self.db.commit()
        catalog = {
            "deepseek-chat": {
                "underlying": "openai/provider-private-model",
                "db_model": True,
                "max_input_tokens": 64000,
                "max_output_tokens": 8192,
                "metadata": {
                    "display_name": "DeepSeek Chat",
                    "provider_key": "deepseek",
                    "provider_display": "DeepSeek",
                    "description": "通用模型",
                    "knowledge_cutoff": "2025-01",
                    "capabilities": {"functionCalling": True, "vision": False},
                    "pricing": {"input": 1.2, "output": 2.4, "unit": "USD"},
                    "cost_tier": "low",
                    "recommended_for": ["general"],
                    "api_key": "catalog-private-key",
                },
            },
            "catalog-only": {
                "underlying": "openai/catalog-only-private-underlying",
                "litellm_provider": "deepseek",
                "db_model": True,
                "max_input_tokens": 32000,
                "max_output_tokens": 4096,
                "metadata": {"display_name": "Catalog Only"},
            },
            "wildcard/*": {"underlying": "openai/*", "db_model": False, "metadata": {}},
        }
        health = {"status": "healthy", "error": None, "checked_at": 1783857600.0}

        with (
            patch("app.services.admin_audit_service.litellm_catalog.list_aliases", return_value=catalog),
            patch(
                "app.services.admin_audit_service.litellm_catalog.get_model_entry",
                side_effect=lambda model_id: catalog.get(model_id),
            ),
            patch("app.services.admin_audit_service.litellm_health.get_health", return_value=health),
            patch(
                "app.services.admin_audit_service.litellm_catalog.get_cache_status",
                return_value={"availability": "available", "has_cache": True},
            ),
            patch(
                "app.services.admin_audit_service.get_agent_tools_disabled_aliases",
                create=True,
                return_value={"deepseek-chat"},
            ) as disabled_aliases,
        ):
            listed = self.client.get("/api/admin/audit/models?page_size=100")
            self.assertEqual(disabled_aliases.call_count, 1)
            filtered = self.client.get(
                "/api/admin/audit/models?page_size=100&provider=%20DeepSeek%20&health_status=healthy"
            )
            unknown_health = self.client.get("/api/admin/audit/models?page_size=100&health_status=unknown")
            active_detail = self.client.get("/api/admin/audit/models/deepseek-chat")
            retired_detail = self.client.get("/api/admin/audit/models/retired/model-v1")
            missing = self.client.get("/api/admin/audit/models/missing/model")

        with (
            patch("app.services.admin_audit_service.litellm_catalog.list_aliases", return_value={}),
            patch("app.services.admin_audit_service.litellm_catalog.get_model_entry", return_value=None),
            patch(
                "app.services.admin_audit_service.litellm_catalog.get_cache_status",
                return_value={"availability": "degraded", "has_cache": False},
            ),
            patch("app.services.admin_audit_service.get_agent_tools_disabled_aliases") as outage_disabled_aliases,
        ):
            catalog_unavailable = self.client.get("/api/admin/audit/models?page_size=100")
            degraded_detail = self.client.get("/api/admin/audit/models/retired/model-v1")

        self.assertEqual(listed.status_code, 200)
        items = {item["model_id"]: item for item in listed.json()["data"]["items"]}
        self.assertEqual(set(items), {"catalog-only", "deepseek-chat", "retired/model-v1"})
        active = items["deepseek-chat"]
        self.assertEqual(active["catalog_status"], "active")
        self.assertEqual(active["catalog_availability"], "available")
        self.assertEqual(active["assistant_message_count"], 1)
        self.assertEqual(active["input_tokens"], 1_500_000_000)
        self.assertEqual(active["output_tokens"], 20)
        self.assertFalse(active["capabilities"]["agentTools"])
        self.assertNotIn("pricing", active)
        self.assertEqual(listed.json()["data"]["catalog_availability"], "available")
        self.assertIsNone(items["catalog-only"]["cost_tier"])
        retired = items["retired/model-v1"]
        self.assertEqual(retired["catalog_status"], "historical")
        self.assertEqual(retired["conversation_count"], 1)
        self.assertEqual(retired["user_count"], 1)
        self.assertEqual(retired["agent_run_count"], 1)
        self.assertEqual(retired["agent_error_count"], 1)
        self.assertEqual(set(retired["capabilities"].values()), {False})
        self.assertEqual(retired["latest_performance_run"]["source"], "admin_imported_performance_run")
        self.assertEqual(retired["latest_performance_run"]["run_id"], "perf-model-retired")
        self.assertNotIn("safe_summary", retired["latest_performance_run"])
        self.assertEqual(
            retired["metric_scope"]["assistant_messages_and_tokens"],
            "persisted_assistant_messages",
        )
        self.assertEqual(active_detail.status_code, 200)
        self.assertEqual(active_detail.json()["data"]["catalog_source"], "litellm_model_info")
        self.assertEqual(retired_detail.status_code, 200)
        self.assertEqual(retired_detail.json()["data"]["model_id"], "retired/model-v1")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(filtered.status_code, 200)
        self.assertEqual(
            [item["model_id"] for item in filtered.json()["data"]["items"]],
            ["catalog-only", "deepseek-chat"],
        )
        self.assertEqual(
            [item["model_id"] for item in unknown_health.json()["data"]["items"]],
            ["retired/model-v1"],
        )
        self.assertEqual(catalog_unavailable.status_code, 200)
        unavailable_items = catalog_unavailable.json()["data"]["items"]
        self.assertEqual({item["model_id"] for item in unavailable_items}, {"deepseek-chat", "retired/model-v1"})
        self.assertEqual({item["catalog_status"] for item in unavailable_items}, {"unknown"})
        self.assertEqual({item["health"]["status"] for item in unavailable_items}, {"unknown"})
        self.assertEqual(catalog_unavailable.json()["data"]["catalog_availability"], "degraded")
        self.assertEqual(degraded_detail.status_code, 200)
        self.assertEqual(degraded_detail.json()["data"]["catalog_status"], "unknown")
        self.assertEqual(degraded_detail.json()["data"]["catalog_availability"], "degraded")
        outage_disabled_aliases.assert_not_called()
        serialized = listed.text + retired_detail.text
        for forbidden in (
            "provider-private-model",
            "catalog-private-key",
            "provider-private-error",
            "total_duration_ms",
            "tool_duration",
            "latency",
            "litellm_params",
        ):
            self.assertNotIn(forbidden, serialized)
        actions = {
            event.action
            for event in self.db.query(AdminAuditEvent).filter(AdminAuditEvent.resource_type == "model").all()
        }
        self.assertEqual(actions, {"admin.audit.models.list", "admin.audit.model.view"})

    def test_model_operations_excludes_invalid_historical_ids_without_breaking_list(self):
        from unittest.mock import patch

        from app.db.models import Conversation

        invalid_ids = ["x" * 201, "bad\nmodel", " padded-model "]
        self.db.add_all(
            [
                Conversation(
                    id=f"conv-invalid-{index}",
                    user_id="user-1",
                    title="异常历史模型",
                    model_id=model_id,
                    created_at=datetime(2026, 7, 12, 12, 0, 0),
                    updated_at=datetime(2026, 7, 12, 12, 0, 0),
                )
                for index, model_id in enumerate(invalid_ids)
            ]
        )
        self.db.commit()

        with patch("app.services.admin_audit_service.litellm_catalog.list_aliases", return_value={}):
            listed = self.client.get("/api/admin/audit/models?page_size=100")
            too_long = self.client.get(f"/api/admin/audit/models/{'x' * 201}")
            padded = self.client.get("/api/admin/audit/models/%20padded-model%20")

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["data"]["excluded_invalid_model_count"], 3)
        self.assertEqual([item["model_id"] for item in listed.json()["data"]["items"]], ["deepseek-chat"])
        self.assertNotIn(invalid_ids[0], listed.text)
        self.assertNotIn(invalid_ids[1], listed.text)
        self.assertEqual(too_long.status_code, 400)
        self.assertEqual(padded.status_code, 400)

    def test_model_provider_options_ignore_current_filters_and_pagination_with_stable_deduplication(self):
        from unittest.mock import patch

        catalog = {
            "alpha-model": {
                "db_model": True,
                "litellm_provider": "alpha",
                "metadata": {"provider_key": "alpha", "provider_display": "Zulu Provider"},
            },
            "zeta-model-a": {
                "db_model": True,
                "litellm_provider": "zeta",
                "metadata": {"provider_key": "zeta", "provider_display": "Alpha Provider"},
            },
            "zeta-model-b": {
                "db_model": True,
                "litellm_provider": "zeta",
                "metadata": {"provider_key": "zeta", "provider_display": "Omega Provider"},
            },
            "gamma-model": {
                "db_model": True,
                "litellm_provider": "gamma",
                "metadata": {},
            },
        }

        with (
            patch("app.services.admin_audit_service.litellm_catalog.list_aliases", return_value=catalog),
            patch(
                "app.services.admin_audit_service.litellm_catalog.get_cache_status",
                return_value={"availability": "available", "has_cache": True},
            ),
            patch(
                "app.services.admin_audit_service.litellm_health.get_health",
                side_effect=lambda model_id: {
                    "status": "healthy" if model_id == "alpha-model" else "unhealthy",
                    "error": None,
                    "checked_at": 1783857600.0,
                },
            ),
            patch(
                "app.services.admin_audit_service.get_agent_tools_disabled_aliases",
                return_value=set(),
            ),
        ):
            active_filtered = self.client.get(
                "/api/admin/audit/models?page=1&page_size=1&q=alpha-model"
                "&provider=alpha&catalog_status=active&health_status=healthy"
            )
            historical_filtered = self.client.get(
                "/api/admin/audit/models?page=1&page_size=1&q=deepseek"
                "&provider=alpha&catalog_status=historical&health_status=healthy"
            )

        expected_options = [
            {"value": "zeta", "label": "Alpha Provider"},
            {"value": "gamma", "label": "gamma"},
            {"value": "alpha", "label": "Zulu Provider"},
        ]
        self.assertEqual(active_filtered.status_code, 200)
        self.assertEqual(
            [item["model_id"] for item in active_filtered.json()["data"]["items"]],
            ["alpha-model"],
        )
        self.assertEqual(active_filtered.json()["data"]["provider_options"], expected_options)
        self.assertEqual(historical_filtered.status_code, 200)
        self.assertEqual(historical_filtered.json()["data"]["items"], [])
        self.assertEqual(historical_filtered.json()["data"]["provider_options"], expected_options)

    def test_model_provider_options_disambiguate_casefold_colliding_labels_deterministically(self):
        from unittest.mock import patch

        catalog = {
            "provider-b-model": {
                "db_model": True,
                "litellm_provider": "provider-b",
                "metadata": {"provider_key": "provider-b", "provider_display": "shared provider"},
            },
            "unique-model": {
                "db_model": True,
                "litellm_provider": "unique",
                "metadata": {"provider_key": "unique", "provider_display": "Unique Provider"},
            },
            "provider-a-model": {
                "db_model": True,
                "litellm_provider": "provider-a",
                "metadata": {"provider_key": "provider-a", "provider_display": "Shared Provider"},
            },
        }
        reversed_catalog = dict(reversed(list(catalog.items())))

        with (
            patch(
                "app.services.admin_audit_service.litellm_catalog.list_aliases",
                side_effect=[catalog, reversed_catalog],
            ),
            patch(
                "app.services.admin_audit_service.litellm_catalog.get_cache_status",
                return_value={"availability": "available", "has_cache": True},
            ),
            patch(
                "app.services.admin_audit_service.litellm_health.get_health",
                return_value={"status": "healthy", "error": None, "checked_at": 1783857600.0},
            ),
            patch(
                "app.services.admin_audit_service.get_agent_tools_disabled_aliases",
                return_value=set(),
            ),
        ):
            first = self.client.get("/api/admin/audit/models?page_size=1&provider=provider-a")
            second = self.client.get("/api/admin/audit/models?page_size=1&provider=provider-b")

        expected_options = [
            {"value": "provider-a", "label": "Shared Provider（provider-a）"},
            {"value": "provider-b", "label": "shared provider（provider-b）"},
            {"value": "unique", "label": "Unique Provider"},
        ]
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["data"]["provider_options"], expected_options)
        self.assertEqual(second.json()["data"]["provider_options"], expected_options)

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

    def test_trajectory_diagnostics_are_audited_bounded_and_separate_from_normal_snapshot(self):
        """若管理员接口绕过审计、泄漏原始工具字段或伪造 span 关联，本端到端契约必须失败。"""
        from app.db.models import AdminAuditEvent, ToolCallLog

        self._add_trajectory_run("run-audit")
        self.db.add(
            ToolCallLog(
                id="tool-audit-web",
                conversation_id="conv-1",
                message_id="msg-run-audit",
                user_id="user-1",
                tool_name="web_search",
                status="failed",
                model_id="model-1",
                provider="provider-1",
                input_params={"query": "安全查询", "authorization": "input-secret"},
                output_data={"result_count": 1, "private_payload": "output-secret"},
                error_message="Bearer error-secret",
                trace_id="run-audit",
                step_number=2,
                created_at=datetime(2026, 7, 11, 12, 1, 0),
            )
        )
        self.db.commit()

        self.current_user.is_superuser = False
        forbidden = self.client.get("/api/admin/audit/conversations/conv-1/runs/run-audit/trajectory")
        self.current_user.id = "user-1"
        normal = self.client.get("/api/conversations/conv-1/runs/run-audit/trajectory")
        self.current_user.id = "admin-1"
        self.current_user.is_superuser = True
        response = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-audit/trajectory",
            headers={"X-Admin-Audit-Reason": "support trajectory"},
        )

        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(normal.status_code, 200)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        data = response.json()["data"]
        self.assertEqual(data["snapshot"], normal.json()["data"])
        self.assertNotEqual(set(data), set(normal.json()["data"]))
        tool = next(item for item in data["tool_calls"] if item["id"] == "tool-audit-web")
        self.assertEqual(tool["association"], "run")
        self.assertNotIn("tool_call_id", tool)
        self.assertEqual(tool["arguments"], {"query": "安全查询"})
        self.assertEqual(tool["result_preview"], {"result_count": 1})
        self.assertEqual(tool["error"], {"type": "execution_failed", "message": "执行失败"})
        self.assertIn("error", tool["redacted_fields"])
        self.assertEqual([item["id"] for item in data["tool_calls"]], ["tool-run-audit", "tool-audit-web"])
        self.assertNotIn("input-secret", response.text)
        self.assertNotIn("output-secret", response.text)
        self.assertNotIn("error-secret", response.text)
        event = self.db.query(AdminAuditEvent).filter(AdminAuditEvent.action == "admin.audit.trajectory.view").one()
        self.assertEqual(event.resource_type, "conversation_run_trajectory")
        self.assertEqual(event.resource_id, "run-audit")
        self.assertEqual(event.target_user_id, "user-1")
        self.assertEqual(event.reason, "support trajectory")
        self.assertEqual(event.extra_metadata, {})

    def test_trajectory_diagnostics_hide_cross_conversation_runs_and_fail_closed_when_audit_fails(self):
        """若 run 归属仅靠管理员权限，或审计失败仍返回诊断，敏感轨迹会泄漏。"""
        from unittest.mock import patch

        from app.db.models import Conversation, Message, User

        self._add_trajectory_run("run-audit-own")
        self.db.add_all(
            [
                User(id="user-2", username="bob", email="bob@example.com"),
                Conversation(
                    id="conv-2",
                    user_id="user-2",
                    title="他人的审计对话",
                    model_id="model-1",
                ),
                Message(
                    id="msg-run-audit-other",
                    conversation_id="conv-2",
                    role="assistant",
                    content=[{"type": "text", "text": "other conversation"}],
                ),
            ]
        )
        self.db.commit()
        self._add_trajectory_run("run-audit-other", conversation_id="conv-2", user_id="user-2")

        cross_run = self.client.get("/api/admin/audit/conversations/conv-1/runs/run-audit-other/trajectory")
        with patch(
            "app.db.admin_audit_repository.AdminAuditRepository.create_audit_event",
            side_effect=RuntimeError("audit unavailable"),
        ):
            audit_failed = self.client.get("/api/admin/audit/conversations/conv-1/runs/run-audit-own/trajectory")

        self.assertEqual(cross_run.status_code, 404)
        self.assertEqual(audit_failed.status_code, 503)
        self.assertNotIn("input-secret", audit_failed.text)
        self.assertNotIn("output-secret", audit_failed.text)
        self.assertNotIn("error-secret", audit_failed.text)

    def test_tool_node_detail_requires_reason_audits_access_and_fails_closed(self):
        """若管理员详情允许无原因读取或审计失败仍返回，敏感诊断访问不可追溯。"""
        from unittest.mock import patch

        from app.db.models import AdminAuditEvent, Conversation, User

        self._add_trajectory_run("run-node-detail")
        self.db.add_all(
            [
                User(id="user-2", username="bob", email="bob@example.com"),
                Conversation(id="conv-2", user_id="user-2", title="他人的对话", model_id="model-1"),
            ]
        )
        self.db.commit()
        self._add_trajectory_run("run-node-other", conversation_id="conv-2", user_id="user-2")

        missing_reason = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-node-detail/node-detail/tool/call-run-node-detail"
        )
        blank_reason = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-node-detail/node-detail/tool/call-run-node-detail",
            headers={"X-Admin-Audit-Reason": "   "},
        )
        self.current_user.is_superuser = False
        forbidden = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-node-detail/node-detail/tool/call-run-node-detail",
            headers={"X-Admin-Audit-Reason": "support node detail"},
        )
        self.current_user.is_superuser = True
        cross_run = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-node-other/node-detail/tool/call-run-node-other",
            headers={"X-Admin-Audit-Reason": "support node detail"},
        )
        response = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-node-detail/node-detail/tool/call-run-node-detail",
            headers={"X-Admin-Audit-Reason": "  support node detail  "},
        )

        self.assertEqual(missing_reason.status_code, 422)
        self.assertEqual(blank_reason.status_code, 400)
        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(cross_run.status_code, 404)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["status"], "available")
        event = (
            self.db.query(AdminAuditEvent)
            .filter(AdminAuditEvent.action == "admin.audit.trajectory.node_detail.view")
            .one()
        )
        self.assertEqual(event.resource_type, "conversation_run_tool_node_detail")
        self.assertEqual(event.resource_id, "run-node-detail:call-run-node-detail")
        self.assertEqual(event.target_user_id, "user-1")
        self.assertEqual(event.reason, "support node detail")
        self.assertEqual(event.extra_metadata, {})

        with patch(
            "app.db.admin_audit_repository.AdminAuditRepository.create_audit_event",
            side_effect=RuntimeError("audit unavailable"),
        ):
            audit_failed = self.client.get(
                "/api/admin/audit/conversations/conv-1/runs/run-node-detail/node-detail/tool/call-run-node-detail",
                headers={"X-Admin-Audit-Reason": "support node detail"},
            )
        self.assertEqual(audit_failed.status_code, 503)
        self.assertNotIn("input-secret", audit_failed.text)
        self.assertNotIn("output-secret", audit_failed.text)
        self.assertNotIn("error-secret", audit_failed.text)

    def test_llm_node_detail_requires_reason_and_records_llm_audit_resource(self):
        from app.db.models import AdminAuditEvent, AgentEvent, AgentLlmRoundDetail, RunTrajectoryMeta

        self._add_trajectory_run("run-llm-node")
        meta = self.db.get(RunTrajectoryMeta, "run-llm-node")
        meta.llm_detail_schema_version = 1
        created = datetime(2026, 7, 11, 12, 0, 1)
        self.db.add_all(
            [
                AgentEvent(
                    conversation_id="conv-1",
                    message_id="msg-run-llm-node",
                    run_id="run-llm-node",
                    sequence=1,
                    event_type="llm_round_started",
                    schema_version=1,
                    event_ts=created,
                    step_id="step-1",
                    payload={"llm_round_id": "round-1", "round_index": 1},
                ),
                AgentEvent(
                    conversation_id="conv-1",
                    message_id="msg-run-llm-node",
                    run_id="run-llm-node",
                    sequence=2,
                    event_type="llm_round_completed",
                    schema_version=1,
                    event_ts=created,
                    step_id="step-1",
                    payload={"llm_round_id": "round-1", "duration_ms": 100},
                ),
                AgentLlmRoundDetail(
                    conversation_id="conv-1",
                    run_id="run-llm-node",
                    message_id="msg-run-llm-node",
                    llm_round_id="round-1",
                    reasoning_text="显式推理",
                    content_text="输出",
                    reasoning_preview="显式推理",
                    output_preview="输出",
                    redacted_fields=[],
                    truncated_fields=[],
                ),
            ]
        )
        self.db.commit()

        missing_reason = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-llm-node/node-detail/llm/round-1"
        )
        response = self.client.get(
            "/api/admin/audit/conversations/conv-1/runs/run-llm-node/node-detail/llm/round-1",
            headers={"X-Admin-Audit-Reason": "review model reasoning"},
        )

        self.assertEqual(missing_reason.status_code, 422)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["node_type"], "llm")
        event = (
            self.db.query(AdminAuditEvent)
            .filter(AdminAuditEvent.resource_type == "conversation_run_llm_node_detail")
            .one()
        )
        self.assertEqual(event.resource_id, "run-llm-node:round-1")
        self.assertEqual(event.reason, "review model reasoning")


if __name__ == "__main__":
    unittest.main()
