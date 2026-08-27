"""普通用户轨迹 API 的鉴权、信封与脱敏边界。"""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from datetime import UTC, datetime, timedelta
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

CAPABILITY_RESOLUTION = {
    "schema_version": 1,
    "router_version": "2026-08-27.1",
    "package_id": "fresh_web",
    "confidence": "high",
    "resolution_mode": "routed",
    "reason_codes": ["fresh_external_fact"],
    "external_tool_names": ["web_search"],
    "effective_plan_mode": "off",
    "include_current_date": True,
    "network_boundary_required": False,
    "bundle_fingerprint": "sha256:" + "a" * 64,
}


class TrajectoryApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")
        cls.client = TestClient(cls.main.app)

    def setUp(self) -> None:
        from app.db.database import Base

        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db = self.Session()
        self.now = datetime(2026, 8, 22, 4, 0, tzinfo=UTC)
        self._seed()

        from app.api.deps import get_current_user, get_db
        from app.core.config import settings

        self.current_user = SimpleNamespace(
            id="user-1", username="alice", email="alice@example.com", is_superuser=False
        )
        self.main.app.dependency_overrides[get_current_user] = lambda: self.current_user
        self.main.app.dependency_overrides[get_db] = lambda: self.db
        self.previous_max_events = settings.MAX_TRAJECTORY_EVENTS_PER_RUN
        self.previous_max_runs = settings.MAX_TRAJECTORY_RUNS_PER_CONVERSATION
        settings.MAX_TRAJECTORY_EVENTS_PER_RUN = 2
        settings.MAX_TRAJECTORY_RUNS_PER_CONVERSATION = 2

    def tearDown(self) -> None:
        from app.core.config import settings

        settings.MAX_TRAJECTORY_EVENTS_PER_RUN = self.previous_max_events
        settings.MAX_TRAJECTORY_RUNS_PER_CONVERSATION = self.previous_max_runs
        self.main.app.dependency_overrides.clear()
        self.db.close()
        self.engine.dispose()

    def _seed(self) -> None:
        from app.db.models import Conversation, TrajectoryLedgerSettings, User

        self.db.add_all(
            [
                User(id="user-1", username="alice", email="alice@example.com"),
                User(id="user-2", username="bob", email="bob@example.com"),
                Conversation(id="conv-1", user_id="user-1", title="我的会话", model_id="model-1"),
                Conversation(id="conv-2", user_id="user-2", title="他人的会话", model_id="model-1"),
                TrajectoryLedgerSettings(singleton_key="default", ledger_enabled_at=self.now - timedelta(days=1)),
            ]
        )
        self.db.commit()

    def _add_run(
        self,
        run_id: str,
        *,
        conversation_id: str = "conv-1",
        user_id: str = "user-1",
        trajectory_status: str = "complete",
        terminal_intent_reason: str | None = None,
        run_config: dict | None = None,
    ) -> None:
        from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta, ToolCallLog

        self.db.add(
            AgentSession(
                id=run_id,
                conversation_id=conversation_id,
                user_id=user_id,
                message_id=f"msg-{run_id}",
                turn_message_id=f"turn-{run_id}",
                attempt_index=1,
                model_id="model-1",
                provider="provider-1",
                status="completed",
                total_steps=2,
                total_tool_calls=3,
                total_duration_ms=123,
                run_config=run_config,
                created_at=self.now,
                terminal_at=self.now,
            )
        )
        if conversation_id == "conv-1":
            self.db.add(
                RunTrajectoryMeta(
                    run_id=run_id,
                    conversation_id=conversation_id,
                    trajectory_status=trajectory_status,
                    event_count=3,
                    expected_last_sequence=2,
                    terminal_intent_pending_at=self.now if terminal_intent_reason else None,
                    terminal_intent_reason=terminal_intent_reason,
                )
            )
            self.db.add_all(
                [
                    AgentEvent(
                        conversation_id=conversation_id,
                        message_id=f"msg-{run_id}",
                        run_id=run_id,
                        sequence=sequence,
                        event_type="step_completed",
                        schema_version=1,
                        event_ts=self.now + timedelta(seconds=sequence),
                        payload={"duration_ms": 10, "safe_summary": f"event-{sequence}"},
                    )
                    for sequence in range(3)
                ]
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
                    created_at=self.now,
                )
            )
        self.db.commit()

    def test_run_list_and_snapshot_use_success_envelope_and_bounded_safe_payload(self):
        """若 router 未挂 /api、未传播 request_id 或读取 ToolCallLog，本端到端契约必须失败。"""
        self._add_run("run-1")

        listing = self.client.get("/api/conversations/conv-1/runs")
        snapshot = self.client.get("/api/conversations/conv-1/runs/run-1/trajectory")

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(listing.json()["code"], "SUCCESS")
        self.assertTrue(listing.json()["request_id"])
        self.assertEqual(listing.json()["data"]["items"][0]["run_id"], "run-1")
        self.assertEqual(snapshot.status_code, 200)
        data = snapshot.json()["data"]
        self.assertTrue(data["truncated"])
        self.assertEqual([item["sequence"] for item in data["records"]], [0, 1])
        self.assertEqual(data["run"]["total_steps"], 2)
        self.assertEqual(data["completeness"]["loaded_event_count"], 2)
        self.assertNotIn("input-secret", snapshot.text)
        self.assertNotIn("output-secret", snapshot.text)
        self.assertNotIn("error-secret", snapshot.text)
        self.assertNotIn("input_params", snapshot.text)
        self.assertNotIn("output_data", snapshot.text)
        self.assertNotIn("terminal_intent", snapshot.text)

    def test_run_list_and_snapshot_expose_only_persisted_safe_capability_resolution(self):
        self._add_run(
            "run-resolution",
            run_config={
                "capability_resolution": CAPABILITY_RESOLUTION,
                "authorization": "Bearer hidden",
                "system_prompt_snapshot": {"content": "PRIVATE 正文"},
            },
        )

        listing = self.client.get("/api/conversations/conv-1/runs")
        snapshot = self.client.get("/api/conversations/conv-1/runs/run-resolution/trajectory")

        self.assertEqual(listing.status_code, 200)
        self.assertEqual(snapshot.status_code, 200)
        self.assertEqual(listing.json()["data"]["items"][0]["capability_resolution"], CAPABILITY_RESOLUTION)
        self.assertEqual(snapshot.json()["data"]["run"]["capability_resolution"], CAPABILITY_RESOLUTION)
        for response in (listing, snapshot):
            self.assertNotIn("Bearer hidden", response.text)
            self.assertNotIn("authorization", response.text)
            self.assertNotIn("PRIVATE 正文", response.text)
            self.assertNotIn("system_prompt_snapshot", response.text)

    def test_legacy_and_invalid_run_capability_resolution_are_null(self):
        self._add_run("run-legacy", run_config={"max_steps": 8})
        invalid_resolutions = (
            (
                "run-invalid-extra",
                {
                    **CAPABILITY_RESOLUTION,
                    "original_message": "用户原文禁止返回",
                },
            ),
            (
                "run-invalid-control",
                {
                    **CAPABILITY_RESOLUTION,
                    "package_id": "mcp_explicit",
                    "reason_codes": ["explicit_authorized_tool_alias"],
                    "external_tool_names": ["update_plan"],
                    "include_current_date": False,
                },
            ),
            (
                "run-invalid-package",
                {
                    **CAPABILITY_RESOLUTION,
                    "package_id": "direct",
                    "reason_codes": ["direct_greeting"],
                    "external_tool_names": ["web_search"],
                    "include_current_date": False,
                },
            ),
        )
        for run_id, resolution in invalid_resolutions:
            self._add_run(run_id, run_config={"capability_resolution": resolution})

        legacy = self.client.get("/api/conversations/conv-1/runs/run-legacy/trajectory")
        self.assertEqual(legacy.status_code, 200)
        self.assertIsNone(legacy.json()["data"]["run"]["capability_resolution"])
        for run_id, _resolution in invalid_resolutions:
            with self.subTest(run_id=run_id):
                invalid = self.client.get(f"/api/conversations/conv-1/runs/{run_id}/trajectory")
                self.assertEqual(invalid.status_code, 200)
                self.assertIsNone(invalid.json()["data"]["run"]["capability_resolution"])
                self.assertNotIn("用户原文禁止返回", invalid.text)

    def test_unauthorized_conversation_and_cross_conversation_run_are_uniformly_not_found(self):
        """若 handler 在 service 之外泄漏资源归属，404 契约会被破坏。"""
        self._add_run("run-own")
        self._add_run("run-other", conversation_id="conv-2", user_id="user-2")

        foreign_conversation = self.client.get("/api/conversations/conv-2/runs")
        cross_run = self.client.get("/api/conversations/conv-1/runs/run-other/trajectory")
        missing_run = self.client.get("/api/conversations/conv-1/runs/no-such-run/trajectory")

        for response in (foreign_conversation, cross_run, missing_run):
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["code"], "NOT_FOUND")
            self.assertEqual(response.json()["message"], "会话或轨迹不存在，或无权访问")

    def test_pending_intent_reason_and_persisted_legacy_meta_do_not_leak_to_normal_response(self):
        """若普通 DTO 读取 terminal intent 或信任已有 legacy meta，此回归必须失败。"""
        self._add_run("run-pending", terminal_intent_reason="write_failed")
        self._add_run("run-legacy-meta", trajectory_status="legacy")

        pending = self.client.get("/api/conversations/conv-1/runs/run-pending/trajectory")
        legacy_meta = self.client.get("/api/conversations/conv-1/runs/run-legacy-meta/trajectory")

        self.assertEqual(pending.status_code, 200)
        self.assertNotIn("write_failed", pending.text)
        self.assertNotIn("terminal_intent_reason", pending.text)
        self.assertEqual(pending.json()["data"]["completeness"]["degraded_reason"], "terminal_outcome_unknown")
        self.assertEqual(legacy_meta.status_code, 200)
        self.assertEqual(legacy_meta.json()["data"]["completeness"]["status"], "degraded")
        self.assertEqual(legacy_meta.json()["data"]["completeness"]["degraded_reason"], "terminal_outcome_unknown")

    def test_tool_node_detail_endpoint_returns_safe_exact_detail_and_uniform_404(self):
        """若普通端点未按会话、用户、run 与 tool_call_id 精确鉴权，详情会越权或误关联。"""
        self._add_run("run-detail")
        self._add_run("run-other", conversation_id="conv-2", user_id="user-2")

        exact = self.client.get("/api/conversations/conv-1/runs/run-detail/node-detail/tool/call-run-detail")
        wrong_node = self.client.get("/api/conversations/conv-1/runs/run-detail/node-detail/tool/tool-run-detail")
        cross_run = self.client.get("/api/conversations/conv-1/runs/run-other/node-detail/tool/call-run-other")

        self.assertEqual(exact.status_code, 200)
        data = exact.json()["data"]
        self.assertEqual(data["status"], "available")
        self.assertEqual(data["node_type"], "tool")
        self.assertEqual(data["detail"]["tool_call_id"], "call-run-detail")
        self.assertIsNone(data["detail"]["payload"])
        self.assertIsNone(data["detail"]["result"])
        self.assertIn("payload", data["redacted_fields"])
        self.assertIn("result", data["redacted_fields"])
        self.assertNotIn('"secret"', exact.text)
        self.assertNotIn("input-secret", exact.text)
        self.assertNotIn("output-secret", exact.text)
        self.assertNotIn("error-secret", exact.text)
        for response in (cross_run,):
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["message"], "会话或轨迹不存在，或无权访问")
        self.assertEqual(wrong_node.status_code, 200)
        self.assertEqual(wrong_node.json()["data"]["status"], "not_recorded")
        self.assertIsNone(wrong_node.json()["data"]["detail"])

    def test_llm_node_detail_endpoint_returns_exact_visible_text_and_uniform_404(self):
        from app.db.models import AgentEvent, AgentLlmRoundDetail, RunTrajectoryMeta

        self._add_run("run-llm-detail")
        meta = self.db.get(RunTrajectoryMeta, "run-llm-detail")
        meta.llm_detail_schema_version = 1
        self.db.add_all(
            [
                AgentEvent(
                    conversation_id="conv-1",
                    message_id="msg-run-llm-detail",
                    run_id="run-llm-detail",
                    sequence=3,
                    event_type="llm_round_started",
                    schema_version=1,
                    event_ts=self.now,
                    step_id="step-1",
                    payload={"llm_round_id": "round-exact", "round_index": 1},
                ),
                AgentEvent(
                    conversation_id="conv-1",
                    message_id="msg-run-llm-detail",
                    run_id="run-llm-detail",
                    sequence=4,
                    event_type="llm_round_completed",
                    schema_version=1,
                    event_ts=self.now,
                    step_id="step-1",
                    payload={"llm_round_id": "round-exact", "duration_ms": 100},
                ),
                AgentLlmRoundDetail(
                    conversation_id="conv-1",
                    run_id="run-llm-detail",
                    message_id="msg-run-llm-detail",
                    llm_round_id="round-exact",
                    reasoning_text="模型显式推理",
                    content_text="模型输出",
                    reasoning_preview="模型显式推理",
                    output_preview="模型输出",
                    redacted_fields=[],
                    truncated_fields=[],
                ),
            ]
        )
        self.db.commit()

        exact = self.client.get("/api/conversations/conv-1/runs/run-llm-detail/node-detail/llm/round-exact")
        wrong_node = self.client.get("/api/conversations/conv-1/runs/run-llm-detail/node-detail/llm/round-other")
        cross_user = self.client.get("/api/conversations/conv-2/runs/run-llm-detail/node-detail/llm/round-exact")

        self.assertEqual(exact.status_code, 200)
        data = exact.json()["data"]
        self.assertEqual(data["node_type"], "llm")
        self.assertEqual(data["detail"]["reasoning_text"], "模型显式推理")
        self.assertEqual(data["detail"]["output_text"], "模型输出")
        for response in (wrong_node, cross_user):
            self.assertEqual(response.status_code, 404)
            self.assertEqual(response.json()["message"], "会话或轨迹不存在，或无权访问")

    def test_system_prompt_detail_returns_persisted_body_only_to_exact_owner(self):
        """正文端点必须有普通用户信封、完整归属校验，且列表和快照不携带正文。"""
        from app.db.models import AgentEvent, AgentSession, AgentSystemPromptSnapshot

        self._add_run("run-prompt")
        self._add_run("run-prompt-old")
        self._add_run("run-prompt-other", conversation_id="conv-2", user_id="user-2")
        self._add_run("run-prompt-inconsistent", user_id="user-2")
        snapshot = {
            "schema_version": 1,
            "template_version": "2026-08-26.1",
            "fingerprint": "b62d9881a561f21dc1dc33e8b8e288ea71d1ed13744258ef986c6cea86b73d39",
            "char_count": 13,
            "sections": [{"section_id": "app_identity", "content": "只属于当前 Run 的正文"}],
        }
        self.db.get(AgentSession, "run-prompt").run_config = {"unrelated_config": "内部配置"}
        self.db.get(AgentSession, "run-prompt-other").run_config = {"unrelated_config": "内部配置"}
        self.db.get(AgentSession, "run-prompt-inconsistent").run_config = {"unrelated_config": "内部配置"}
        self.db.add_all(
            [
                AgentSystemPromptSnapshot(
                    run_id="run-prompt",
                    conversation_id="conv-1",
                    user_id="user-1",
                    snapshot=snapshot,
                ),
                AgentSystemPromptSnapshot(
                    run_id="run-prompt-other",
                    conversation_id="conv-2",
                    user_id="user-2",
                    snapshot=snapshot,
                ),
                AgentSystemPromptSnapshot(
                    run_id="run-prompt-inconsistent",
                    conversation_id="conv-1",
                    user_id="user-2",
                    snapshot=snapshot,
                ),
                AgentEvent(
                    conversation_id="conv-1",
                    message_id="msg-run-prompt",
                    run_id="run-prompt",
                    sequence=3,
                    event_type="system_prompt_prepared",
                    schema_version=1,
                    event_ts=self.now,
                    payload={"status": "ready", "detail_status": "available", "fingerprint": snapshot["fingerprint"]},
                ),
            ]
        )
        self.db.commit()
        self.db.close()
        self.db = self.Session()

        exact = self.client.get("/api/conversations/conv-1/runs/run-prompt/node-detail/system-prompt")

        self.assertEqual(exact.status_code, 200)
        self.assertEqual(exact.headers.get("cache-control"), "private, no-store")
        self.assertEqual(exact.json()["code"], "SUCCESS")
        self.assertTrue(exact.json()["request_id"])
        data = exact.json()["data"]
        self.assertEqual(data["node_type"], "system_prompt")
        self.assertEqual(data["status"], "available")
        self.assertEqual(data["available_sections"], ["summary", "prompt"])
        self.assertEqual(data["detail"], {key: value for key, value in snapshot.items() if key != "schema_version"})
        self.assertEqual(data["redacted_fields"], [])
        self.assertEqual(data["truncated_fields"], [])
        for path in (
            "/api/conversations/conv-2/runs/run-prompt-other/node-detail/system-prompt",
            "/api/conversations/conv-1/runs/run-prompt-other/node-detail/system-prompt",
            "/api/conversations/conv-2/runs/run-prompt/node-detail/system-prompt",
            "/api/conversations/conv-1/runs/run-prompt-inconsistent/node-detail/system-prompt",
            "/api/conversations/conv-1/runs/missing-run/node-detail/system-prompt",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404)
                self.assertEqual(response.json()["code"], "NOT_FOUND")
                self.assertEqual(response.json()["message"], "会话或轨迹不存在，或无权访问")
                self.assertNotIn("只属于当前 Run 的正文", response.text)
        old = self.client.get("/api/conversations/conv-1/runs/run-prompt-old/node-detail/system-prompt")
        self.assertEqual(old.status_code, 200)
        self.assertEqual(old.headers.get("cache-control"), "private, no-store")
        self.assertEqual(old.json()["data"]["status"], "not_recorded")
        self.assertEqual(old.json()["data"]["reason"], "system_prompt_not_recorded")
        for path in ("/api/conversations/conv-1/runs", "/api/conversations/conv-1/runs/run-prompt/trajectory"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn("只属于当前 Run 的正文", response.text)
            self.assertNotIn("system_prompt_snapshot", response.text)


if __name__ == "__main__":
    unittest.main()
