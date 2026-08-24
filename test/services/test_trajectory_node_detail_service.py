"""Tool Node Detail 的精确关联、四态与安全投影契约。"""

from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.database import Base
from app.db.models import AgentSession, Conversation, ToolCallLog, TrajectoryLedgerSettings, User
from app.db.trajectory_repository import TrajectoryRepository
from app.services.trajectory_query_service import TrajectoryQueryService


class TrajectoryNodeDetailServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "trajectory-node-detail.sqlite3"
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.now = datetime(2026, 8, 24, 4, 0, tzinfo=UTC)
        with self.Session() as db:
            db.add_all(
                [
                    User(id="user-1", username="alice", email="alice@example.com"),
                    User(id="user-2", username="bob", email="bob@example.com"),
                    Conversation(id="conv-1", user_id="user-1", title="我的会话", model_id="model-1"),
                    Conversation(id="conv-2", user_id="user-2", title="他人的会话", model_id="model-1"),
                    TrajectoryLedgerSettings(
                        singleton_key="default",
                        ledger_enabled_at=self.now - timedelta(days=30),
                        trajectory_detail_enabled_at=self.now - timedelta(days=1),
                    ),
                ]
            )
            db.commit()

    def tearDown(self) -> None:
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _service(self, *, grace_seconds: float = 5) -> TrajectoryQueryService:
        db = self.Session()
        self.addCleanup(db.close)
        return TrajectoryQueryService(
            TrajectoryRepository(db),
            max_events_per_run=10,
            max_runs_per_conversation=10,
            detail_settle_grace_seconds=grace_seconds,
            now_provider=lambda: self.now,
        )

    def _set_watermark(self, value: datetime | None) -> None:
        with self.Session() as db:
            row = db.get(TrajectoryLedgerSettings, "default")
            assert row is not None
            row.trajectory_detail_enabled_at = value
            db.commit()

    def _run(
        self,
        run_id: str,
        *,
        conversation_id: str = "conv-1",
        user_id: str = "user-1",
        status: str = "completed",
        created_at: datetime | None = None,
        terminal_at: datetime | None = None,
    ) -> None:
        run_created_at = created_at or self.now - timedelta(minutes=1)
        with self.Session() as db:
            db.add(
                AgentSession(
                    id=run_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_id=f"msg-{run_id}",
                    model_id="model-1",
                    provider="provider-1",
                    status=status,
                    created_at=run_created_at,
                    terminal_at=terminal_at,
                )
            )
            db.commit()

    def _tool(
        self,
        run_id: str,
        tool_call_id: str,
        *,
        tool_name: str = "web_search",
        conversation_id: str = "conv-1",
        user_id: str = "user-1",
        input_params: dict | None = None,
        output_data: dict | None = None,
        duration_ms: int | None = 125,
        error_message: str | None = None,
    ) -> None:
        with self.Session() as db:
            db.add(
                ToolCallLog(
                    id=f"log-{run_id}-{tool_call_id}",
                    conversation_id=conversation_id,
                    message_id=f"msg-{run_id}",
                    user_id=user_id,
                    tool_name=tool_name,
                    status="failed" if error_message else "success",
                    duration_ms=duration_ms,
                    model_id="model-1",
                    provider="provider-1",
                    input_params=input_params,
                    output_data=output_data,
                    error_message=error_message,
                    trace_id=run_id,
                    tool_call_id=tool_call_id,
                    step_number=1,
                    created_at=self.now,
                )
            )
            db.commit()

    def test_available_precedes_watermark_and_uses_safe_allowlist_projection(self):
        """若先判水位或回传原始工具日志，水位前的精确日志会丢失并泄漏凭据。"""
        self._set_watermark(self.now)
        self._run("run-before", created_at=self.now - timedelta(days=1), terminal_at=self.now - timedelta(days=1))
        self._tool(
            "run-before",
            "call-exact",
            input_params={"query": "安全查询", "authorization": "Bearer input-secret"},
            output_data={
                "result_count": 1,
                "private_payload": "output-secret",
                "sources": [
                    {
                        "title": "标题",
                        "url": "https://example.com/path?token=secret",
                        "status": "ok",
                        "raw_content": "source-secret",
                    }
                ],
            },
            error_message="Bearer error-secret",
        )

        response = self._service().get_user_tool_node_detail("conv-1", "run-before", "call-exact", "user-1")

        assert response is not None and response.detail is not None
        self.assertEqual(response.status, "available")
        self.assertEqual(response.node_type, "tool")
        self.assertEqual(response.available_sections, ["summary", "payload", "result", "timing"])
        self.assertEqual(response.detail.tool_call_id, "call-exact")
        self.assertEqual(response.detail.payload, {"query": "安全查询"})
        self.assertEqual(response.detail.result["result_count"], 1)
        self.assertEqual(response.detail.error, {"type": "execution_failed", "message": "执行失败"})
        self.assertIn("payload", response.redacted_fields)
        self.assertIn("result", response.redacted_fields)
        self.assertIn("error", response.redacted_fields)
        encoded = json.dumps(response.model_dump(mode="json"), ensure_ascii=False)
        for secret in ("input-secret", "output-secret", "source-secret", "error-secret"):
            self.assertNotIn(secret, encoded)
        for hidden_key in ("authorization", "private_payload", "raw_content"):
            self.assertNotIn(hidden_key, encoded)
        for internal_field in ("log-run-before-call-exact", "trace_id", "step_number", "model_id", "provider"):
            self.assertNotIn(internal_field, encoded)
        self.assertNotIn("schema", response.available_sections)

    def test_unknown_tool_never_returns_raw_payload_or_result(self):
        """若未知工具默认透传输入输出，普通详情端点会成为原始凭据旁路。"""
        self._run("run-unknown", terminal_at=self.now - timedelta(minutes=1))
        self._tool(
            "run-unknown",
            "call-unknown",
            tool_name="future_private_tool",
            input_params={"account": "input-secret"},
            output_data={"raw": "output-secret"},
            duration_ms=None,
        )

        response = self._service().get_user_tool_node_detail("conv-1", "run-unknown", "call-unknown", "user-1")

        assert response is not None and response.detail is not None
        self.assertEqual(response.status, "available")
        self.assertIsNone(response.detail.payload)
        self.assertIsNone(response.detail.result)
        self.assertEqual(response.available_sections, ["summary"])
        self.assertIn("payload", response.redacted_fields)
        self.assertIn("result", response.redacted_fields)
        self.assertNotIn("account", response.model_dump_json())
        self.assertNotIn("raw", response.model_dump_json())
        self.assertNotIn("input-secret", response.model_dump_json())
        self.assertNotIn("output-secret", response.model_dump_json())

    def test_url_read_only_returns_existing_allowlist_fields(self):
        """若 url_read 绕过既有 allowlist，正文或凭据会进入普通详情。"""
        self._run("run-url-read", terminal_at=self.now - timedelta(minutes=1))
        self._tool(
            "run-url-read",
            "call-url-read",
            tool_name="url_read",
            input_params={
                "url": "https://example.com/page?token=input-secret",
                "reason": "核对文档",
                "authorization": "Bearer input-secret",
            },
            output_data={
                "url": "https://example.com/page?token=output-secret",
                "title": "示例",
                "status": "success",
                "content_length": 120,
                "content": "output-secret",
            },
        )

        response = self._service().get_user_tool_node_detail("conv-1", "run-url-read", "call-url-read", "user-1")

        assert response is not None and response.detail is not None
        self.assertEqual(response.status, "available")
        self.assertEqual(response.detail.tool_name, "url_read")
        self.assertEqual(set(response.detail.payload or {}), {"url", "reason"})
        self.assertEqual(set(response.detail.result or {}), {"url", "title", "status", "content_length"})
        self.assertIn("payload", response.redacted_fields)
        self.assertIn("payload.url.query.token", response.redacted_fields)
        self.assertIn("result", response.redacted_fields)
        self.assertIn("result.url.query.token", response.redacted_fields)
        self.assertNotIn("authorization", response.model_dump_json())
        self.assertNotIn('"content":', response.model_dump_json())
        self.assertNotIn("input-secret", response.model_dump_json())
        self.assertNotIn("output-secret", response.model_dump_json())

    def test_mcp_and_amap_details_exclude_internal_binding_metadata(self):
        """若普通详情沿用管理员宽投影，MCP 服务绑定与版本哈希会泄漏。"""
        for index, tool_name in enumerate(("mcp_internal_alias", "local_place_search"), start=1):
            run_id = f"run-internal-{index}"
            call_id = f"call-internal-{index}"
            self._run(run_id, terminal_at=self.now - timedelta(minutes=1))
            self._tool(
                run_id,
                call_id,
                tool_name=tool_name,
                input_params={
                    "mcp_server_id": "server-amap-secret",
                    "remote_tool_name": "maps_text_search-secret",
                    "provider": "amap-internal-provider",
                    "config_version": 7,
                    "definition_sha256": "a" * 64,
                    "argument_count": 3,
                    "internal_request_id": "request-secret",
                },
                output_data={
                    "mcp_server_id": "server-amap-secret",
                    "remote_tool_name": "maps_text_search-secret",
                    "provider": "amap-internal-provider",
                    "config_version": 7,
                    "definition_sha256": "a" * 64,
                    "status": "failed",
                    "payload_bytes": 128,
                    "error_code": "rate_limited",
                    "subcall_attempt_count": 2,
                    "remote_tools_attempted": ["maps_geo-secret"],
                    "internal_response_id": "response-secret",
                },
                error_message="MCP upstream failed with server-secret",
            )

            response = self._service().get_user_tool_node_detail("conv-1", run_id, call_id, "user-1")

            assert response is not None and response.detail is not None
            self.assertEqual(response.detail.payload, {"argument_count": 3})
            self.assertEqual(
                response.detail.result,
                {
                    "status": "failed",
                    "payload_bytes": 128,
                    "error_code": "rate_limited",
                    "subcall_attempt_count": 2,
                },
            )
            self.assertEqual(response.detail.error, {"type": "execution_failed", "message": "执行失败"})
            self.assertIn("payload", response.redacted_fields)
            self.assertIn("result", response.redacted_fields)
            self.assertIn("error", response.redacted_fields)
            encoded = response.model_dump_json()
            for hidden_key in (
                "mcp_server_id",
                "remote_tool_name",
                "provider",
                "config_version",
                "definition_sha256",
                "remote_tools_attempted",
                "internal_request_id",
                "internal_response_id",
            ):
                self.assertNotIn(hidden_key, encoded)
            for hidden_value in (
                "server-amap-secret",
                "maps_text_search-secret",
                "amap-internal-provider",
                "request-secret",
                "maps_geo-secret",
                "response-secret",
                "server-secret",
                "a" * 64,
            ):
                self.assertNotIn(hidden_value, encoded)

    def test_missing_exact_log_converges_in_deterministic_four_state_order(self):
        """若状态顺序或 terminal_at 边界改变，旧数据会被伪装或新缺失会无限 pending。"""
        self._set_watermark(None)
        self._run("run-no-watermark", terminal_at=self.now - timedelta(minutes=1))
        no_watermark = self._service().get_user_tool_node_detail("conv-1", "run-no-watermark", "missing", "user-1")

        self._set_watermark(self.now - timedelta(hours=1))
        self._run(
            "run-before-watermark",
            created_at=self.now - timedelta(hours=2),
            terminal_at=self.now - timedelta(hours=2),
        )
        before_watermark = self._service().get_user_tool_node_detail(
            "conv-1", "run-before-watermark", "missing", "user-1"
        )
        self._run("run-running", status="running", terminal_at=None)
        running = self._service().get_user_tool_node_detail("conv-1", "run-running", "missing", "user-1")
        self._run("run-settling", terminal_at=self.now - timedelta(seconds=5))
        settling = self._service().get_user_tool_node_detail("conv-1", "run-settling", "missing", "user-1")
        self._run("run-degraded", terminal_at=self.now - timedelta(seconds=5, microseconds=1))
        degraded = self._service().get_user_tool_node_detail("conv-1", "run-degraded", "missing", "user-1")
        self._run("run-no-terminal", terminal_at=None)
        no_terminal = self._service().get_user_tool_node_detail("conv-1", "run-no-terminal", "missing", "user-1")

        self.assertEqual(no_watermark.status, "not_recorded")
        self.assertEqual(before_watermark.status, "not_recorded")
        self.assertEqual(running.status, "pending")
        self.assertEqual(settling.status, "pending")
        self.assertEqual(degraded.status, "degraded")
        self.assertEqual(no_terminal.status, "degraded")
        for response in (no_watermark, before_watermark, running, settling, degraded, no_terminal):
            assert response is not None
            self.assertIsNone(response.detail)
            self.assertEqual(response.available_sections, [])
            self.assertEqual(response.redacted_fields, [])

    def test_same_step_and_name_with_different_tool_call_id_never_falls_back(self):
        """若按 step/name/time 猜测，错误工具的 Payload 会被关联到当前节点。"""
        self._run("run-no-fallback", terminal_at=self.now - timedelta(minutes=1))
        self._tool(
            "run-no-fallback",
            "call-other",
            input_params={"query": "不应关联"},
            output_data={"result_count": 1},
        )

        response = self._service().get_user_tool_node_detail("conv-1", "run-no-fallback", "call-requested", "user-1")

        assert response is not None
        self.assertEqual(response.status, "degraded")
        self.assertIsNone(response.detail)

    def test_user_and_admin_queries_both_validate_run_and_tool_ownership(self):
        """若管理员只按 tool_call_id 查询或普通查询漏 user 条件，跨会话详情会泄漏。"""
        self._run("run-other", conversation_id="conv-2", user_id="user-2", terminal_at=self.now)
        self._tool(
            "run-other",
            "call-other-owner",
            conversation_id="conv-2",
            user_id="user-2",
            input_params={"query": "他人查询"},
            output_data={"result_count": 1},
        )

        service = self._service()
        self.assertIsNone(service.get_user_tool_node_detail("conv-2", "run-other", "call-other-owner", "user-1"))
        self.assertIsNone(service.get_user_tool_node_detail("conv-1", "run-other", "call-other-owner", "user-1"))
        self.assertIsNone(service.get_admin_tool_node_detail("conv-1", "run-other", "call-other-owner"))
        admin_response = service.get_admin_tool_node_detail("conv-2", "run-other", "call-other-owner")
        assert admin_response is not None
        self.assertEqual(admin_response.status, "available")

        self._run("run-inconsistent-owner", conversation_id="conv-1", user_id="user-2", terminal_at=self.now)
        self._tool(
            "run-inconsistent-owner",
            "call-inconsistent-owner",
            conversation_id="conv-1",
            user_id="user-2",
            input_params={"query": "不一致归属"},
            output_data={"result_count": 1},
        )
        self.assertIsNone(
            service.get_admin_tool_node_detail("conv-1", "run-inconsistent-owner", "call-inconsistent-owner")
        )

    def test_schema_and_configuration_contracts_are_strict(self):
        """若 DTO 允许额外字段或 grace 为负，状态与安全边界将不可预测。"""
        from app.schemas import trajectory

        self.assertTrue(hasattr(trajectory, "ToolNodeDetail"))
        self.assertTrue(hasattr(trajectory, "TrajectoryNodeDetailResponse"))
        self.assertEqual(settings.TRAJECTORY_DETAIL_SETTLE_GRACE_SECONDS, 5)

        db = self.Session()
        self.addCleanup(db.close)
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            TrajectoryQueryService(
                TrajectoryRepository(db),
                max_events_per_run=10,
                max_runs_per_conversation=10,
                detail_settle_grace_seconds=-0.1,
            )


if __name__ == "__main__":
    unittest.main()
