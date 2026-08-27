"""Tool Node Detail 的精确关联、四态与安全投影契约。"""

from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.db.database import Base
from app.db.models import (
    AgentEvent,
    AgentLlmRoundDetail,
    AgentSession,
    AgentSystemPromptSnapshot,
    Conversation,
    RunTrajectoryMeta,
    ToolCallLog,
    TrajectoryLedgerSettings,
    User,
)
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

    @staticmethod
    def _system_prompt_snapshot() -> dict:
        return {
            "schema_version": 1,
            "template_version": "2026-08-26.1",
            "fingerprint": "ea74d6bfa2a7c788583abe258af4d4ded20982b8907b8b1ece9cb46ad05aee90",
            "char_count": 1438,
            "sections": [
                {"section_id": "current_date", "content": "历史日期：2026-08-26（Asia/Shanghai）"},
                {"section_id": "user_preferences", "content": "保持详细回答。\n" + "完整规则" * 350},
            ],
        }

    def _system_prompt(
        self,
        run_id: str,
        *,
        snapshot: object | None = None,
        metadata: dict | None = None,
    ) -> None:
        with self.Session() as db:
            run = db.get(AgentSession, run_id)
            assert run is not None
            if snapshot is not None:
                run.run_config = {"unrelated_config": "内部配置"}
                db.add(
                    AgentSystemPromptSnapshot(
                        run_id=run_id,
                        conversation_id=run.conversation_id,
                        user_id=run.user_id,
                        snapshot=snapshot,
                    )
                )
            if metadata is not None:
                db.add(
                    AgentEvent(
                        conversation_id=run.conversation_id,
                        message_id=run.message_id,
                        run_id=run_id,
                        sequence=0,
                        event_type="system_prompt_prepared",
                        schema_version=1,
                        event_ts=self.now,
                        payload=metadata,
                    )
                )
            db.commit()

    def test_system_prompt_detail_reads_exact_persisted_body_after_session_refresh_and_template_changes(self):
        """若重新执行当前模板、截断正文或读取会话配置，历史正文读取必须失败。"""
        self._run("run-system-prompt")
        snapshot = self._system_prompt_snapshot()
        self._system_prompt(
            "run-system-prompt",
            snapshot=snapshot,
            metadata={"status": "ready", "detail_status": "available", "fingerprint": snapshot["fingerprint"]},
        )

        first = self._service().get_user_system_prompt_node_detail("conv-1", "run-system-prompt", "user-1")
        with (
            patch("app.ai.prompts.system_prompt.TEMPLATE_VERSION", "2099-new-template"),
            patch("app.ai.prompts.system_prompt.build_base_sections", side_effect=AssertionError("不能重建历史提示词")),
        ):
            refreshed = self._service().get_user_system_prompt_node_detail("conv-1", "run-system-prompt", "user-1")

        assert first is not None and refreshed is not None
        self.assertEqual(first.model_dump(), refreshed.model_dump())
        self.assertEqual(first.status, "available")
        self.assertEqual(first.node_type, "system_prompt")
        self.assertEqual(first.available_sections, ["summary", "prompt"])
        expected_detail = {key: value for key, value in snapshot.items() if key != "schema_version"}
        self.assertEqual(first.detail.model_dump(), expected_detail)
        self.assertEqual(first.redacted_fields, [])
        self.assertEqual(first.truncated_fields, [])
        self.assertIsNone(first.reason)
        self.assertNotIn("内部配置", first.model_dump_json())

    def test_system_prompt_detail_requires_exact_conversation_run_and_user_ownership(self):
        """若省略任一归属条件，其他 Run 或不一致用户的持久正文会泄漏。"""
        self._run("run-owned")
        self._run("run-foreign", conversation_id="conv-2", user_id="user-2")
        self._run("run-inconsistent", user_id="user-2")
        for run_id in ("run-owned", "run-foreign", "run-inconsistent"):
            self._system_prompt(run_id, snapshot=self._system_prompt_snapshot())

        service = self._service()
        for conversation_id, run_id, user_id in (
            ("conv-2", "run-foreign", "user-1"),
            ("conv-1", "run-foreign", "user-1"),
            ("conv-2", "run-owned", "user-2"),
            ("conv-1", "run-owned", "user-2"),
            ("conv-1", "run-inconsistent", "user-1"),
            ("conv-1", "run-inconsistent", "user-2"),
            ("conv-1", "missing-run", "user-1"),
        ):
            with self.subTest(conversation_id=conversation_id, run_id=run_id, user_id=user_id):
                self.assertIsNone(service.get_user_system_prompt_node_detail(conversation_id, run_id, user_id))
        own = service.get_user_system_prompt_node_detail("conv-1", "run-owned", "user-1")
        assert own is not None
        self.assertEqual(own.status, "available")

    def test_system_prompt_detail_distinguishes_unrecorded_assembly_failure_and_missing_snapshot(self):
        """旧 Run 不伪造正文，新记录缺失与实际组装失败必须明确降级。"""
        cases = (
            (None, "not_recorded", "system_prompt_not_recorded"),
            ({"status": "ready"}, "not_recorded", "system_prompt_not_recorded"),
            ({"status": "failed"}, "degraded", "system_prompt_assembly_failed"),
            ({"status": "ready", "detail_status": "available"}, "degraded", "system_prompt_detail_missing"),
            ({"status": "ready", "detail_status": "degraded"}, "degraded", "system_prompt_detail_missing"),
        )
        for index, (metadata, status, reason) in enumerate(cases):
            with self.subTest(metadata=metadata):
                run_id = f"run-system-status-{index}"
                self._run(run_id)
                self._system_prompt(run_id, metadata=metadata)
                response = self._service().get_user_system_prompt_node_detail("conv-1", run_id, "user-1")
                assert response is not None
                self.assertEqual(response.status, status)
                self.assertEqual(response.node_type, "system_prompt")
                self.assertEqual(response.reason, reason)
                self.assertIsNone(response.detail)
                self.assertEqual(response.available_sections, [])

    def test_system_prompt_detail_waits_for_async_ledger_before_classifying_an_old_run(self):
        """SSE 先于账本落库时，应短暂等待；终态过宽限且无记录才视为旧 Run。"""
        cases = (
            ("running", None, "pending", "system_prompt_detail_settling"),
            ("completed", self.now - timedelta(seconds=5), "pending", "system_prompt_detail_settling"),
            ("completed", self.now - timedelta(seconds=6), "not_recorded", "system_prompt_not_recorded"),
        )
        for index, (run_status, terminal_at, expected_status, reason) in enumerate(cases):
            with self.subTest(run_status=run_status, terminal_at=terminal_at):
                run_id = f"run-system-ledger-lag-{index}"
                self._run(run_id, status=run_status, terminal_at=terminal_at)
                response = self._service().get_user_system_prompt_node_detail("conv-1", run_id, "user-1")
                assert response is not None
                self.assertEqual(response.status, expected_status)
                self.assertEqual(response.reason, reason)
                self.assertIsNone(response.detail)
                self.assertEqual(response.available_sections, [])

                if expected_status == "pending":
                    self._system_prompt(run_id, metadata={"status": "ready", "detail_status": "degraded"})
                    settled = self._service().get_user_system_prompt_node_detail("conv-1", run_id, "user-1")
                    assert settled is not None
                    self.assertEqual(settled.status, "degraded")
                    self.assertEqual(settled.reason, "system_prompt_detail_missing")

    def test_system_prompt_snapshot_is_available_before_event_and_failed_event_never_exposes_body(self):
        """写入与事件间隙可读快照，但失败事件不能被偶存正文覆盖为成功。"""
        self._run("run-before-event")
        self._run("run-failed-event")
        snapshot = self._system_prompt_snapshot()
        self._system_prompt("run-before-event", snapshot=snapshot)
        self._system_prompt("run-failed-event", snapshot=snapshot, metadata={"status": "failed"})

        before_event = self._service().get_user_system_prompt_node_detail("conv-1", "run-before-event", "user-1")
        failed = self._service().get_user_system_prompt_node_detail("conv-1", "run-failed-event", "user-1")

        assert before_event is not None and failed is not None
        self.assertEqual(before_event.status, "available")
        self.assertEqual(failed.status, "degraded")
        self.assertEqual(failed.reason, "system_prompt_assembly_failed")
        self.assertIsNone(failed.detail)

    def test_system_prompt_detail_rejects_corrupt_schema_sections_hash_and_character_count(self):
        """若直接信任持久字典、事件 hash 或字符数，损坏正文会被当成可用历史。"""
        valid = self._system_prompt_snapshot()
        invalid_values = [
            [],
            {**valid, "schema_version": 2},
            {**valid, "schema_version": True},
            {**valid, "template_version": 123},
            {**valid, "char_count": str(valid["char_count"])},
            {**valid, "char_count": valid["char_count"] + 1},
            {**valid, "sections": []},
            {**valid, "sections": list(reversed(valid["sections"]))},
            {**valid, "fingerprint": "0" * 64},
            {**valid, "unexpected_body": "不得放行的字段"},
        ]
        duplicate = deepcopy(valid)
        duplicate["sections"][1]["section_id"] = duplicate["sections"][0]["section_id"]
        invalid_values.append(duplicate)
        malformed = deepcopy(valid)
        malformed["sections"][0]["content"] = ["内容不是文本"]
        invalid_values.append(malformed)
        empty_id = deepcopy(valid)
        empty_id["sections"][0]["section_id"] = ""
        invalid_values.append(empty_id)
        for index, snapshot in enumerate(invalid_values):
            with self.subTest(index=index):
                run_id = f"run-system-invalid-{index}"
                self._run(run_id)
                self._system_prompt(
                    run_id,
                    snapshot=snapshot,
                    metadata={
                        "status": "ready",
                        "detail_status": "available",
                        "fingerprint": snapshot.get("fingerprint") if isinstance(snapshot, dict) else None,
                    },
                )
                response = self._service().get_user_system_prompt_node_detail("conv-1", run_id, "user-1")
                assert response is not None
                self.assertEqual(response.status, "degraded")
                self.assertEqual(response.reason, "system_prompt_detail_invalid")
                self.assertIsNone(response.detail)
                self.assertEqual(response.available_sections, [])

    def test_system_prompt_detail_rejects_event_metadata_mismatch(self):
        """快照内容虽自洽，但若与组装事件的正文或版本信息不同，不能当作事件正文。"""
        snapshot = self._system_prompt_snapshot()
        mismatches = {
            "fingerprint": "f" * 64,
            "section_ids": ["other", "user_preferences"],
            "template_version": "other-version",
            "char_count": 0,
        }
        for field, value in mismatches.items():
            with self.subTest(field=field):
                run_id = f"run-system-mismatch-{field}"
                self._run(run_id)
                self._system_prompt(
                    run_id,
                    snapshot=snapshot,
                    metadata={
                        "status": "ready",
                        "detail_status": "available",
                        "fingerprint": snapshot["fingerprint"],
                        field: value,
                    },
                )

                response = self._service().get_user_system_prompt_node_detail("conv-1", run_id, "user-1")

                assert response is not None
                self.assertEqual(response.status, "degraded")
                self.assertEqual(response.reason, "system_prompt_detail_invalid")
                self.assertIsNone(response.detail)

    def test_general_trajectory_and_other_node_details_do_not_read_system_prompt_body(self):
        """若共用 Run 查询增加 config 或触发全列懒加载，普通读取会无故加载完整正文。"""
        self._run("run-system-private")
        self._system_prompt("run-system-private", snapshot=self._system_prompt_snapshot())
        statements = []

        def capture_sql(connection, cursor, statement, parameters, context, executemany):
            statements.append(statement)

        event.listen(self.engine, "before_cursor_execute", capture_sql)
        try:
            service = self._service()
            listing = service.list_runs("conv-1", "user-1")
            snapshot = service.get_user_snapshot("conv-1", "run-system-private", "user-1")
            tool = service.get_user_tool_node_detail("conv-1", "run-system-private", "absent-tool", "user-1")
            llm = service.get_user_llm_node_detail("conv-1", "run-system-private", "absent-round", "user-1")
        finally:
            event.remove(self.engine, "before_cursor_execute", capture_sql)

        assert listing is not None and snapshot is not None and tool is not None
        self.assertIsNone(llm)
        self.assertTrue(statements)
        sql = "\n".join(statements)
        self.assertNotRegex(sql, r"(?:SELECT|,)\s*agent_sessions\.config(?:\s+AS\s+\w+)?\s*(?:,|FROM)")
        self.assertIn("JSON_EXTRACT(agent_sessions.config", sql)
        self.assertNotIn("agent_system_prompt_snapshots.snapshot", "\n".join(statements))
        self.assertNotIn("完整规则", listing.model_dump_json() + snapshot.model_dump_json() + tool.model_dump_json())

    def _llm_round(
        self,
        run_id: str,
        llm_round_id: str,
        *,
        terminal_at: datetime | None,
        detail: tuple[str, str] | None = None,
        schema_version: int | None = 1,
    ) -> None:
        with self.Session() as db:
            db.add(
                RunTrajectoryMeta(
                    run_id=run_id,
                    conversation_id="conv-1",
                    message_id=f"msg-{run_id}",
                    trajectory_status="complete" if terminal_at is not None else "recording",
                    event_count=2 if terminal_at is not None else 1,
                    llm_detail_schema_version=schema_version,
                )
            )
            db.add(
                AgentEvent(
                    conversation_id="conv-1",
                    message_id=f"msg-{run_id}",
                    run_id=run_id,
                    sequence=0,
                    event_type="llm_round_started",
                    schema_version=1,
                    event_ts=self.now - timedelta(seconds=10),
                    step_id="step-1",
                    payload={"llm_round_id": llm_round_id, "round_index": 1},
                )
            )
            if terminal_at is not None:
                db.add(
                    AgentEvent(
                        conversation_id="conv-1",
                        message_id=f"msg-{run_id}",
                        run_id=run_id,
                        sequence=1,
                        event_type="llm_round_completed",
                        schema_version=1,
                        event_ts=terminal_at,
                        step_id="step-1",
                        payload={"llm_round_id": llm_round_id, "duration_ms": 100},
                    )
                )
            if detail is not None:
                reasoning, output = detail
                db.add(
                    AgentLlmRoundDetail(
                        conversation_id="conv-1",
                        run_id=run_id,
                        message_id=f"msg-{run_id}",
                        llm_round_id=llm_round_id,
                        reasoning_text=reasoning or None,
                        content_text=output or None,
                        reasoning_preview=reasoning[:200] or None,
                        output_preview=output[:200] or None,
                        redacted_fields=["reasoning_text"] if reasoning else [],
                        truncated_fields=["content_text"] if output else [],
                    )
                )
            db.commit()

    def test_llm_detail_uses_exact_round_identity_and_exposes_thinking_output_sections(self):
        self._run("run-llm", terminal_at=self.now - timedelta(seconds=20))
        self._llm_round(
            "run-llm",
            "round-exact",
            terminal_at=self.now - timedelta(seconds=10),
            detail=("显式推理", "最终输出"),
        )

        response = self._service().get_user_llm_node_detail("conv-1", "run-llm", "round-exact", "user-1")
        missing = self._service().get_user_llm_node_detail("conv-1", "run-llm", "round-other", "user-1")

        self.assertIsNotNone(response)
        self.assertIsNone(missing)
        self.assertEqual(response.status, "available")
        self.assertEqual(response.node_type, "llm")
        self.assertEqual(response.available_sections, ["summary", "thinking", "output", "timing"])
        self.assertEqual(response.detail.reasoning_text, "显式推理")
        self.assertEqual(response.detail.output_text, "最终输出")
        self.assertEqual(response.redacted_fields, ["reasoning_text"])
        self.assertEqual(response.truncated_fields, ["content_text"])

    def test_llm_detail_status_uses_schema_version_and_round_terminal_grace(self):
        self._run("run-old", terminal_at=self.now - timedelta(minutes=1))
        self._llm_round(
            "run-old",
            "round-old",
            terminal_at=self.now - timedelta(minutes=1),
            schema_version=None,
        )
        self._run("run-live", status="running")
        self._llm_round("run-live", "round-live", terminal_at=None)
        self._run("run-settling", terminal_at=self.now)
        self._llm_round("run-settling", "round-settling", terminal_at=self.now - timedelta(seconds=5))
        self._run("run-degraded", terminal_at=self.now)
        self._llm_round(
            "run-degraded",
            "round-degraded",
            terminal_at=self.now - timedelta(seconds=5, microseconds=1),
        )

        service = self._service()
        old = service.get_user_llm_node_detail("conv-1", "run-old", "round-old", "user-1")
        live = service.get_user_llm_node_detail("conv-1", "run-live", "round-live", "user-1")
        settling = service.get_user_llm_node_detail("conv-1", "run-settling", "round-settling", "user-1")
        degraded = service.get_user_llm_node_detail("conv-1", "run-degraded", "round-degraded", "user-1")

        self.assertIsNone(old)
        self.assertEqual((live.status, live.reason), ("pending", "llm_round_in_progress"))
        self.assertEqual((settling.status, settling.reason), ("pending", "llm_detail_settling"))
        self.assertEqual((degraded.status, degraded.reason), ("degraded", "llm_detail_missing"))

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
