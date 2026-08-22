"""普通用户轨迹查询服务的有界、安全读取契约。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import (
    AgentEvent,
    AgentSession,
    Conversation,
    RunTrajectoryMeta,
    ToolCallLog,
    TrajectoryLedgerSettings,
    User,
)
from app.db.trajectory_repository import TrajectoryRepository
from app.services.trajectory_query_service import TrajectoryQueryService


class TrajectoryQueryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "trajectory-query.sqlite3"
        self.engine = create_engine(f"sqlite:///{database_path}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.now = datetime(2026, 8, 22, 4, 0, tzinfo=UTC)
        with self.Session() as db:
            db.add_all(
                [
                    User(id="user-1", username="alice", email="alice@example.com"),
                    User(id="user-2", username="bob", email="bob@example.com"),
                    Conversation(id="conv-1", user_id="user-1", title="我的会话", model_id="model-1"),
                    Conversation(id="conv-2", user_id="user-2", title="他人的会话", model_id="model-1"),
                    TrajectoryLedgerSettings(singleton_key="default", ledger_enabled_at=self.now - timedelta(days=1)),
                ]
            )
            db.commit()

    def tearDown(self) -> None:
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _service(self, *, max_events: int = 2, max_runs: int = 2) -> TrajectoryQueryService:
        db = self.Session()
        self.addCleanup(db.close)
        return TrajectoryQueryService(
            TrajectoryRepository(db),
            max_events_per_run=max_events,
            max_runs_per_conversation=max_runs,
        )

    def _run(
        self,
        run_id: str,
        *,
        conversation_id: str = "conv-1",
        user_id: str = "user-1",
        created_at: datetime | None = None,
        turn_message_id: str | None = "turn-1",
        attempt_index: int | None = 1,
    ) -> None:
        with self.Session() as db:
            db.add(
                AgentSession(
                    id=run_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_id=f"msg-{run_id}",
                    turn_message_id=turn_message_id,
                    attempt_index=attempt_index,
                    model_id="model-1",
                    provider="provider-1",
                    status="completed",
                    total_steps=7,
                    total_tool_calls=11,
                    total_duration_ms=345,
                    created_at=created_at or self.now,
                    terminal_at=created_at or self.now,
                )
            )
            db.commit()

    def _meta(
        self,
        run_id: str,
        *,
        status: str = "complete",
        pending: bool = False,
        pending_reason: str | None = None,
    ) -> None:
        with self.Session() as db:
            db.add(
                RunTrajectoryMeta(
                    run_id=run_id,
                    conversation_id="conv-1",
                    trajectory_status=status,
                    event_count=3,
                    expected_last_sequence=2,
                    terminal_intent_pending_at=self.now if pending else None,
                    terminal_intent_reason=pending_reason,
                )
            )
            db.commit()

    def _event(self, run_id: str, sequence: int) -> None:
        with self.Session() as db:
            db.add(
                AgentEvent(
                    conversation_id="conv-1",
                    message_id=f"msg-{run_id}",
                    run_id=run_id,
                    sequence=sequence,
                    event_type="step_completed",
                    schema_version=1,
                    event_ts=self.now + timedelta(seconds=sequence),
                    payload={"duration_ms": 10, "safe_summary": f"event-{sequence}"},
                )
            )
            db.commit()

    def test_snapshot_caps_events_and_reports_authoritative_summary_and_completeness(self):
        """若改成无界 events 查询、伪造摘要或遗漏完整性字段，快照必须失败。"""
        self._run("run-1")
        self._meta("run-1", pending=True)
        for sequence in range(3):
            self._event("run-1", sequence)

        snapshot = self._service(max_events=2).get_user_snapshot("conv-1", "run-1", "user-1")

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertTrue(snapshot.truncated)
        self.assertEqual([record.sequence for record in snapshot.records], [0, 1])
        self.assertEqual(snapshot.run.total_steps, 7)
        self.assertEqual(snapshot.run.total_tool_calls, 11)
        self.assertEqual(snapshot.run.duration_ms, 345)
        self.assertEqual(snapshot.completeness.status, "degraded")
        self.assertEqual(snapshot.completeness.degraded_reason, "terminal_outcome_unknown")
        self.assertEqual(snapshot.completeness.event_count, 3)
        self.assertEqual(snapshot.completeness.expected_last_sequence, 2)
        self.assertEqual(snapshot.completeness.loaded_event_count, 2)
        self.assertEqual((snapshot.completeness.first_sequence, snapshot.completeness.last_sequence), (0, 1))

    def test_snapshot_at_exact_event_cap_is_not_truncated_and_legacy_empty_is_explainable(self):
        """若 limit+1 边界错判，或 legacy 空账本误作不存在，本测试必须失败。"""
        self._run("run-exact")
        self._meta("run-exact")
        self._event("run-exact", 0)
        self._event("run-exact", 1)
        self._run("run-legacy", created_at=self.now - timedelta(days=2), turn_message_id=None, attempt_index=None)

        exact = self._service(max_events=2).get_user_snapshot("conv-1", "run-exact", "user-1")
        legacy = self._service().get_user_snapshot("conv-1", "run-legacy", "user-1")

        self.assertFalse(exact.truncated)
        self.assertEqual(exact.completeness.loaded_event_count, 2)
        self.assertIsNotNone(legacy)
        assert legacy is not None
        self.assertEqual(legacy.records, [])
        self.assertEqual((legacy.completeness.status, legacy.completeness.degraded_reason), ("legacy", "not_recorded"))

    def test_service_hides_cross_user_conversation_and_cross_conversation_run_as_not_found(self):
        """若仓库没有在查询中同时限定会话和用户，资源存在性会泄漏。"""
        self._run("run-own")
        self._run("run-other", conversation_id="conv-2", user_id="user-2", turn_message_id="turn-other")

        service = self._service()

        self.assertIsNone(service.list_runs("conv-1", "user-2"))
        self.assertIsNone(service.get_user_snapshot("conv-1", "run-other", "user-1"))
        self.assertIsNone(service.get_user_snapshot("conv-2", "run-own", "user-1"))

    def test_run_list_caps_before_grouping_then_orders_recent_groups_and_attempts(self):
        """若先无界取全量或把 NULL attempt 伪装成 1，本分组排序契约必须失败。"""
        self._run(
            "run-old-attempt", created_at=self.now - timedelta(minutes=3), turn_message_id="turn-a", attempt_index=2
        )
        self._run(
            "run-new-attempt", created_at=self.now - timedelta(minutes=2), turn_message_id="turn-a", attempt_index=1
        )
        self._run("run-null", created_at=self.now - timedelta(minutes=1), turn_message_id=None, attempt_index=None)
        self._meta("run-old-attempt")
        self._meta("run-new-attempt")
        self._meta("run-null")

        result = self._service(max_runs=2).list_runs("conv-1", "user-1")

        self.assertTrue(result.truncated)
        self.assertEqual([item.run_id for item in result.items], ["run-null", "run-new-attempt"])
        self.assertIsNone(result.items[0].attempt_index)
        self.assertEqual(result.items[1].attempt_index, 1)

    def test_service_rejects_non_positive_bounds(self):
        """若允许零或负数上限，读侧会失去可预测的有界语义。"""
        db = self.Session()
        self.addCleanup(db.close)
        with self.assertRaisesRegex(ValueError, "必须大于 0"):
            TrajectoryQueryService(TrajectoryRepository(db), max_events_per_run=0, max_runs_per_conversation=1)
        with self.assertRaisesRegex(ValueError, "必须大于 0"):
            TrajectoryQueryService(TrajectoryRepository(db), max_events_per_run=1, max_runs_per_conversation=0)

    def test_pending_reason_and_persisted_legacy_meta_are_not_exposed_as_user_status(self):
        """若普通 helper 读取 intent reason 或信任已有 legacy meta，安全状态会泄漏或失真。"""
        self._run("run-pending")
        self._meta("run-pending", pending=True, pending_reason="write_failed")
        self._run("run-legacy-meta", created_at=self.now - timedelta(days=2), turn_message_id="turn-legacy")
        self._meta("run-legacy-meta", status="legacy")

        service = self._service()
        pending = service.get_user_snapshot("conv-1", "run-pending", "user-1")
        legacy_meta = service.get_user_snapshot("conv-1", "run-legacy-meta", "user-1")

        assert pending is not None
        assert legacy_meta is not None
        self.assertEqual(
            (pending.completeness.status, pending.completeness.degraded_reason),
            ("degraded", "terminal_outcome_unknown"),
        )
        self.assertEqual(
            (legacy_meta.completeness.status, legacy_meta.completeness.degraded_reason),
            ("degraded", "terminal_outcome_unknown"),
        )

    def test_run_list_uses_owner_aware_outer_join_and_one_watermark_query(self):
        """若正常非空列表额外预检会话，读侧查询数会从两次退化为三次。"""
        self._run("run-count")
        self._meta("run-count")
        statements: list[str] = []

        def capture(_conn, _cursor, statement, _parameters, _context, _executemany) -> None:
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(self.engine, "before_cursor_execute", capture)
        try:
            result = self._service().list_runs("conv-1", "user-1")
        finally:
            event.remove(self.engine, "before_cursor_execute", capture)

        self.assertIsNotNone(result)
        self.assertEqual(len(statements), 2)

    def test_run_list_applies_max_plus_one_limit_in_database_query(self):
        """若删除 repository 的 SQL LIMIT，响应切片仍可假绿而此数据库边界测试必须失败。"""
        self._run("run-list-limit-1", turn_message_id="turn-list-1")
        self._run("run-list-limit-2", turn_message_id="turn-list-2")
        self._run("run-list-limit-3", turn_message_id="turn-list-3")
        for run_id in ("run-list-limit-1", "run-list-limit-2", "run-list-limit-3"):
            self._meta(run_id)
        statements: list[tuple[str, tuple[object, ...] | list[object]]] = []

        def capture(_conn, _cursor, statement, parameters, _context, _executemany) -> None:
            if "FROM conversations" in statement:
                statements.append((statement, parameters))

        event.listen(self.engine, "before_cursor_execute", capture)
        try:
            result = self._service(max_runs=2).list_runs("conv-1", "user-1")
        finally:
            event.remove(self.engine, "before_cursor_execute", capture)

        self.assertIsNotNone(result)
        self.assertEqual(len(statements), 1)
        statement, parameters = statements[0]
        self.assertIn("LIMIT", statement.upper())
        self.assertIn(3, parameters)

    def test_snapshot_applies_max_plus_one_limit_in_database_query(self):
        """若删除 event SQL LIMIT，快照仍可在 service 切片而掩盖无界读取。"""
        self._run("run-event-limit")
        self._meta("run-event-limit")
        for sequence in range(3):
            self._event("run-event-limit", sequence)
        statements: list[tuple[str, tuple[object, ...] | list[object]]] = []

        def capture(_conn, _cursor, statement, parameters, _context, _executemany) -> None:
            if "FROM agent_events" in statement:
                statements.append((statement, parameters))

        event.listen(self.engine, "before_cursor_execute", capture)
        try:
            snapshot = self._service(max_events=2).get_user_snapshot("conv-1", "run-event-limit", "user-1")
        finally:
            event.remove(self.engine, "before_cursor_execute", capture)

        self.assertIsNotNone(snapshot)
        self.assertEqual(len(statements), 1)
        statement, parameters = statements[0]
        self.assertIn("LIMIT", statement.upper())
        self.assertIn(3, parameters)

    def test_run_list_keeps_all_latest_attempts_in_a_turn_and_sorts_them_ascending(self):
        """若分组逻辑只覆盖一个 attempt 或按时间倒序输出，同一 turn 的重试顺序会错误。"""
        self._run("run-turn-a-2", created_at=self.now - timedelta(minutes=1), turn_message_id="turn-a", attempt_index=2)
        self._run("run-turn-a-1", created_at=self.now - timedelta(minutes=2), turn_message_id="turn-a", attempt_index=1)
        self._run("run-turn-b-1", created_at=self.now, turn_message_id="turn-b", attempt_index=1)
        for run_id in ("run-turn-a-2", "run-turn-a-1", "run-turn-b-1"):
            self._meta(run_id)

        result = self._service(max_runs=3).list_runs("conv-1", "user-1")

        assert result is not None
        self.assertEqual(
            [(item.turn_message_id, item.attempt_index) for item in result.items],
            [("turn-b", 1), ("turn-a", 1), ("turn-a", 2)],
        )

    def test_admin_snapshot_bounds_tool_diagnostics_and_marks_truncation(self):
        """若管理员工具诊断无界读取，单次历史查看可被大量 ToolCallLog 放大。"""
        self._run("run-admin")
        self._meta("run-admin")
        with self.Session() as db:
            db.add_all(
                [
                    ToolCallLog(
                        id=f"tool-admin-{index}",
                        conversation_id="conv-1",
                        message_id="msg-run-admin",
                        user_id="user-1",
                        tool_name="private_tool",
                        status="success",
                        model_id="model-1",
                        provider="provider-1",
                        trace_id="run-admin",
                        step_number=index,
                        created_at=self.now + timedelta(seconds=index),
                    )
                    for index in range(2)
                ]
            )
            db.commit()

        snapshot = self._service(max_events=1).get_admin_snapshot("conv-1", "run-admin")

        assert snapshot is not None
        self.assertTrue(snapshot.tool_calls_truncated)
        self.assertEqual([item.id for item in snapshot.tool_calls], ["tool-admin-0"])
