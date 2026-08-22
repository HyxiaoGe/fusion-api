"""普通用户轨迹查询服务的有界、安全读取契约。"""

from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentEvent, AgentSession, Conversation, RunTrajectoryMeta, TrajectoryLedgerSettings, User
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

    def _meta(self, run_id: str, *, status: str = "complete", pending: bool = False) -> None:
        with self.Session() as db:
            db.add(
                RunTrajectoryMeta(
                    run_id=run_id,
                    conversation_id="conv-1",
                    trajectory_status=status,
                    event_count=3,
                    expected_last_sequence=2,
                    terminal_intent_pending_at=self.now if pending else None,
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
