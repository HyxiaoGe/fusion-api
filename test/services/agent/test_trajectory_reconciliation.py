import asyncio
import concurrent.futures
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from sqlalchemy import create_engine, select
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta, TrajectoryLedgerSettings
from app.services.agent.trajectory_reconciliation import (
    DEFAULT_RECONCILIATION_STALE_GRACE,
    TERMINAL_OUTCOME_UNKNOWN_REASON,
    build_reconciliation_candidate_query,
    classify_missing_trajectory_meta,
    reconcile_trajectory_batch,
    resolve_ledger_watermark,
    resolve_run_trajectory_status,
)
from app.services.agent.trajectory_recorder import TrajectoryRecorder

_DEFAULT_TERMINAL_AT = object()


class TrajectoryReconciliationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "trajectory-reconciliation.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.now = datetime(2026, 8, 22, 4, 0, tzinfo=UTC)
        self.stale_before = self.now - DEFAULT_RECONCILIATION_STALE_GRACE

    def tearDown(self) -> None:
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _add_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        created_at: datetime | None = None,
        terminal_at: datetime | None | object = _DEFAULT_TERMINAL_AT,
    ) -> None:
        if terminal_at is _DEFAULT_TERMINAL_AT:
            terminal_at = None if status == "running" else self.stale_before - timedelta(seconds=1)
        with self.Session() as db:
            db.add(
                AgentSession(
                    id=run_id,
                    conversation_id="conv-1",
                    message_id=f"msg-{run_id}",
                    user_id="user-1",
                    model_id="model-1",
                    provider="provider-1",
                    status=status,
                    created_at=created_at or self.now,
                    terminal_at=terminal_at,
                )
            )
            db.commit()

    def _add_meta(
        self,
        run_id: str,
        *,
        trajectory_status: str = "recording",
        expected_last_sequence: int | None = None,
        degraded_reason: str | None = None,
        pending: bool = False,
        intent_status: str | None = None,
        intent_reason: str | None = None,
        intent_version: int | None = None,
        updated_at: datetime | None = None,
        pending_at: datetime | None = None,
    ) -> None:
        with self.Session() as db:
            run = db.get(AgentSession, run_id)
            db.add(
                RunTrajectoryMeta(
                    run_id=run_id,
                    conversation_id=run.conversation_id,
                    message_id=run.message_id,
                    trajectory_status=trajectory_status,
                    expected_last_sequence=expected_last_sequence,
                    degraded_reason=degraded_reason,
                    finalized_at=self.now if trajectory_status == "complete" else None,
                    terminal_intent_id="intent-1" if pending else None,
                    terminal_intent_status=intent_status if pending else None,
                    terminal_intent_reason=intent_reason if pending else None,
                    terminal_intent_version=intent_version if pending else None,
                    terminal_intent_pending_at=(pending_at or self.stale_before - timedelta(seconds=1))
                    if pending
                    else None,
                    updated_at=updated_at or self.stale_before - timedelta(seconds=1),
                )
            )
            db.commit()

    def _add_events(self, run_id: str, sequences: list[int]) -> None:
        with self.Session() as db:
            run = db.get(AgentSession, run_id)
            for sequence in sequences:
                db.add(
                    AgentEvent(
                        conversation_id=run.conversation_id,
                        message_id=run.message_id,
                        run_id=run_id,
                        sequence=sequence,
                        event_type="step_started",
                        schema_version=1,
                        event_ts=self.now + timedelta(milliseconds=sequence),
                        recorded_at=self.now,
                        payload={"type": "step_started", "sequence": sequence},
                    )
                )
            db.commit()

    def _meta(self, run_id: str) -> RunTrajectoryMeta | None:
        with self.Session() as db:
            return db.get(RunTrajectoryMeta, run_id)

    def test_postgresql_candidate_query_uses_stable_limit_and_skip_locked(self):
        compiled = str(
            build_reconciliation_candidate_query(
                batch_size=37,
                stale_before=self.stale_before,
            ).compile(
                dialect=postgresql.dialect(),
                compile_kwargs={"literal_binds": True},
            )
        ).upper()

        self.assertIn("FOR UPDATE OF RUN_TRAJECTORY_META, AGENT_SESSIONS SKIP LOCKED", compiled)
        self.assertIn("LIMIT 37", compiled)
        self.assertIn("AGENT_SESSIONS.STATUS != 'RUNNING'", compiled)
        self.assertIn("AGENT_SESSIONS.TERMINAL_AT <=", compiled)
        self.assertIn("RUN_TRAJECTORY_META.UPDATED_AT <=", compiled)
        self.assertIn("RUN_TRAJECTORY_META.TERMINAL_INTENT_PENDING_AT <=", compiled)

    def test_fresh_terminal_rows_wait_for_grace_before_reconciliation(self):
        self._add_run("fresh-recording", terminal_at=self.now)
        self._add_meta(
            "fresh-recording",
            expected_last_sequence=0,
            updated_at=self.now,
        )
        self._add_events("fresh-recording", [0])
        self._add_run("fresh-complete-pending", terminal_at=self.now)
        self._add_meta(
            "fresh-complete-pending",
            trajectory_status="complete",
            expected_last_sequence=0,
            pending=True,
            intent_status="complete",
            intent_version=1,
            updated_at=self.now,
            pending_at=self.now,
        )
        self._add_events("fresh-complete-pending", [0])
        self._add_run("fresh-meta-missing", terminal_at=self.now)

        fresh = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        self.assertEqual(fresh.processed, 0)
        self.assertEqual(self._meta("fresh-recording").trajectory_status, "recording")
        self.assertEqual(self._meta("fresh-complete-pending").trajectory_status, "complete")
        self.assertIsNotNone(self._meta("fresh-complete-pending").terminal_intent_pending_at)
        self.assertIsNone(self._meta("fresh-meta-missing"))

        stale = reconcile_trajectory_batch(
            session_factory=self.Session,
            now=self.now + DEFAULT_RECONCILIATION_STALE_GRACE + timedelta(seconds=1),
        )

        self.assertEqual(stale.recording_completed, 1)
        self.assertEqual(stale.pending_degraded, 1)
        self.assertEqual(stale.meta_missing_degraded, 1)

    def test_fresh_terminal_grace_allows_late_first_write_and_finalize_to_win(self):
        self._add_run("late-recorder", terminal_at=self.now)
        self._add_meta("late-recorder", updated_at=self.now)

        result = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        self.assertEqual(result.processed, 0)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        recorder = TrajectoryRecorder(
            run_id="late-recorder",
            conversation_id="conv-1",
            message_id="msg-late-recorder",
            session_factory=self.Session,
            executor=executor,
        )
        event = {
            "schema_version": 1,
            "type": "step_started",
            "run_id": "late-recorder",
            "parent_run_id": None,
            "step_id": "step-late",
            "parent_step_id": None,
            "tool_call_id": None,
            "sequence": 0,
            "trace_id": "trace-late",
            "ts": self.now.timestamp(),
            "step_number": 1,
        }

        async def finish_late_recorder() -> None:
            await recorder.record_chunk("conv-1", "agent_event", event)
            await recorder.finalize(0)

        asyncio.run(finish_late_recorder())
        executor.shutdown(wait=True)

        meta = self._meta("late-recorder")
        self.assertEqual(meta.trajectory_status, "complete")
        self.assertIsNone(meta.terminal_intent_pending_at)

    def test_conflicting_missing_meta_insert_never_overwrites_recorder_or_rolls_back_other_work(self):
        self._add_run("recorder-won")
        self._add_meta(
            "recorder-won",
            trajectory_status="complete",
            expected_last_sequence=0,
        )
        self._add_run("other-pending")
        self._add_meta(
            "other-pending",
            trajectory_status="complete",
            expected_last_sequence=0,
            pending=True,
            intent_status="complete",
            intent_version=1,
        )

        forced_stale_snapshot = select(AgentSession).where(AgentSession.id == "recorder-won")
        with patch(
            "app.services.agent.trajectory_reconciliation._build_missing_meta_query",
            return_value=forced_stale_snapshot,
        ):
            result = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        self.assertEqual(result.pending_degraded, 1)
        self.assertEqual(result.meta_missing_degraded, 0)
        recorder_meta = self._meta("recorder-won")
        self.assertEqual(recorder_meta.trajectory_status, "complete")
        self.assertIsNone(recorder_meta.degraded_reason)
        self.assertEqual(self._meta("other-pending").trajectory_status, "degraded")

    def test_any_pending_state_is_conservatively_degraded_and_clears_all_intent_fields(self):
        for index, status in enumerate(("recording", "complete", "degraded")):
            run_id = f"pending-{status}"
            self._add_run(run_id)
            self._add_meta(
                run_id,
                trajectory_status=status,
                expected_last_sequence=0,
                degraded_reason="write_failed" if index == 2 else None,
                pending=True,
                intent_status="complete",
                intent_reason="recorder_timeout",
                intent_version=1,
            )
            self._add_events(run_id, [0])

        result = reconcile_trajectory_batch(session_factory=self.Session, now=self.now, batch_size=10)

        self.assertEqual(result.pending_degraded, 3)
        for status in ("recording", "complete", "degraded"):
            meta = self._meta(f"pending-{status}")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertIsNone(meta.finalized_at)
            self.assertIn(meta.degraded_reason, {"recorder_timeout", "write_failed"})
            self.assertIsNone(meta.terminal_intent_id)
            self.assertIsNone(meta.terminal_intent_status)
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertIsNone(meta.terminal_intent_version)
            self.assertIsNone(meta.terminal_intent_pending_at)

    def test_unknown_pending_contract_never_raises_or_becomes_complete(self):
        self._add_run("unknown-pending")
        self._add_meta(
            "unknown-pending",
            trajectory_status="future_status",
            expected_last_sequence=0,
            degraded_reason="untrusted database text",
            pending=True,
            intent_status="future_target",
            intent_reason="contains secret-like arbitrary text",
            intent_version=999,
        )
        self._add_events("unknown-pending", [0])

        reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        meta = self._meta("unknown-pending")
        self.assertEqual(meta.trajectory_status, "degraded")
        self.assertEqual(meta.degraded_reason, TERMINAL_OUTCOME_UNKNOWN_REASON)
        self.assertIsNone(meta.finalized_at)
        self.assertIsNone(meta.terminal_intent_id)
        self.assertIsNone(meta.terminal_intent_pending_at)

    def test_stale_recording_completes_only_when_count_min_and_max_all_match(self):
        for run_id, expected, sequences in (
            ("contiguous", 1, [0, 1]),
            ("missing-expected", None, [0]),
            ("sequence-hole", 2, [0, 2]),
        ):
            self._add_run(run_id)
            self._add_meta(run_id, expected_last_sequence=expected)
            self._add_events(run_id, sequences)

        first = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)
        second = reconcile_trajectory_batch(session_factory=self.Session, now=self.now + timedelta(seconds=1))

        self.assertEqual(first.recording_completed, 1)
        self.assertEqual(first.recording_degraded, 2)
        self.assertEqual(second.processed, 0)
        complete = self._meta("contiguous")
        self.assertEqual(complete.trajectory_status, "complete")
        self.assertEqual(complete.finalized_at, self.now.replace(tzinfo=None))
        self.assertIsNone(complete.degraded_reason)
        self.assertEqual(self._meta("missing-expected").degraded_reason, "expected_sequence_missing")
        self.assertEqual(self._meta("sequence-hole").degraded_reason, "sequence_mismatch")

    def test_running_business_run_is_never_reconciled(self):
        self._add_run("still-running", status="running")
        self._add_meta(
            "still-running",
            trajectory_status="complete",
            expected_last_sequence=0,
            pending=True,
            intent_status="complete",
            intent_version=1,
        )
        self._add_events("still-running", [0])

        result = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        self.assertEqual(result.processed, 0)
        meta = self._meta("still-running")
        self.assertEqual(meta.trajectory_status, "complete")
        self.assertIsNotNone(meta.terminal_intent_pending_at)

    def test_persistent_watermark_distinguishes_legacy_from_new_meta_missing(self):
        watermark = self.now - timedelta(hours=1)
        with self.Session() as db:
            db.add(
                TrajectoryLedgerSettings(
                    singleton_key="default",
                    ledger_enabled_at=watermark,
                    created_at=watermark,
                )
            )
            db.commit()
        self._add_run("legacy-run", created_at=watermark - timedelta(seconds=1))
        self._add_run("new-run", created_at=watermark)

        result = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        self.assertEqual(result.legacy_not_recorded, 0)
        self.assertEqual(result.meta_missing_degraded, 1)
        self.assertIsNone(self._meta("legacy-run"))
        with self.Session() as db:
            legacy = resolve_run_trajectory_status(db, "legacy-run")
        self.assertEqual(legacy.trajectory_status, "legacy")
        self.assertEqual(legacy.degraded_reason, "not_recorded")
        new_meta = self._meta("new-run")
        self.assertEqual(new_meta.trajectory_status, "degraded")
        self.assertEqual(new_meta.degraded_reason, "meta_missing")
        self.assertIsNone(new_meta.finalized_at)

    def test_missing_or_invalid_watermark_is_conservatively_degraded(self):
        self._add_run("settings-missing")

        missing_result = reconcile_trajectory_batch(session_factory=self.Session, now=self.now)

        self.assertEqual(missing_result.meta_missing_degraded, 1)
        self.assertEqual(self._meta("settings-missing").degraded_reason, "ledger_settings_missing")
        self.assertEqual(resolve_ledger_watermark([]).degraded_reason, "ledger_settings_missing")
        duplicate = [
            ("default", self.now),
            ("default", self.now + timedelta(seconds=1)),
        ]
        self.assertEqual(resolve_ledger_watermark(duplicate).degraded_reason, "ledger_settings_invalid")

    def test_naive_sqlite_timestamps_are_normalized_as_utc(self):
        watermark = self.now.replace(tzinfo=None)

        legacy = classify_missing_trajectory_meta(
            run_created_at=(self.now - timedelta(seconds=1)).replace(tzinfo=None),
            ledger_enabled_at=watermark,
        )
        current = classify_missing_trajectory_meta(
            run_created_at=self.now.replace(tzinfo=None),
            ledger_enabled_at=watermark,
        )

        self.assertEqual(legacy.trajectory_status, "legacy")
        self.assertEqual(legacy.degraded_reason, "not_recorded")
        self.assertEqual(current.trajectory_status, "degraded")
        self.assertEqual(current.degraded_reason, "meta_missing")


if __name__ == "__main__":
    unittest.main()
