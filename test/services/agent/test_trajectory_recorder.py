import asyncio
import concurrent.futures
import gc
import tempfile
import threading
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import Mock, patch

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as SqlAlchemySession
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta
from app.services.agent.trajectory_recorder import (
    TRAJECTORY_CONNECT_TIMEOUT_SECONDS,
    TRAJECTORY_LOCK_TIMEOUT_MS,
    TRAJECTORY_STATEMENT_TIMEOUT_MS,
    TRAJECTORY_WAIT_TIMEOUT_SECONDS,
    TrajectoryRecorder,
    _FinalizeHandshake,
    create_trajectory_session_factory,
)


def _event(sequence: int, event_type: str = "step_started", **fields):
    return {
        "schema_version": 1,
        "type": event_type,
        "run_id": "run-1",
        "parent_run_id": None,
        "step_id": "step-1",
        "parent_step_id": None,
        "tool_call_id": None,
        "sequence": sequence,
        "trace_id": "trace-1",
        "ts": 1_700_000_000.0 + sequence,
        **({"step_number": sequence + 1} if event_type == "step_started" else {}),
        **fields,
    }


def _assert_all_permits_available(test_case: unittest.TestCase, semaphore: threading.BoundedSemaphore) -> None:
    acquired = [semaphore.acquire(blocking=False) for _ in range(5)]
    test_case.assertEqual(acquired, [True, True, True, True, False])
    for _ in range(4):
        semaphore.release()


class ManualExecutor(concurrent.futures.Executor):
    """仅由测试显式推进任务，覆盖 worker 尚未启动的取消/超时竞态。"""

    def __init__(self) -> None:
        self.submitted = threading.Event()
        self._calls: list[tuple[concurrent.futures.Future, object, tuple]] = []

    def submit(self, fn, /, *args, **kwargs):
        if kwargs:
            raise AssertionError("测试 executor 不接受 kwargs")
        future = concurrent.futures.Future()
        self._calls.append((future, fn, args))
        self.submitted.set()
        return future

    def run_next(self) -> None:
        future, fn, args = self._calls.pop(0)
        if not future.set_running_or_notify_cancel():
            return
        try:
            future.set_result(fn(*args))
        except BaseException as error:
            future.set_exception(error)

    @property
    def pending_count(self) -> int:
        return len(self._calls)


class FailingSubmitExecutor(concurrent.futures.Executor):
    def submit(self, fn, /, *args, **kwargs):
        raise RuntimeError("executor 已关闭")


class RecorderDatabaseTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "trajectory.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            db.add(
                AgentSession(
                    id="run-1",
                    conversation_id="conv-1",
                    message_id="msg-1",
                    user_id="user-1",
                    model_id="gpt-4",
                    provider="openai",
                    status="running",
                )
            )
            db.commit()
        self.executors: list[concurrent.futures.ThreadPoolExecutor] = []

    def tearDown(self) -> None:
        for executor in self.executors:
            executor.shutdown(wait=True)
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _recorder(self, *, session_factory=None, worker=None, semaphore=None) -> TrajectoryRecorder:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.executors.append(executor)
        return TrajectoryRecorder(
            run_id="run-1",
            conversation_id="conv-1",
            message_id="msg-1",
            session_factory=session_factory or self.Session,
            executor=executor,
            semaphore=semaphore or threading.BoundedSemaphore(4),
            worker=worker,
        )

    async def test_uses_independent_short_sessions_and_counts_duplicate_sequence_once(self):
        created_sessions: list[SqlAlchemySession] = []

        def session_factory():
            session = self.Session()
            created_sessions.append(session)
            return session

        recorder = self._recorder(session_factory=session_factory)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        with self.Session() as db:
            self.assertEqual(db.query(AgentEvent).filter_by(run_id="run-1").count(), 1)
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertIsNotNone(meta)
            self.assertEqual(meta.event_count, 1)
        self.assertEqual(len(created_sessions), 2)
        self.assertTrue(all(session is not created_sessions[0] for session in created_sessions[1:]))
        self.assertTrue(all(not session.is_active or session.in_transaction() is False for session in created_sessions))

    async def test_commit_failure_after_insert_rolls_back_event_and_meta_together(self):
        class FlushThenFailSession(SqlAlchemySession):
            def commit(self):
                self.flush()
                raise RuntimeError("commit 前故障注入")

        failing_factory = sessionmaker(
            bind=self.engine,
            class_=FlushThenFailSession,
            expire_on_commit=False,
        )
        recorder = self._recorder(session_factory=failing_factory)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        with self.Session() as db:
            self.assertEqual(db.query(AgentEvent).filter_by(run_id="run-1").count(), 0)
            self.assertIsNone(db.get(RunTrajectoryMeta, "run-1"))
        self.assertEqual(recorder.degraded_reason, "write_failed")

    async def test_finalize_marks_complete_only_for_contiguous_sequence_range(self):
        recorder = self._recorder()
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await recorder.record_chunk("conv-1", "agent_event", _event(1))

        await recorder.finalize(1)

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertEqual(meta.expected_last_sequence, 1)
            self.assertIsNotNone(meta.finalized_at)
            self.assertIsNone(meta.degraded_reason)
            self.assertIsNone(meta.terminal_intent_status)
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertIsNone(meta.terminal_intent_version)
            self.assertIsNone(meta.terminal_intent_pending_at)

    async def test_terminal_ack_failure_leaves_complete_with_durable_pending_intent(self):
        ack_attempted = threading.Event()
        factory_calls = 0

        class FailingAckSession(SqlAlchemySession):
            def commit(self):
                ack_attempted.set()
                self.flush()
                raise RuntimeError("intent ack 提交失败")

        failing_ack_factory = sessionmaker(
            bind=self.engine,
            class_=FailingAckSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return failing_ack_factory() if factory_calls == 5 else self.Session()

        recorder = self._recorder(session_factory=session_factory)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        await recorder.finalize(0)

        self.assertTrue(ack_attempted.is_set())
        self.assertIsNone(recorder.degraded_reason)
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertIsNotNone(meta.finalized_at)
            self.assertEqual(meta.terminal_intent_status, "complete")
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertEqual(meta.terminal_intent_version, 1)
            self.assertIsNotNone(meta.terminal_intent_pending_at)

    async def test_finalize_persists_pending_intent_before_assessment_begins(self):
        assessment_started = threading.Event()
        release_assessment = threading.Event()
        factory_calls = 0

        class BlockingAssessmentSession(SqlAlchemySession):
            def execute(self, *args, **kwargs):
                assessment_started.set()
                release_assessment.wait(timeout=2)
                return super().execute(*args, **kwargs)

        blocking_assessment_factory = sessionmaker(
            bind=self.engine,
            class_=BlockingAssessmentSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return blocking_assessment_factory() if factory_calls == 3 else self.Session()

        recorder = self._recorder(session_factory=session_factory)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(assessment_started.wait, 1))

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "recording")
            self.assertEqual(meta.expected_last_sequence, 0)
            self.assertEqual(meta.terminal_intent_status, "complete")
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertEqual(meta.terminal_intent_version, 1)
            self.assertIsNotNone(meta.terminal_intent_pending_at)

        release_assessment.set()
        await finalize_task
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertIsNone(meta.terminal_intent_pending_at)

    async def test_finalize_never_downgrades_existing_degraded_pending_intent(self):
        recorder = self._recorder()
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            meta.expected_last_sequence = 0
            meta.terminal_intent_status = "degraded"
            meta.terminal_intent_reason = "recorder_cancelled"
            meta.terminal_intent_version = 1
            meta.terminal_intent_pending_at = datetime.now(UTC)
            db.commit()

        await recorder.finalize(0)

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "recording")
            self.assertEqual(meta.terminal_intent_status, "degraded")
            self.assertEqual(meta.terminal_intent_reason, "recorder_cancelled")
            self.assertEqual(meta.terminal_intent_version, 1)
            self.assertIsNotNone(meta.terminal_intent_pending_at)

    async def test_finalize_marks_gap_as_degraded(self):
        recorder = self._recorder()
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await recorder.record_chunk("conv-1", "agent_event", _event(2))

        await recorder.finalize(2)

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.expected_last_sequence, 2)
            self.assertIsNone(meta.finalized_at)
            self.assertEqual(meta.degraded_reason, "finalize_mismatch")

    async def test_finalize_never_overwrites_persisted_degraded_with_complete(self):
        recorder = self._recorder()
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            meta.trajectory_status = "degraded"
            meta.degraded_reason = "prior_persistent_failure"
            db.commit()

        await recorder.finalize(0)

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "prior_persistent_failure")
            self.assertIsNone(meta.finalized_at)

    async def test_finalize_checks_latch_before_matching_count_min_max(self):
        recorder = self._recorder()
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await recorder.record_chunk("conv-1", "agent_event", _event(1, event_type="future_event"))

        await recorder.finalize(0)

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.event_count, 1)
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "unsupported_event_type")
            self.assertIsNone(meta.finalized_at)

    async def test_late_last_event_success_never_clears_latch_or_completes_run(self):
        first_call = True
        late_started = threading.Event()
        allow_late_commit = threading.Event()
        late_finished = threading.Event()

        def controlled_worker(operation):
            nonlocal first_call
            if first_call:
                first_call = False
                operation()
                return
            late_started.set()
            allow_late_commit.wait(timeout=2)
            operation()
            late_finished.set()

        recorder = self._recorder(worker=controlled_worker)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        late_task = asyncio.create_task(recorder.record_chunk("conv-1", "agent_event", _event(1)))
        self.assertTrue(await asyncio.to_thread(late_started.wait, 1))
        await late_task
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        allow_late_commit.set()
        self.assertTrue(await asyncio.to_thread(late_finished.wait, 1))

        await recorder.finalize(1)

        with self.Session() as db:
            self.assertEqual(db.query(AgentEvent).filter_by(run_id="run-1").count(), 2)
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_timeout")
            self.assertIsNone(meta.finalized_at)

    async def test_finalize_timeout_late_complete_is_corrected_to_degraded(self):
        finalize_started = threading.Event()
        release_finalize = threading.Event()
        finalize_finished = threading.Event()
        worker_calls = 0
        semaphore = threading.BoundedSemaphore(4)

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            if worker_calls == 1:
                return operation()
            finalize_started.set()
            release_finalize.wait(timeout=2)
            try:
                return operation()
            finally:
                finalize_finished.set()

        recorder = self._recorder(worker=controlled_worker, semaphore=semaphore)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(finalize_started.wait, 1))

        await finalize_task
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        release_finalize.set()
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_timeout")
            self.assertIsNone(meta.finalized_at)
        _assert_all_permits_available(self, semaphore)

    async def test_finalize_cancel_late_complete_is_corrected_and_cancel_reraised(self):
        finalize_started = threading.Event()
        release_finalize = threading.Event()
        finalize_finished = threading.Event()
        worker_calls = 0
        semaphore = threading.BoundedSemaphore(4)

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            if worker_calls == 1:
                return operation()
            finalize_started.set()
            release_finalize.wait(timeout=2)
            try:
                return operation()
            finally:
                finalize_finished.set()

        recorder = self._recorder(worker=controlled_worker, semaphore=semaphore)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(finalize_started.wait, 1))

        finalize_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await finalize_task
        self.assertEqual(recorder.degraded_reason, "recorder_cancelled")
        release_finalize.set()
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_cancelled")
            self.assertIsNone(meta.finalized_at)
            self.assertIsNone(meta.terminal_intent_status)
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertIsNone(meta.terminal_intent_version)
            self.assertIsNone(meta.terminal_intent_pending_at)
        _assert_all_permits_available(self, semaphore)

    async def test_finalize_cancel_after_ack_wins_before_terminal_decision(self):
        decision_entered = threading.Event()
        release_decision = threading.Event()
        finalize_finished = threading.Event()
        worker_calls = 0

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    finalize_finished.set()

        recorder = self._recorder(worker=controlled_worker)
        original_decision = recorder._seal_latch_for_terminal_decision

        def blocked_decision():
            decision_entered.set()
            release_decision.wait(timeout=2)
            return original_decision()

        recorder._seal_latch_for_terminal_decision = blocked_decision
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(decision_entered.wait, 1))

        finalize_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await finalize_task
        release_decision.set()
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))

        self.assertEqual(recorder.degraded_reason, "recorder_cancelled")
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_cancelled")
            self.assertIsNone(meta.finalized_at)

    async def test_finalize_rejects_new_record_while_assessment_commit_is_in_flight(self):
        commit_ready = threading.Event()
        release_commit = threading.Event()
        finalize_finished = threading.Event()
        factory_calls = 0
        worker_calls = 0

        class BlockingCommitSession(SqlAlchemySession):
            def commit(self):
                commit_ready.set()
                release_commit.wait(timeout=2)
                super().commit()

        blocking_factory = sessionmaker(bind=self.engine, class_=BlockingCommitSession, expire_on_commit=False)

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return self.Session() if factory_calls == 1 else blocking_factory()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    finalize_finished.set()

        recorder = self._recorder(session_factory=session_factory, worker=controlled_worker)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(commit_ready.wait, 1))

        await asyncio.to_thread(
            lambda: asyncio.run(
                recorder.record_chunk("conv-1", "agent_event", _event(1, event_type="future_event"))
            )
        )
        self.assertIsNone(recorder.degraded_reason)
        release_commit.set()
        await finalize_task
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))

        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertIsNone(meta.degraded_reason)
            self.assertIsNotNone(meta.finalized_at)

    async def test_failed_terminal_transition_keeps_recording_reachable_and_consumes_exception(self):
        finalize_started = threading.Event()
        release_finalize = threading.Event()
        finalize_finished = threading.Event()
        terminal_attempted = threading.Event()
        factory_calls = 0
        worker_calls = 0
        loop_errors: list[dict] = []
        loop = asyncio.get_running_loop()
        previous_handler = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
        self.addCleanup(loop.set_exception_handler, previous_handler)

        class FailingTerminalSession(SqlAlchemySession):
            def commit(self):
                terminal_attempted.set()
                self.flush()
                raise RuntimeError("终态提交失败")

        failing_factory = sessionmaker(
            bind=self.engine,
            class_=FailingTerminalSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return failing_factory() if factory_calls == 5 else self.Session()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            if worker_calls == 1:
                return operation()
            finalize_started.set()
            release_finalize.wait(timeout=2)
            try:
                return operation()
            finally:
                finalize_finished.set()

        recorder = self._recorder(session_factory=session_factory, worker=controlled_worker)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(finalize_started.wait, 1))

        await finalize_task
        release_finalize.set()
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))
        self.assertTrue(terminal_attempted.is_set())
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "recording")
            self.assertEqual(meta.expected_last_sequence, 0)
            self.assertIsNone(meta.finalized_at)
            self.assertEqual(meta.terminal_intent_status, "degraded")
            self.assertEqual(meta.terminal_intent_reason, "recorder_timeout")
            self.assertEqual(meta.terminal_intent_version, 1)
            self.assertIsNotNone(meta.terminal_intent_pending_at)

        del finalize_task
        gc.collect()
        await asyncio.sleep(0)
        self.assertFalse(
            [context for context in loop_errors if "never retrieved" in context.get("message", "").lower()]
        )

    async def test_terminal_commit_response_loss_reconciles_complete_without_false_latch(self):
        commit_persisted = threading.Event()
        factory_calls = 0

        class CommitThenLoseResponseSession(SqlAlchemySession):
            def commit(self):
                super().commit()
                commit_persisted.set()
                raise RuntimeError("终态 commit 响应丢失")

        response_loss_factory = sessionmaker(
            bind=self.engine,
            class_=CommitThenLoseResponseSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return response_loss_factory() if factory_calls == 4 else self.Session()

        recorder = self._recorder(session_factory=session_factory)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        await recorder.finalize(0)

        self.assertTrue(commit_persisted.is_set())
        self.assertIsNone(recorder.degraded_reason)
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertEqual(meta.expected_last_sequence, 0)
            self.assertIsNotNone(meta.finalized_at)
            self.assertIsNone(meta.degraded_reason)

    async def test_terminal_commit_and_reconciliation_failure_stays_unknown_without_false_latch(self):
        commit_persisted = threading.Event()
        reconciliation_attempted = threading.Event()
        factory_calls = 0

        class CommitThenLoseResponseSession(SqlAlchemySession):
            def commit(self):
                super().commit()
                commit_persisted.set()
                raise RuntimeError("终态 commit 响应丢失")

        class FailingReconciliationSession(SqlAlchemySession):
            def execute(self, *args, **kwargs):
                reconciliation_attempted.set()
                raise RuntimeError("终态对账查询失败")

        response_loss_factory = sessionmaker(
            bind=self.engine,
            class_=CommitThenLoseResponseSession,
            expire_on_commit=False,
        )
        failing_reconciliation_factory = sessionmaker(
            bind=self.engine,
            class_=FailingReconciliationSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 4:
                return response_loss_factory()
            if factory_calls == 5:
                return failing_reconciliation_factory()
            return self.Session()

        recorder = self._recorder(session_factory=session_factory)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        await recorder.finalize(0)

        self.assertTrue(commit_persisted.is_set())
        self.assertTrue(reconciliation_attempted.is_set())
        self.assertIsNone(recorder.degraded_reason)
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertEqual(meta.expected_last_sequence, 0)
            self.assertIsNotNone(meta.finalized_at)
            self.assertIsNone(meta.degraded_reason)

    async def test_terminal_wait_timeout_returns_bounded_and_late_worker_converges_degraded(self):
        terminal_commit_started = threading.Event()
        release_terminal_commit = threading.Event()
        terminal_worker_finished = threading.Event()
        factory_calls = 0
        worker_calls = 0
        semaphore = threading.BoundedSemaphore(4)

        class BlockingTerminalSession(SqlAlchemySession):
            def commit(self):
                terminal_commit_started.set()
                release_terminal_commit.wait(timeout=2)
                super().commit()

        blocking_factory = sessionmaker(
            bind=self.engine,
            class_=BlockingTerminalSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return blocking_factory() if factory_calls == 4 else self.Session()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    terminal_worker_finished.set()

        recorder = self._recorder(
            session_factory=session_factory,
            worker=controlled_worker,
            semaphore=semaphore,
        )
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(terminal_commit_started.wait, 1))

        try:
            await asyncio.wait_for(
                asyncio.shield(finalize_task),
                timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS * 3,
            )
        except BaseException:
            release_terminal_commit.set()
            await finalize_task
            raise

        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        self.assertFalse(terminal_worker_finished.is_set())
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "recording")
            self.assertEqual(meta.expected_last_sequence, 0)
            self.assertIsNone(meta.finalized_at)
            self.assertEqual(meta.terminal_intent_status, "complete")
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertEqual(meta.terminal_intent_version, 1)
            self.assertIsNotNone(meta.terminal_intent_pending_at)

        release_terminal_commit.set()
        self.assertTrue(await asyncio.to_thread(terminal_worker_finished.wait, 1))
        await asyncio.sleep(0)
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_timeout")
            self.assertIsNone(meta.finalized_at)
        _assert_all_permits_available(self, semaphore)

    async def test_terminal_timeout_response_loss_and_first_reconciliation_failure_converges_degraded(self):
        terminal_commit_started = threading.Event()
        release_terminal_commit = threading.Event()
        terminal_worker_finished = threading.Event()
        first_reconciliation_failed = threading.Event()
        factory_calls = 0
        worker_calls = 0
        semaphore = threading.BoundedSemaphore(4)

        class CommitThenLoseResponseSession(SqlAlchemySession):
            def commit(self):
                terminal_commit_started.set()
                release_terminal_commit.wait(timeout=2)
                super().commit()
                raise RuntimeError("终态 commit 响应丢失")

        class FailFirstReconciliationSession(SqlAlchemySession):
            def execute(self, *args, **kwargs):
                first_reconciliation_failed.set()
                raise RuntimeError("首次终态对账失败")

        response_loss_factory = sessionmaker(
            bind=self.engine,
            class_=CommitThenLoseResponseSession,
            expire_on_commit=False,
        )
        first_failure_factory = sessionmaker(
            bind=self.engine,
            class_=FailFirstReconciliationSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 4:
                return response_loss_factory()
            if factory_calls == 5:
                return first_failure_factory()
            return self.Session()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    terminal_worker_finished.set()

        recorder = self._recorder(
            session_factory=session_factory,
            worker=controlled_worker,
            semaphore=semaphore,
        )
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(terminal_commit_started.wait, 1))

        await asyncio.wait_for(
            asyncio.shield(finalize_task),
            timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS * 3,
        )
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        pending = recorder.pending_terminal_reconciliation
        self.assertIsNotNone(pending)
        self.assertEqual(pending.target_status, "degraded")
        self.assertEqual(pending.degraded_reason, "recorder_timeout")
        release_terminal_commit.set()
        self.assertTrue(await asyncio.to_thread(terminal_worker_finished.wait, 1))

        self.assertTrue(first_reconciliation_failed.is_set())
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_timeout")
            self.assertIsNone(meta.finalized_at)
        self.assertIsNone(recorder.pending_terminal_reconciliation)
        _assert_all_permits_available(self, semaphore)

    async def test_terminal_cancel_response_loss_and_first_reconciliation_failure_converges_degraded(self):
        terminal_commit_started = threading.Event()
        release_terminal_commit = threading.Event()
        terminal_worker_finished = threading.Event()
        first_reconciliation_failed = threading.Event()
        release_first_reconciliation = threading.Event()
        factory_calls = 0
        worker_calls = 0
        semaphore = threading.BoundedSemaphore(4)

        class CommitThenLoseResponseSession(SqlAlchemySession):
            def commit(self):
                terminal_commit_started.set()
                release_terminal_commit.wait(timeout=2)
                super().commit()
                raise RuntimeError("终态 commit 响应丢失")

        class FailFirstReconciliationSession(SqlAlchemySession):
            def execute(self, *args, **kwargs):
                first_reconciliation_failed.set()
                release_first_reconciliation.wait(timeout=2)
                raise RuntimeError("首次终态对账失败")

        response_loss_factory = sessionmaker(
            bind=self.engine,
            class_=CommitThenLoseResponseSession,
            expire_on_commit=False,
        )
        first_failure_factory = sessionmaker(
            bind=self.engine,
            class_=FailFirstReconciliationSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 4:
                return response_loss_factory()
            if factory_calls == 5:
                return first_failure_factory()
            return self.Session()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    terminal_worker_finished.set()

        recorder = self._recorder(
            session_factory=session_factory,
            worker=controlled_worker,
            semaphore=semaphore,
        )
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(terminal_commit_started.wait, 1))
        release_terminal_commit.set()
        self.assertTrue(await asyncio.to_thread(first_reconciliation_failed.wait, 1))

        finalize_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await finalize_task
        self.assertEqual(recorder.degraded_reason, "recorder_cancelled")
        pending = recorder.pending_terminal_reconciliation
        self.assertIsNotNone(pending)
        self.assertEqual(pending.target_status, "degraded")
        self.assertEqual(pending.degraded_reason, "recorder_cancelled")
        release_first_reconciliation.set()
        self.assertTrue(await asyncio.to_thread(terminal_worker_finished.wait, 1))

        self.assertTrue(first_reconciliation_failed.is_set())
        self.assertEqual(recorder.degraded_reason, "recorder_cancelled")
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_cancelled")
            self.assertIsNone(meta.finalized_at)
            self.assertIsNone(meta.terminal_intent_status)
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertIsNone(meta.terminal_intent_version)
            self.assertIsNone(meta.terminal_intent_pending_at)
        self.assertIsNone(recorder.pending_terminal_reconciliation)
        _assert_all_permits_available(self, semaphore)

    async def test_persistent_terminal_reconciliation_failure_exposes_request_without_latch_complete_conflict(self):
        terminal_commit_started = threading.Event()
        release_terminal_commit = threading.Event()
        terminal_worker_finished = threading.Event()
        reconciliation_attempts = 0
        factory_calls = 0
        worker_calls = 0
        semaphore = threading.BoundedSemaphore(4)

        class CommitThenLoseResponseSession(SqlAlchemySession):
            def commit(self):
                terminal_commit_started.set()
                release_terminal_commit.wait(timeout=2)
                super().commit()
                raise RuntimeError("终态 commit 响应丢失")

        class UnavailableReconciliationSession(SqlAlchemySession):
            def execute(self, *args, **kwargs):
                nonlocal reconciliation_attempts
                reconciliation_attempts += 1
                raise RuntimeError("终态数据库持续不可达")

        response_loss_factory = sessionmaker(
            bind=self.engine,
            class_=CommitThenLoseResponseSession,
            expire_on_commit=False,
        )
        unavailable_factory = sessionmaker(
            bind=self.engine,
            class_=UnavailableReconciliationSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            if factory_calls == 4:
                return response_loss_factory()
            if factory_calls >= 5:
                return unavailable_factory()
            return self.Session()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    terminal_worker_finished.set()

        recorder = self._recorder(
            session_factory=session_factory,
            worker=controlled_worker,
            semaphore=semaphore,
        )
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(terminal_commit_started.wait, 1))

        await asyncio.wait_for(
            asyncio.shield(finalize_task),
            timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS * 3,
        )
        release_terminal_commit.set()
        self.assertTrue(await asyncio.to_thread(terminal_worker_finished.wait, 1))

        self.assertGreaterEqual(reconciliation_attempts, 2)
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        request = recorder.pending_terminal_reconciliation
        self.assertIsNotNone(request)
        self.assertEqual(request.run_id, "run-1")
        self.assertEqual(request.expected_last_sequence, 0)
        self.assertEqual(request.degraded_reason, "recorder_timeout")
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertEqual(meta.expected_last_sequence, 0)
            self.assertIsNotNone(meta.finalized_at)
            self.assertIsNone(meta.degraded_reason)
            self.assertEqual(meta.terminal_intent_status, "complete")
            self.assertIsNone(meta.terminal_intent_reason)
            self.assertIsNotNone(meta.terminal_intent_pending_at)
            self.assertEqual(meta.terminal_intent_version, 1)
        _assert_all_permits_available(self, semaphore)

    async def test_finalize_start_closes_record_admission_before_assessment(self):
        finalize_started = threading.Event()
        release_finalize = threading.Event()
        finalize_finished = threading.Event()
        worker_calls = 0

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            if call_index == 2:
                finalize_started.set()
                release_finalize.wait(timeout=2)
            try:
                return operation()
            finally:
                if call_index == 2:
                    finalize_finished.set()

        recorder = self._recorder(worker=controlled_worker)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(finalize_started.wait, 1))

        await recorder.record_chunk("conv-1", "agent_event", _event(1))
        release_finalize.set()
        await finalize_task
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))

        self.assertEqual(worker_calls, 2)
        self.assertIsNone(recorder.degraded_reason)
        with self.Session() as db:
            self.assertEqual(db.query(AgentEvent).filter_by(run_id="run-1").count(), 1)
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")

    async def test_post_terminal_decision_record_and_mark_cannot_create_latch_complete_combo(self):
        terminal_committed = threading.Event()
        release_terminal_worker = threading.Event()
        finalize_finished = threading.Event()
        factory_calls = 0
        worker_calls = 0

        class ObserveTerminalCommitSession(SqlAlchemySession):
            def commit(self):
                status = self.execute(
                    select(RunTrajectoryMeta.trajectory_status).where(RunTrajectoryMeta.run_id == "run-1")
                ).scalar_one_or_none()
                super().commit()
                if status in {"complete", "degraded"}:
                    terminal_committed.set()
                    release_terminal_worker.wait(timeout=2)

        observed_factory = sessionmaker(
            bind=self.engine,
            class_=ObserveTerminalCommitSession,
            expire_on_commit=False,
        )

        def session_factory():
            nonlocal factory_calls
            factory_calls += 1
            return self.Session() if factory_calls == 1 else observed_factory()

        def controlled_worker(operation):
            nonlocal worker_calls
            worker_calls += 1
            call_index = worker_calls
            try:
                return operation()
            finally:
                if call_index == 2:
                    finalize_finished.set()

        recorder = self._recorder(session_factory=session_factory, worker=controlled_worker)
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        finalize_task = asyncio.create_task(recorder.finalize(0))
        self.assertTrue(await asyncio.to_thread(terminal_committed.wait, 1))

        await recorder.record_chunk("conv-1", "agent_event", _event(1))
        await asyncio.to_thread(recorder._mark_degraded, "post_terminal_decision")
        release_terminal_worker.set()
        await finalize_task
        self.assertTrue(await asyncio.to_thread(finalize_finished.wait, 1))

        self.assertIsNone(recorder.degraded_reason)
        with self.Session() as db:
            self.assertEqual(db.query(AgentEvent).filter_by(run_id="run-1").count(), 1)
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "complete")
            self.assertIsNotNone(meta.finalized_at)

    async def test_finalize_worker_without_event_loop_acknowledgement_persists_timeout_degraded(self):
        recorder = self._recorder()
        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        loop = asyncio.get_running_loop()
        handshake = _FinalizeHandshake(
            assessment_future=loop.create_future(),
            acknowledgement=threading.Event(),
            expected_last_sequence=0,
        )

        result = await asyncio.to_thread(recorder._finalize_with_handshake, handshake)

        self.assertEqual(result, "degraded")
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        with self.Session() as db:
            meta = db.get(RunTrajectoryMeta, "run-1")
            self.assertEqual(meta.trajectory_status, "degraded")
            self.assertEqual(meta.degraded_reason, "recorder_timeout")
            self.assertIsNone(meta.finalized_at)


class RecorderConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    def _recorder(self, *, executor, semaphore, worker=None) -> TrajectoryRecorder:
        return TrajectoryRecorder(
            run_id="run-1",
            conversation_id="conv-1",
            message_id="msg-1",
            session_factory=Mock(side_effect=AssertionError("并发测试不得访问数据库")),
            executor=executor,
            semaphore=semaphore,
            worker=worker or (lambda operation: operation()),
        )

    async def test_full_admission_fails_open_without_submitting(self):
        semaphore = threading.BoundedSemaphore(4)
        for _ in range(4):
            self.assertTrue(semaphore.acquire(blocking=False))
        executor = Mock()
        recorder = self._recorder(executor=executor, semaphore=semaphore)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        executor.submit.assert_not_called()
        self.assertEqual(recorder.degraded_reason, "admission_full")
        for _ in range(4):
            semaphore.release()

    async def test_submit_failure_releases_callers_permit(self):
        semaphore = threading.BoundedSemaphore(4)
        recorder = self._recorder(executor=FailingSubmitExecutor(), semaphore=semaphore)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        self.assertEqual(recorder.degraded_reason, "write_failed")
        _assert_all_permits_available(self, semaphore)

    async def test_worker_success_and_failure_release_the_only_permit(self):
        for error in (None, RuntimeError("worker failed")):
            with self.subTest(error=error):
                semaphore = threading.BoundedSemaphore(4)
                executor = ManualExecutor()

                def worker(_operation):
                    if error is not None:
                        raise error

                recorder = self._recorder(executor=executor, semaphore=semaphore, worker=worker)
                task = asyncio.create_task(recorder.record_chunk("conv-1", "agent_event", _event(0)))
                self.assertTrue(await asyncio.to_thread(executor.submitted.wait, 1))
                executor.run_next()
                await task
                _assert_all_permits_available(self, semaphore)

    async def test_real_wait_for_timeout_shields_late_failure_and_preserves_first_latch(self):
        semaphore = threading.BoundedSemaphore(4)
        executor = ManualExecutor()

        def worker(_operation):
            raise RuntimeError("迟到异常")

        recorder = self._recorder(executor=executor, semaphore=semaphore, worker=worker)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))

        self.assertEqual(recorder.degraded_reason, "recorder_timeout")
        self.assertEqual(executor.pending_count, 1)
        executor.run_next()
        await asyncio.sleep(0)
        _assert_all_permits_available(self, semaphore)

        await recorder.record_chunk("conv-1", "agent_event", _event(1))
        self.assertEqual(executor.pending_count, 0)
        self.assertEqual(recorder.degraded_reason, "recorder_timeout")

    async def test_cancel_before_worker_starts_relatches_reraises_and_eventually_releases(self):
        semaphore = threading.BoundedSemaphore(4)
        executor = ManualExecutor()
        recorder = self._recorder(executor=executor, semaphore=semaphore)
        task = asyncio.create_task(recorder.record_chunk("conv-1", "agent_event", _event(0)))
        self.assertTrue(await asyncio.to_thread(executor.submitted.wait, 1))

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertEqual(recorder.degraded_reason, "recorder_cancelled")
        executor.run_next()
        await asyncio.sleep(0)
        _assert_all_permits_available(self, semaphore)

    async def test_cancel_running_worker_relatches_reraises_and_eventually_releases(self):
        semaphore = threading.BoundedSemaphore(4)
        started = threading.Event()
        release_worker = threading.Event()
        finished = threading.Event()

        def worker(_operation):
            started.set()
            release_worker.wait(timeout=2)
            finished.set()

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.addCleanup(executor.shutdown, wait=True)
        recorder = self._recorder(executor=executor, semaphore=semaphore, worker=worker)
        task = asyncio.create_task(recorder.record_chunk("conv-1", "agent_event", _event(0)))
        self.assertTrue(await asyncio.to_thread(started.wait, 1))

        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(recorder.degraded_reason, "recorder_cancelled")

        release_worker.set()
        self.assertTrue(await asyncio.to_thread(finished.wait, 1))
        await asyncio.sleep(0)
        _assert_all_permits_available(self, semaphore)

    async def test_unknown_event_latches_and_never_reaches_worker(self):
        semaphore = threading.BoundedSemaphore(4)
        executor = Mock()
        recorder = self._recorder(executor=executor, semaphore=semaphore)

        await recorder.record_chunk("conv-1", "agent_event", _event(0, event_type="future_event"))

        self.assertEqual(recorder.degraded_reason, "unsupported_event_type")
        executor.submit.assert_not_called()
        _assert_all_permits_available(self, semaphore)


class RecorderConfigurationTests(unittest.TestCase):
    def test_production_session_factory_configures_fixed_postgresql_timeouts(self):
        engine = object()
        factory = object()
        with (
            patch("app.services.agent.trajectory_recorder.create_engine", return_value=engine) as create_engine_mock,
            patch("app.services.agent.trajectory_recorder.sessionmaker", return_value=factory) as sessionmaker_mock,
        ):
            result = create_trajectory_session_factory("postgresql://db.example/fusion")

        self.assertIs(result, factory)
        create_engine_mock.assert_called_once_with(
            "postgresql://db.example/fusion",
            connect_args={
                "connect_timeout": TRAJECTORY_CONNECT_TIMEOUT_SECONDS,
                "options": (
                    f"-c statement_timeout={TRAJECTORY_STATEMENT_TIMEOUT_MS} "
                    f"-c lock_timeout={TRAJECTORY_LOCK_TIMEOUT_MS}"
                ),
            },
            pool_pre_ping=True,
        )
        sessionmaker_mock.assert_called_once_with(autocommit=False, autoflush=False, bind=engine)
        self.assertEqual(TRAJECTORY_WAIT_TIMEOUT_SECONDS, 0.25)

    def test_sqlite_session_factory_does_not_claim_postgresql_timeout_support(self):
        with (
            patch("app.services.agent.trajectory_recorder.create_engine", return_value=object()) as create_engine_mock,
            patch("app.services.agent.trajectory_recorder.sessionmaker", return_value=object()),
        ):
            create_trajectory_session_factory("sqlite:///:memory:")

        self.assertEqual(create_engine_mock.call_args.kwargs, {"pool_pre_ping": True})


if __name__ == "__main__":
    unittest.main()
