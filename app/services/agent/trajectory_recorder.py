"""有界、fail-open 的脱敏轨迹账本记录器。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import create_engine, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import sessionmaker

from app.core.config import settings
from app.core.logger import app_logger
from app.db.models import AgentEvent, RunTrajectoryMeta
from app.services.agent.trajectory_payload import UnsupportedTrajectoryEventError, build_trajectory_payload

TRAJECTORY_MAX_WORKERS = 4
TRAJECTORY_WAIT_TIMEOUT_SECONDS = 0.25
TRAJECTORY_STATEMENT_TIMEOUT_MS = 200
TRAJECTORY_LOCK_TIMEOUT_MS = 100
TRAJECTORY_CONNECT_TIMEOUT_SECONDS = 1
TRAJECTORY_TERMINAL_RECONCILIATION_ATTEMPTS = 2
TRAJECTORY_TERMINAL_INTENT_VERSION = 1

_EXECUTOR = ThreadPoolExecutor(max_workers=TRAJECTORY_MAX_WORKERS, thread_name_prefix="trajectory-recorder")
_ADMISSION_SEMAPHORE = threading.BoundedSemaphore(TRAJECTORY_MAX_WORKERS)
_DEFAULT_FACTORY_LOCK = threading.Lock()
_DEFAULT_SESSION_FACTORY: Callable[[], Any] | None = None

_T = TypeVar("_T")


class _TerminalOutcomeUnknownError(RuntimeError):
    """终态 commit 与后续对账均失败，持久化结果不可判定。"""


@dataclass(frozen=True)
class TrajectoryTerminalReconciliation:
    """供后续 stale coordinator 消费的幂等终态请求。"""

    run_id: str
    expected_last_sequence: int
    target_status: str
    degraded_reason: str | None


@dataclass(frozen=True)
class _FinalizeHandshake:
    assessment_future: asyncio.Future[bool]
    acknowledgement: threading.Event
    expected_last_sequence: int


def create_trajectory_session_factory(database_url: str) -> Callable[[], Any]:
    """创建账本专用 Session factory；PostgreSQL 超时由连接参数固定下发。"""
    engine_kwargs: dict[str, Any] = {"pool_pre_ping": True}
    if database_url.startswith(("postgresql://", "postgresql+psycopg2://")):
        engine_kwargs["connect_args"] = {
            "connect_timeout": TRAJECTORY_CONNECT_TIMEOUT_SECONDS,
            "options": (
                f"-c statement_timeout={TRAJECTORY_STATEMENT_TIMEOUT_MS} "
                f"-c lock_timeout={TRAJECTORY_LOCK_TIMEOUT_MS}"
            ),
        }
    engine = create_engine(database_url, **engine_kwargs)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def _new_default_session() -> Any:
    global _DEFAULT_SESSION_FACTORY
    if _DEFAULT_SESSION_FACTORY is None:
        with _DEFAULT_FACTORY_LOCK:
            if _DEFAULT_SESSION_FACTORY is None:
                _DEFAULT_SESSION_FACTORY = create_trajectory_session_factory(str(settings.DATABASE_URL))
    return _DEFAULT_SESSION_FACTORY()


def _execute_operation(operation: Callable[[], _T]) -> _T:
    return operation()


class TrajectoryRecorder:
    """每个 run 一个实例；数据库工作只使用独立短 Session。"""

    def __init__(
        self,
        *,
        run_id: str,
        conversation_id: str,
        message_id: str | None,
        session_factory: Callable[[], Any] = _new_default_session,
        executor: Executor = _EXECUTOR,
        semaphore: threading.BoundedSemaphore = _ADMISSION_SEMAPHORE,
        worker: Callable[[Callable[[], Any]], Any] | None = None,
        logger: Any = app_logger,
    ) -> None:
        self.run_id = run_id
        self.conversation_id = conversation_id
        self.message_id = message_id
        self._session_factory = session_factory
        self._executor = executor
        self._semaphore = semaphore
        self._worker = worker or _execute_operation
        self._logger = logger
        self._state_lock = threading.Lock()
        self._degraded_reason: str | None = None
        self._finalize_started = False
        self._latch_sealed = False
        self._terminal_transition_finished = False
        self._terminal_failure_reason: str | None = None
        self._pending_terminal_reconciliation: TrajectoryTerminalReconciliation | None = None
        self._active_records = 0
        self._record_drain_waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []

    @property
    def degraded_reason(self) -> str | None:
        with self._state_lock:
            return self._degraded_reason or self._terminal_failure_reason

    @property
    def pending_terminal_reconciliation(self) -> TrajectoryTerminalReconciliation | None:
        with self._state_lock:
            return self._pending_terminal_reconciliation

    def degraded_latch(self, run_id: str | None = None) -> bool:
        return (run_id is None or run_id == self.run_id) and self.degraded_reason is not None

    async def record_chunk(
        self,
        conversation_id: str,
        chunk_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if conversation_id != self.conversation_id or chunk_type != "agent_event":
            return
        if not self._try_admit_record():
            return
        try:
            try:
                stored_payload = build_trajectory_payload(payload)
            except UnsupportedTrajectoryEventError:
                self._mark_degraded("unsupported_event_type")
                return
            if stored_payload.get("run_id") != self.run_id:
                self._mark_degraded("invalid_event")
                return

            await self._run_isolated(lambda: self._write_event(stored_payload))
        finally:
            self._finish_record()

    async def finalize(self, expected_last_sequence: int) -> None:
        """关闭新写入并等待已接纳调用定论，再执行 assessment/ack/终态转换。"""
        if not await self._close_record_admission():
            return
        await self._run_finalize_isolated(expected_last_sequence)

    def _mark_degraded(self, reason: str) -> bool:
        """终态决议封口前首次原因胜出；封口后拒绝新的 latch 来源。"""
        with self._state_lock:
            if self._latch_sealed:
                return False
            if self._degraded_reason is None:
                self._degraded_reason = reason
            return True

    def _force_terminal_failure_latch(
        self,
        reason: str = "write_failed",
        *,
        expected_last_sequence: int | None = None,
    ) -> bool:
        """终态完成线性化前保留失败证据，完成后拒绝制造 latch 冲突。"""
        with self._state_lock:
            if self._terminal_transition_finished:
                return False
            if self._degraded_reason is None and self._terminal_failure_reason is None:
                self._terminal_failure_reason = reason
            if expected_last_sequence is not None:
                pending_reason = self._degraded_reason or self._terminal_failure_reason
                self._pending_terminal_reconciliation = TrajectoryTerminalReconciliation(
                    run_id=self.run_id,
                    expected_last_sequence=expected_last_sequence,
                    target_status="degraded",
                    degraded_reason=pending_reason,
                )
            return True

    def _try_admit_record(self) -> bool:
        with self._state_lock:
            if self._finalize_started or self._degraded_reason is not None:
                return False
            self._active_records += 1
            return True

    def _finish_record(self) -> None:
        waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []
        with self._state_lock:
            self._active_records -= 1
            if self._active_records == 0:
                waiters, self._record_drain_waiters = self._record_drain_waiters, []
        for loop, waiter in waiters:
            try:
                loop.call_soon_threadsafe(self._resolve_waiter, waiter)
            except RuntimeError:
                continue

    async def _close_record_admission(self) -> bool:
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[None] | None = None
        with self._state_lock:
            if self._finalize_started:
                return False
            self._finalize_started = True
            if self._active_records:
                waiter = loop.create_future()
                self._record_drain_waiters.append((loop, waiter))
        if waiter is None:
            return True
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            self._mark_degraded("recorder_cancelled")
            with self._state_lock:
                self._record_drain_waiters = [
                    item for item in self._record_drain_waiters if item[1] is not waiter
                ]
            waiter.cancel()
            raise
        return True

    @staticmethod
    def _resolve_waiter(waiter: asyncio.Future[None]) -> None:
        if not waiter.done():
            waiter.set_result(None)

    def _seal_latch_for_terminal_decision(self) -> str | None:
        with self._state_lock:
            self._latch_sealed = True
            return self._degraded_reason

    async def _run_isolated(self, operation: Callable[[], _T]) -> _T | None:
        if not self._semaphore.acquire(blocking=False):
            self._mark_degraded("admission_full")
            return None

        loop = asyncio.get_running_loop()
        try:
            future = loop.run_in_executor(self._executor, self._worker_entry, operation)
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._semaphore.release()
            self._mark_degraded("write_failed")
            self._log_failure("提交账本 worker 失败", error)
            return None

        try:
            return await asyncio.wait_for(
                asyncio.shield(future),
                timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            self._mark_degraded("recorder_cancelled")
            self._consume_late(future)
            raise
        except asyncio.TimeoutError:
            self._mark_degraded("recorder_timeout")
            self._consume_late(future)
            return None
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._mark_degraded("write_failed")
            self._log_failure("轨迹账本写入失败", error)
            return None

    async def _run_finalize_isolated(self, expected_last_sequence: int) -> None:
        if not self._semaphore.acquire(blocking=False):
            self._mark_degraded("admission_full")
            return None

        loop = asyncio.get_running_loop()
        assessment_future: asyncio.Future[bool] = loop.create_future()
        handshake = _FinalizeHandshake(
            assessment_future=assessment_future,
            acknowledgement=threading.Event(),
            expected_last_sequence=expected_last_sequence,
        )
        try:
            worker_future = loop.run_in_executor(
                self._executor,
                self._worker_entry,
                lambda: self._finalize_with_handshake(handshake),
            )
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._semaphore.release()
            assessment_future.cancel()
            self._mark_degraded("write_failed")
            self._log_failure("提交 finalize worker 失败", error)
            return

        try:
            assessment_complete = await asyncio.wait_for(
                asyncio.shield(assessment_future),
                timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            self._mark_degraded("recorder_cancelled")
            handshake.acknowledgement.set()
            self._consume_late(assessment_future)
            self._consume_late(worker_future)
            raise
        except asyncio.TimeoutError:
            self._mark_degraded("recorder_timeout")
            handshake.acknowledgement.set()
            self._consume_late(assessment_future)
            self._consume_late(worker_future)
            return
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._mark_degraded("write_failed")
            handshake.acknowledgement.set()
            self._consume_late(worker_future)
            self._log_failure("轨迹账本 finalize 失败", error)
            return

        if not assessment_complete:
            self._mark_degraded("finalize_mismatch")
        handshake.acknowledgement.set()
        try:
            terminal_status = await asyncio.wait_for(
                asyncio.shield(worker_future),
                timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS,
            )
            if terminal_status == "unknown":
                self._force_terminal_failure_latch(
                    "write_failed",
                    expected_last_sequence=expected_last_sequence,
                )
        except asyncio.CancelledError:
            self._force_terminal_failure_latch(
                "recorder_cancelled",
                expected_last_sequence=expected_last_sequence,
            )
            self._consume_late(worker_future)
            raise
        except asyncio.TimeoutError:
            self._force_terminal_failure_latch(
                "recorder_timeout",
                expected_last_sequence=expected_last_sequence,
            )
            self._consume_late(worker_future)
            return
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._log_failure("轨迹账本终态写入失败", error)

    def _worker_entry(self, operation: Callable[[], _T]) -> _T:
        try:
            return self._worker(operation)
        finally:
            self._semaphore.release()

    def _finalize_with_handshake(self, handshake: _FinalizeHandshake) -> str:
        assessment_complete = False
        assessment_error: BaseException | None = None
        intent_persisted = False
        try:
            with self._state_lock:
                initial_degraded_reason = self._degraded_reason
            self._persist_terminal_intent_transaction(
                expected_last_sequence=handshake.expected_last_sequence,
                degraded_reason=initial_degraded_reason,
            )
            intent_persisted = True
            assessment_complete = self._assess_finalize_transaction(handshake.expected_last_sequence)
        except BaseException as error:
            assessment_error = error

        self._publish_finalize_outcome(
            handshake.assessment_future,
            result=assessment_complete if assessment_error is None else None,
            error=assessment_error,
        )
        acknowledged = handshake.acknowledgement.wait(timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS)
        if not acknowledged:
            self._mark_degraded("recorder_timeout")

        latched_reason = self._seal_latch_for_terminal_decision()
        if not intent_persisted:
            if assessment_error is None:  # pragma: no cover - 防御性保护
                raise RuntimeError("终态 intent 未持久化")
            raise assessment_error
        try:
            if latched_reason is not None:
                self._upgrade_terminal_intent_transaction(
                    expected_last_sequence=handshake.expected_last_sequence,
                    degraded_reason=latched_reason,
                )
            try:
                terminal_status = self._write_terminal_transaction(
                    expected_last_sequence=handshake.expected_last_sequence,
                    assessment_complete=assessment_complete,
                    degraded_reason=latched_reason,
                )
            except _TerminalOutcomeUnknownError:
                terminal_status = self._retry_unknown_terminal_transition(
                    expected_last_sequence=handshake.expected_last_sequence,
                    assessment_complete=assessment_complete,
                )
            if terminal_status == "unknown":
                return terminal_status
            while True:
                with self._state_lock:
                    final_latched_reason = self._degraded_reason or self._terminal_failure_reason
                    if terminal_status != "complete" or final_latched_reason is None:
                        if terminal_status == "degraded" and self._degraded_reason is None:
                            self._degraded_reason = self._terminal_failure_reason
                        self._terminal_failure_reason = None
                        self._terminal_transition_finished = True
                        break
                self._upgrade_terminal_intent_transaction(
                    expected_last_sequence=handshake.expected_last_sequence,
                    degraded_reason=final_latched_reason,
                )
                try:
                    terminal_status = self._write_terminal_transaction(
                        expected_last_sequence=handshake.expected_last_sequence,
                        assessment_complete=False,
                        degraded_reason=final_latched_reason,
                        allow_complete_correction=True,
                    )
                except _TerminalOutcomeUnknownError:
                    terminal_status = self._retry_unknown_terminal_transition(
                        expected_last_sequence=handshake.expected_last_sequence,
                        assessment_complete=False,
                    )
                    if terminal_status == "unknown":
                        return terminal_status
            terminal_reason = final_latched_reason if terminal_status == "degraded" else None
            intent_acknowledged = self._ack_terminal_intent_transaction(
                expected_last_sequence=handshake.expected_last_sequence,
                terminal_status=terminal_status,
                degraded_reason=terminal_reason,
            )
            with self._state_lock:
                if intent_acknowledged:
                    self._pending_terminal_reconciliation = None
                else:
                    self._pending_terminal_reconciliation = TrajectoryTerminalReconciliation(
                        run_id=self.run_id,
                        expected_last_sequence=handshake.expected_last_sequence,
                        target_status=terminal_status,
                        degraded_reason=terminal_reason,
                    )
        except BaseException:
            self._force_terminal_failure_latch()
            raise
        if assessment_error is not None:
            raise assessment_error
        return terminal_status

    def _retry_unknown_terminal_transition(
        self,
        *,
        expected_last_sequence: int,
        assessment_complete: bool,
    ) -> str:
        """有限次重试未知终态；耗尽后发布显式 reconciliation 请求。"""
        for _ in range(TRAJECTORY_TERMINAL_RECONCILIATION_ATTEMPTS):
            with self._state_lock:
                degraded_reason = self._degraded_reason or self._terminal_failure_reason
            try:
                if degraded_reason is not None:
                    self._upgrade_terminal_intent_transaction(
                        expected_last_sequence=expected_last_sequence,
                        degraded_reason=degraded_reason,
                    )
                return self._write_terminal_transaction(
                    expected_last_sequence=expected_last_sequence,
                    assessment_complete=assessment_complete and degraded_reason is None,
                    degraded_reason=degraded_reason,
                    allow_complete_correction=degraded_reason is not None,
                )
            except (OSError, RuntimeError, _TerminalOutcomeUnknownError):
                continue

        with self._state_lock:
            degraded_reason = self._degraded_reason or self._terminal_failure_reason
            if degraded_reason is None:
                self._terminal_failure_reason = "write_failed"
                degraded_reason = self._terminal_failure_reason
            self._pending_terminal_reconciliation = TrajectoryTerminalReconciliation(
                run_id=self.run_id,
                expected_last_sequence=expected_last_sequence,
                target_status="degraded" if degraded_reason is not None else "complete",
                degraded_reason=degraded_reason,
            )
        try:
            self._upgrade_terminal_intent_transaction(
                expected_last_sequence=expected_last_sequence,
                degraded_reason=degraded_reason,
            )
        except BaseException:
            pass
        return "unknown"

    @staticmethod
    def _publish_finalize_outcome(
        future: asyncio.Future[bool],
        *,
        result: bool | None,
        error: BaseException | None,
    ) -> None:
        def _publish() -> None:
            if future.done():
                return
            if error is not None:
                future.set_exception(error)
            elif result is not None:
                future.set_result(result)

        try:
            future.get_loop().call_soon_threadsafe(_publish)
        except RuntimeError:
            return

    @staticmethod
    def _consume_late(future: asyncio.Future[Any]) -> None:
        def _consume(done: asyncio.Future[Any]) -> None:
            try:
                done.exception()
            except BaseException:
                return

        future.add_done_callback(_consume)

    def _write_event(self, payload: Mapping[str, Any]) -> None:
        event_ts = datetime.fromtimestamp(float(payload["ts"]), tz=UTC)
        now = datetime.now(UTC)
        session = self._session_factory()
        try:
            self._insert_meta_if_missing(session, now=now)
            event_statement = self._insert_do_nothing(
                session,
                AgentEvent,
                {
                    "conversation_id": self.conversation_id,
                    "message_id": self.message_id,
                    "run_id": self.run_id,
                    "sequence": int(payload["sequence"]),
                    "event_type": str(payload["type"]),
                    "schema_version": int(payload["schema_version"]),
                    "step_id": payload.get("step_id"),
                    "tool_call_id": payload.get("tool_call_id"),
                    "parent_step_id": payload.get("parent_step_id"),
                    "trace_id": payload.get("trace_id"),
                    "event_ts": event_ts,
                    "recorded_at": now,
                    "payload": dict(payload),
                },
                conflict_columns=("run_id", "sequence"),
            )
            inserted = session.execute(event_statement).rowcount == 1
            if inserted:
                session.execute(
                    update(RunTrajectoryMeta)
                    .where(RunTrajectoryMeta.run_id == self.run_id)
                    .values(
                        event_count=RunTrajectoryMeta.event_count + 1,
                        first_event_ts=func.coalesce(RunTrajectoryMeta.first_event_ts, event_ts),
                        last_event_ts=event_ts,
                        updated_at=now,
                    )
                )
            session.commit()
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def _persist_terminal_intent_transaction(
        self,
        *,
        expected_last_sequence: int,
        degraded_reason: str | None,
    ) -> None:
        """在 assessment 前用独立已确认事务持久化可恢复的终态意图。"""
        now = datetime.now(UTC)
        session = self._session_factory()
        target_status = "degraded" if degraded_reason is not None else "complete"
        operation_error: BaseException | None = None
        try:
            self._insert_meta_if_missing(session, now=now)
            statement = (
                update(RunTrajectoryMeta)
                .where(RunTrajectoryMeta.run_id == self.run_id)
                .where(RunTrajectoryMeta.trajectory_status.in_(("recording", "complete")))
            )
            if degraded_reason is None:
                statement = statement.where(
                    or_(
                        RunTrajectoryMeta.terminal_intent_status.is_(None),
                        RunTrajectoryMeta.terminal_intent_status == "complete",
                    )
                )
                intent_reason = None
            else:
                intent_reason = func.coalesce(
                    RunTrajectoryMeta.terminal_intent_reason,
                    degraded_reason,
                )
            result = session.execute(
                statement.values(
                    expected_last_sequence=expected_last_sequence,
                    terminal_intent_status=target_status,
                    terminal_intent_reason=intent_reason,
                    terminal_intent_version=TRAJECTORY_TERMINAL_INTENT_VERSION,
                    terminal_intent_pending_at=func.coalesce(
                        RunTrajectoryMeta.terminal_intent_pending_at,
                        now,
                    ),
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise RuntimeError("终态 intent 不满足 recording/complete 持久化条件")
            session.commit()
            return
        except BaseException as error:
            operation_error = error
            try:
                session.rollback()
            except BaseException:
                pass
        finally:
            session.close()

        persisted = self._terminal_intent_is_pending(
            expected_last_sequence=expected_last_sequence,
            target_status=target_status,
            degraded_reason=degraded_reason,
        )
        if persisted is True:
            return
        if persisted is None:
            raise _TerminalOutcomeUnknownError("终态 intent commit 与对账结果均未知") from operation_error
        if operation_error is None:  # pragma: no cover - try 分支已直接返回
            raise RuntimeError("终态 intent 事务缺少执行结果")
        raise operation_error

    def _upgrade_terminal_intent_transaction(
        self,
        *,
        expected_last_sequence: int,
        degraded_reason: str,
    ) -> None:
        """把 pending intent 单调升级为 degraded，首次原因不可被覆盖。"""
        now = datetime.now(UTC)
        session = self._session_factory()
        operation_error: BaseException | None = None
        try:
            result = session.execute(
                update(RunTrajectoryMeta)
                .where(RunTrajectoryMeta.run_id == self.run_id)
                .where(RunTrajectoryMeta.expected_last_sequence == expected_last_sequence)
                .where(RunTrajectoryMeta.trajectory_status.in_(("recording", "complete")))
                .where(RunTrajectoryMeta.terminal_intent_pending_at.is_not(None))
                .where(RunTrajectoryMeta.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION)
                .values(
                    terminal_intent_status="degraded",
                    terminal_intent_reason=func.coalesce(
                        RunTrajectoryMeta.terminal_intent_reason,
                        degraded_reason,
                    ),
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                raise RuntimeError("pending terminal intent 不可升级")
            session.commit()
            return
        except BaseException as error:
            operation_error = error
            try:
                session.rollback()
            except BaseException:
                pass
        finally:
            session.close()

        persisted = self._terminal_intent_is_pending(
            expected_last_sequence=expected_last_sequence,
            target_status="degraded",
            degraded_reason=degraded_reason,
        )
        if persisted is True:
            return
        if persisted is None:
            raise _TerminalOutcomeUnknownError("终态 intent 升级与对账结果均未知") from operation_error
        if operation_error is None:  # pragma: no cover - try 分支已直接返回
            raise RuntimeError("终态 intent 升级缺少执行结果")
        raise operation_error

    def _assess_finalize_transaction(self, expected_last_sequence: int) -> bool:
        """只读取 COUNT/MIN/MAX；pending intent 已由前置独立事务确认。"""
        session = self._session_factory()
        try:
            count, minimum, maximum = session.execute(
                select(
                    func.count(AgentEvent.event_id),
                    func.min(AgentEvent.sequence),
                    func.max(AgentEvent.sequence),
                ).where(AgentEvent.run_id == self.run_id)
            ).one()
            assessment_complete = (
                expected_last_sequence >= 0
                and count == expected_last_sequence + 1
                and minimum == 0
                and maximum == expected_last_sequence
            )
            session.commit()
            return assessment_complete
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def _write_terminal_transaction(
        self,
        *,
        expected_last_sequence: int,
        assessment_complete: bool,
        degraded_reason: str | None,
        allow_complete_correction: bool = False,
    ) -> str:
        """幂等转换终态；commit 结果不确定时用新 Session 对账。"""
        now = datetime.now(UTC)
        session = self._session_factory()
        terminal_status: str
        reason: str | None
        operation_error: BaseException | None = None
        try:
            if assessment_complete and degraded_reason is None:
                statement = (
                    update(RunTrajectoryMeta)
                    .where(RunTrajectoryMeta.run_id == self.run_id)
                    .where(RunTrajectoryMeta.trajectory_status == "recording")
                    .where(RunTrajectoryMeta.degraded_reason.is_(None))
                    .where(RunTrajectoryMeta.expected_last_sequence == expected_last_sequence)
                    .where(RunTrajectoryMeta.terminal_intent_status == "complete")
                    .where(RunTrajectoryMeta.terminal_intent_reason.is_(None))
                    .where(RunTrajectoryMeta.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION)
                    .where(RunTrajectoryMeta.terminal_intent_pending_at.is_not(None))
                    .values(
                        trajectory_status="complete",
                        expected_last_sequence=expected_last_sequence,
                        finalized_at=now,
                        degraded_reason=None,
                        updated_at=now,
                    )
                )
                terminal_status = "complete"
                reason = None
            else:
                reason = degraded_reason or "finalize_mismatch"
                eligible_status = (
                    RunTrajectoryMeta.trajectory_status.in_(("recording", "complete"))
                    if allow_complete_correction
                    else RunTrajectoryMeta.trajectory_status == "recording"
                )
                statement = (
                    update(RunTrajectoryMeta)
                    .where(RunTrajectoryMeta.run_id == self.run_id)
                    .where(eligible_status)
                    .where(RunTrajectoryMeta.expected_last_sequence == expected_last_sequence)
                    .where(RunTrajectoryMeta.terminal_intent_status == "degraded")
                    .where(RunTrajectoryMeta.terminal_intent_reason == reason)
                    .where(RunTrajectoryMeta.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION)
                    .where(RunTrajectoryMeta.terminal_intent_pending_at.is_not(None))
                    .values(
                        trajectory_status="degraded",
                        expected_last_sequence=expected_last_sequence,
                        finalized_at=None,
                        degraded_reason=reason,
                        updated_at=now,
                    )
                )
                terminal_status = "degraded"
            result = session.execute(statement)
            if result.rowcount != 1:
                raise RuntimeError("终态转换条件不满足")
            session.commit()
            return terminal_status
        except BaseException as error:
            operation_error = error
            try:
                session.rollback()
            except BaseException:
                pass
        finally:
            session.close()

        persisted = self._terminal_intent_is_persisted(
            expected_last_sequence=expected_last_sequence,
            terminal_status=terminal_status,
            degraded_reason=reason,
        )
        if persisted is True:
            return terminal_status
        if persisted is None:
            raise _TerminalOutcomeUnknownError("终态 commit 与对账结果均未知") from operation_error
        if operation_error is None:  # pragma: no cover - try 分支已直接返回
            raise RuntimeError("终态事务缺少执行结果")
        raise operation_error

    def _terminal_intent_is_persisted(
        self,
        *,
        expected_last_sequence: int,
        terminal_status: str,
        degraded_reason: str | None,
    ) -> bool | None:
        """用独立短 Session 判定 commit 响应丢失后的真实持久化结果。"""
        try:
            session = self._session_factory()
        except BaseException:
            return None
        try:
            row = session.execute(
                select(
                    RunTrajectoryMeta.trajectory_status,
                    RunTrajectoryMeta.expected_last_sequence,
                    RunTrajectoryMeta.finalized_at,
                    RunTrajectoryMeta.degraded_reason,
                    RunTrajectoryMeta.terminal_intent_status,
                    RunTrajectoryMeta.terminal_intent_reason,
                    RunTrajectoryMeta.terminal_intent_version,
                    RunTrajectoryMeta.terminal_intent_pending_at,
                ).where(RunTrajectoryMeta.run_id == self.run_id)
            ).one_or_none()
        except BaseException:
            return None
        finally:
            session.close()
        if row is None or row.expected_last_sequence != expected_last_sequence:
            return False
        if terminal_status == "complete":
            return (
                row.trajectory_status == "complete"
                and row.finalized_at is not None
                and row.degraded_reason is None
                and row.terminal_intent_status == "complete"
                and row.terminal_intent_reason is None
                and row.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION
                and row.terminal_intent_pending_at is not None
            )
        return (
            row.trajectory_status == "degraded"
            and row.finalized_at is None
            and row.degraded_reason == degraded_reason
            and row.terminal_intent_status == "degraded"
            and row.terminal_intent_reason == degraded_reason
            and row.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION
            and row.terminal_intent_pending_at is not None
        )

    def _terminal_intent_is_pending(
        self,
        *,
        expected_last_sequence: int,
        target_status: str,
        degraded_reason: str | None,
    ) -> bool | None:
        """对账 intent 的持久化真相；查询失败返回 unknown。"""
        try:
            session = self._session_factory()
        except BaseException:
            return None
        try:
            row = session.execute(
                select(
                    RunTrajectoryMeta.expected_last_sequence,
                    RunTrajectoryMeta.terminal_intent_status,
                    RunTrajectoryMeta.terminal_intent_reason,
                    RunTrajectoryMeta.terminal_intent_version,
                    RunTrajectoryMeta.terminal_intent_pending_at,
                ).where(RunTrajectoryMeta.run_id == self.run_id)
            ).one_or_none()
        except BaseException:
            return None
        finally:
            session.close()
        return bool(
            row is not None
            and row.expected_last_sequence == expected_last_sequence
            and row.terminal_intent_status == target_status
            and row.terminal_intent_reason == degraded_reason
            and row.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION
            and row.terminal_intent_pending_at is not None
        )

    def _ack_terminal_intent_transaction(
        self,
        *,
        expected_last_sequence: int,
        terminal_status: str,
        degraded_reason: str | None,
    ) -> bool:
        """终态明确后用独立幂等事务清除 pending；未知时保留给 stale 扫描。"""
        now = datetime.now(UTC)
        session = self._session_factory()
        operation_error: BaseException | None = None
        try:
            statement = (
                update(RunTrajectoryMeta)
                .where(RunTrajectoryMeta.run_id == self.run_id)
                .where(RunTrajectoryMeta.expected_last_sequence == expected_last_sequence)
                .where(RunTrajectoryMeta.trajectory_status == terminal_status)
                .where(RunTrajectoryMeta.terminal_intent_status == terminal_status)
                .where(RunTrajectoryMeta.terminal_intent_version == TRAJECTORY_TERMINAL_INTENT_VERSION)
                .where(RunTrajectoryMeta.terminal_intent_pending_at.is_not(None))
            )
            if terminal_status == "complete":
                statement = statement.where(RunTrajectoryMeta.finalized_at.is_not(None)).where(
                    RunTrajectoryMeta.degraded_reason.is_(None)
                ).where(RunTrajectoryMeta.terminal_intent_reason.is_(None))
            else:
                statement = statement.where(RunTrajectoryMeta.finalized_at.is_(None)).where(
                    RunTrajectoryMeta.degraded_reason == degraded_reason
                ).where(RunTrajectoryMeta.terminal_intent_reason == degraded_reason)
            result = session.execute(
                statement.values(
                    terminal_intent_status=None,
                    terminal_intent_reason=None,
                    terminal_intent_version=None,
                    terminal_intent_pending_at=None,
                    updated_at=now,
                )
            )
            if result.rowcount != 1:
                session.rollback()
                return self._terminal_intent_is_cleared(
                    expected_last_sequence=expected_last_sequence,
                    terminal_status=terminal_status,
                    degraded_reason=degraded_reason,
                ) is True
            session.commit()
            return True
        except BaseException as error:
            operation_error = error
            try:
                session.rollback()
            except BaseException:
                pass
        finally:
            session.close()

        cleared = self._terminal_intent_is_cleared(
            expected_last_sequence=expected_last_sequence,
            terminal_status=terminal_status,
            degraded_reason=degraded_reason,
        )
        if cleared is True:
            return True
        if operation_error is not None:
            self._log_failure("轨迹账本终态 intent ack 失败", operation_error)
        return False

    def _terminal_intent_is_cleared(
        self,
        *,
        expected_last_sequence: int,
        terminal_status: str,
        degraded_reason: str | None,
    ) -> bool | None:
        try:
            session = self._session_factory()
        except BaseException:
            return None
        try:
            row = session.execute(
                select(
                    RunTrajectoryMeta.trajectory_status,
                    RunTrajectoryMeta.expected_last_sequence,
                    RunTrajectoryMeta.finalized_at,
                    RunTrajectoryMeta.degraded_reason,
                    RunTrajectoryMeta.terminal_intent_status,
                    RunTrajectoryMeta.terminal_intent_reason,
                    RunTrajectoryMeta.terminal_intent_version,
                    RunTrajectoryMeta.terminal_intent_pending_at,
                ).where(RunTrajectoryMeta.run_id == self.run_id)
            ).one_or_none()
        except BaseException:
            return None
        finally:
            session.close()
        if row is None:
            return False
        terminal_matches = (
            row.trajectory_status == terminal_status
            and row.expected_last_sequence == expected_last_sequence
            and row.degraded_reason == degraded_reason
            and ((terminal_status == "complete" and row.finalized_at is not None) or
                 (terminal_status == "degraded" and row.finalized_at is None))
        )
        return bool(
            terminal_matches
            and row.terminal_intent_status is None
            and row.terminal_intent_reason is None
            and row.terminal_intent_version is None
            and row.terminal_intent_pending_at is None
        )

    def _insert_meta_if_missing(self, session: Any, *, now: datetime) -> None:
        statement = self._insert_do_nothing(
            session,
            RunTrajectoryMeta,
            {
                "run_id": self.run_id,
                "conversation_id": self.conversation_id,
                "message_id": self.message_id,
                "trajectory_status": "recording",
                "event_count": 0,
                "updated_at": now,
            },
            conflict_columns=("run_id",),
        )
        session.execute(statement)

    @staticmethod
    def _insert_do_nothing(
        session: Any,
        model: type[Any],
        values: Mapping[str, Any],
        *,
        conflict_columns: tuple[str, ...],
    ) -> Any:
        dialect_name = session.get_bind().dialect.name
        if dialect_name == "postgresql":
            statement = postgresql_insert(model).values(**values)
        elif dialect_name == "sqlite":
            statement = sqlite_insert(model).values(**values)
        else:
            raise RuntimeError(f"轨迹账本不支持数据库方言: {dialect_name}")
        return statement.on_conflict_do_nothing(index_elements=list(conflict_columns))

    def _log_failure(self, message: str, error: BaseException) -> None:
        self._logger.warning(f"{message}: run_id={self.run_id}, error_type={type(error).__name__}")
