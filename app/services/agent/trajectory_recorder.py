"""有界、fail-open 的脱敏轨迹账本记录器。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, TypeVar

from sqlalchemy import create_engine, func, select, update
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

_EXECUTOR = ThreadPoolExecutor(max_workers=TRAJECTORY_MAX_WORKERS, thread_name_prefix="trajectory-recorder")
_ADMISSION_SEMAPHORE = threading.BoundedSemaphore(TRAJECTORY_MAX_WORKERS)
_DEFAULT_FACTORY_LOCK = threading.Lock()
_DEFAULT_SESSION_FACTORY: Callable[[], Any] | None = None

_T = TypeVar("_T")


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
        self._active_records = 0
        self._record_drain_waiters: list[tuple[asyncio.AbstractEventLoop, asyncio.Future[None]]] = []

    @property
    def degraded_reason(self) -> str | None:
        with self._state_lock:
            return self._degraded_reason

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

    def _force_terminal_failure_latch(self) -> None:
        """终态事务失败时保留内存证据；此时数据库仍是 recording。"""
        with self._state_lock:
            if self._degraded_reason is None:
                self._degraded_reason = "write_failed"

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
            await asyncio.shield(worker_future)
        except asyncio.CancelledError:
            self._mark_degraded("recorder_cancelled")
            self._consume_late(worker_future)
            raise
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
        try:
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
        try:
            terminal_status = self._write_terminal_transaction(
                expected_last_sequence=handshake.expected_last_sequence,
                assessment_complete=assessment_complete,
                degraded_reason=latched_reason,
            )
        except BaseException:
            self._force_terminal_failure_latch()
            raise
        if assessment_error is not None:
            raise assessment_error
        return terminal_status

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

    def _assess_finalize_transaction(self, expected_last_sequence: int) -> bool:
        """只持久化完整性评估输入，禁止在 ack 前写入任何终态。"""
        now = datetime.now(UTC)
        session = self._session_factory()
        try:
            self._insert_meta_if_missing(session, now=now)
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
            session.execute(
                update(RunTrajectoryMeta)
                .where(RunTrajectoryMeta.run_id == self.run_id)
                .where(RunTrajectoryMeta.trajectory_status == "recording")
                .values(
                    expected_last_sequence=expected_last_sequence,
                    updated_at=now,
                )
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
    ) -> str:
        """从 recording 单次转换终态；提交失败由 rollback 保持协调器可达。"""
        now = datetime.now(UTC)
        session = self._session_factory()
        try:
            if assessment_complete and degraded_reason is None:
                statement = (
                    update(RunTrajectoryMeta)
                    .where(RunTrajectoryMeta.run_id == self.run_id)
                    .where(RunTrajectoryMeta.trajectory_status == "recording")
                    .where(RunTrajectoryMeta.degraded_reason.is_(None))
                    .values(
                        trajectory_status="complete",
                        expected_last_sequence=expected_last_sequence,
                        finalized_at=now,
                        degraded_reason=None,
                        updated_at=now,
                    )
                )
                terminal_status = "complete"
            else:
                reason = degraded_reason or "finalize_mismatch"
                statement = (
                    update(RunTrajectoryMeta)
                    .where(RunTrajectoryMeta.run_id == self.run_id)
                    .where(RunTrajectoryMeta.trajectory_status == "recording")
                    .values(
                        trajectory_status="degraded",
                        expected_last_sequence=expected_last_sequence,
                        finalized_at=None,
                        degraded_reason=reason,
                        updated_at=now,
                    )
                )
                terminal_status = "degraded"
            session.execute(statement)
            session.commit()
            return terminal_status
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

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
