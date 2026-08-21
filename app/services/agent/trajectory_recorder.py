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
    operation_future: asyncio.Future[str]
    acknowledgement: threading.Event
    expected_last_sequence: int
    initial_latched_reason: str | None


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
        self._latch_lock = threading.Lock()
        self._degraded_reason: str | None = None

    @property
    def degraded_reason(self) -> str | None:
        with self._latch_lock:
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
        if self.degraded_latch():
            return
        try:
            stored_payload = build_trajectory_payload(payload)
        except UnsupportedTrajectoryEventError:
            self._mark_degraded("unsupported_event_type")
            return
        if stored_payload.get("run_id") != self.run_id:
            self._mark_degraded("invalid_event")
            return

        await self._run_isolated(lambda: self._write_event(stored_payload))

    async def finalize(self, expected_last_sequence: int) -> None:
        """latch 优先持久化完整性，再在无 latch 时校验 COUNT/MIN/MAX。"""
        result = await self._run_finalize_isolated(expected_last_sequence)
        if result == "finalize_mismatch":
            self._mark_degraded("finalize_mismatch")

    def _mark_degraded(self, reason: str) -> None:
        with self._latch_lock:
            if self._degraded_reason is None:
                self._degraded_reason = reason

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

    async def _run_finalize_isolated(self, expected_last_sequence: int) -> str | None:
        if not self._semaphore.acquire(blocking=False):
            self._mark_degraded("admission_full")
            return None

        loop = asyncio.get_running_loop()
        operation_future: asyncio.Future[str] = loop.create_future()
        handshake = _FinalizeHandshake(
            operation_future=operation_future,
            acknowledgement=threading.Event(),
            expected_last_sequence=expected_last_sequence,
            initial_latched_reason=self.degraded_reason,
        )
        try:
            worker_future = loop.run_in_executor(
                self._executor,
                self._worker_entry,
                lambda: self._finalize_with_handshake(handshake),
            )
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._semaphore.release()
            operation_future.cancel()
            self._mark_degraded("write_failed")
            self._log_failure("提交 finalize worker 失败", error)
            return None

        try:
            result = await asyncio.wait_for(
                asyncio.shield(operation_future),
                timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            self._mark_degraded("recorder_cancelled")
            handshake.acknowledgement.set()
            self._consume_late(operation_future)
            self._consume_late(worker_future)
            raise
        except asyncio.TimeoutError:
            self._mark_degraded("recorder_timeout")
            handshake.acknowledgement.set()
            self._consume_late(operation_future)
            self._consume_late(worker_future)
            return None
        except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
            self._mark_degraded("write_failed")
            handshake.acknowledgement.set()
            self._consume_late(worker_future)
            self._log_failure("轨迹账本 finalize 失败", error)
            return None

        handshake.acknowledgement.set()
        self._consume_late(worker_future)
        return result

    def _worker_entry(self, operation: Callable[[], _T]) -> _T:
        try:
            return self._worker(operation)
        finally:
            self._semaphore.release()

    def _finalize_with_handshake(self, handshake: _FinalizeHandshake) -> str:
        result: str | None = None
        operation_error: BaseException | None = None
        try:
            result = self._finalize_transaction(
                expected_last_sequence=handshake.expected_last_sequence,
                latched_reason=handshake.initial_latched_reason,
            )
        except BaseException as error:
            operation_error = error

        self._publish_finalize_outcome(
            handshake.operation_future,
            result=result,
            error=operation_error,
        )
        acknowledged = handshake.acknowledgement.wait(timeout=TRAJECTORY_WAIT_TIMEOUT_SECONDS)
        if not acknowledged:
            self._mark_degraded("recorder_timeout")

        latched_reason = self.degraded_reason
        if latched_reason is not None:
            self._persist_degraded_transaction(
                expected_last_sequence=handshake.expected_last_sequence,
                degraded_reason=latched_reason,
            )
        if operation_error is not None:
            raise operation_error
        if result is None:  # pragma: no cover - 上述分支已覆盖全部结果
            raise RuntimeError("finalize worker 未返回结果")
        return result

    @staticmethod
    def _publish_finalize_outcome(
        future: asyncio.Future[str],
        *,
        result: str | None,
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

    def _finalize_transaction(
        self,
        *,
        expected_last_sequence: int,
        latched_reason: str | None,
    ) -> str:
        now = datetime.now(UTC)
        session = self._session_factory()
        try:
            self._insert_meta_if_missing(session, now=now)
            if latched_reason is not None:
                status = "degraded"
                degraded_reason = latched_reason
                finalized_at = None
            else:
                count, minimum, maximum = session.execute(
                    select(
                        func.count(AgentEvent.event_id),
                        func.min(AgentEvent.sequence),
                        func.max(AgentEvent.sequence),
                    ).where(AgentEvent.run_id == self.run_id)
                ).one()
                is_complete = (
                    expected_last_sequence >= 0
                    and count == expected_last_sequence + 1
                    and minimum == 0
                    and maximum == expected_last_sequence
                )
                status = "complete" if is_complete else "degraded"
                degraded_reason = None if is_complete else "finalize_mismatch"
                finalized_at = now if is_complete else None

            meta_update = (
                update(RunTrajectoryMeta)
                .where(RunTrajectoryMeta.run_id == self.run_id)
                .values(
                    trajectory_status=status,
                    expected_last_sequence=expected_last_sequence,
                    finalized_at=finalized_at,
                    degraded_reason=degraded_reason,
                    updated_at=now,
                )
            )
            if status == "complete":
                meta_update = meta_update.where(RunTrajectoryMeta.trajectory_status != "degraded")
            session.execute(meta_update)
            session.commit()
            return status if status == "complete" else str(degraded_reason)
        except BaseException:
            session.rollback()
            raise
        finally:
            session.close()

    def _persist_degraded_transaction(
        self,
        *,
        expected_last_sequence: int,
        degraded_reason: str,
    ) -> None:
        now = datetime.now(UTC)
        session = self._session_factory()
        try:
            self._insert_meta_if_missing(session, now=now)
            session.execute(
                update(RunTrajectoryMeta)
                .where(RunTrajectoryMeta.run_id == self.run_id)
                .values(
                    trajectory_status="degraded",
                    expected_last_sequence=expected_last_sequence,
                    finalized_at=None,
                    degraded_reason=degraded_reason,
                    updated_at=now,
                )
            )
            session.commit()
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
