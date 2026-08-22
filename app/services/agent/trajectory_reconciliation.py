"""轨迹账本终态协调与 legacy 判定。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.core.logger import app_logger
from app.db.models import AgentEvent, AgentSession, RunTrajectoryMeta, TrajectoryLedgerSettings
from app.schemas.trajectory import UserTrajectoryMetaRow

DEFAULT_RECONCILIATION_BATCH_SIZE = 100
DEFAULT_RECONCILIATION_STALE_GRACE = timedelta(seconds=60)
TERMINAL_OUTCOME_UNKNOWN_REASON = "terminal_outcome_unknown"

_SAFE_DEGRADED_REASONS = frozenset(
    {
        "admission_full",
        "expected_sequence_missing",
        "finalize_mismatch",
        "invalid_event",
        "ledger_settings_invalid",
        "ledger_settings_missing",
        "meta_missing",
        "recorder_cancelled",
        "recorder_timeout",
        "sequence_mismatch",
        TERMINAL_OUTCOME_UNKNOWN_REASON,
        "unsupported_event_type",
        "write_failed",
    }
)


@dataclass(frozen=True)
class TrajectoryStatusAssessment:
    """可由 P1 直接复用的完整性判定，不包含事件内容。"""

    trajectory_status: str
    degraded_reason: str | None


@dataclass(frozen=True)
class LedgerWatermarkResolution:
    ledger_enabled_at: datetime | None
    degraded_reason: str | None


@dataclass(frozen=True)
class TrajectoryReconciliationResult:
    processed: int = 0
    pending_degraded: int = 0
    recording_completed: int = 0
    recording_degraded: int = 0
    legacy_not_recorded: int = 0
    meta_missing_degraded: int = 0


def _aware_utc(value: datetime) -> datetime:
    """SQLite 可能返回 naive 值；显式按 UTC 解释，绝不依赖宿主机时区。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def resolve_ledger_watermark(rows: Sequence[Any]) -> LedgerWatermarkResolution:
    """验证数据库中的不可变单例水位；异常形态一律保守。"""
    if not rows:
        return LedgerWatermarkResolution(None, "ledger_settings_missing")
    if len(rows) != 1:
        return LedgerWatermarkResolution(None, "ledger_settings_invalid")

    row = rows[0]
    if isinstance(row, tuple):
        try:
            singleton_key, enabled_at = row
        except ValueError:
            return LedgerWatermarkResolution(None, "ledger_settings_invalid")
    else:
        singleton_key = getattr(row, "singleton_key", None)
        enabled_at = getattr(row, "ledger_enabled_at", None)
    if singleton_key != "default" or not isinstance(enabled_at, datetime):
        return LedgerWatermarkResolution(None, "ledger_settings_invalid")
    return LedgerWatermarkResolution(_aware_utc(enabled_at), None)


def classify_missing_trajectory_meta(
    *,
    run_created_at: datetime,
    ledger_enabled_at: datetime | None,
    ledger_error: str | None = None,
) -> TrajectoryStatusAssessment:
    """仅以持久化水位判断无 meta 的终态 run。"""
    if ledger_error is not None or ledger_enabled_at is None:
        reason = ledger_error if ledger_error in _SAFE_DEGRADED_REASONS else "ledger_settings_invalid"
        return TrajectoryStatusAssessment("degraded", reason)
    if _aware_utc(run_created_at) < _aware_utc(ledger_enabled_at):
        return TrajectoryStatusAssessment("legacy", "not_recorded")
    return TrajectoryStatusAssessment("degraded", "meta_missing")


def build_reconciliation_candidate_query(*, batch_size: int, stale_before: datetime):
    """构造锁定 meta 候选的 PostgreSQL 兼容查询。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    return (
        select(RunTrajectoryMeta)
        .join(AgentSession, AgentSession.id == RunTrajectoryMeta.run_id)
        .where(AgentSession.status != "running")
        .where(AgentSession.terminal_at.is_not(None))
        .where(AgentSession.terminal_at <= stale_before)
        .where(
            or_(
                and_(
                    RunTrajectoryMeta.trajectory_status == "recording",
                    RunTrajectoryMeta.terminal_intent_pending_at.is_(None),
                    RunTrajectoryMeta.updated_at <= stale_before,
                ),
                and_(
                    RunTrajectoryMeta.terminal_intent_pending_at.is_not(None),
                    RunTrajectoryMeta.terminal_intent_pending_at <= stale_before,
                    RunTrajectoryMeta.updated_at <= stale_before,
                ),
            )
        )
        .order_by(RunTrajectoryMeta.updated_at, RunTrajectoryMeta.run_id)
        .limit(batch_size)
        .with_for_update(of=(RunTrajectoryMeta, AgentSession), skip_locked=True)
    )


def _build_missing_meta_query(
    *,
    batch_size: int,
    ledger_enabled_at: datetime | None,
    stale_before: datetime,
):
    statement = (
        select(AgentSession)
        .outerjoin(RunTrajectoryMeta, RunTrajectoryMeta.run_id == AgentSession.id)
        .where(AgentSession.status != "running")
        .where(AgentSession.terminal_at.is_not(None))
        .where(AgentSession.terminal_at <= stale_before)
        .where(RunTrajectoryMeta.run_id.is_(None))
    )
    # 水位有效时只需持久收敛上线后的缺失；历史 run 由读取侧纯判定为 legacy，
    # 避免每批历史数据长期占满协调窗口。
    if ledger_enabled_at is not None:
        statement = statement.where(AgentSession.created_at >= ledger_enabled_at)
    return (
        statement.order_by(AgentSession.created_at, AgentSession.id)
        .limit(batch_size)
        .with_for_update(of=AgentSession, skip_locked=True)
    )


def _safe_pending_reason(meta: RunTrajectoryMeta) -> str:
    for candidate in (meta.degraded_reason, meta.terminal_intent_reason):
        if candidate in _SAFE_DEGRADED_REASONS:
            return str(candidate)
    return TERMINAL_OUTCOME_UNKNOWN_REASON


def _clear_terminal_intent(meta: RunTrajectoryMeta) -> None:
    meta.terminal_intent_id = None
    meta.terminal_intent_status = None
    meta.terminal_intent_reason = None
    meta.terminal_intent_version = None
    meta.terminal_intent_pending_at = None


def _reconcile_meta(session: Any, meta: RunTrajectoryMeta, *, now: datetime) -> str:
    if meta.terminal_intent_pending_at is not None:
        meta.trajectory_status = "degraded"
        meta.finalized_at = None
        meta.degraded_reason = _safe_pending_reason(meta)
        _clear_terminal_intent(meta)
        meta.updated_at = now
        return "pending_degraded"

    expected = meta.expected_last_sequence
    if expected is None:
        meta.trajectory_status = "degraded"
        meta.finalized_at = None
        meta.degraded_reason = "expected_sequence_missing"
        meta.updated_at = now
        return "recording_degraded"

    count, minimum, maximum = session.execute(
        select(
            func.count(AgentEvent.event_id),
            func.min(AgentEvent.sequence),
            func.max(AgentEvent.sequence),
        ).where(AgentEvent.run_id == meta.run_id)
    ).one()
    is_complete = expected >= 0 and count == expected + 1 and minimum == 0 and maximum == expected
    if is_complete:
        meta.trajectory_status = "complete"
        meta.finalized_at = now
        meta.degraded_reason = None
        outcome = "recording_completed"
    else:
        meta.trajectory_status = "degraded"
        meta.finalized_at = None
        meta.degraded_reason = "sequence_mismatch"
        outcome = "recording_degraded"
    meta.updated_at = now
    return outcome


def _watermark_from_session(session: Any) -> LedgerWatermarkResolution:
    rows = session.execute(
        select(
            TrajectoryLedgerSettings.singleton_key,
            TrajectoryLedgerSettings.ledger_enabled_at,
        )
    ).all()
    normalized = [(row.singleton_key, row.ledger_enabled_at) for row in rows]
    return resolve_ledger_watermark(normalized)


def _insert_missing_meta_do_nothing(session: Any, values: dict[str, Any]) -> bool:
    """Recorder 与协调器竞态时只允许先到者创建 meta。"""
    dialect_name = session.get_bind().dialect.name
    if dialect_name == "postgresql":
        statement = postgresql_insert(RunTrajectoryMeta).values(**values)
    elif dialect_name == "sqlite":
        statement = sqlite_insert(RunTrajectoryMeta).values(**values)
    else:
        raise RuntimeError(f"轨迹协调不支持数据库方言: {dialect_name}")
    result = session.execute(statement.on_conflict_do_nothing(index_elements=["run_id"]))
    return result.rowcount == 1


def resolve_run_trajectory_status(session: Any, run_id: str) -> TrajectoryStatusAssessment:
    """读取侧完整性契约：pending complete 不得对外宣称 complete。"""
    meta = session.get(RunTrajectoryMeta, run_id)
    if meta is not None:
        return resolve_trajectory_status_from_rows(None, meta, None)
    run = session.get(AgentSession, run_id)
    watermark = _watermark_from_session(session)
    if run is None or not isinstance(run.created_at, datetime):
        return TrajectoryStatusAssessment("degraded", "meta_missing")
    return resolve_trajectory_status_from_rows(run, meta, watermark)


def resolve_trajectory_status_from_rows(
    run: AgentSession | None,
    meta: RunTrajectoryMeta | None,
    watermark: LedgerWatermarkResolution | datetime | None,
) -> TrajectoryStatusAssessment:
    """由已批量读取的 run/meta/水位判定完整性，避免 P1 列表 N+1 查询。"""
    if meta is not None:
        if meta.terminal_intent_pending_at is not None:
            return TrajectoryStatusAssessment("degraded", _safe_pending_reason(meta))
        if meta.trajectory_status in {"recording", "complete", "degraded", "legacy"}:
            return TrajectoryStatusAssessment(meta.trajectory_status, meta.degraded_reason)
        return TrajectoryStatusAssessment("degraded", TERMINAL_OUTCOME_UNKNOWN_REASON)

    if run is None or not isinstance(run.created_at, datetime):
        return TrajectoryStatusAssessment("degraded", "meta_missing")
    if isinstance(watermark, LedgerWatermarkResolution):
        ledger_enabled_at = watermark.ledger_enabled_at
        ledger_error = watermark.degraded_reason
    else:
        ledger_enabled_at = watermark
        ledger_error = None if isinstance(watermark, datetime) else "ledger_settings_missing"
    return classify_missing_trajectory_meta(
        run_created_at=run.created_at,
        ledger_enabled_at=ledger_enabled_at,
        ledger_error=ledger_error,
    )


def resolve_user_trajectory_status_from_rows(
    run_created_at: datetime,
    meta: UserTrajectoryMetaRow | None,
    watermark: LedgerWatermarkResolution,
) -> TrajectoryStatusAssessment:
    """普通 P1 读取侧状态判定；pending 与未知 meta 均只返回安全通用原因。"""
    if meta is None:
        return classify_missing_trajectory_meta(
            run_created_at=run_created_at,
            ledger_enabled_at=watermark.ledger_enabled_at,
            ledger_error=watermark.degraded_reason,
        )
    if meta.has_pending_terminal_intent:
        return TrajectoryStatusAssessment("degraded", TERMINAL_OUTCOME_UNKNOWN_REASON)
    if meta.trajectory_status == "degraded":
        reason = (
            meta.degraded_reason if meta.degraded_reason in _SAFE_DEGRADED_REASONS else TERMINAL_OUTCOME_UNKNOWN_REASON
        )
        return TrajectoryStatusAssessment("degraded", reason)
    if meta.trajectory_status in {"recording", "complete"}:
        return TrajectoryStatusAssessment(meta.trajectory_status, None)
    return TrajectoryStatusAssessment("degraded", TERMINAL_OUTCOME_UNKNOWN_REASON)


def reconcile_trajectory_batch(
    *,
    session_factory: Callable[[], Any],
    now: datetime | None = None,
    batch_size: int = DEFAULT_RECONCILIATION_BATCH_SIZE,
    stale_grace: timedelta = DEFAULT_RECONCILIATION_STALE_GRACE,
) -> TrajectoryReconciliationResult:
    """在一个短事务中幂等收敛一批终态 run。"""
    if batch_size <= 0:
        raise ValueError("batch_size 必须大于 0")
    if stale_grace < timedelta(0):
        raise ValueError("stale_grace 不能为负数")
    reconciled_at = _aware_utc(now or datetime.now(UTC))
    stale_before = reconciled_at - stale_grace
    counters = {
        "pending_degraded": 0,
        "recording_completed": 0,
        "recording_degraded": 0,
        "legacy_not_recorded": 0,
        "meta_missing_degraded": 0,
    }
    session = session_factory()
    try:
        meta_rows = (
            session.execute(
                build_reconciliation_candidate_query(
                    batch_size=batch_size,
                    stale_before=stale_before,
                )
            )
            .scalars()
            .all()
        )
        for meta in meta_rows:
            counters[_reconcile_meta(session, meta, now=reconciled_at)] += 1

        remaining = batch_size - len(meta_rows)
        if remaining > 0:
            watermark = _watermark_from_session(session)
            missing_runs = (
                session.execute(
                    _build_missing_meta_query(
                        batch_size=remaining,
                        ledger_enabled_at=watermark.ledger_enabled_at,
                        stale_before=stale_before,
                    )
                )
                .scalars()
                .all()
            )
            for run in missing_runs:
                assessment = classify_missing_trajectory_meta(
                    run_created_at=run.created_at,
                    ledger_enabled_at=watermark.ledger_enabled_at,
                    ledger_error=watermark.degraded_reason,
                )
                if assessment.trajectory_status == "legacy":
                    counters["legacy_not_recorded"] += 1
                    continue
                inserted = _insert_missing_meta_do_nothing(
                    session,
                    {
                        "run_id": run.id,
                        "conversation_id": run.conversation_id,
                        "message_id": run.message_id,
                        "trajectory_status": "degraded",
                        "event_count": 0,
                        "finalized_at": None,
                        "degraded_reason": assessment.degraded_reason,
                        "updated_at": reconciled_at,
                    },
                )
                if inserted:
                    counters["meta_missing_degraded"] += 1

        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()

    return TrajectoryReconciliationResult(
        processed=sum(counters.values()),
        **counters,
    )


async def reconcile_trajectory_best_effort(
    *,
    session_factory: Callable[[], Any] | None = None,
    logger: Any = app_logger,
) -> TrajectoryReconciliationResult:
    """供 scheduler 调用；失败只写安全摘要，不传播数据库异常。"""
    if session_factory is None:
        from app.db.database import SessionLocal

        session_factory = SessionLocal
    try:
        return await asyncio.to_thread(
            reconcile_trajectory_batch,
            session_factory=session_factory,
        )
    except Exception:  # noqa: BLE001 — scheduler 辅助任务必须 fail-open
        logger.error("轨迹账本协调任务失败")
        return TrajectoryReconciliationResult()
