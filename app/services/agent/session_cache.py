"""派生状态写入：agent_sessions / agent_steps 行的 INSERT/UPDATE。

emitter 不碰 DB；本模块由 stream_handler (Task 9) 在 emit 调用点平行调用。
所有函数 async，但内部用同步 SQLAlchemy session（沿用项目惯例）。
"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.models import AgentSession, AgentStep, Conversation

_ATTEMPT_ALLOCATION_RETRIES = 3
_ATTEMPT_UNIQUE_INDEX = "uq_agent_sessions_turn_attempt"


async def write_session_started(
    *,
    run_id: str,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    turn_message_id: str,
    message_id: str | None = None,
    previous_run_id: str | None = None,
    run_config: dict | None = None,
) -> None:
    """run 启动时 UPSERT agent_sessions 行（status='running' 占位）。

    幂等：同 run_id 二次调用（任务重试 / 恢复 / superseded）会更新已有行
    而不是抛 PK 冲突，避免后续 finally 的 write_session_status 跑不到。

    AgentSession 表的 user_id / model_id / provider 都是 NOT NULL，
    必须由调用方提供。终态由 write_session_status 在 finally 块更新。
    """
    if not turn_message_id:
        raise ValueError("turn_message_id 不能为空")

    for allocation_try in range(1, _ATTEMPT_ALLOCATION_RETRIES + 1):
        with SessionLocal() as session:
            existing = session.get(AgentSession, run_id)
            if existing is not None:
                _reset_existing_session(
                    existing,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    model_id=model_id,
                    provider=provider,
                    message_id=message_id,
                    run_config=run_config,
                )
                session.commit()
                return
            try:
                row = _allocate_new_session(
                    session,
                    run_id=run_id,
                    conversation_id=conversation_id,
                    user_id=user_id,
                    model_id=model_id,
                    provider=provider,
                    message_id=message_id,
                    turn_message_id=turn_message_id,
                    previous_run_id=previous_run_id,
                    run_config=run_config,
                )
                session.add(row)
                session.commit()
                return
            except IntegrityError as error:
                session.rollback()
                if not _is_attempt_unique_conflict(error) or allocation_try == _ATTEMPT_ALLOCATION_RETRIES:
                    raise
                logger.warning(
                    f"轨迹 attempt 分配冲突，准备重试: turn_message_id={turn_message_id} "
                    f"allocation_try={allocation_try}"
                )


def _allocate_new_session(
    session,
    *,
    run_id: str,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    message_id: str | None,
    turn_message_id: str,
    previous_run_id: str | None,
    run_config: dict | None,
) -> AgentSession:
    locked_conversation_id = session.execute(
        select(Conversation.id).where(Conversation.id == conversation_id).with_for_update()
    ).scalar_one_or_none()
    if locked_conversation_id is None:
        raise ValueError(f"conversation 不存在: {conversation_id}")

    max_attempt = session.execute(
        select(func.max(AgentSession.attempt_index)).where(
            AgentSession.conversation_id == conversation_id,
            AgentSession.turn_message_id == turn_message_id,
        )
    ).scalar_one()
    attempt_index = int(max_attempt or 0) + 1
    resolved_previous_run_id = previous_run_id
    if resolved_previous_run_id is None and attempt_index > 1:
        resolved_previous_run_id = session.execute(
            select(AgentSession.id)
            .where(
                AgentSession.conversation_id == conversation_id,
                AgentSession.turn_message_id == turn_message_id,
            )
            .order_by(
                AgentSession.attempt_index.desc().nullslast(),
                AgentSession.created_at.desc(),
                AgentSession.id.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        if resolved_previous_run_id is not None:
            logger.info(
                "TRAJECTORY_PREVIOUS_RUN_FALLBACK "
                f"turn_message_id={turn_message_id} run_id={run_id} previous_run_id={resolved_previous_run_id}"
            )

    return AgentSession(
        id=run_id,
        conversation_id=conversation_id,
        user_id=user_id,
        model_id=model_id,
        provider=provider,
        message_id=message_id,
        turn_message_id=turn_message_id,
        previous_run_id=resolved_previous_run_id,
        attempt_index=attempt_index,
        run_config=run_config,
        status="running",
        total_steps=0,
        total_tool_calls=0,
    )


def _reset_existing_session(
    existing: AgentSession,
    *,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    message_id: str | None,
    run_config: dict | None,
) -> None:
    """幂等恢复仅重置运行状态，不改动已分配的 turn/lineage/index。"""

    existing.conversation_id = conversation_id
    existing.user_id = user_id
    existing.model_id = model_id
    existing.provider = provider
    existing.message_id = message_id
    existing.run_config = run_config
    existing.status = "running"
    existing.limit_reason = None
    existing.error_message = None
    existing.total_duration_ms = None
    existing.total_steps = 0
    existing.total_tool_calls = 0


def _is_attempt_unique_conflict(error: IntegrityError) -> bool:
    constraint_name = getattr(getattr(error.orig, "diag", None), "constraint_name", None)
    if constraint_name == _ATTEMPT_UNIQUE_INDEX:
        return True
    return "agent_sessions.turn_message_id, agent_sessions.attempt_index" in str(error.orig)


async def write_step_started(*, run_id: str, step_id: str, step_number: int) -> None:
    """step 开始时插入 agent_steps 行（status='running'）。

    duration_ms / tool_names 留空（None / [], 由 write_step_completed 填）；
    避免 INSERT 时填 0 导致 'WHERE duration_ms<X' 误扫到 running step。
    """
    with SessionLocal() as session:
        step_row = AgentStep(
            id=step_id,
            trace_id=run_id,
            step_number=step_number,
            status="running",
            tool_names=[],
            duration_ms=None,
        )
        session.add(step_row)
        session.commit()


async def write_step_completed(
    *, step_id: str, tool_names: list[str] | None = None, tool_calls_count: int | None = None, duration_ms: int = 0
) -> None:
    """step 正常结束时 update agent_steps 行。

    tool_names / tool_calls_count 为 None 时不更新对应字段（沿用原值）。
    row 不存在时 silently return + log warning。
    """
    with SessionLocal() as session:
        row = session.get(AgentStep, step_id)
        if row is None:
            logger.warning(f"write_step_completed: agent_steps row missing step_id={step_id}")
            return
        row.status = "completed"
        if tool_names is not None:
            row.tool_names = tool_names
        if tool_calls_count is not None:
            row.tool_calls_count = tool_calls_count
        row.duration_ms = duration_ms
        session.commit()


async def write_step_terminal(*, step_id: str, status: str) -> None:
    """step 异常结束（failed / interrupted）时 update。

    row 不存在时 silently return + log warning。
    """
    if status not in ("failed", "interrupted"):
        raise ValueError(f"invalid step terminal status: {status!r}")
    with SessionLocal() as session:
        row = session.get(AgentStep, step_id)
        if row is None:
            logger.warning(f"write_step_terminal: agent_steps row missing step_id={step_id}")
            return
        row.status = status
        session.commit()


async def write_session_status(
    *,
    run_id: str,
    status: str,
    total_steps: int,
    total_tool_calls: int,
    total_duration_ms: int | None = None,
    limit_reason: str | None = None,
) -> None:
    """run 终态写入 agent_sessions 行。

    total_duration_ms 为 None 时不更新该字段（兼容某些不计时的路径）。
    row 不存在时 silently return + log warning。
    """
    if status not in ("completed", "limit_reached", "incomplete", "interrupted", "error"):
        raise ValueError(f"invalid session terminal status: {status!r}")
    with SessionLocal() as session:
        row = session.get(AgentSession, run_id)
        if row is None:
            logger.warning(f"write_session_status: agent_sessions row missing run_id={run_id}")
            return
        row.status = status
        row.total_steps = total_steps
        row.total_tool_calls = total_tool_calls
        row.limit_reason = limit_reason if status == "limit_reached" else None
        if total_duration_ms is not None:
            row.total_duration_ms = total_duration_ms
        session.commit()
