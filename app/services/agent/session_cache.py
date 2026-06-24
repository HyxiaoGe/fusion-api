"""派生状态写入：agent_sessions / agent_steps 行的 INSERT/UPDATE。

emitter 不碰 DB；本模块由 stream_handler (Task 9) 在 emit 调用点平行调用。
所有函数 async，但内部用同步 SQLAlchemy session（沿用项目惯例）。
"""

from __future__ import annotations

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.models import AgentSession, AgentStep


async def write_session_started(
    *,
    run_id: str,
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    message_id: str | None = None,
) -> None:
    """run 启动时 UPSERT agent_sessions 行（status='running' 占位）。

    幂等：同 run_id 二次调用（任务重试 / 恢复 / superseded）会更新已有行
    而不是抛 PK 冲突，避免后续 finally 的 write_session_status 跑不到。

    AgentSession 表的 user_id / model_id / provider 都是 NOT NULL，
    必须由调用方提供。终态由 write_session_status 在 finally 块更新。
    """
    with SessionLocal() as session:
        existing = session.get(AgentSession, run_id)
        if existing is not None:
            # 已有行：更新元信息 + 重置为 running 占位
            existing.conversation_id = conversation_id
            existing.user_id = user_id
            existing.model_id = model_id
            existing.provider = provider
            existing.message_id = message_id
            existing.status = "running"
            existing.total_steps = 0
            existing.total_tool_calls = 0
            session.commit()
            return
        row = AgentSession(
            id=run_id,
            conversation_id=conversation_id,
            user_id=user_id,
            model_id=model_id,
            provider=provider,
            message_id=message_id,
            status="running",  # 占位，终态由 write_session_status 更新
            total_steps=0,
            total_tool_calls=0,
        )
        session.add(row)
        session.commit()


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
    *, run_id: str, status: str, total_steps: int, total_tool_calls: int, total_duration_ms: int | None = None
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
        if total_duration_ms is not None:
            row.total_duration_ms = total_duration_ms
        session.commit()
