"""派生状态写入：agent_sessions / agent_steps 行的 INSERT/UPDATE。

emitter 不碰 DB；本模块由 stream_handler (Task 9) 在 emit 调用点平行调用。
所有函数 async，但内部用同步 SQLAlchemy session（沿用项目惯例）。
"""
from __future__ import annotations

from app.db.database import SessionLocal
from app.db.models import AgentSession, AgentStep


async def write_session_started(*, run_id: str, conversation_id: str,
                                user_id: str, model_id: str,
                                provider: str) -> None:
    """run 启动时立即创建 agent_sessions 行（status='running' 占位）。

    AgentSession 表的 user_id / model_id / provider 都是 NOT NULL，
    必须由调用方提供。终态由 write_session_status 在 finally 块更新。
    """
    with SessionLocal() as session:
        row = AgentSession(
            id=run_id,
            conversation_id=conversation_id,
            user_id=user_id,
            model_id=model_id,
            provider=provider,
            status="running",  # 占位，终态由 write_session_status 更新
            total_steps=0,
            total_tool_calls=0,
        )
        session.add(row)
        session.commit()


async def write_step_started(*, run_id: str, step_id: str,
                             step_number: int) -> None:
    """step 开始时插入 agent_steps 行（status='running'）。"""
    with SessionLocal() as session:
        step_row = AgentStep(
            id=step_id,
            trace_id=run_id,
            step_number=step_number,
            status="running",
            tool_names=[],
            duration_ms=0,
        )
        session.add(step_row)
        session.commit()


async def write_step_completed(*, step_id: str,
                               tool_names: list[str] | None = None,
                               duration_ms: int = 0) -> None:
    """step 正常结束时 update agent_steps 行。

    row 不存在时 silently return（极少发生：write_step_started 失败但 emitter 仍发了事件）。
    """
    with SessionLocal() as session:
        row = session.get(AgentStep, step_id)
        if row is None:
            return
        row.status = "completed"
        if tool_names is not None:
            row.tool_names = tool_names
        row.duration_ms = duration_ms
        session.commit()


async def write_step_terminal(*, step_id: str, status: str) -> None:
    """step 异常结束（failed / interrupted）时 update。

    row 不存在时 silently return。
    """
    assert status in ("failed", "interrupted"), f"invalid step terminal status: {status!r}"
    with SessionLocal() as session:
        row = session.get(AgentStep, step_id)
        if row is None:
            return
        row.status = status
        session.commit()


async def write_session_status(*, run_id: str, status: str,
                               total_steps: int, total_tool_calls: int) -> None:
    """run 终态写入 agent_sessions 行。

    row 不存在时 silently return。
    """
    assert status in ("completed", "limit_reached", "interrupted", "error"), \
        f"invalid session terminal status: {status!r}"
    with SessionLocal() as session:
        row = session.get(AgentSession, run_id)
        if row is None:
            return
        row.status = status
        row.total_steps = total_steps
        row.total_tool_calls = total_tool_calls
        session.commit()
