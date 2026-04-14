"""
Agent 日志写入服务 — 异步记录工具调用、Agent 步骤与会话

提供三个写入函数：
- log_tool_call(): 单次工具调用日志
- log_agent_step(): Agent 单步执行日志
- log_agent_session(): Agent 会话汇总日志

所有函数均使用独立 DB Session，失败时静默处理不阻塞主流程。
"""

import uuid as _uuid
from typing import Optional

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.models import AgentSession, AgentStep, ToolCallLog


async def log_tool_call(
    conversation_id: str,
    message_id: Optional[str],
    user_id: str,
    tool_name: str,
    status: str,
    duration_ms: Optional[int],
    model_id: str,
    provider: str,
    input_params: Optional[dict] = None,
    output_data: Optional[dict] = None,
    error_message: Optional[str] = None,
    metadata: Optional[dict] = None,
    log_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    step_number: Optional[int] = None,
) -> None:
    """异步记录工具调用日志，失败时静默处理不影响主流程"""
    try:
        db = SessionLocal()
        log = ToolCallLog(
            id=log_id or str(_uuid.uuid4()),
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            tool_name=tool_name,
            status=status,
            error_message=error_message,
            duration_ms=duration_ms,
            model_id=model_id,
            provider=provider,
            input_params=input_params,
            output_data=output_data,
            extra_metadata=metadata,
            trace_id=trace_id,
            step_number=step_number,
        )
        db.add(log)
        db.commit()
        logger.info(f"工具调用日志已记录: tool={tool_name}, status={status}, duration={duration_ms}ms")
    except Exception as e:
        logger.error(f"写入工具调用日志失败: {e}")
        db.rollback()
    finally:
        db.close()


async def log_agent_step(
    trace_id: str,
    step_number: int,
    tool_calls_count: int,
    tool_names: list[str],
    duration_ms: Optional[int] = None,
) -> None:
    """异步记录 Agent 单步执行，失败时静默处理"""
    try:
        db = SessionLocal()
        step = AgentStep(
            trace_id=trace_id,
            step_number=step_number,
            tool_calls_count=tool_calls_count,
            tool_names=tool_names,
            duration_ms=duration_ms,
        )
        db.add(step)
        db.commit()
        logger.info(f"Agent step 已记录: trace={trace_id}, step={step_number}, tools={tool_names}")
    except Exception as e:
        logger.error(f"写入 Agent step 日志失败: {e}")
        db.rollback()
    finally:
        db.close()


async def log_agent_session(
    trace_id: str,
    conversation_id: str,
    message_id: Optional[str],
    user_id: str,
    model_id: str,
    provider: str,
    total_steps: int,
    total_tool_calls: int,
    total_duration_ms: Optional[int],
    status: str,
    limit_reason: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """异步记录 Agent 会话汇总，失败时静默处理"""
    try:
        db = SessionLocal()
        session = AgentSession(
            id=trace_id,
            conversation_id=conversation_id,
            message_id=message_id,
            user_id=user_id,
            model_id=model_id,
            provider=provider,
            total_steps=total_steps,
            total_tool_calls=total_tool_calls,
            total_duration_ms=total_duration_ms,
            status=status,
            limit_reason=limit_reason,
            error_message=error_message,
        )
        db.add(session)
        db.commit()
        logger.info(f"Agent session 已记录: trace={trace_id}, steps={total_steps}, tools={total_tool_calls}, status={status}")
    except Exception as e:
        logger.error(f"写入 Agent session 日志失败: {e}")
        db.rollback()
    finally:
        db.close()
