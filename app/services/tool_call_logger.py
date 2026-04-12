"""
工具调用统计日志 — 异步写入服务

在后台任务中记录每次 tool_call 的执行情况，不阻塞主流程。
使用独立 DB Session，与调用方 session 完全隔离。
"""

import uuid as _uuid
from typing import Optional

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.models import ToolCallLog


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
) -> None:
    """异步记录工具调用日志，失败时静默处理不影响主流程"""
    db = SessionLocal()
    try:
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
        )
        db.add(log)
        db.commit()
        logger.info(f"工具调用日志已记录: tool={tool_name}, status={status}, duration={duration_ms}ms")
    except Exception as e:
        logger.error(f"写入工具调用日志失败: {e}")
        db.rollback()
    finally:
        db.close()
