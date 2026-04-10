"""
后台生成任务管理器

维护 conversation_id → asyncio.Task 的映射。
同一 conversation 发新消息时，取消旧任务。
任务完成后自动从 registry 移除。
"""

import asyncio
from typing import Optional

from app.core.logger import app_logger as logger

# 模块级 task registry，进程内全局唯一
# key: conversation_id, value: (task, task_id)
_tasks: dict[str, tuple[asyncio.Task, str]] = {}


def register_task(conversation_id: str, task: asyncio.Task, task_id: str) -> None:
    """
    注册新任务，取消同一 conversation 的旧任务。
    task_id 由调用方预生成（避免时序问题）。
    """
    # 取消同一 conversation 的旧任务
    if conversation_id in _tasks:
        old_task, old_id = _tasks[conversation_id]
        if not old_task.done():
            old_task.cancel()
            logger.info(f"取消旧任务: conv_id={conversation_id}, old_task_id={old_id}")

    _tasks[conversation_id] = (task, task_id)

    # 任务完成后自动清理
    def _cleanup(fut: asyncio.Task):
        if _tasks.get(conversation_id) == (task, task_id):
            _tasks.pop(conversation_id, None)

    task.add_done_callback(_cleanup)


def get_task(conversation_id: str) -> Optional[asyncio.Task]:
    """获取当前运行中的任务"""
    entry = _tasks.get(conversation_id)
    return entry[0] if entry else None


def cancel_task(conversation_id: str) -> bool:
    """主动取消任务（用户手动 stop）"""
    entry = _tasks.get(conversation_id)
    if entry and not entry[0].done():
        entry[0].cancel()
        return True
    return False
