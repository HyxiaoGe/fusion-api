"""推荐问题独立后台任务及进程重启恢复。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.db.models import Message as MessageModel
from app.services.suggested_question_service import (
    SUGGESTION_STATUS_PENDING,
    SuggestedQuestionClaim,
    SuggestedQuestionService,
)

SessionFactory = Callable[[], Any]
ScheduleFn = Callable[[SuggestedQuestionClaim], Any]

_worker_tasks: set[asyncio.Task] = set()
_shutdown_in_progress = False


def schedule_suggested_question_generation(
    claim: SuggestedQuestionClaim,
    *,
    session_factory: SessionFactory | None = None,
) -> asyncio.Task:
    """脱离聊天 task registry 启动任务，下一轮消息不会取消它。"""

    factory = session_factory or SessionLocal
    coroutine = run_suggested_question_generation_worker(claim, session_factory=factory)
    try:
        task = asyncio.create_task(
            coroutine,
            name=f"suggested-questions:{claim.message_id}:{claim.revision}",
        )
    except Exception:
        coroutine.close()
        _mark_failed_with_new_session(claim, factory)
        raise
    _worker_tasks.add(task)
    task.add_done_callback(_worker_tasks.discard)
    return task


async def run_suggested_question_generation_worker(
    claim: SuggestedQuestionClaim,
    *,
    session_factory: SessionFactory | None = None,
) -> None:
    """使用独立 DB session 生成；异常与非停机取消会把当前 revision 收敛到 failed。"""

    factory = session_factory or SessionLocal
    db = factory()
    try:
        await SuggestedQuestionService(db).generate_claimed_questions(claim)
    except asyncio.CancelledError:
        db.rollback()
        db.close()
        db = None
        # graceful shutdown 保留 pending，下一进程启动后按同 revision 恢复；
        # 其他意外取消才收敛为 failed，避免运行中永久 pending。
        if not _shutdown_in_progress:
            _mark_failed_with_new_session(claim, factory)
        raise
    except Exception as error:  # noqa: BLE001 — 独立辅助任务不得污染事件循环
        db.rollback()
        db.close()
        db = None
        _mark_failed_with_new_session(claim, factory)
        logger.warning(
            "推荐问题后台任务失败: message_id=%s revision=%s error_type=%s",
            claim.message_id,
            claim.revision,
            type(error).__name__,
        )
    finally:
        if db is not None:
            db.close()


async def recover_pending_suggested_questions(
    *,
    session_factory: SessionFactory | None = None,
    schedule_fn: ScheduleFn | None = None,
) -> int:
    """应用启动时重新调度 orphan pending；同 revision 重复执行由 CAS 保证幂等。"""

    factory = session_factory or SessionLocal
    scheduler = schedule_fn or (lambda claim: schedule_suggested_question_generation(claim, session_factory=factory))
    db = factory()
    try:
        messages = (
            db.query(MessageModel)
            .filter(
                MessageModel.role == "assistant",
                MessageModel.suggested_questions_status == SUGGESTION_STATUS_PENDING,
            )
            .order_by(MessageModel.created_at.asc(), MessageModel.id.asc())
            .all()
        )
        claims = [
            SuggestedQuestionClaim(
                message_id=message.id,
                conversation_id=message.conversation_id,
                revision=int(message.suggested_questions_revision or 0),
                model_id=message.model_id or message.conversation.model_id,
            )
            for message in messages
        ]
    except Exception as error:  # noqa: BLE001 — 恢复失败不能阻止 API 启动
        db.rollback()
        logger.warning("恢复 pending 推荐问题失败: error_type=%s", type(error).__name__)
        return 0
    finally:
        db.close()

    scheduled = 0
    for claim in claims:
        try:
            scheduler(claim)
            scheduled += 1
        except Exception as error:  # noqa: BLE001 — 单条失败不阻断其余恢复
            _mark_failed_with_new_session(claim, factory)
            logger.warning(
                "重新调度 pending 推荐问题失败: message_id=%s revision=%s error_type=%s",
                claim.message_id,
                claim.revision,
                type(error).__name__,
            )
    if scheduled:
        logger.info("已恢复 pending 推荐问题任务: count=%s", scheduled)
    return scheduled


async def stop_suggested_question_workers() -> None:
    """关闭应用时取消独立任务并保留 pending，交给下一进程按原 revision 恢复。"""

    global _shutdown_in_progress

    tasks = [task for task in _worker_tasks if not task.done()]
    _shutdown_in_progress = True
    try:
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        _shutdown_in_progress = False


def fail_claimed_suggested_questions(claim: SuggestedQuestionClaim) -> None:
    """流终态失败时用独立 session 收敛已领取但尚未调度的 revision。"""

    _mark_failed_with_new_session(claim, SessionLocal)


def _mark_failed_with_new_session(claim: SuggestedQuestionClaim, session_factory: SessionFactory) -> None:
    db = session_factory()
    try:
        SuggestedQuestionService(db).mark_generation_failed(claim)
    except Exception as error:  # noqa: BLE001 — 进程重启恢复仍可再次收敛 orphan pending
        db.rollback()
        logger.warning(
            "收敛推荐问题 pending 失败: message_id=%s revision=%s error_type=%s",
            claim.message_id,
            claim.revision,
            type(error).__name__,
        )
    finally:
        db.close()
