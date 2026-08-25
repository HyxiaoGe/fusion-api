"""LLM round 正文详情的辅助后台写入器。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable

from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.core.logger import app_logger
from app.db.database import SessionLocal
from app.db.models import AgentLlmRoundDetail
from app.utils.user_visible_content import sanitize_user_visible_reasoning

LLM_DETAIL_PREVIEW_MAX_CHARS = 200
LLM_DETAIL_TEXT_MAX_CHARS = 100_000

SessionFactory = Callable[[], Any]


@dataclass(frozen=True)
class LlmRoundDetailDraft:
    conversation_id: str
    run_id: str
    message_id: str | None
    llm_round_id: str
    reasoning_text: str
    content_text: str


_worker_tasks: set[asyncio.Task[None]] = set()


def schedule_llm_round_detail(
    draft: LlmRoundDetailDraft,
    *,
    session_factory: SessionFactory | None = None,
    on_degraded: Callable[[str], Any] | None = None,
) -> asyncio.Task[None]:
    """调度 fail-open 详情写入；调用方无需等待数据库。"""

    factory = session_factory or SessionLocal
    coroutine = _run_llm_round_detail_worker(
        draft,
        session_factory=factory,
        on_degraded=on_degraded,
    )
    try:
        task = asyncio.create_task(
            coroutine,
            name=f"llm-round-detail:{draft.run_id}:{draft.llm_round_id}",
        )
    except BaseException:
        coroutine.close()
        _notify_degraded(on_degraded)
        raise
    _worker_tasks.add(task)
    task.add_done_callback(_release_worker_task)
    return task


async def stop_llm_round_detail_workers() -> None:
    """应用关闭时取消并观察所有尚未结束的辅助写入。"""

    tasks = [task for task in _worker_tasks if not task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_llm_round_detail_worker(
    draft: LlmRoundDetailDraft,
    *,
    session_factory: SessionFactory,
    on_degraded: Callable[[str], Any] | None,
) -> None:
    try:
        await asyncio.to_thread(_write_llm_round_detail, draft, session_factory)
    except asyncio.CancelledError:
        raise
    except Exception as error:  # noqa: BLE001 — 辅助详情不可中断 Agent 主链路
        _notify_degraded(on_degraded)
        app_logger.warning(
            "LLM round 详情写入失败: run_id=%s llm_round_id=%s error_type=%s",
            draft.run_id,
            draft.llm_round_id,
            type(error).__name__,
        )


def _write_llm_round_detail(draft: LlmRoundDetailDraft, session_factory: SessionFactory) -> None:
    sanitized_reasoning = sanitize_user_visible_reasoning(draft.reasoning_text, final=True)
    reasoning_text, reasoning_truncated = _truncate_text(sanitized_reasoning)
    content_text, content_truncated = _truncate_text(draft.content_text)
    redacted_fields = ["reasoning_text"] if sanitized_reasoning != draft.reasoning_text else []
    truncated_fields = [
        field_name
        for field_name, truncated in (
            ("reasoning_text", reasoning_truncated),
            ("content_text", content_truncated),
        )
        if truncated
    ]
    values = {
        "conversation_id": draft.conversation_id,
        "run_id": draft.run_id,
        "message_id": draft.message_id,
        "llm_round_id": draft.llm_round_id,
        "reasoning_text": reasoning_text or None,
        "content_text": content_text or None,
        "reasoning_preview": _preview(reasoning_text),
        "output_preview": _preview(content_text),
        "redacted_fields": redacted_fields,
        "truncated_fields": truncated_fields,
    }

    session = session_factory()
    try:
        dialect_name = session.get_bind().dialect.name
        if dialect_name == "postgresql":
            statement = postgresql_insert(AgentLlmRoundDetail).values(**values)
        elif dialect_name == "sqlite":
            statement = sqlite_insert(AgentLlmRoundDetail).values(**values)
        else:
            raise RuntimeError(f"LLM round 详情不支持数据库方言: {dialect_name}")
        session.execute(
            statement.on_conflict_do_nothing(
                index_elements=["run_id", "llm_round_id"],
            )
        )
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()


def _truncate_text(text: str) -> tuple[str, bool]:
    if len(text) <= LLM_DETAIL_TEXT_MAX_CHARS:
        return text, False
    return text[:LLM_DETAIL_TEXT_MAX_CHARS], True


def _preview(text: str) -> str | None:
    if not text:
        return None
    return text[:LLM_DETAIL_PREVIEW_MAX_CHARS]


def _release_worker_task(task: asyncio.Task[None]) -> None:
    _worker_tasks.discard(task)
    if task.cancelled():
        return
    try:
        task.exception()
    except BaseException:
        return


def _notify_degraded(callback: Callable[[str], Any] | None) -> None:
    if callback is None:
        return
    try:
        callback("llm_detail_write_failed")
    except BaseException as error:  # noqa: BLE001 — 降级通知本身也不能击穿主链路
        app_logger.warning(
            "LLM round 详情降级通知失败: error_type=%s",
            type(error).__name__,
        )
