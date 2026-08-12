from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models import KnowledgeStorageCleanupTask, KnowledgeStorageUploadIntent
from app.db.storage_cleanup_repository import (
    RegisteredStorageUploadIntent,
    StorageCleanupRepository,
)
from app.utils.time import utc_now

logger = logging.getLogger(__name__)

# asyncio 只保存 Task 的弱引用。请求取消后上传仍可能在线程中执行，因此必须由模块级
# 集合持有 lifecycle，直到真实 SDK 调用返回且持久 intent 已完成交接。
_ACTIVE_STORAGE_UPLOAD_LIFECYCLES: set[asyncio.Task[None]] = set()


class StorageUploadFenceLost(RuntimeError):
    """对象上传返回时，持久 intent 已无法阻止 cleanup Worker 接管。"""


class GuardedStorageUpload:
    """把不可取消的对象上传与请求生命周期、持久清理 intent 解耦。

    硬终止后只能依赖最后一次续租留下的保护窗口；若远端 PUT 在该窗口之后才提交，
    现有数据库无法证明其终态。这一极端不确定窗口需要存储供应商的幂等状态查询或
    持久上传回执才能彻底消除，当前代际 key 至少保证旧补偿不会伤及后续上传。
    """

    def __init__(
        self,
        *,
        registration: RegisteredStorageUploadIntent,
        session_factory: Callable[[], Session],
        storage,
        storage_key: str,
        content: bytes,
        mimetype: str,
        hold_seconds: float,
    ):
        self.registration = registration
        self.session_factory = session_factory
        self.storage = storage
        self.storage_key = storage_key
        self.content = content
        self.mimetype = mimetype
        self.hold_seconds = max(1.0, hold_seconds)
        self._terminal = asyncio.Event()
        self._stop_renewal = asyncio.Event()
        self._terminal_state: str | None = None
        self._fence_lost = False
        self._upload_result: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(
            self._run(),
            name=f"knowledge-storage-upload-{registration.intent_id}",
        )
        _ACTIVE_STORAGE_UPLOAD_LIFECYCLES.add(self._task)
        self._task.add_done_callback(_discard_storage_upload_lifecycle)

    async def wait_upload(self) -> str:
        """等待上传结果，但调用方取消时不能把 lifecycle 一并取消。"""
        return await asyncio.shield(self._upload_result)

    async def wait_finished(self) -> None:
        """正常终态下等待续租协程退出，避免把已完成任务遗留到 shutdown。"""
        await asyncio.shield(self._task)

    def mark_request_resolved(self) -> None:
        """请求事务已经提交或已完成冲突补偿，可停止续租。"""
        if self._terminal_state is None:
            self._terminal_state = "resolved"
            self._terminal.set()

    def detach_request(self) -> None:
        """请求已取消；真实上传结束后由独立 finalizer 收敛。"""
        if self._terminal_state is None:
            self._terminal_state = "detached"
            self._terminal.set()

    async def _run(self) -> None:
        renewal_task = asyncio.create_task(
            self._renew_loop(),
            name=f"knowledge-storage-upload-renew-{self.registration.intent_id}",
        )
        try:
            try:
                result = await self.storage.upload(self.storage_key, self.content, self.mimetype)
            except asyncio.CancelledError:
                if not self._upload_result.done():
                    self._upload_result.cancel()
                raise
            except Exception as exc:
                if not self._upload_result.done():
                    self._upload_result.set_exception(exc)
            else:
                # SDK 返回时用新事务重新取得持久 fence。数据库失联期间 intent 可能
                # 已过期且 cleanup Worker 已接管；此时绝不能继续提交文档记录。
                try:
                    fence_renewed = await asyncio.to_thread(self._renew_once)
                except Exception:
                    logger.exception(
                        "知识库上传返回后无法确认持久 fence: task_id=%s intent_id=%s",
                        self.registration.task_id,
                        self.registration.intent_id,
                    )
                    fence_renewed = False
                if self._fence_lost or not fence_renewed:
                    self._fence_lost = True
                    if not self._upload_result.done():
                        self._upload_result.set_exception(StorageUploadFenceLost("对象上传完成时持久清理 fence 已失效"))
                elif not self._upload_result.done():
                    self._upload_result.set_result(result)

            # SDK 返回并不代表请求侧数据库事务已经结束。正常请求必须继续持有 intent，
            # 直到 Service 明确发出提交或冲突补偿的终态信号。
            await self._terminal.wait()
            if self._terminal_state == "detached":
                await asyncio.to_thread(self._finalize_detached_upload)
        finally:
            self._stop_renewal.set()
            await renewal_task
            # detach 后没有调用方继续读取 Future；主动观察异常，避免事件循环警告。
            if (
                self._terminal_state == "detached"
                and self._upload_result.done()
                and not self._upload_result.cancelled()
            ):
                self._upload_result.exception()

    async def _renew_loop(self) -> None:
        interval_seconds = min(5.0, self.hold_seconds / 3)
        while True:
            try:
                await asyncio.wait_for(self._stop_renewal.wait(), timeout=interval_seconds)
                return
            except TimeoutError:
                pass
            try:
                renewed = await asyncio.to_thread(self._renew_once)
                if not renewed and not self._terminal.is_set():
                    self._fence_lost = True
                    logger.warning(
                        "知识库上传 intent 已失效，禁止提交文档并等待清理: task_id=%s intent_id=%s",
                        self.registration.task_id,
                        self.registration.intent_id,
                    )
            except Exception:
                # 数据库短暂不可用时继续尝试；若进程崩溃，续租自然停止，既有持久清理
                # Worker 会在 lease 到期后接管。
                logger.exception(
                    "知识库上传 intent 续租失败，将继续重试: task_id=%s intent_id=%s",
                    self.registration.task_id,
                    self.registration.intent_id,
                )

    def _renew_once(self) -> bool:
        with self.session_factory() as db:
            task = (
                db.query(KnowledgeStorageCleanupTask)
                .filter(KnowledgeStorageCleanupTask.id == self.registration.task_id)
                .populate_existing()
                .with_for_update()
                .first()
            )
            if task is None:
                db.rollback()
                return False
            intent = (
                db.query(KnowledgeStorageUploadIntent)
                .filter(
                    KnowledgeStorageUploadIntent.id == self.registration.intent_id,
                    KnowledgeStorageUploadIntent.cleanup_task_id == self.registration.task_id,
                )
                .populate_existing()
                .with_for_update()
                .first()
            )
            if intent is None:
                db.rollback()
                return False
            now = utc_now()
            guard_until = now + timedelta(seconds=self.hold_seconds)
            intent.expires_at = guard_until
            task.available_at = max(
                self._as_utc(task.available_at),
                self._as_utc(guard_until),
            )
            task.updated_at = now
            db.commit()
            return True

    def _finalize_detached_upload(self) -> None:
        """用新 session 精确复核代际 key，不复用已关闭的请求 session。"""
        try:
            with self.session_factory() as db:
                repo = StorageCleanupRepository(db)
                if repo.mark_document_committed(self.registration):
                    return
                repo.resolve_known_conflict(
                    self.registration,
                    cleanup_succeeded=False,
                )
        except Exception:
            # finalizer 失败时保留持久记录。停止续租后 intent 会到期，由 cleanup Worker
            # 再次按精确 backend + generation key 检查引用并删除孤立对象。
            logger.exception(
                "知识库取消上传 finalizer 失败，等待持久清理接管: task_id=%s intent_id=%s",
                self.registration.task_id,
                self.registration.intent_id,
            )

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def start_guarded_storage_upload(
    *,
    registration: RegisteredStorageUploadIntent,
    session_factory: Callable[[], Session],
    storage,
    storage_key: str,
    content: bytes,
    mimetype: str,
    hold_seconds: float,
) -> GuardedStorageUpload:
    return GuardedStorageUpload(
        registration=registration,
        session_factory=session_factory,
        storage=storage,
        storage_key=storage_key,
        content=content,
        mimetype=mimetype,
        hold_seconds=hold_seconds,
    )


def active_storage_upload_lifecycle_count() -> int:
    return len(_ACTIVE_STORAGE_UPLOAD_LIFECYCLES)


async def drain_storage_upload_lifecycles() -> None:
    """优雅关闭时等待所有真实 SDK 上传及其 finalizer 完成。"""
    while _ACTIVE_STORAGE_UPLOAD_LIFECYCLES:
        await asyncio.gather(*tuple(_ACTIVE_STORAGE_UPLOAD_LIFECYCLES), return_exceptions=True)


async def shutdown_storage_upload_lifecycles() -> None:
    """停止接收请求后安全 drain；不能取消仍在线程中的 SDK 调用。"""
    await drain_storage_upload_lifecycles()


def _discard_storage_upload_lifecycle(task: asyncio.Task[None]) -> None:
    _ACTIVE_STORAGE_UPLOAD_LIFECYCLES.discard(task)
    if task.cancelled():
        return
    try:
        error = task.exception()
    except asyncio.CancelledError:
        return
    if error is not None:
        logger.error(
            "知识库上传 lifecycle 异常退出: error_type=%s",
            type(error).__name__,
            exc_info=(type(error), error, error.__traceback__),
        )
