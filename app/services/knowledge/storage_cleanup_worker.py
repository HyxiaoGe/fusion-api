from __future__ import annotations

import logging
import uuid
from typing import Callable

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.storage_cleanup_repository import StorageCleanupRepository
from app.services.storage import get_storage_for_backend

logger = logging.getLogger(__name__)


class KnowledgeStorageCleanupWorker:
    """处理上传与数据库提交脱节后遗留的对象。"""

    def __init__(self, session_factory: Callable[[], Session], *, worker_id: str | None = None):
        self.session_factory = session_factory
        self.worker_id = worker_id or f"knowledge-storage-cleanup-{uuid.uuid4()}"

    async def run_once(self) -> bool:
        with self.session_factory() as db:
            claimed = StorageCleanupRepository(db).claim_task(
                worker_id=self.worker_id,
                lease_seconds=settings.KNOWLEDGE_WORKER_LEASE_SECONDS,
                retry_base_seconds=settings.KNOWLEDGE_WORKER_RETRY_BASE_SECONDS,
                retry_max_seconds=settings.KNOWLEDGE_WORKER_RETRY_MAX_SECONDS,
            )
        if claimed is None:
            return False
        with self.session_factory() as db:
            repo = StorageCleanupRepository(db)
            decision, task = repo.prepare_delete(claimed.task.id, claimed.lease_token)
            if decision != "delete" or task is None:
                return True
            try:
                # MinIO/OSS 的 SDK 删除在线程中不可取消。事务行锁必须覆盖到真实返回，
                # 不能因 asyncio 超时提前释放同 key fencing。
                await self._delete_idempotently(task.storage_backend, task.storage_key)
            except Exception as exc:
                retry_delay = min(
                    settings.KNOWLEDGE_WORKER_RETRY_BASE_SECONDS * (2 ** max(claimed.task.attempt_count - 1, 0)),
                    settings.KNOWLEDGE_WORKER_RETRY_MAX_SECONDS,
                )
                repo.fail_delete(
                    task,
                    claimed.lease_token,
                    error_code="KNOWLEDGE_STORAGE_DELETE_FAILED",
                    error_summary="孤立对象删除失败",
                    retry_delay_seconds=retry_delay,
                )
                logger.warning(
                    "知识库孤立对象清理失败: task_id=%s backend=%s attempt=%s error_type=%s",
                    task.id,
                    task.storage_backend,
                    task.attempt_count,
                    type(exc).__name__,
                )
            else:
                repo.complete_delete(task, claimed.lease_token)
        return True

    @staticmethod
    async def _delete_idempotently(storage_backend: str, storage_key: str) -> None:
        storage = get_storage_for_backend(storage_backend)
        if not await storage.exists(storage_key):
            return
        if await storage.delete(storage_key):
            return
        if not await storage.exists(storage_key):
            return
        raise RuntimeError("storage object still exists after delete")
