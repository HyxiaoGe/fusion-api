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
            storage_backend = task.storage_backend
            storage_key = task.storage_key
            try:
                storage = get_storage_for_backend(storage_backend)
                if not await storage.exists(storage_key):
                    deleted = False
                else:
                    # 删除阶段先持久化再发 RPC；即使 RPC 成功后的 completed 提交失败，
                    # 重领任务也能用 delete_started_at + absent 安全收敛。
                    locked_task = repo.mark_delete_started(
                        task,
                        claimed.lease_token,
                        lease_seconds=settings.KNOWLEDGE_WORKER_LEASE_SECONDS,
                    )
                    if locked_task is None:
                        return True
                    task = locked_task
                    if await storage.delete(storage_key):
                        deleted = True
                    elif not await storage.exists(storage_key):
                        deleted = True
                    else:
                        raise RuntimeError("storage object still exists after delete")
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
                    claimed.task.id,
                    storage_backend,
                    claimed.task.attempt_count,
                    type(exc).__name__,
                )
            else:
                if not deleted:
                    if repo.complete_absent_if_definitive(task, claimed.lease_token):
                        return True
                    retry_delay = min(
                        settings.KNOWLEDGE_WORKER_RETRY_BASE_SECONDS * (2 ** max(claimed.task.attempt_count - 1, 0)),
                        settings.KNOWLEDGE_WORKER_RETRY_MAX_SECONDS,
                    )
                    repo.fail_delete(
                        task,
                        claimed.lease_token,
                        error_code="KNOWLEDGE_STORAGE_DELETE_UNCERTAIN",
                        error_summary="对象当前不可见，等待迟到上传或精确引用收敛",
                        retry_delay_seconds=retry_delay,
                    )
                    return True
                repo.complete_delete(task, claimed.lease_token)
        return True
