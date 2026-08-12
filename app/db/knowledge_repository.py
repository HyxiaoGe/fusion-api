from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Iterable, Sequence

from sqlalchemy import and_, exists, func, insert, not_, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, aliased

from app.db.models import (
    KnowledgeBase,
    KnowledgeChunkManifest,
    KnowledgeDocument,
    KnowledgeIndexTask,
    KnowledgeIndexVersion,
    User,
)
from app.db.storage_cleanup_repository import (
    RegisteredStorageUploadIntent,
    StorageCleanupRepository,
    StorageUploadFenceConflict,
)
from app.utils.time import utc_now


@dataclass(frozen=True)
class ClaimedKnowledgeTask:
    task: KnowledgeIndexTask
    lease_token: str


@dataclass(frozen=True)
class ReadyKnowledgeDocument:
    document: KnowledgeDocument
    active_version: KnowledgeIndexVersion


class KnowledgeBaseWriteConflict(RuntimeError):
    """文档提交前知识库已进入不可写状态。"""

    def __init__(self, message: str, *, cleanup_storage: bool, duplicate: bool = False):
        self.cleanup_storage = cleanup_storage
        self.duplicate = duplicate
        super().__init__(message)


class KnowledgeBaseLimitExceeded(RuntimeError):
    """用户级行锁保护下确认知识库数量已达上限。"""


class KnowledgeRepository:
    """知识库 PostgreSQL 事实源与持久任务租约操作。"""

    def __init__(self, db: Session):
        self.db = db

    def create_knowledge_base(
        self,
        knowledge_base: KnowledgeBase,
        *,
        max_bases: int | None = None,
    ) -> KnowledgeBase:
        if max_bases is not None:
            user = self.db.query(User).filter(User.id == knowledge_base.user_id).with_for_update().first()
            if user is None:
                raise KnowledgeBaseWriteConflict("用户不存在", cleanup_storage=False)
            if self.count_knowledge_bases(knowledge_base.user_id) >= max_bases:
                self.db.rollback()
                raise KnowledgeBaseLimitExceeded
        self.db.add(knowledge_base)
        try:
            self.db.commit()
            self.db.refresh(knowledge_base)
            return knowledge_base
        except Exception:
            self.db.rollback()
            raise

    def list_knowledge_bases(
        self,
        *,
        user_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[KnowledgeBase], int]:
        query = self.db.query(KnowledgeBase).filter(
            KnowledgeBase.user_id == user_id,
            KnowledgeBase.status != "deleted",
        )
        total = query.count()
        rows = (
            query.order_by(KnowledgeBase.updated_at.desc(), KnowledgeBase.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return rows, total

    def count_knowledge_bases(self, user_id: str) -> int:
        return (
            self.db.query(KnowledgeBase)
            .filter(KnowledgeBase.user_id == user_id, KnowledgeBase.status != "deleted")
            .count()
        )

    def get_knowledge_bases_by_ids(self, user_id: str, knowledge_base_ids: Sequence[str]) -> list[KnowledgeBase]:
        if not knowledge_base_ids:
            return []
        return (
            self.db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.id.in_(knowledge_base_ids),
                KnowledgeBase.status != "deleted",
            )
            .all()
        )

    def get_knowledge_base(
        self,
        knowledge_base_id: str,
        user_id: str,
        *,
        include_deleted: bool = False,
        lock: bool = False,
    ) -> KnowledgeBase | None:
        query = self.db.query(KnowledgeBase).filter(
            KnowledgeBase.id == knowledge_base_id,
            KnowledgeBase.user_id == user_id,
        )
        if not include_deleted:
            query = query.filter(KnowledgeBase.status != "deleted")
        if lock:
            query = query.with_for_update()
        return query.first()

    def update_knowledge_base(self, row: KnowledgeBase, **updates: object) -> KnowledgeBase:
        for key, value in updates.items():
            setattr(row, key, value)
        row.updated_at = utc_now()
        try:
            self.db.commit()
            self.db.refresh(row)
            return row
        except Exception:
            self.db.rollback()
            raise

    def document_stats(self, knowledge_base_id: str) -> dict[str, int]:
        rows = (
            self.db.query(KnowledgeDocument.status, func.count(KnowledgeDocument.id))
            .filter(
                KnowledgeDocument.knowledge_base_id == knowledge_base_id,
                KnowledgeDocument.status != "deleted",
            )
            .group_by(KnowledgeDocument.status)
            .all()
        )
        by_status = {str(status): int(count) for status, count in rows}
        return {
            "total": sum(by_status.values()),
            "ready": by_status.get("ready", 0),
            "failed": by_status.get("failed", 0),
            "processing": sum(
                by_status.get(status, 0)
                for status in ("queued", "parsing", "chunking", "embedding", "writing", "deleting")
            ),
        }

    def get_duplicate_document(self, knowledge_base_id: str, checksum_sha256: str) -> KnowledgeDocument | None:
        return (
            self.db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.knowledge_base_id == knowledge_base_id,
                KnowledgeDocument.dedupe_key == checksum_sha256,
                KnowledgeDocument.status != "deleted",
            )
            .first()
        )

    def count_documents(self, knowledge_base_id: str) -> int:
        return (
            self.db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.knowledge_base_id == knowledge_base_id,
                KnowledgeDocument.status != "deleted",
            )
            .count()
        )

    def _has_active_checksum_reference(self, document: KnowledgeDocument) -> bool:
        return (
            self.db.query(KnowledgeDocument.id)
            .filter(
                KnowledgeDocument.knowledge_base_id == document.knowledge_base_id,
                KnowledgeDocument.status != "deleted",
                KnowledgeDocument.checksum_sha256 == document.checksum_sha256,
                KnowledgeDocument.storage_backend == document.storage_backend,
            )
            .first()
            is not None
        )

    def create_document_with_task(
        self,
        document: KnowledgeDocument,
        version: KnowledgeIndexVersion,
        task: KnowledgeIndexTask,
        *,
        max_documents: int | None = None,
        upload_registration: RegisteredStorageUploadIntent | None = None,
    ) -> tuple[KnowledgeDocument, KnowledgeIndexVersion, KnowledgeIndexTask]:
        knowledge_base = (
            self.db.query(KnowledgeBase)
            .filter(
                KnowledgeBase.id == document.knowledge_base_id,
                KnowledgeBase.user_id == document.user_id,
            )
            .populate_existing()
            .with_for_update()
            .first()
        )
        if knowledge_base is None or knowledge_base.status != "active":
            self.db.rollback()
            raise KnowledgeBaseWriteConflict("知识库已进入删除流程", cleanup_storage=True)
        cleanup_repo = StorageCleanupRepository(self.db)
        cleanup_task = (
            cleanup_repo.lock_succeeded_upload_for_document(upload_registration)
            if upload_registration is not None
            else None
        )
        if cleanup_task is not None and (
            cleanup_task.storage_backend != document.storage_backend or cleanup_task.storage_key != document.storage_key
        ):
            self.db.rollback()
            raise StorageUploadFenceConflict("对象上传 fence 与文档存储代际不一致")
        if max_documents is not None:
            active_count = (
                self.db.query(KnowledgeDocument)
                .filter(
                    KnowledgeDocument.knowledge_base_id == document.knowledge_base_id,
                    KnowledgeDocument.status != "deleted",
                )
                .count()
            )
            if active_count >= max_documents:
                duplicate = self._has_active_checksum_reference(document)
                self.db.rollback()
                raise KnowledgeBaseWriteConflict(
                    "相同内容的活动文档已存在于该知识库" if duplicate else "已达到每个知识库的文档数量上限",
                    # 每次上传使用独立 document_id 代际 key，重复内容也可安全清理本次对象。
                    cleanup_storage=True,
                    duplicate=duplicate,
                )
        self.db.add_all([document, version, task])
        try:
            self.db.flush()
            if upload_registration is not None and cleanup_task is not None:
                cleanup_repo.complete_document_write_locked(upload_registration, cleanup_task)
            self.db.commit()
            self.db.refresh(document)
            self.db.refresh(version)
            self.db.refresh(task)
            return document, version, task
        except Exception:
            self.db.rollback()
            raise

    def list_documents(
        self,
        *,
        knowledge_base_id: str,
        user_id: str,
        page: int,
        page_size: int,
    ) -> tuple[list[KnowledgeDocument], int]:
        query = self.db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == knowledge_base_id,
            KnowledgeDocument.user_id == user_id,
            KnowledgeDocument.status != "deleted",
        )
        total = query.count()
        rows = (
            query.order_by(KnowledgeDocument.updated_at.desc(), KnowledgeDocument.id.desc())
            .offset((page - 1) * page_size)
            .limit(page_size)
            .all()
        )
        return rows, total

    def get_document(
        self,
        document_id: str,
        user_id: str,
        *,
        include_deleted: bool = False,
        lock: bool = False,
    ) -> KnowledgeDocument | None:
        query = self.db.query(KnowledgeDocument).filter(
            KnowledgeDocument.id == document_id,
            KnowledgeDocument.user_id == user_id,
        )
        if not include_deleted:
            query = query.filter(KnowledgeDocument.status != "deleted")
        if lock:
            query = query.with_for_update()
        return query.first()

    def get_documents_by_ids(self, user_id: str, document_ids: Sequence[str]) -> list[KnowledgeDocument]:
        if not document_ids:
            return []
        return (
            self.db.query(KnowledgeDocument)
            .filter(
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.id.in_(document_ids),
            )
            .all()
        )

    def get_ready_documents(self, user_id: str, knowledge_base_ids: Sequence[str]) -> list[ReadyKnowledgeDocument]:
        if not knowledge_base_ids:
            return []
        rows = (
            self.db.query(KnowledgeDocument, KnowledgeIndexVersion)
            .join(KnowledgeBase, KnowledgeBase.id == KnowledgeDocument.knowledge_base_id)
            .join(
                KnowledgeIndexVersion,
                and_(
                    KnowledgeIndexVersion.id == KnowledgeDocument.active_index_version,
                    KnowledgeIndexVersion.document_id == KnowledgeDocument.id,
                    KnowledgeIndexVersion.knowledge_base_id == KnowledgeDocument.knowledge_base_id,
                    KnowledgeIndexVersion.user_id == KnowledgeDocument.user_id,
                ),
            )
            .filter(
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.knowledge_base_id.in_(knowledge_base_ids),
                KnowledgeDocument.status == "ready",
                KnowledgeDocument.active_index_version.isnot(None),
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.status == "active",
                KnowledgeIndexVersion.user_id == user_id,
                KnowledgeIndexVersion.status == "active",
                KnowledgeIndexVersion.deleted_at.is_(None),
            )
            .order_by(KnowledgeDocument.id.asc())
            .all()
        )
        return [ReadyKnowledgeDocument(document=document, active_version=version) for document, version in rows]

    def revalidate_retrieval_documents(
        self,
        *,
        user_id: str,
        knowledge_base_ids: Sequence[str],
        document_ids: Sequence[str],
        index_versions: Sequence[str],
    ) -> list[KnowledgeDocument]:
        """Milvus 返回后重新读取 PostgreSQL，封住删除/版本切换竞态。"""
        if not document_ids or not index_versions:
            return []
        return (
            self.db.query(KnowledgeDocument)
            .join(KnowledgeBase, KnowledgeBase.id == KnowledgeDocument.knowledge_base_id)
            .filter(
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.knowledge_base_id.in_(knowledge_base_ids),
                KnowledgeDocument.id.in_(document_ids),
                KnowledgeDocument.status == "ready",
                KnowledgeDocument.active_index_version.in_(index_versions),
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.status == "active",
            )
            .all()
        )

    def get_authorized_chunks(
        self,
        *,
        user_id: str,
        knowledge_base_ids: Sequence[str],
        chunk_ids: Sequence[str],
    ) -> list[KnowledgeChunkManifest]:
        if not chunk_ids:
            return []
        return (
            self.db.query(KnowledgeChunkManifest)
            .join(KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunkManifest.document_id)
            .join(KnowledgeBase, KnowledgeBase.id == KnowledgeChunkManifest.knowledge_base_id)
            .filter(
                KnowledgeChunkManifest.user_id == user_id,
                KnowledgeChunkManifest.knowledge_base_id.in_(knowledge_base_ids),
                KnowledgeChunkManifest.chunk_id.in_(chunk_ids),
                KnowledgeDocument.user_id == user_id,
                KnowledgeDocument.status == "ready",
                KnowledgeDocument.active_index_version == KnowledgeChunkManifest.index_version,
                KnowledgeBase.user_id == user_id,
                KnowledgeBase.status == "active",
            )
            .all()
        )

    def replace_chunk_manifest(
        self,
        *,
        task_id: str,
        lease_token: str,
        version: KnowledgeIndexVersion,
        filename: str,
        chunks: Sequence[object],
        batch_size: int = 500,
    ) -> bool:
        if batch_size < 1:
            raise ValueError("manifest 批量写入大小必须大于零")
        task = self._lock_task(task_id)
        if not self._lease_matches(task, lease_token, utc_now()):
            self.db.rollback()
            return False
        try:
            self.db.query(KnowledgeChunkManifest).filter(KnowledgeChunkManifest.index_version == version.id).delete(
                synchronize_session=False
            )
            manifest_table = KnowledgeChunkManifest.__table__
            for offset in range(0, len(chunks), batch_size):
                batch = chunks[offset : offset + batch_size]
                rows = [
                    {
                        "chunk_id": chunk.chunk_id,
                        "knowledge_base_id": version.knowledge_base_id,
                        "document_id": version.document_id,
                        "user_id": version.user_id,
                        "index_version": version.id,
                        "ordinal": chunk.ordinal,
                        "text": chunk.text,
                        "text_sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                        "filename": filename,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "page": chunk.page,
                        "section": chunk.section,
                    }
                    for chunk in batch
                ]
                self.db.execute(insert(manifest_table), rows)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def enqueue_document_delete(self, document: KnowledgeDocument, *, max_attempts: int) -> KnowledgeIndexTask:
        document.status = "deleting"
        document.updated_at = utc_now()
        task = KnowledgeIndexTask(
            knowledge_base_id=document.knowledge_base_id,
            document_id=document.id,
            user_id=document.user_id,
            task_type="delete_document",
            status="pending",
            phase="queued",
            max_attempts=max_attempts,
        )
        self._cancel_non_delete_document_tasks(document.id)
        self.db.add(task)
        try:
            self.db.commit()
            self.db.refresh(task)
            return task
        except Exception:
            self.db.rollback()
            raise

    def enqueue_knowledge_base_delete(
        self,
        knowledge_base: KnowledgeBase,
        *,
        max_attempts: int,
    ) -> KnowledgeIndexTask:
        knowledge_base.status = "deleting"
        knowledge_base.updated_at = utc_now()
        self.db.query(KnowledgeDocument).filter(
            KnowledgeDocument.knowledge_base_id == knowledge_base.id,
            KnowledgeDocument.status.notin_(("deleted", "deleting")),
        ).update(
            {"status": "deleting", "updated_at": utc_now()},
            synchronize_session=False,
        )
        self.db.query(KnowledgeIndexTask).filter(
            KnowledgeIndexTask.knowledge_base_id == knowledge_base.id,
            KnowledgeIndexTask.task_type != "delete_knowledge_base",
            KnowledgeIndexTask.status.in_(("pending", "retry")),
        ).update(
            {"status": "cancelled", "completed_at": utc_now(), "updated_at": utc_now()},
            synchronize_session=False,
        )
        task = KnowledgeIndexTask(
            knowledge_base_id=knowledge_base.id,
            user_id=knowledge_base.user_id,
            task_type="delete_knowledge_base",
            status="pending",
            phase="queued",
            max_attempts=max_attempts,
        )
        self.db.add(task)
        try:
            self.db.commit()
            self.db.refresh(task)
            return task
        except Exception:
            self.db.rollback()
            raise

    def retry_document(
        self,
        document: KnowledgeDocument,
        version: KnowledgeIndexVersion,
        task: KnowledgeIndexTask,
    ) -> KnowledgeIndexTask:
        previous_version = self.get_index_version(document.desired_index_version)
        cleanup_task = None
        if (
            previous_version is not None
            and previous_version.id != document.active_index_version
            and previous_version.status in {"building", "failed"}
        ):
            previous_version.status = "deleting"
            previous_version.error_code = document.error_code
            previous_version.error_summary = document.error_summary
            cleanup_task = KnowledgeIndexTask(
                knowledge_base_id=previous_version.knowledge_base_id,
                document_id=previous_version.document_id,
                user_id=previous_version.user_id,
                task_type="delete_index_version",
                index_version=previous_version.id,
                status="pending",
                phase="queued",
                max_attempts=task.max_attempts,
            )
        self.db.query(KnowledgeIndexTask).filter(
            KnowledgeIndexTask.document_id == document.id,
            KnowledgeIndexTask.task_type == "index_document",
            KnowledgeIndexTask.status.in_(("pending", "retry")),
        ).update(
            {"status": "cancelled", "completed_at": utc_now(), "updated_at": utc_now()},
            synchronize_session=False,
        )
        document.desired_index_version = version.id
        if document.active_index_version is None:
            document.parser_version = version.parser_version
            document.chunker_version = version.chunker_version
            document.embedding_provider = version.embedding_provider
            document.embedding_model = version.embedding_model
            document.embedding_revision = version.embedding_revision
            document.embedding_dimension = version.embedding_dimension
            document.distance_metric = version.distance_metric
        document.status = "ready" if document.active_index_version else "queued"
        document.error_code = None
        document.error_summary = None
        document.updated_at = utc_now()
        self.db.add_all([row for row in (version, task, cleanup_task) if row is not None])
        try:
            self.db.commit()
            self.db.refresh(task)
            return task
        except Exception:
            self.db.rollback()
            raise

    def claim_task(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        external_write_grace_seconds: int = 0,
        now: datetime | None = None,
    ) -> ClaimedKnowledgeTask | None:
        now = now or utc_now()
        self._reconcile_failed_index_cleanup_tasks(
            now,
            base_delay_seconds=external_write_grace_seconds,
        )
        self._fail_exhausted_expired_tasks(now, external_write_grace_seconds=external_write_grace_seconds)
        index_reclaim_before = now - timedelta(seconds=max(0, external_write_grace_seconds))
        claimable = or_(
            and_(
                KnowledgeIndexTask.status.in_(("pending", "retry")),
                KnowledgeIndexTask.available_at <= now,
            ),
            and_(
                KnowledgeIndexTask.status == "running",
                or_(
                    and_(
                        KnowledgeIndexTask.task_type == "index_document",
                        KnowledgeIndexTask.lease_expires_at < index_reclaim_before,
                    ),
                    and_(
                        KnowledgeIndexTask.task_type != "index_document",
                        KnowledgeIndexTask.lease_expires_at < now,
                    ),
                ),
            ),
        )
        running_index_task = aliased(KnowledgeIndexTask)
        blocked_by_running_index = or_(
            and_(
                KnowledgeIndexTask.task_type == "delete_document",
                exists().where(
                    running_index_task.document_id == KnowledgeIndexTask.document_id,
                    running_index_task.task_type == "index_document",
                    running_index_task.status == "running",
                ),
            ),
            and_(
                KnowledgeIndexTask.task_type == "delete_knowledge_base",
                exists().where(
                    running_index_task.knowledge_base_id == KnowledgeIndexTask.knowledge_base_id,
                    running_index_task.task_type == "index_document",
                    running_index_task.status == "running",
                ),
            ),
            and_(
                KnowledgeIndexTask.task_type == "delete_index_version",
                exists().where(
                    running_index_task.index_version == KnowledgeIndexTask.index_version,
                    running_index_task.task_type == "index_document",
                    # 即使租约已过期也保持串行；索引任务在 grace 后先重领并收敛，
                    # 避免清理与旧不可取消的 Milvus upsert 或重领 Worker 并发。
                    running_index_task.status == "running",
                ),
            ),
        )
        running_delete_task = aliased(KnowledgeIndexTask)
        blocked_index_by_delete = and_(
            KnowledgeIndexTask.task_type == "index_document",
            exists().where(
                running_delete_task.knowledge_base_id == KnowledgeIndexTask.knowledge_base_id,
                running_delete_task.status == "running",
                or_(
                    running_delete_task.task_type == "delete_knowledge_base",
                    and_(
                        running_delete_task.task_type == "delete_document",
                        running_delete_task.document_id == KnowledgeIndexTask.document_id,
                    ),
                ),
            ),
        )
        query = (
            self.db.query(KnowledgeIndexTask)
            .filter(
                claimable,
                KnowledgeIndexTask.attempt_count < KnowledgeIndexTask.max_attempts,
                not_(or_(blocked_by_running_index, blocked_index_by_delete)),
            )
            .order_by(KnowledgeIndexTask.available_at.asc(), KnowledgeIndexTask.created_at.asc())
        )
        if self.db.get_bind().dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        else:
            query = query.with_for_update()
        task = query.first()
        if task is None:
            self.db.commit()
            return None
        lease_token = secrets.token_urlsafe(32)
        task.status = "running"
        task.phase = "queued"
        task.attempt_count += 1
        task.lease_owner = worker_id
        task.lease_token_hash = self._hash_token(lease_token)
        task.lease_expires_at = now + timedelta(seconds=lease_seconds)
        task.heartbeat_at = now
        task.error_code = None
        task.error_summary = None
        task.updated_at = now
        self.db.commit()
        self.db.refresh(task)
        return ClaimedKnowledgeTask(task=task, lease_token=lease_token)

    def index_task_write_state(
        self,
        *,
        task_id: str,
        lease_token: str,
        document_id: str,
        index_version: str,
    ) -> str:
        """确认索引外部写入仍被当前租约和文档期望版本授权。"""
        task = self._lock_task(task_id)
        if (
            not self._lease_matches(task, lease_token, utc_now())
            or task is None
            or task.task_type != "index_document"
            or task.document_id != document_id
            or task.index_version != index_version
        ):
            self.db.rollback()
            return "lease_lost"
        document = (
            self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).with_for_update().first()
        )
        if (
            document is None
            or document.desired_index_version != index_version
            or document.status in {"deleting", "deleted"}
        ):
            self.db.rollback()
            return "stale"
        self.db.rollback()
        return "allowed"

    def renew_lease(
        self,
        task_id: str,
        lease_token: str,
        *,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> bool:
        now = now or utc_now()
        task = self._lock_task(task_id)
        if not self._lease_matches(task, lease_token, now):
            self.db.rollback()
            return False
        task.heartbeat_at = now
        task.lease_expires_at = now + timedelta(seconds=lease_seconds)
        task.updated_at = now
        self.db.commit()
        return True

    def update_task_phase(self, task_id: str, lease_token: str, phase: str) -> bool:
        task = self._lock_task(task_id)
        if not self._lease_matches(task, lease_token, utc_now()):
            self.db.rollback()
            return False
        task.phase = phase
        task.updated_at = utc_now()
        if task.document_id:
            document = self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id == task.document_id).first()
            if (
                document is not None
                and task.task_type == "index_document"
                and document.desired_index_version == task.index_version
                and document.status not in {"deleting", "deleted"}
                and document.active_index_version is None
            ):
                document.status = phase
                document.updated_at = utc_now()
        self.db.commit()
        return True

    def complete_task(self, task_id: str, lease_token: str) -> bool:
        task = self._lock_task(task_id)
        if not self._lease_matches(task, lease_token, utc_now()):
            self.db.rollback()
            return False
        now = utc_now()
        task.status = "completed"
        task.phase = "completed"
        task.completed_at = now
        task.updated_at = now
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.heartbeat_at = None
        self.db.commit()
        return True

    def fail_task(
        self,
        task_id: str,
        lease_token: str,
        *,
        error_code: str,
        error_summary: str,
        retryable: bool,
        retry_delay_seconds: int,
        external_write_grace_seconds: int = 0,
    ) -> bool:
        task = self._lock_task(task_id)
        if not self._lease_matches(task, lease_token, utc_now()):
            self.db.rollback()
            return False
        now = utc_now()
        will_retry = retryable and task.attempt_count < task.max_attempts
        task.status = "retry" if will_retry else "failed"
        task.available_at = now + timedelta(seconds=retry_delay_seconds) if will_retry else now
        task.error_code = error_code[:120]
        task.error_summary = error_summary[:500]
        task.completed_at = None if will_retry else now
        task.updated_at = now
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.heartbeat_at = None
        if task.document_id and task.task_type == "index_document":
            document = self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id == task.document_id).first()
            if (
                document is not None
                and document.desired_index_version == task.index_version
                and document.status not in {"deleting", "deleted"}
                and document.active_index_version is None
            ):
                document.status = "queued" if will_retry else "failed"
                document.error_code = task.error_code
                document.error_summary = task.error_summary
                document.attempt_count = task.attempt_count
                document.updated_at = now
                if not will_retry and task.index_version:
                    version = self.get_index_version(task.index_version)
                    if version is not None and version.status == "building":
                        version.status = "failed"
                        version.error_code = task.error_code
                        version.error_summary = task.error_summary
            if not will_retry:
                self._schedule_final_index_cleanup(
                    task,
                    now,
                    external_write_grace_seconds=external_write_grace_seconds,
                )
        self.db.commit()
        return True

    def finalize_document_index(
        self,
        *,
        task_id: str,
        lease_token: str,
        document_id: str,
        index_version: str,
        chunk_count: int,
        cleanup_max_attempts: int,
    ) -> str:
        """原子激活版本、登记旧版本清理并完成原任务。"""
        task = self._lock_task(task_id)
        if not self._lease_matches(task, lease_token, utc_now()):
            self.db.rollback()
            return "lease_lost"
        document = (
            self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).with_for_update().first()
        )
        if (
            document is None
            or document.desired_index_version != index_version
            or document.status in {"deleting", "deleted"}
        ):
            self.db.rollback()
            return "stale"
        version = (
            self.db.query(KnowledgeIndexVersion)
            .filter(
                KnowledgeIndexVersion.id == index_version,
                KnowledgeIndexVersion.document_id == document_id,
                KnowledgeIndexVersion.status == "building",
            )
            .with_for_update()
            .first()
        )
        if version is None:
            self.db.rollback()
            return "stale"
        previous_version = None
        if document.active_index_version and document.active_index_version != index_version:
            previous_version = (
                self.db.query(KnowledgeIndexVersion)
                .filter(KnowledgeIndexVersion.id == document.active_index_version)
                .with_for_update()
                .first()
            )
        now = utc_now()
        document.active_index_version = index_version
        document.chunk_count = chunk_count
        document.parser_version = version.parser_version
        document.chunker_version = version.chunker_version
        document.embedding_provider = version.embedding_provider
        document.embedding_model = version.embedding_model
        document.embedding_revision = version.embedding_revision
        document.embedding_dimension = version.embedding_dimension
        document.distance_metric = version.distance_metric
        document.attempt_count = task.attempt_count
        document.status = "ready"
        document.error_code = None
        document.error_summary = None
        document.ready_at = now
        document.updated_at = now
        version.status = "active"
        version.chunk_count = chunk_count
        version.activated_at = now
        if previous_version is not None:
            previous_version.status = "deleting"
            self.db.add(
                KnowledgeIndexTask(
                    knowledge_base_id=previous_version.knowledge_base_id,
                    document_id=previous_version.document_id,
                    user_id=previous_version.user_id,
                    task_type="delete_index_version",
                    index_version=previous_version.id,
                    status="pending",
                    phase="queued",
                    max_attempts=cleanup_max_attempts,
                )
            )
        task.status = "completed"
        task.phase = "completed"
        task.completed_at = now
        task.updated_at = now
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.heartbeat_at = None
        self.db.commit()
        return "activated"

    def get_index_version(self, version_id: str) -> KnowledgeIndexVersion | None:
        return self.db.query(KnowledgeIndexVersion).filter(KnowledgeIndexVersion.id == version_id).first()

    def get_task(self, task_id: str, user_id: str) -> KnowledgeIndexTask | None:
        return (
            self.db.query(KnowledgeIndexTask)
            .filter(KnowledgeIndexTask.id == task_id, KnowledgeIndexTask.user_id == user_id)
            .first()
        )

    def mark_index_version_deleted(self, version_id: str, *, task_id: str, lease_token: str) -> bool:
        task = self._lock_task(task_id)
        if (
            not self._lease_matches(task, lease_token, utc_now())
            or task is None
            or task.task_type not in {"index_document", "delete_index_version"}
            or task.index_version != version_id
        ):
            self.db.rollback()
            return False
        version = (
            self.db.query(KnowledgeIndexVersion)
            .filter(KnowledgeIndexVersion.id == version_id)
            .with_for_update()
            .first()
        )
        if version is None:
            self.db.rollback()
            return False
        version.status = "deleted"
        version.deleted_at = utc_now()
        self.db.query(KnowledgeChunkManifest).filter(KnowledgeChunkManifest.index_version == version_id).delete(
            synchronize_session=False
        )
        self.db.commit()
        return True

    def mark_document_deleted(self, document_id: str, *, task_id: str, lease_token: str) -> bool:
        task = self._lock_task(task_id)
        if (
            not self._lease_matches(task, lease_token, utc_now())
            or task is None
            or task.task_type != "delete_document"
            or task.document_id != document_id
        ):
            self.db.rollback()
            return False
        document = (
            self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id == document_id).with_for_update().first()
        )
        if document is None:
            self.db.rollback()
            return False
        now = utc_now()
        document.status = "deleted"
        document.dedupe_key = self._soft_delete_tombstone(
            document.checksum_sha256,
            document.id,
            max_length=120,
        )
        document.deleted_at = now
        document.updated_at = now
        document.active_index_version = None
        document.chunk_count = 0
        self.db.query(KnowledgeIndexVersion).filter(
            KnowledgeIndexVersion.document_id == document.id,
            KnowledgeIndexVersion.status != "deleted",
        ).update(
            {"status": "deleted", "deleted_at": now},
            synchronize_session=False,
        )
        self.db.query(KnowledgeChunkManifest).filter(KnowledgeChunkManifest.document_id == document.id).delete(
            synchronize_session=False
        )
        self.db.commit()
        return True

    def mark_knowledge_base_deleted(self, knowledge_base_id: str, *, task_id: str, lease_token: str) -> bool:
        task = self._lock_task(task_id)
        if (
            not self._lease_matches(task, lease_token, utc_now())
            or task is None
            or task.task_type != "delete_knowledge_base"
            or task.knowledge_base_id != knowledge_base_id
        ):
            self.db.rollback()
            return False
        row = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == knowledge_base_id).with_for_update().first()
        if row is None:
            self.db.rollback()
            return False
        now = utc_now()
        row.status = "deleted"
        row.name_normalized = self._soft_delete_tombstone(
            row.name_normalized,
            row.id,
            max_length=200,
        )
        row.deleted_at = now
        row.updated_at = now
        self.db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == row.id).update(
            {
                "status": "deleted",
                "deleted_at": now,
                "updated_at": now,
                "active_index_version": None,
                "chunk_count": 0,
            },
            synchronize_session=False,
        )
        self.db.query(KnowledgeIndexVersion).filter(
            KnowledgeIndexVersion.knowledge_base_id == row.id,
            KnowledgeIndexVersion.status != "deleted",
        ).update(
            {"status": "deleted", "deleted_at": now},
            synchronize_session=False,
        )
        self.db.query(KnowledgeChunkManifest).filter(KnowledgeChunkManifest.knowledge_base_id == row.id).delete(
            synchronize_session=False
        )
        self.db.query(KnowledgeIndexTask).filter(
            KnowledgeIndexTask.knowledge_base_id == row.id,
            KnowledgeIndexTask.status.in_(("pending", "retry")),
        ).update(
            {"status": "cancelled", "completed_at": now, "updated_at": now},
            synchronize_session=False,
        )
        self.db.commit()
        return True

    def documents_for_cleanup(self, knowledge_base_id: str) -> list[KnowledgeDocument]:
        return self.db.query(KnowledgeDocument).filter(KnowledgeDocument.knowledge_base_id == knowledge_base_id).all()

    def _cancel_non_delete_document_tasks(self, document_id: str) -> None:
        now = utc_now()
        self.db.query(KnowledgeIndexTask).filter(
            KnowledgeIndexTask.document_id == document_id,
            KnowledgeIndexTask.task_type == "index_document",
            KnowledgeIndexTask.status.in_(("pending", "retry")),
        ).update(
            {"status": "cancelled", "completed_at": now, "updated_at": now},
            synchronize_session=False,
        )

    @staticmethod
    def _soft_delete_tombstone(value: str, row_id: str, *, max_length: int) -> str:
        suffix = f"#deleted#{hashlib.sha256(row_id.encode('utf-8')).hexdigest()}"
        prefix_length = max(0, max_length - len(suffix))
        return f"{value[:prefix_length]}{suffix}"[-max_length:]

    def _expired_exhausted_tasks_query(self, now: datetime):
        query = (
            self.db.query(KnowledgeIndexTask)
            .filter(
                KnowledgeIndexTask.status == "running",
                KnowledgeIndexTask.lease_expires_at < now,
                KnowledgeIndexTask.attempt_count >= KnowledgeIndexTask.max_attempts,
            )
            .populate_existing()
        )
        if self.db.get_bind().dialect.name == "postgresql":
            return query.with_for_update(skip_locked=True)
        return query.with_for_update()

    def _failed_index_cleanup_tasks_query(self):
        query = (
            self.db.query(KnowledgeIndexTask)
            .filter(
                KnowledgeIndexTask.task_type == "delete_index_version",
                KnowledgeIndexTask.status == "failed",
                KnowledgeIndexTask.index_version.is_not(None),
            )
            .order_by(
                KnowledgeIndexTask.index_version.asc(),
                KnowledgeIndexTask.updated_at.asc(),
                KnowledgeIndexTask.id.asc(),
            )
            .populate_existing()
            .limit(100)
        )
        if self.db.get_bind().dialect.name == "postgresql":
            return query.with_for_update(skip_locked=True)
        return query.with_for_update()

    def _reconcile_failed_index_cleanup_tasks(self, now: datetime, *, base_delay_seconds: int) -> None:
        rows: Iterable[KnowledgeIndexTask] = self._failed_index_cleanup_tasks_query().all()
        for task in rows:
            if task.status != "failed" or task.task_type != "delete_index_version" or not task.index_version:
                continue
            version = (
                self.db.query(KnowledgeIndexVersion)
                .filter(
                    KnowledgeIndexVersion.id == task.index_version,
                    KnowledgeIndexVersion.status == "deleting",
                )
                .populate_existing()
                .with_for_update()
                .first()
            )
            if version is None:
                continue
            active_cleanup = (
                self.db.query(KnowledgeIndexTask.id)
                .filter(
                    KnowledgeIndexTask.id != task.id,
                    KnowledgeIndexTask.task_type == "delete_index_version",
                    KnowledgeIndexTask.index_version == task.index_version,
                    KnowledgeIndexTask.status.in_(("pending", "running", "retry")),
                )
                .first()
            )
            if active_cleanup is not None:
                continue
            task.status = "retry"
            task.phase = "queued"
            # Reconciler 每轮只授予一次额外尝试，保留累计尝试次数且避免失败时热循环。
            task.max_attempts = max(task.max_attempts, task.attempt_count + 1)
            task.available_at = now + timedelta(
                seconds=self._cleanup_reconcile_delay_seconds(
                    attempt_count=task.attempt_count,
                    base_delay_seconds=base_delay_seconds,
                )
            )
            task.completed_at = None
            task.updated_at = now
            task.lease_owner = None
            task.lease_token_hash = None
            task.lease_expires_at = None
            task.heartbeat_at = None

    @staticmethod
    def _cleanup_reconcile_delay_seconds(*, attempt_count: int, base_delay_seconds: int) -> int:
        base = max(1, base_delay_seconds)
        exponent = min(max(attempt_count - 1, 0), 8)
        return min(3600, base * (2**exponent))

    def _fail_exhausted_expired_tasks(self, now: datetime, *, external_write_grace_seconds: int) -> None:
        rows: Iterable[KnowledgeIndexTask] = self._expired_exhausted_tasks_query(now).all()
        for task in rows:
            grace_boundary = (
                self._as_utc(task.lease_expires_at) + timedelta(seconds=max(0, external_write_grace_seconds))
                if task.lease_expires_at is not None
                else None
            )
            if (
                task.status != "running"
                or task.attempt_count < task.max_attempts
                or task.lease_expires_at is None
                or self._as_utc(task.lease_expires_at) >= self._as_utc(now)
                or (
                    task.task_type == "index_document"
                    and grace_boundary is not None
                    and grace_boundary >= self._as_utc(now)
                )
            ):
                continue
            task.status = "failed"
            task.error_code = "KNOWLEDGE_TASK_RETRY_EXHAUSTED"
            task.error_summary = "任务租约过期且已达到最大尝试次数"
            task.completed_at = now
            task.updated_at = now
            task.lease_owner = None
            task.lease_token_hash = None
            task.lease_expires_at = None
            task.heartbeat_at = None
            if task.document_id and task.task_type == "index_document":
                document = self.db.query(KnowledgeDocument).filter(KnowledgeDocument.id == task.document_id).first()
                if (
                    document is not None
                    and document.desired_index_version == task.index_version
                    and document.status not in {"deleting", "deleted"}
                    and document.active_index_version is None
                ):
                    document.status = "failed"
                    document.error_code = task.error_code
                    document.error_summary = task.error_summary
                    document.attempt_count = task.attempt_count
                    document.updated_at = now
                    if task.index_version:
                        version = self.get_index_version(task.index_version)
                        if version is not None and version.status == "building":
                            version.status = "failed"
                            version.error_code = task.error_code
                            version.error_summary = task.error_summary
                self._schedule_final_index_cleanup(
                    task,
                    now,
                    external_write_grace_seconds=external_write_grace_seconds,
                )

    def _schedule_final_index_cleanup(
        self,
        task: KnowledgeIndexTask,
        now: datetime,
        *,
        external_write_grace_seconds: int,
    ) -> None:
        if task.task_type != "index_document" or not task.index_version:
            return
        available_at = now + timedelta(seconds=max(0, external_write_grace_seconds))
        self.db.query(KnowledgeIndexTask).filter(
            KnowledgeIndexTask.status.in_(("pending", "retry")),
            or_(
                and_(
                    KnowledgeIndexTask.task_type == "delete_document",
                    KnowledgeIndexTask.document_id == task.document_id,
                ),
                and_(
                    KnowledgeIndexTask.task_type == "delete_knowledge_base",
                    KnowledgeIndexTask.knowledge_base_id == task.knowledge_base_id,
                ),
            ),
            KnowledgeIndexTask.available_at < available_at,
        ).update({"available_at": available_at, "updated_at": now}, synchronize_session=False)
        existing = (
            self.db.query(KnowledgeIndexTask)
            .filter(
                KnowledgeIndexTask.task_type == "delete_index_version",
                KnowledgeIndexTask.index_version == task.index_version,
                KnowledgeIndexTask.status.in_(("pending", "running", "retry")),
            )
            .first()
        )
        if existing is not None:
            if existing.status != "running" and existing.available_at < available_at:
                existing.available_at = available_at
                existing.updated_at = now
            return
        version = self.get_index_version(task.index_version)
        if version is None or version.status == "active":
            return
        version.status = "deleting"
        self.db.add(
            KnowledgeIndexTask(
                knowledge_base_id=task.knowledge_base_id,
                document_id=task.document_id,
                user_id=task.user_id,
                task_type="delete_index_version",
                index_version=task.index_version,
                status="pending",
                phase="queued",
                max_attempts=task.max_attempts,
                available_at=available_at,
            )
        )

    def _lock_task(self, task_id: str) -> KnowledgeIndexTask | None:
        return self.db.query(KnowledgeIndexTask).filter(KnowledgeIndexTask.id == task_id).with_for_update().first()

    @classmethod
    def _lease_matches(
        cls,
        task: KnowledgeIndexTask | None,
        lease_token: str,
        now: datetime,
    ) -> bool:
        if task is None or task.status != "running" or not task.lease_token_hash:
            return False
        if not secrets.compare_digest(task.lease_token_hash, cls._hash_token(lease_token)):
            return False
        return task.lease_expires_at is not None and cls._as_utc(task.lease_expires_at) >= cls._as_utc(now)

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()


def is_unique_violation(exc: Exception) -> bool:
    return isinstance(exc, IntegrityError)
