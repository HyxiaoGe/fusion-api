from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import and_, exists, not_, or_
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session

from app.db.models import (
    File,
    KnowledgeDocument,
    KnowledgeStorageCleanupTask,
    KnowledgeStorageUploadIntent,
)
from app.utils.time import utc_now


@dataclass(frozen=True)
class RegisteredStorageUploadIntent:
    task_id: str
    intent_id: str


@dataclass(frozen=True)
class ClaimedStorageCleanupTask:
    task: KnowledgeStorageCleanupTask
    lease_token: str


class StorageCleanupBusy(RuntimeError):
    """旧清理 Worker 已取得租约，新的同 key 上传必须等待其收敛。"""


class StorageUploadFenceConflict(RuntimeError):
    """文档提交时上传 intent 已失效、未成功或清理任务已取得所有权。"""


class StorageCleanupRepository:
    """对象清理补偿的 intent guard、租约和有界重试状态机。"""

    def __init__(self, db: Session):
        self.db = db

    def register_upload_intent(
        self,
        *,
        storage_backend: str,
        storage_key: str,
        hold_seconds: int,
        max_attempts: int,
        cleanup_scope: str = "knowledge",
        now: datetime | None = None,
    ) -> RegisteredStorageUploadIntent:
        if cleanup_scope not in {"knowledge", "file"}:
            raise ValueError("对象清理 scope 无效")
        requested_now = now
        registration_started_at = requested_now or utc_now()
        guard_until = registration_started_at + timedelta(seconds=max(1, hold_seconds))
        try:
            task = self._lock_task_for_object(storage_backend, storage_key)
        except OperationalError as exc:
            self.db.rollback()
            raise StorageCleanupBusy("同一对象的清理任务正在运行") from exc
        if task is None:
            task = KnowledgeStorageCleanupTask(
                id=str(uuid.uuid4()),
                storage_backend=storage_backend,
                storage_key=storage_key,
                max_attempts=max_attempts,
                available_at=guard_until,
                status="completed" if cleanup_scope == "file" else "pending",
                file_status="pending" if cleanup_scope == "file" else None,
            )
            self.db.add(task)
            try:
                self.db.flush()
            except IntegrityError:
                # 并发注册同一内容寻址 key 时复用唯一任务。
                self.db.rollback()
                try:
                    task = self._lock_task_for_object(storage_backend, storage_key)
                except OperationalError as exc:
                    self.db.rollback()
                    raise StorageCleanupBusy("同一对象的清理任务正在运行") from exc
                if task is None:
                    raise
        if (task.file_status is not None) != (cleanup_scope == "file"):
            self.db.rollback()
            raise StorageCleanupBusy("同一对象的清理 scope 不一致")
        if self._task_state(task) == "running":
            self.db.rollback()
            raise StorageCleanupBusy("同一对象的清理任务正在运行")
        registered_at = requested_now or utc_now()
        guard_until = registered_at + timedelta(seconds=max(1, hold_seconds))
        self._delete_expired_intents(task.id, registered_at)
        self._set_task_state(task, "pending")
        task.attempt_count = 0
        task.max_attempts = max_attempts
        task.available_at = max(self._as_utc(task.available_at), self._as_utc(guard_until))
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.heartbeat_at = None
        task.error_code = None
        task.error_summary = None
        task.completed_at = None
        task.updated_at = registered_at
        intent = KnowledgeStorageUploadIntent(
            id=str(uuid.uuid4()),
            cleanup_task_id=task.id,
            expires_at=guard_until,
            outcome="uploading",
        )
        self.db.add(intent)
        self.db.commit()
        return RegisteredStorageUploadIntent(task_id=task.id, intent_id=intent.id)

    def record_upload_outcome(
        self,
        registration: RegisteredStorageUploadIntent,
        *,
        outcome: str,
        hold_seconds: float,
    ) -> bool:
        """在持久 fence 内记录 SDK 的确定结果，并续租到请求事务完成。"""
        if outcome not in {"succeeded", "failed", "uncertain"}:
            raise ValueError("上传结果只能是 succeeded、failed 或 uncertain")
        task = self._lock_task(registration.task_id)
        if task is None or self._task_state(task) in {"running", "completed"} or task.attempt_count != 0:
            self.db.rollback()
            return False
        intent = self._lock_intent(registration)
        if intent is None or intent.outcome not in {"uploading", "uncertain"}:
            self.db.rollback()
            return False
        now = utc_now()
        guard_until = now + timedelta(seconds=max(1.0, hold_seconds))
        intent.outcome = outcome
        intent.outcome_at = now
        intent.expires_at = guard_until
        task.available_at = max(self._as_utc(task.available_at), self._as_utc(guard_until))
        task.updated_at = now
        self.db.commit()
        return True

    def release_detached_upload(self, registration: RegisteredStorageUploadIntent) -> None:
        """取消请求的单调收敛：completed 不回退，不确定结果保留供 Worker 复核。"""
        task = self._lock_task(registration.task_id)
        if task is None:
            self.db.rollback()
            return
        if self._task_state(task) == "completed":
            self._delete_intent(registration.intent_id, task.id)
            self.db.commit()
            return
        intent = self._lock_intent(registration)
        now = utc_now()
        if self._has_storage_reference(task):
            self._complete_locked(task, now)
            return
        if intent is not None and intent.outcome == "failed":
            self._complete_locked(task, now)
            return
        if intent is not None and intent.outcome == "succeeded":
            self._delete_intent(registration.intent_id, task.id)
        elif intent is not None:
            # unknown/uncertain 不能因一次 absent 就完成；保留过期 outcome，Worker
            # 可在对象迟到后删除，或由后续明确结果推进为 failed。
            intent.expires_at = now
        if self._task_state(task) != "running":
            self._set_task_state(task, "pending")
            task.available_at = now
            self._clear_lease(task)
            task.completed_at = None
            task.updated_at = now
        self.db.commit()

    def complete_absent_if_definitive(
        self,
        task: KnowledgeStorageCleanupTask,
        lease_token: str,
    ) -> bool:
        """只有 PUT 明确失败时，对象 absent 才是确定终态。"""
        if not self._lease_token_matches(task, lease_token):
            self.db.rollback()
            return False
        outcomes = self._upload_outcomes(task.id)
        if task.delete_started_at is None and (not outcomes or any(outcome != "failed" for outcome in outcomes)):
            return False
        self._complete_locked(task, utc_now())
        return True

    def mark_delete_started(
        self,
        task: KnowledgeStorageCleanupTask,
        lease_token: str,
        *,
        lease_seconds: int,
    ) -> KnowledgeStorageCleanupTask | None:
        """持久化删除回执后重新锁行，并在真实 RPC 返回前持续持有。"""
        locked = self._lock_task(task.id)
        if not self._lease_token_matches(locked, lease_token):
            self.db.rollback()
            return None
        now = utc_now()
        locked.delete_started_at = now
        locked.lease_expires_at = now + timedelta(seconds=max(1, lease_seconds))
        locked.heartbeat_at = now
        locked.updated_at = now
        self.db.commit()

        # delete_started_at 必须先独立持久化，才能在 RPC 成功后进程崩溃时凭 absent
        # 安全收敛。随后重新取得同一行的排他锁；一旦旧 token 已被重领替换，绝不
        # 再发送 DELETE。持锁跨越真实 RPC 可阻止过期租约被重领和同 key 重新上传。
        locked = self._lock_task(task.id)
        if not self._lease_token_matches(locked, lease_token):
            self.db.rollback()
            return None
        now = utc_now()
        locked.lease_expires_at = now + timedelta(seconds=max(1, lease_seconds))
        locked.heartbeat_at = now
        locked.updated_at = now
        self.db.flush()
        return locked

    def lock_succeeded_upload_for_document(
        self,
        registration: RegisteredStorageUploadIntent,
    ) -> KnowledgeStorageCleanupTask:
        """在调用方文档事务内取得 task/intent fence；本方法不提交。"""
        return self.lock_succeeded_uploads_for_reference([registration])[0][1]

    def lock_succeeded_uploads_for_reference(
        self,
        registrations: list[RegisteredStorageUploadIntent],
    ) -> list[tuple[RegisteredStorageUploadIntent, KnowledgeStorageCleanupTask]]:
        """按稳定顺序锁定一个引用事务需要的全部成功上传 fence。"""
        locked: list[tuple[RegisteredStorageUploadIntent, KnowledgeStorageCleanupTask]] = []
        seen: set[str] = set()
        for registration in sorted(registrations, key=lambda item: (item.task_id, item.intent_id)):
            if registration.task_id in seen:
                self.db.rollback()
                raise StorageUploadFenceConflict("同一引用事务不能重复使用 cleanup task")
            seen.add(registration.task_id)
            task = self._lock_task(registration.task_id)
            intent = self._lock_intent(registration)
            now = utc_now()
            if (
                task is None
                or self._task_state(task) in {"running", "completed"}
                or task.attempt_count != 0
                or intent is None
                or intent.outcome != "succeeded"
                or self._as_utc(intent.expires_at) <= self._as_utc(now)
            ):
                self.db.rollback()
                raise StorageUploadFenceConflict("对象上传 fence 已失效")
            locked.append((registration, task))
        return locked

    def complete_reference_writes_locked(
        self,
        locked_uploads: list[tuple[RegisteredStorageUploadIntent, KnowledgeStorageCleanupTask]],
    ) -> None:
        """在引用 INSERT/UPDATE 的同一事务内完成全部 cleanup fence。"""
        now = utc_now()
        for _registration, task in locked_uploads:
            self._set_completed(task, now)

    def complete_document_write_locked(
        self,
        registration: RegisteredStorageUploadIntent,
        task: KnowledgeStorageCleanupTask,
    ) -> None:
        """与文档 INSERT 处于同一事务，原子移除 intent 并完成 cleanup task。"""
        self.complete_reference_writes_locked([(registration, task)])

    def complete_referenced_upload(self, registration: RegisteredStorageUploadIntent) -> bool:
        """旧文件 API 已提交 File 引用后完成持久 cleanup fence。"""
        task = self._lock_task(registration.task_id)
        if task is None:
            self.db.rollback()
            return False
        if self._task_state(task) == "completed":
            self._delete_intent(registration.intent_id, task.id)
            self.db.commit()
            return True
        self._lock_intent(registration)
        if not self._has_storage_reference(task):
            self.db.rollback()
            return False
        self._complete_locked(task, utc_now())
        return True

    def resolve_known_conflict(
        self,
        registration: RegisteredStorageUploadIntent,
        *,
        cleanup_succeeded: bool,
    ) -> None:
        task = self._lock_task(registration.task_id)
        if task is None:
            self.db.rollback()
            return
        if self._task_state(task) == "completed":
            self._delete_intent(registration.intent_id, task.id)
            self.db.commit()
            return
        now = utc_now()
        self._delete_intent(registration.intent_id, task.id)
        self._delete_expired_intents(task.id, now)
        if self._has_storage_reference(task):
            self._complete_locked(task, now)
            return
        remaining = self._active_intents(task.id, now)
        if cleanup_succeeded and not remaining:
            self._complete_locked(task, now)
            return
        self._set_task_state(task, "pending")
        task.available_at = min((self._as_utc(row.expires_at) for row in remaining), default=self._as_utc(now))
        self._clear_lease(task)
        task.completed_at = None
        task.updated_at = now
        self.db.commit()

    def claim_task(
        self,
        *,
        worker_id: str,
        lease_seconds: int,
        retry_base_seconds: int = 5,
        retry_max_seconds: int = 300,
        now: datetime | None = None,
    ) -> ClaimedStorageCleanupTask | None:
        requested_now = now
        now = requested_now or utc_now()
        self._fail_exhausted_expired_tasks(now)
        self._reconcile_failed_tasks(
            now,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds,
        )
        active_intent = exists().where(
            KnowledgeStorageUploadIntent.cleanup_task_id == KnowledgeStorageCleanupTask.id,
            KnowledgeStorageUploadIntent.expires_at > now,
        )
        claimable = or_(
            and_(
                KnowledgeStorageCleanupTask.status.in_(("pending", "retry")),
                KnowledgeStorageCleanupTask.available_at <= now,
            ),
            and_(
                KnowledgeStorageCleanupTask.file_status.in_(("pending", "retry")),
                KnowledgeStorageCleanupTask.available_at <= now,
            ),
            and_(
                KnowledgeStorageCleanupTask.status == "running",
                KnowledgeStorageCleanupTask.lease_expires_at < now,
            ),
            and_(
                KnowledgeStorageCleanupTask.file_status == "running",
                KnowledgeStorageCleanupTask.lease_expires_at < now,
            ),
        )
        query = (
            self.db.query(KnowledgeStorageCleanupTask)
            .filter(
                claimable,
                KnowledgeStorageCleanupTask.attempt_count < KnowledgeStorageCleanupTask.max_attempts,
                not_(active_intent),
            )
            .order_by(
                KnowledgeStorageCleanupTask.available_at.asc(),
                KnowledgeStorageCleanupTask.created_at.asc(),
                KnowledgeStorageCleanupTask.id.asc(),
            )
        )
        query = (
            query.with_for_update(skip_locked=True)
            if self.db.get_bind().dialect.name == "postgresql"
            else query.with_for_update()
        )
        task = query.first()
        if task is None:
            self.db.commit()
            return None
        lease_started_at = requested_now or utc_now()
        lease_token = secrets.token_urlsafe(32)
        self._set_task_state(task, "running")
        task.attempt_count += 1
        task.lease_owner = worker_id
        task.lease_token_hash = self._hash_token(lease_token)
        task.lease_expires_at = lease_started_at + timedelta(seconds=lease_seconds)
        task.heartbeat_at = lease_started_at
        task.error_code = None
        task.error_summary = None
        task.updated_at = lease_started_at
        expire_on_commit = self.db.expire_on_commit
        try:
            self.db.expire_on_commit = False
            self.db.commit()
        finally:
            self.db.expire_on_commit = expire_on_commit
        return ClaimedStorageCleanupTask(task=task, lease_token=lease_token)

    def prepare_delete(self, task_id: str, lease_token: str) -> tuple[str, KnowledgeStorageCleanupTask | None]:
        task = self._lock_task(task_id)
        now = utc_now()
        if not self._lease_matches(task, lease_token, now):
            self.db.rollback()
            return "lease_lost", None
        active_intents = self._active_intents(task.id, now)
        if active_intents:
            self._set_task_state(task, "pending")
            task.available_at = min(self._as_utc(row.expires_at) for row in active_intents)
            task.attempt_count = max(0, task.attempt_count - 1)
            self._clear_lease(task)
            task.updated_at = now
            self.db.commit()
            return "guarded", None
        if self._has_storage_reference(task):
            self._complete_locked(task, now)
            return "referenced", None
        return "delete", task

    def complete_delete(self, task: KnowledgeStorageCleanupTask, lease_token: str) -> bool:
        task = self._lock_task(task.id)
        if not self._lease_token_matches(task, lease_token):
            self.db.rollback()
            return False
        self._complete_locked(task, utc_now())
        return True

    def fail_delete(
        self,
        task: KnowledgeStorageCleanupTask,
        lease_token: str,
        *,
        error_code: str,
        error_summary: str,
        retry_delay_seconds: int,
    ) -> bool:
        task = self._lock_task(task.id)
        if not self._lease_token_matches(task, lease_token):
            self.db.rollback()
            return False
        now = utc_now()
        will_retry = task.attempt_count < task.max_attempts
        self._set_task_state(task, "retry" if will_retry else "failed")
        task.available_at = now + timedelta(seconds=retry_delay_seconds) if will_retry else now
        task.error_code = error_code[:120]
        task.error_summary = error_summary[:500]
        task.completed_at = None if will_retry else now
        task.updated_at = now
        self._clear_lease(task)
        self.db.commit()
        return True

    def _lock_task_for_object(self, storage_backend: str, storage_key: str) -> KnowledgeStorageCleanupTask | None:
        return self._task_for_object_query(storage_backend, storage_key).first()

    def _task_for_object_query(self, storage_backend: str, storage_key: str):
        query = (
            self.db.query(KnowledgeStorageCleanupTask)
            .filter(
                KnowledgeStorageCleanupTask.storage_backend == storage_backend,
                KnowledgeStorageCleanupTask.storage_key == storage_key,
            )
            .populate_existing()
        )
        query = (
            query.with_for_update(nowait=True)
            if self.db.get_bind().dialect.name == "postgresql"
            else query.with_for_update()
        )
        return query

    def _expired_exhausted_tasks_query(self, now: datetime):
        query = (
            self.db.query(KnowledgeStorageCleanupTask)
            .filter(
                or_(
                    KnowledgeStorageCleanupTask.status == "running",
                    KnowledgeStorageCleanupTask.file_status == "running",
                ),
                KnowledgeStorageCleanupTask.lease_expires_at < now,
                KnowledgeStorageCleanupTask.attempt_count >= KnowledgeStorageCleanupTask.max_attempts,
            )
            .order_by(
                KnowledgeStorageCleanupTask.lease_expires_at.asc(),
                KnowledgeStorageCleanupTask.updated_at.asc(),
                KnowledgeStorageCleanupTask.id.asc(),
            )
            .populate_existing()
            .limit(100)
        )
        return (
            query.with_for_update(skip_locked=True)
            if self.db.get_bind().dialect.name == "postgresql"
            else query.with_for_update()
        )

    def _fail_exhausted_expired_tasks(self, now: datetime) -> None:
        for task in self._expired_exhausted_tasks_query(now).all():
            if (
                self._task_state(task) != "running"
                or task.lease_expires_at is None
                or self._as_utc(task.lease_expires_at) >= self._as_utc(now)
                or task.attempt_count < task.max_attempts
            ):
                continue
            self._set_task_state(task, "failed")
            task.available_at = now
            task.error_code = "KNOWLEDGE_STORAGE_CLEANUP_INTERRUPTED"
            task.error_summary = "孤立对象清理 Worker 在最后一次尝试中断"
            task.completed_at = now
            task.updated_at = now
            self._clear_lease(task)

    def _lock_task(self, task_id: str) -> KnowledgeStorageCleanupTask | None:
        return self._task_by_id_query(task_id).first()

    def _task_by_id_query(self, task_id: str):
        return (
            self.db.query(KnowledgeStorageCleanupTask)
            .filter(KnowledgeStorageCleanupTask.id == task_id)
            .populate_existing()
            .with_for_update()
        )

    def _lock_intent(
        self,
        registration: RegisteredStorageUploadIntent,
    ) -> KnowledgeStorageUploadIntent | None:
        return self._intent_query(registration).first()

    def _intent_query(self, registration: RegisteredStorageUploadIntent):
        return (
            self.db.query(KnowledgeStorageUploadIntent)
            .filter(
                KnowledgeStorageUploadIntent.id == registration.intent_id,
                KnowledgeStorageUploadIntent.cleanup_task_id == registration.task_id,
            )
            .populate_existing()
            .with_for_update()
        )

    def _has_storage_reference(self, task: KnowledgeStorageCleanupTask) -> bool:
        knowledge_reference = (
            self.db.query(KnowledgeDocument.id)
            .filter(
                KnowledgeDocument.storage_backend == task.storage_backend,
                KnowledgeDocument.storage_key == task.storage_key,
                KnowledgeDocument.status != "deleted",
            )
            .first()
            is not None
        )
        if knowledge_reference:
            return True
        return (
            self.db.query(File.id)
            .filter(
                File.storage_backend == task.storage_backend,
                or_(File.storage_key == task.storage_key, File.thumbnail_key == task.storage_key),
            )
            .first()
            is not None
        )

    def _active_intents(self, task_id: str, now: datetime) -> list[KnowledgeStorageUploadIntent]:
        return (
            self.db.query(KnowledgeStorageUploadIntent)
            .filter(
                KnowledgeStorageUploadIntent.cleanup_task_id == task_id,
                KnowledgeStorageUploadIntent.expires_at > now,
            )
            .order_by(KnowledgeStorageUploadIntent.expires_at.asc())
            .all()
        )

    def _upload_outcomes(self, task_id: str) -> list[str]:
        return [
            str(outcome)
            for (outcome,) in self.db.query(KnowledgeStorageUploadIntent.outcome)
            .filter(KnowledgeStorageUploadIntent.cleanup_task_id == task_id)
            .all()
        ]

    def _delete_expired_intents(self, task_id: str, now: datetime) -> None:
        self.db.query(KnowledgeStorageUploadIntent).filter(
            KnowledgeStorageUploadIntent.cleanup_task_id == task_id,
            KnowledgeStorageUploadIntent.expires_at <= now,
        ).delete(synchronize_session=False)

    def _delete_intent(self, intent_id: str, task_id: str) -> None:
        self.db.query(KnowledgeStorageUploadIntent).filter(
            KnowledgeStorageUploadIntent.id == intent_id,
            KnowledgeStorageUploadIntent.cleanup_task_id == task_id,
        ).delete(synchronize_session=False)

    def _complete_locked(self, task: KnowledgeStorageCleanupTask, now: datetime) -> None:
        self._set_completed(task, now)
        self.db.commit()

    def _set_completed(self, task: KnowledgeStorageCleanupTask, now: datetime) -> None:
        # 一个内容寻址 task 可被并发请求共享。任一受保护引用或删除完成后，该 task
        # 已进入不可逆终态，必须清除全部 intent，避免败方 detach 永久遗留记录。
        self.db.query(KnowledgeStorageUploadIntent).filter(
            KnowledgeStorageUploadIntent.cleanup_task_id == task.id
        ).delete(synchronize_session=False)
        self._set_task_state(task, "completed")
        task.completed_at = now
        task.updated_at = now
        task.error_code = None
        task.error_summary = None
        self._clear_lease(task)

    def _reconcile_failed_tasks(
        self,
        now: datetime,
        *,
        retry_base_seconds: int,
        retry_max_seconds: int,
    ) -> None:
        query = (
            self.db.query(KnowledgeStorageCleanupTask)
            .filter(
                or_(
                    KnowledgeStorageCleanupTask.status == "failed",
                    KnowledgeStorageCleanupTask.file_status == "failed",
                )
            )
            .order_by(KnowledgeStorageCleanupTask.updated_at.asc(), KnowledgeStorageCleanupTask.id.asc())
            .limit(100)
        )
        query = (
            query.with_for_update(skip_locked=True)
            if self.db.get_bind().dialect.name == "postgresql"
            else query.with_for_update()
        )
        for task in query.all():
            delay = min(
                max(1, retry_base_seconds) * (2 ** min(max(task.attempt_count - 1, 0), 8)),
                max(1, retry_max_seconds),
            )
            self._set_task_state(task, "retry")
            task.max_attempts = max(task.max_attempts, task.attempt_count + 1)
            task.available_at = now + timedelta(seconds=delay)
            task.completed_at = None
            task.updated_at = now
            self._clear_lease(task)

    @staticmethod
    def _task_state(task: KnowledgeStorageCleanupTask) -> str:
        return str(task.file_status or task.status)

    @staticmethod
    def _set_task_state(task: KnowledgeStorageCleanupTask, state: str) -> None:
        if task.file_status is None:
            task.status = state
            return
        # 物理 status 永远保持旧 Worker 不会领取的 completed；新 Worker 只按
        # file_status 驱动状态机，从而允许安全回滚到不理解 File 引用的镜像。
        task.status = "completed"
        task.file_status = state

    @staticmethod
    def _clear_lease(task: KnowledgeStorageCleanupTask) -> None:
        task.lease_owner = None
        task.lease_token_hash = None
        task.lease_expires_at = None
        task.heartbeat_at = None

    @classmethod
    def _lease_matches(
        cls,
        task: KnowledgeStorageCleanupTask | None,
        lease_token: str,
        now: datetime,
    ) -> bool:
        if task is None or cls._task_state(task) != "running" or not task.lease_token_hash:
            return False
        if not secrets.compare_digest(task.lease_token_hash, cls._hash_token(lease_token)):
            return False
        return task.lease_expires_at is not None and cls._as_utc(task.lease_expires_at) >= cls._as_utc(now)

    @classmethod
    def _lease_token_matches(cls, task: KnowledgeStorageCleanupTask | None, lease_token: str) -> bool:
        if task is None or cls._task_state(task) != "running" or not task.lease_token_hash:
            return False
        return secrets.compare_digest(task.lease_token_hash, cls._hash_token(lease_token))

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
