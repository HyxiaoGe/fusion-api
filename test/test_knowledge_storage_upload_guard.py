import asyncio
import gc
import hashlib
import io
import threading
import unittest
import weakref
from datetime import UTC, timedelta
from unittest.mock import MagicMock, patch

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import Headers

from app.db.database import Base
from app.db.models import (
    KnowledgeBase,
    KnowledgeDocument,
    KnowledgeStorageCleanupTask,
    KnowledgeStorageUploadIntent,
    User,
)
from app.db.storage_cleanup_repository import StorageCleanupRepository
from app.schemas.response import ApiException
from app.services.knowledge.service import KnowledgeService
from app.services.knowledge.storage_cleanup_worker import KnowledgeStorageCleanupWorker
from app.services.knowledge.storage_upload_guard import (
    StorageUploadFenceLost,
    active_storage_upload_lifecycle_count,
    drain_storage_upload_lifecycles,
    start_guarded_storage_upload,
)
from app.utils.time import utc_now


class _BlockingThreadStorage:
    """真实占用 to_thread，模拟 wait_for 无法取消的对象存储 SDK。"""

    def __init__(self):
        self.upload_started = threading.Event()
        self.upload_release = threading.Event()
        self._lock = threading.Lock()
        self.object_exists = False
        self.deleted_keys: list[str] = []

    async def upload(self, key: str, _content: bytes, _mimetype: str) -> str:
        return await asyncio.to_thread(self._blocking_upload, key)

    def _blocking_upload(self, key: str) -> str:
        self.upload_started.set()
        if not self.upload_release.wait(timeout=10):
            raise TimeoutError("测试上传线程未获释放")
        with self._lock:
            self.object_exists = True
        return key

    async def exists(self, _key: str) -> bool:
        with self._lock:
            return self.object_exists

    async def delete(self, key: str) -> bool:
        with self._lock:
            self.deleted_keys.append(key)
            self.object_exists = False
        return True


class _ImmediateStorage:
    async def upload(self, key: str, _content: bytes, _mimetype: str) -> str:
        return key


class _ToggleSessionFactory:
    def __init__(self, factory):
        self.factory = factory
        self.available = False

    def __call__(self):
        if not self.available:
            raise OperationalError("connect", {}, RuntimeError("database unavailable"))
        return self.factory()


class KnowledgeStorageUploadGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            db.add(User(id="user-1", username="guard-user", email="guard@example.com"))
            db.add(
                KnowledgeBase(
                    id="kb-1",
                    user_id="user-1",
                    name="Guard",
                    name_normalized="guard",
                    description="",
                    business_type="general",
                    status="active",
                    embedding_provider="litellm",
                    embedding_model="embed-v1",
                    embedding_revision="embed-r1",
                    embedding_dimension=2,
                    distance_metric="COSINE",
                )
            )
            db.commit()

    def tearDown(self):
        self.engine.dispose()

    async def test_sdk_return_before_request_terminal_keeps_renewing_and_registry_holds_task(self):
        with self.Session() as db:
            registration = StorageCleanupRepository(db).register_upload_intent(
                storage_backend="local",
                storage_key="knowledge/guard/normal",
                hold_seconds=1,
                max_attempts=3,
            )
        with self.Session() as db:
            initial_expiry = db.get(KnowledgeStorageUploadIntent, registration.intent_id).expires_at

        lifecycle = start_guarded_storage_upload(
            registration=registration,
            session_factory=self.Session,
            storage=_ImmediateStorage(),
            storage_key="knowledge/guard/normal",
            content=b"content",
            mimetype="text/plain",
            hold_seconds=1,
        )
        await lifecycle.wait_upload()
        lifecycle_ref = weakref.ref(lifecycle)
        del lifecycle
        gc.collect()

        self.assertEqual(active_storage_upload_lifecycle_count(), 1)
        self.assertIsNotNone(lifecycle_ref())
        await asyncio.sleep(1.1)
        with self.Session() as db:
            renewed_expiry = db.get(KnowledgeStorageUploadIntent, registration.intent_id).expires_at
            self.assertGreater(renewed_expiry, initial_expiry)
            self.assertIsNone(StorageCleanupRepository(db).claim_task(worker_id="worker", lease_seconds=10))

        with self.Session() as db:
            StorageCleanupRepository(db).resolve_known_conflict(
                registration,
                cleanup_succeeded=True,
            )
        retained_lifecycle = lifecycle_ref()
        self.assertIsNotNone(retained_lifecycle)
        retained_lifecycle.mark_request_resolved()
        await retained_lifecycle.wait_finished()
        self.assertEqual(active_storage_upload_lifecycle_count(), 0)

    async def test_wait_for_cancel_keeps_intent_until_blocking_upload_returns_then_cleans_orphan(self):
        content = b"cancelled upload"
        storage = _BlockingThreadStorage()
        service_db = self.Session()
        service = KnowledgeService(
            service_db,
            storage=storage,
            embedding=MagicMock(),
            vector_store=MagicMock(),
            session_factory=self.Session,
            storage_upload_hold_seconds=1,
        )
        service.vector_store.collection_name.return_value = "knowledge_v1_d2"
        upload = UploadFile(
            filename="manual.txt",
            file=io.BytesIO(content),
            headers=Headers({"content-type": "text/plain"}),
        )

        try:
            with (
                patch("app.services.knowledge.service.settings.KNOWLEDGE_BASE_ENABLED", True),
                patch("app.services.knowledge.service.settings.STORAGE_BACKEND", "local"),
                patch.object(service, "_require_enabled", return_value=None),
            ):
                request_task = asyncio.create_task(service.upload_document("user-1", "kb-1", upload))
                started = await asyncio.to_thread(storage.upload_started.wait, 2)
                self.assertTrue(started)
                with self.assertRaises(TimeoutError):
                    await asyncio.wait_for(request_task, timeout=0.02)
        finally:
            # 模拟 FastAPI 在超时响应后关闭请求依赖；后台只能使用注入的新 session。
            service_db.close()

        self.assertEqual(active_storage_upload_lifecycle_count(), 1)
        with self.Session() as db:
            intent = db.query(KnowledgeStorageUploadIntent).one()
            registration_task_id = intent.cleanup_task_id
            initial_expiry = intent.expires_at

        # 底层线程持续时间超过初始一秒 hold，期间 intent 必须被续租且不可 claim。
        await asyncio.sleep(1.1)
        with self.Session() as db:
            renewed_intent = db.query(KnowledgeStorageUploadIntent).one()
            self.assertGreater(renewed_intent.expires_at, initial_expiry)
            self.assertGreater(
                self._as_utc(renewed_intent.expires_at),
                self._as_utc(utc_now()),
            )
            self.assertIsNone(StorageCleanupRepository(db).claim_task(worker_id="early", lease_seconds=10))

        drain_task = asyncio.create_task(drain_storage_upload_lifecycles())
        await asyncio.sleep(0)
        self.assertFalse(drain_task.done())
        storage.upload_release.set()
        await asyncio.wait_for(drain_task, timeout=3)
        self.assertEqual(active_storage_upload_lifecycle_count(), 0)

        # 迟到 PUT 已真实成功，但 finalizer 不会把无引用对象误标 completed；它释放
        # intent，让持久 cleanup Worker 删除精确 generation key。
        self.assertTrue(storage.object_exists)
        with self.Session() as db:
            cleanup_task = db.get(KnowledgeStorageCleanupTask, registration_task_id)
            self.assertEqual(cleanup_task.status, "pending")
            self.assertEqual(db.query(KnowledgeStorageUploadIntent).count(), 0)

        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="cleanup-1")
        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            self.assertTrue(await worker.run_once())

        self.assertFalse(storage.object_exists)
        self.assertEqual(len(storage.deleted_keys), 1)
        expected_checksum = hashlib.sha256(content).hexdigest()
        self.assertIn(expected_checksum, storage.deleted_keys[0])
        with self.Session() as db:
            cleanup_task = db.get(KnowledgeStorageCleanupTask, registration_task_id)
            self.assertEqual(cleanup_task.status, "completed")

    async def test_database_outage_cannot_commit_after_cleanup_takes_over_upload_fence(self):
        content = b"database outage upload"
        storage = _BlockingThreadStorage()
        lifecycle_sessions = _ToggleSessionFactory(self.Session)
        service_db = self.Session()
        service = KnowledgeService(
            service_db,
            storage=storage,
            embedding=MagicMock(),
            vector_store=MagicMock(),
            session_factory=lifecycle_sessions,
            storage_upload_hold_seconds=1,
        )
        service.vector_store.collection_name.return_value = "knowledge_v1_d2"
        upload = UploadFile(
            filename="manual.txt",
            file=io.BytesIO(content),
            headers=Headers({"content-type": "text/plain"}),
        )
        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="cleanup-outage")

        try:
            with (
                patch("app.services.knowledge.service.settings.KNOWLEDGE_BASE_ENABLED", True),
                patch("app.services.knowledge.service.settings.STORAGE_BACKEND", "local"),
                patch.object(service, "_require_enabled", return_value=None),
                patch(
                    "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
                    return_value=storage,
                ),
                patch("app.services.knowledge.storage_upload_guard.logger.exception"),
            ):
                request_task = asyncio.create_task(service.upload_document("user-1", "kb-1", upload))
                self.assertTrue(await asyncio.to_thread(storage.upload_started.wait, 2))
                await asyncio.sleep(1.1)

                # DB 恢复后 cleanup 先取得过期任务。对象尚不可见不能完成，只能退避复核。
                self.assertTrue(await worker.run_once())
                with self.Session() as db:
                    task = db.query(KnowledgeStorageCleanupTask).one()
                    self.assertIn(task.status, {"retry", "failed"})
                    self.assertEqual(task.error_code, "KNOWLEDGE_STORAGE_DELETE_UNCERTAIN")
                    intent = db.query(KnowledgeStorageUploadIntent).one()
                    self.assertEqual(intent.outcome, "uploading")

                lifecycle_sessions.available = True
                storage.upload_release.set()
                with self.assertRaises(ApiException) as raised:
                    await request_task
                self.assertEqual(raised.exception.code, "KNOWLEDGE_STORAGE_BUSY")
                self.assertEqual(raised.exception.status_code, 503)
                await drain_storage_upload_lifecycles()
        finally:
            service_db.close()

        # fresh-session finalizer 保留不确定 outcome 并将代际置为待清理；迟到 PUT
        # 出现后仍能删除，不能因第一次 absent 丢失持久证据。
        self.assertTrue(storage.object_exists)
        with self.Session() as db:
            task = db.query(KnowledgeStorageCleanupTask).one()
            self.assertEqual(task.status, "pending")
            self.assertEqual(db.query(KnowledgeDocument).count(), 0)
            intent = db.query(KnowledgeStorageUploadIntent).one()
            self.assertEqual(intent.outcome, "uploading")
            task.available_at = utc_now()
            db.commit()
        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            self.assertTrue(await worker.run_once())
        self.assertFalse(storage.object_exists)
        with self.Session() as db:
            self.assertEqual(db.query(KnowledgeStorageCleanupTask).one().status, "completed")

    async def test_renewal_cannot_revive_fence_after_cleanup_has_taken_over(self):
        with self.Session() as db:
            registration = StorageCleanupRepository(db).register_upload_intent(
                storage_backend="local",
                storage_key="knowledge/guard/taken-over",
                hold_seconds=1,
                max_attempts=3,
                now=utc_now(),
            )
            intent = db.get(KnowledgeStorageUploadIntent, registration.intent_id)
            intent.expires_at = utc_now() - timedelta(seconds=1)
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            task.available_at = utc_now() - timedelta(seconds=1)
            db.commit()
        with self.Session() as db:
            claimed = StorageCleanupRepository(db).claim_task(worker_id="cleanup", lease_seconds=30)
        self.assertIsNotNone(claimed)

        lifecycle = start_guarded_storage_upload(
            registration=registration,
            session_factory=self.Session,
            storage=_ImmediateStorage(),
            storage_key="knowledge/guard/taken-over",
            content=b"content",
            mimetype="text/plain",
            hold_seconds=1,
        )
        with self.assertRaises(StorageUploadFenceLost):
            await lifecycle.wait_upload()
        lifecycle.detach_request()
        await lifecycle.wait_finished()

        with self.Session() as db:
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(task.status, "running")
            self.assertEqual(task.attempt_count, 1)

    @staticmethod
    def _as_utc(value):
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


if __name__ == "__main__":
    unittest.main()
