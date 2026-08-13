import asyncio
import hashlib
import io
import unittest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError, OperationalError
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
from app.db.storage_cleanup_repository import StorageCleanupBusy, StorageCleanupRepository
from app.schemas.response import ApiException
from app.services.knowledge.service import KnowledgeService
from app.services.knowledge.storage_cleanup_worker import KnowledgeStorageCleanupWorker
from app.utils.time import utc_now


class KnowledgeStorageCleanupTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            db.add(User(id="user-1", username="cleanup-user", email="cleanup@example.com"))
            db.add(
                KnowledgeBase(
                    id="kb-1",
                    user_id="user-1",
                    name="Manual",
                    name_normalized="manual",
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

    def _register(self, *, key: str = "knowledge/orphan", hold_seconds: int = 60, max_attempts: int = 3):
        with self.Session() as db:
            return StorageCleanupRepository(db).register_upload_intent(
                storage_backend="local",
                storage_key=key,
                hold_seconds=hold_seconds,
                max_attempts=max_attempts,
            )

    def _activate(self, registration, *, cleanup_succeeded: bool = False) -> None:
        with self.Session() as db:
            StorageCleanupRepository(db).resolve_known_conflict(
                registration,
                cleanup_succeeded=cleanup_succeeded,
            )

    def test_intent_guard_is_not_claimed_before_expiry(self):
        now = utc_now()
        with self.Session() as db:
            StorageCleanupRepository(db).register_upload_intent(
                storage_backend="local",
                storage_key="knowledge/guarded",
                hold_seconds=60,
                max_attempts=3,
                now=now,
            )
        with self.Session() as db:
            claimed = StorageCleanupRepository(db).claim_task(
                worker_id="worker-1",
                lease_seconds=30,
                now=now + timedelta(seconds=59),
            )
        self.assertIsNone(claimed)

    def test_register_upload_intent_guard_uses_fresh_clock_after_object_lock(self):
        registration_started_at = utc_now()
        object_locked_at = registration_started_at + timedelta(seconds=9)
        with (
            self.Session() as db,
            patch(
                "app.db.storage_cleanup_repository.utc_now",
                side_effect=(registration_started_at, object_locked_at),
            ) as clock,
        ):
            registration = StorageCleanupRepository(db).register_upload_intent(
                storage_backend="local",
                storage_key="knowledge/fresh-intent",
                hold_seconds=10,
                max_attempts=3,
            )

        self.assertEqual(clock.call_count, 2)
        with self.Session() as db:
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            intent = db.get(KnowledgeStorageUploadIntent, registration.intent_id)
            expected_expiry = (object_locked_at + timedelta(seconds=10)).replace(tzinfo=None)
            self.assertEqual(task.available_at, expected_expiry)
            self.assertEqual(intent.expires_at, expected_expiry)

    def test_cleanup_claim_lease_uses_fresh_clock_after_task_lock(self):
        registration = self._register(key="knowledge/fresh-lease", hold_seconds=1)
        self._activate(registration)
        query_started_at = utc_now() + timedelta(seconds=2)
        lock_acquired_at = query_started_at + timedelta(seconds=9)
        with (
            self.Session() as db,
            patch(
                "app.db.storage_cleanup_repository.utc_now",
                side_effect=(query_started_at, lock_acquired_at),
            ) as clock,
        ):
            claimed = StorageCleanupRepository(db).claim_task(
                worker_id="worker-1",
                lease_seconds=10,
            )

        self.assertIsNotNone(claimed)
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(claimed.task.heartbeat_at, lock_acquired_at.replace(tzinfo=None))
        self.assertEqual(
            claimed.task.lease_expires_at,
            (lock_acquired_at + timedelta(seconds=10)).replace(tzinfo=None),
        )

    def test_running_task_rejects_same_key_registration_without_revoking_lease(self):
        registration = self._register(key="knowledge/running", hold_seconds=1)
        self._activate(registration)
        with self.Session() as db:
            claimed = StorageCleanupRepository(db).claim_task(
                worker_id="worker-1",
                lease_seconds=30,
                now=utc_now() + timedelta(seconds=2),
            )
        self.assertIsNotNone(claimed)

        with self.Session() as db:
            with self.assertRaises(StorageCleanupBusy):
                StorageCleanupRepository(db).register_upload_intent(
                    storage_backend="local",
                    storage_key="knowledge/running",
                    hold_seconds=60,
                    max_attempts=3,
                )
        with self.Session() as db:
            task = db.query(KnowledgeStorageCleanupTask).filter_by(id=claimed.task.id).one()
            self.assertEqual(task.status, "running")
            self.assertEqual(task.lease_owner, "worker-1")
            self.assertIsNotNone(task.lease_token_hash)

    def test_postgres_registration_uses_nowait_and_maps_lock_contention_to_busy(self):
        with self.Session() as db, patch.object(self.engine.dialect, "name", "postgresql"):
            repo = StorageCleanupRepository(db)
            statement = repo._task_for_object_query("local", "knowledge/locked").statement
        compiled = str(statement.compile(dialect=postgresql.dialect()))
        self.assertIn("FOR UPDATE NOWAIT", compiled)

        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            lock_error = OperationalError("select", {}, RuntimeError("lock not available"))
            with (
                patch.object(repo, "_lock_task_for_object", side_effect=lock_error),
                self.assertRaises(StorageCleanupBusy),
            ):
                repo.register_upload_intent(
                    storage_backend="local",
                    storage_key="knowledge/locked",
                    hold_seconds=60,
                    max_attempts=3,
                )

    def test_document_fence_locks_cleanup_task_then_upload_intent_on_postgres(self):
        registration = self._register(key="knowledge/document-fence", hold_seconds=60)
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            task_sql = str(repo._task_by_id_query(registration.task_id).statement.compile(dialect=postgresql.dialect()))
            intent_sql = str(repo._intent_query(registration).statement.compile(dialect=postgresql.dialect()))

        self.assertIn("FOR UPDATE", task_sql)
        self.assertIn("knowledge_storage_cleanup_tasks", task_sql)
        self.assertIn("FOR UPDATE", intent_sql)
        self.assertIn("knowledge_storage_upload_intents", intent_sql)

    def test_terminal_task_is_reused_for_later_same_key_intent(self):
        first = self._register(key="knowledge/reused", hold_seconds=1)
        self._activate(first, cleanup_succeeded=True)
        with self.Session() as db:
            self.assertEqual(db.get(KnowledgeStorageCleanupTask, first.task_id).status, "completed")

        second = self._register(key="knowledge/reused", hold_seconds=60)

        self.assertEqual(second.task_id, first.task_id)
        self.assertNotEqual(second.intent_id, first.intent_id)
        with self.Session() as db:
            task = db.get(KnowledgeStorageCleanupTask, first.task_id)
            self.assertEqual(task.status, "pending")
            self.assertEqual(task.attempt_count, 0)
            self.assertEqual(db.query(KnowledgeStorageUploadIntent).filter_by(cleanup_task_id=task.id).count(), 1)

    def test_finalizer_never_reopens_completed_cleanup_task(self):
        registration = self._register(key="knowledge/completed-finalizer", hold_seconds=60)
        self._activate(registration, cleanup_succeeded=True)

        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            repo.release_detached_upload(registration)
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(task.status, "completed")
            self.assertIsNotNone(task.completed_at)
            self.assertEqual(db.query(KnowledgeStorageUploadIntent).count(), 0)

    async def test_definitive_put_failure_absent_completes_without_polling(self):
        registration = self._register(key="knowledge/definitive-failure", hold_seconds=60)
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            self.assertTrue(
                repo.record_upload_outcome(
                    registration,
                    outcome="failed",
                    hold_seconds=60,
                )
            )
            repo.release_detached_upload(registration)

        with self.Session() as db:
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(task.status, "completed")
            self.assertEqual(db.query(KnowledgeStorageUploadIntent).count(), 0)

    async def test_uncertain_put_failure_absent_keeps_persistent_recheck(self):
        registration = self._register(key="knowledge/uncertain-failure", hold_seconds=1)
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            self.assertTrue(
                repo.record_upload_outcome(
                    registration,
                    outcome="uncertain",
                    hold_seconds=1,
                )
            )
            repo.release_detached_upload(registration)

        storage = MagicMock()
        storage.exists = AsyncMock(return_value=False)
        storage.delete = AsyncMock(return_value=False)
        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="uncertain-worker")
        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            self.assertTrue(await worker.run_once())

        with self.Session() as db:
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertIn(task.status, {"retry", "failed"})
            self.assertEqual(task.error_code, "KNOWLEDGE_STORAGE_DELETE_UNCERTAIN")
            intent = db.get(KnowledgeStorageUploadIntent, registration.intent_id)
            self.assertIsNotNone(intent)
            self.assertEqual(intent.outcome, "uncertain")

    async def test_document_reference_completes_without_deleting_object(self):
        registration = self._register(key="knowledge/referenced", hold_seconds=1)
        with self.Session() as db:
            db.add(
                KnowledgeDocument(
                    id="doc-1",
                    knowledge_base_id="kb-1",
                    user_id="user-1",
                    original_filename="manual.txt",
                    mimetype="text/plain",
                    size=10,
                    checksum_sha256="a" * 64,
                    dedupe_key="a" * 64,
                    storage_backend="local",
                    storage_key="knowledge/referenced",
                    status="queued",
                    parser_version="parser-v1",
                    chunker_version="chunker-v1",
                    embedding_provider="litellm",
                    embedding_model="embed-v1",
                    embedding_revision="embed-r1",
                    embedding_dimension=2,
                    distance_metric="COSINE",
                    desired_index_version="version-1",
                )
            )
            intent = db.get(KnowledgeStorageUploadIntent, registration.intent_id)
            intent.expires_at = utc_now() - timedelta(seconds=1)
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            task.available_at = utc_now() - timedelta(seconds=1)
            db.commit()
        storage = MagicMock()
        storage.exists = AsyncMock(return_value=True)
        storage.delete = AsyncMock(side_effect=[False, True])
        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="cleanup-1")

        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            handled = await worker.run_once()

        self.assertTrue(handled)
        storage.delete.assert_not_awaited()
        with self.Session() as db:
            self.assertEqual(db.get(KnowledgeStorageCleanupTask, registration.task_id).status, "completed")
            self.assertEqual(db.query(KnowledgeStorageUploadIntent).count(), 0)

    async def test_delete_failure_retries_and_failed_cycle_is_reconciled(self):
        registration = self._register(key="knowledge/retry", hold_seconds=1, max_attempts=1)
        self._activate(registration)
        storage = MagicMock()
        storage.exists = AsyncMock(return_value=True)
        storage.delete = AsyncMock(side_effect=[RuntimeError("temporary"), True])
        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="cleanup-1")

        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            self.assertTrue(await worker.run_once())
            with self.Session() as db:
                task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
                self.assertEqual(task.status, "failed")
                self.assertEqual(task.attempt_count, 1)
            self.assertFalse(await worker.run_once())
            with self.Session() as db:
                task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
                self.assertEqual(task.status, "retry")
                self.assertEqual(task.max_attempts, 2)
                task.available_at = utc_now() - timedelta(seconds=1)
                db.commit()
            self.assertTrue(await worker.run_once())

        with self.Session() as db:
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(task.status, "completed")
            self.assertEqual(task.attempt_count, 2)
        self.assertEqual(storage.delete.await_count, 2)

    async def test_blocked_delete_keeps_same_key_upload_fenced_until_sdk_returns(self):
        registration = self._register(key="knowledge/blocked", hold_seconds=1)
        self._activate(registration)
        delete_started = asyncio.Event()
        delete_release = asyncio.Event()

        async def blocked_delete(_key):
            delete_started.set()
            await delete_release.wait()
            return True

        storage = MagicMock()
        storage.exists = AsyncMock(return_value=True)
        storage.delete = AsyncMock(side_effect=blocked_delete)
        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="cleanup-1")
        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            cleanup = asyncio.create_task(worker.run_once())
            await delete_started.wait()
            with self.Session() as db:
                with self.assertRaises(StorageCleanupBusy):
                    StorageCleanupRepository(db).register_upload_intent(
                        storage_backend="local",
                        storage_key="knowledge/blocked",
                        hold_seconds=60,
                        max_attempts=3,
                    )
            delete_release.set()
            self.assertTrue(await cleanup)

        later = self._register(key="knowledge/blocked", hold_seconds=60)
        self.assertEqual(later.task_id, registration.task_id)

    def test_expired_running_lease_is_reclaimed_after_crash(self):
        registration = self._register(key="knowledge/crash", hold_seconds=1)
        self._activate(registration)
        now = utc_now() + timedelta(seconds=2)
        with self.Session() as db:
            first = StorageCleanupRepository(db).claim_task(
                worker_id="worker-1",
                lease_seconds=10,
                now=now,
            )
        with self.Session() as db:
            before_expiry = StorageCleanupRepository(db).claim_task(
                worker_id="worker-2",
                lease_seconds=10,
                now=now + timedelta(seconds=9),
            )
            reclaimed = StorageCleanupRepository(db).claim_task(
                worker_id="worker-2",
                lease_seconds=10,
                now=now + timedelta(seconds=11),
            )

        self.assertIsNotNone(first)
        self.assertIsNone(before_expiry)
        self.assertIsNotNone(reclaimed)
        self.assertNotEqual(first.lease_token, reclaimed.lease_token)
        self.assertEqual(reclaimed.task.attempt_count, 2)

    def test_last_attempt_crash_is_reconciled_once_with_backoff_and_skip_locked(self):
        registration = self._register(key="knowledge/last-attempt-crash", hold_seconds=1, max_attempts=1)
        self._activate(registration)
        claim_at = utc_now() + timedelta(seconds=2)
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            first = repo.claim_task(
                worker_id="worker-1",
                lease_seconds=10,
                now=claim_at,
            )
        self.assertIsNotNone(first)
        self.assertEqual(first.task.attempt_count, 1)
        self.assertEqual(first.task.max_attempts, 1)

        reconcile_at = claim_at + timedelta(seconds=11)
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            with patch.object(self.engine.dialect, "name", "postgresql"):
                statement = repo._expired_exhausted_tasks_query(reconcile_at).statement
            compiled = str(statement.compile(dialect=postgresql.dialect()))
            self.assertIn("FOR UPDATE SKIP LOCKED", compiled)

            self.assertIsNone(
                repo.claim_task(
                    worker_id="worker-2",
                    lease_seconds=10,
                    retry_base_seconds=5,
                    retry_max_seconds=60,
                    now=reconcile_at,
                )
            )
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(task.status, "retry")
            self.assertEqual(task.attempt_count, 1)
            self.assertEqual(task.max_attempts, 2)
            self.assertEqual(task.error_code, "KNOWLEDGE_STORAGE_CLEANUP_INTERRUPTED")
            self.assertIsNone(task.lease_owner)
            self.assertIsNone(task.lease_token_hash)
            retry_at = task.available_at

        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            self.assertIsNone(
                repo.claim_task(
                    worker_id="worker-2",
                    lease_seconds=10,
                    retry_base_seconds=5,
                    retry_max_seconds=60,
                    now=reconcile_at + timedelta(seconds=1),
                )
            )
            task = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(task.attempt_count, 1)
            self.assertEqual(task.max_attempts, 2)
            self.assertEqual(task.available_at, retry_at)

    def test_confirmed_delete_completes_after_wall_clock_lease_expiry_while_lock_is_held(self):
        registration = self._register(key="knowledge/slow-external-call", hold_seconds=1, max_attempts=1)
        self._activate(registration)
        claim_at = utc_now() + timedelta(seconds=2)
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            first = repo.claim_task(worker_id="worker-1", lease_seconds=10, now=claim_at)
        self.assertIsNotNone(first)

        # prepare_delete 的事务仍持有 task 行锁，外部删除已经确认成功。即使墙钟越过
        # lease，只要 token 未被替换，该持锁事务必须可以完成。
        with self.Session() as db:
            repo = StorageCleanupRepository(db)
            decision, task = repo.prepare_delete(first.task.id, first.lease_token)
            self.assertEqual(decision, "delete")
            task.lease_expires_at = utc_now() - timedelta(seconds=1)
            self.assertTrue(repo.complete_delete(task, first.lease_token))

        with self.Session() as db:
            completed = db.get(KnowledgeStorageCleanupTask, registration.task_id)
            self.assertEqual(completed.status, "completed")
            self.assertEqual(completed.attempt_count, 1)

    async def test_api_rejects_upload_while_same_key_cleanup_is_running(self):
        content = b"knowledge content"
        checksum = hashlib.sha256(content).hexdigest()
        storage_key = f"knowledge/v1/users/user-1/bases/kb-1/objects/{checksum}/busy-generation"
        registration = self._register(key=storage_key, hold_seconds=1)
        self._activate(registration)
        with self.Session() as db:
            claimed = StorageCleanupRepository(db).claim_task(
                worker_id="worker-1",
                lease_seconds=30,
                now=utc_now() + timedelta(seconds=2),
            )
        self.assertIsNotNone(claimed)
        storage = MagicMock()
        storage.upload = AsyncMock(return_value=storage_key)
        service_db = self.Session()
        service = KnowledgeService(
            service_db,
            storage=storage,
            embedding=MagicMock(),
            vector_store=MagicMock(),
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
                patch(
                    "app.services.knowledge.service.uuid.uuid4",
                    side_effect=["busy-generation", "version-1"],
                ),
                patch.object(
                    service,
                    "_require_enabled",
                    return_value=None,
                ),
            ):
                with self.assertRaises(ApiException) as raised:
                    await service.upload_document("user-1", "kb-1", upload)
        finally:
            service_db.close()

        self.assertEqual(raised.exception.code, "KNOWLEDGE_STORAGE_BUSY")
        self.assertEqual(raised.exception.status_code, 503)
        storage.upload.assert_not_awaited()

    async def test_duplicate_integrity_race_persists_cleanup_and_preserves_active_object(self):
        content = b"concurrent content"
        checksum = hashlib.sha256(content).hexdigest()
        losing_key_prefix = f"knowledge/v1/users/user-1/bases/kb-1/objects/{checksum}/"
        storage = MagicMock()
        storage.upload = AsyncMock(side_effect=lambda key, *_args: key)
        storage.exists = AsyncMock(return_value=True)
        storage.delete = AsyncMock(side_effect=[False, True])
        service_db = self.Session()
        service = KnowledgeService(
            service_db,
            storage=storage,
            embedding=MagicMock(),
            vector_store=MagicMock(),
        )
        service.vector_store.collection_name.return_value = "knowledge_v1_d2"

        def concurrent_commit(*_args, **_kwargs):
            service_db.add(
                KnowledgeDocument(
                    id="concurrent-doc",
                    knowledge_base_id="kb-1",
                    user_id="user-1",
                    original_filename="manual.txt",
                    mimetype="text/plain",
                    size=len(content),
                    checksum_sha256=checksum,
                    dedupe_key=checksum,
                    storage_backend="local",
                    storage_key=f"{losing_key_prefix}active-generation",
                    status="queued",
                    parser_version="parser-v1",
                    chunker_version="chunker-v1",
                    embedding_provider="litellm",
                    embedding_model="embed-v1",
                    embedding_revision="embed-r1",
                    embedding_dimension=2,
                    distance_metric="COSINE",
                    desired_index_version="concurrent-version",
                )
            )
            service_db.commit()
            raise IntegrityError("insert", {}, RuntimeError("duplicate"))

        service.repo.create_document_with_task = MagicMock(side_effect=concurrent_commit)
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
                with self.assertRaises(ApiException) as raised:
                    await service.upload_document("user-1", "kb-1", upload)
        finally:
            service_db.close()

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_DUPLICATE")
        with self.Session() as db:
            task = (
                db.query(KnowledgeStorageCleanupTask)
                .filter(
                    KnowledgeStorageCleanupTask.storage_key.like(f"{losing_key_prefix}%"),
                    KnowledgeStorageCleanupTask.storage_key.not_like("%/active-generation"),
                )
                .one()
            )
            self.assertEqual(task.status, "pending")
            task.available_at = utc_now() - timedelta(seconds=1)
            db.commit()

        worker = KnowledgeStorageCleanupWorker(self.Session, worker_id="cleanup-1")
        with patch(
            "app.services.knowledge.storage_cleanup_worker.get_storage_for_backend",
            return_value=storage,
        ):
            self.assertTrue(await worker.run_once())

        self.assertEqual(storage.delete.await_count, 2)
        self.assertTrue(
            all(call.args[0] != f"{losing_key_prefix}active-generation" for call in storage.delete.await_args_list)
        )
        with self.Session() as db:
            task = (
                db.query(KnowledgeStorageCleanupTask)
                .filter(
                    KnowledgeStorageCleanupTask.storage_key.like(f"{losing_key_prefix}%"),
                    KnowledgeStorageCleanupTask.storage_key.not_like("%/active-generation"),
                )
                .one()
            )
            self.assertEqual(task.status, "completed")
            self.assertEqual(db.query(KnowledgeStorageUploadIntent).filter_by(cleanup_task_id=task.id).count(), 0)


if __name__ == "__main__":
    unittest.main()
