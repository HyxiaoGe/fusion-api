import hashlib
import unittest
from datetime import timedelta
from unittest.mock import patch

from sqlalchemy import create_engine, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.knowledge_repository import (
    KnowledgeBaseLimitExceeded,
    KnowledgeBaseWriteConflict,
    KnowledgeRepository,
)
from app.db.models import (
    KnowledgeBase,
    KnowledgeChunkManifest,
    KnowledgeDocument,
    KnowledgeIndexTask,
    KnowledgeIndexVersion,
    KnowledgeStorageCleanupTask,
    KnowledgeStorageUploadIntent,
    User,
)
from app.db.storage_cleanup_repository import StorageCleanupRepository, StorageUploadFenceConflict
from app.services.knowledge.chunker import KnowledgeChunk
from app.utils.time import utc_now


class KnowledgeRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        event.listen(
            self.engine,
            "connect",
            lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"),
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.db = self.Session()
        self.db.add_all(
            [
                User(id="user-1", username="user-one", email="one@example.com"),
                User(id="user-2", username="user-two", email="two@example.com"),
            ]
        )
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _base(self, *, base_id="kb-1", user_id="user-1", normalized="manual"):
        return KnowledgeBase(
            id=base_id,
            user_id=user_id,
            name="Manual",
            name_normalized=normalized,
            description="",
            business_type="general",
            status="active",
            embedding_provider="litellm",
            embedding_model="embed-v1",
            embedding_revision="embed-r1",
            embedding_dimension=2,
            distance_metric="COSINE",
        )

    def _document_bundle(
        self,
        *,
        document_id="doc-1",
        checksum="a" * 64,
        storage_key=None,
        version_id=None,
    ):
        version_id = version_id or f"version-{document_id}"
        document = KnowledgeDocument(
            id=document_id,
            knowledge_base_id="kb-1",
            user_id="user-1",
            original_filename="manual.txt",
            mimetype="text/plain",
            size=10,
            checksum_sha256=checksum,
            dedupe_key=checksum,
            storage_backend="local",
            storage_key=storage_key or f"knowledge/{document_id}",
            status="queued",
            parser_version="parser-v1",
            chunker_version="chunker-v1",
            embedding_provider="litellm",
            embedding_model="embed-v1",
            embedding_revision="embed-r1",
            embedding_dimension=2,
            distance_metric="COSINE",
            desired_index_version=version_id,
        )
        version = KnowledgeIndexVersion(
            id=version_id,
            knowledge_base_id="kb-1",
            document_id=document_id,
            user_id="user-1",
            status="building",
            parser_version="parser-v1",
            chunker_version="chunker-v1",
            chunk_size=1200,
            chunk_overlap=200,
            embedding_provider="litellm",
            embedding_model="embed-v1",
            embedding_revision="embed-r1",
            embedding_dimension=2,
            distance_metric="COSINE",
            collection_name="knowledge_v1_d2",
        )
        task = KnowledgeIndexTask(
            id=f"task-{document_id}",
            knowledge_base_id="kb-1",
            document_id=document_id,
            user_id="user-1",
            task_type="index_document",
            index_version=version_id,
            max_attempts=3,
        )
        return document, version, task

    @staticmethod
    def _chunks(count, *, prefix="chunk"):
        return [
            KnowledgeChunk(
                chunk_id=hashlib.sha256(f"{prefix}-{index}".encode()).hexdigest(),
                ordinal=index,
                text=f"正文-{index}",
                char_start=index * 10,
                char_end=index * 10 + 4,
                page=None,
                section=None,
            )
            for index in range(count)
        ]

    def test_names_are_unique_per_user_only(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        with self.assertRaises(IntegrityError):
            repo.create_knowledge_base(self._base(base_id="kb-2"))

        other_user = repo.create_knowledge_base(self._base(base_id="kb-3", user_id="user-2"))
        self.assertEqual(other_user.user_id, "user-2")

    def test_create_limit_is_checked_inside_locked_write(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base(), max_bases=1)

        with self.assertRaises(KnowledgeBaseLimitExceeded):
            repo.create_knowledge_base(
                self._base(base_id="kb-2", normalized="another"),
                max_bases=1,
            )

    def test_repository_never_returns_another_users_base(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())

        self.assertIsNone(repo.get_knowledge_base("kb-1", "user-2"))

    def test_ready_documents_load_active_versions_in_one_query_and_preserve_scope(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        expected_versions = {}
        for index in range(6):
            document, version, task = self._document_bundle(
                document_id=f"doc-{index}",
                checksum=f"{index:064x}",
            )
            repo.create_document_with_task(document, version, task)
            document.status = "ready"
            document.active_index_version = version.id
            version.status = "active"
            expected_versions[document.id] = version.id

        inactive_document, inactive_version, inactive_task = self._document_bundle(
            document_id="doc-inactive-version",
            checksum="f" * 64,
        )
        repo.create_document_with_task(inactive_document, inactive_version, inactive_task)
        inactive_document.status = "ready"
        inactive_document.active_index_version = inactive_version.id
        inactive_version.status = "superseded"

        deleted_version_document, deleted_version, deleted_version_task = self._document_bundle(
            document_id="doc-deleted-version",
            checksum="e" * 64,
        )
        repo.create_document_with_task(deleted_version_document, deleted_version, deleted_version_task)
        deleted_version_document.status = "ready"
        deleted_version_document.active_index_version = deleted_version.id
        deleted_version.status = "active"
        deleted_version.deleted_at = utc_now()
        self.db.commit()
        self.db.expire_all()

        statements = []

        def count_selects(_connection, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(self.engine, "before_cursor_execute", count_selects)
        try:
            rows = repo.get_ready_documents("user-1", ["kb-1"])
            loaded_versions = {row.document.id: row.active_version.id for row in rows}
        finally:
            event.remove(self.engine, "before_cursor_execute", count_selects)

        self.assertEqual(loaded_versions, expected_versions)
        self.assertEqual(len(statements), 1)
        self.assertEqual(repo.get_ready_documents("user-2", ["kb-1"]), [])

    def test_replace_chunk_manifest_uses_bounded_core_batches(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        claimed = repo.claim_task(worker_id="worker-1", lease_seconds=60)
        insert_sizes = []

        def capture_insert(_connection, _cursor, statement, parameters, _context, executemany):
            if statement.lstrip().upper().startswith("INSERT INTO KNOWLEDGE_CHUNK_MANIFESTS"):
                insert_sizes.append(len(parameters) if executemany else 1)

        event.listen(self.engine, "before_cursor_execute", capture_insert)
        try:
            replaced = repo.replace_chunk_manifest(
                task_id=task.id,
                lease_token=claimed.lease_token,
                version=version,
                filename=document.original_filename,
                chunks=self._chunks(5),
                batch_size=2,
            )
        finally:
            event.remove(self.engine, "before_cursor_execute", capture_insert)

        self.assertTrue(replaced)
        self.assertEqual(insert_sizes, [2, 2, 1])
        self.assertEqual(self.db.query(KnowledgeChunkManifest).count(), 5)

    def test_replace_chunk_manifest_lost_lease_never_writes(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        repo.claim_task(worker_id="worker-1", lease_seconds=60)
        insert_count = 0

        def capture_insert(_connection, _cursor, statement, _parameters, _context, _executemany):
            nonlocal insert_count
            if statement.lstrip().upper().startswith("INSERT INTO KNOWLEDGE_CHUNK_MANIFESTS"):
                insert_count += 1

        event.listen(self.engine, "before_cursor_execute", capture_insert)
        try:
            replaced = repo.replace_chunk_manifest(
                task_id=task.id,
                lease_token="invalid-lease-token",
                version=version,
                filename=document.original_filename,
                chunks=self._chunks(3),
                batch_size=2,
            )
        finally:
            event.remove(self.engine, "before_cursor_execute", capture_insert)

        self.assertFalse(replaced)
        self.assertEqual(insert_count, 0)
        self.assertEqual(self.db.query(KnowledgeChunkManifest).count(), 0)

    def test_replace_chunk_manifest_rolls_back_all_batches_on_later_failure(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        claimed = repo.claim_task(worker_id="worker-1", lease_seconds=60)
        old_chunks = self._chunks(1, prefix="old")
        self.assertTrue(
            repo.replace_chunk_manifest(
                task_id=task.id,
                lease_token=claimed.lease_token,
                version=version,
                filename=document.original_filename,
                chunks=old_chunks,
                batch_size=1,
            )
        )
        insert_number = 0

        def fail_second_insert(_connection, _cursor, statement, _parameters, _context, _executemany):
            nonlocal insert_number
            if not statement.lstrip().upper().startswith("INSERT INTO KNOWLEDGE_CHUNK_MANIFESTS"):
                return
            insert_number += 1
            if insert_number == 2:
                raise RuntimeError("模拟后续 manifest 批次写入失败")

        event.listen(self.engine, "before_cursor_execute", fail_second_insert)
        try:
            with self.assertRaisesRegex(RuntimeError, "后续 manifest 批次"):
                repo.replace_chunk_manifest(
                    task_id=task.id,
                    lease_token=claimed.lease_token,
                    version=version,
                    filename=document.original_filename,
                    chunks=self._chunks(5, prefix="new"),
                    batch_size=2,
                )
        finally:
            event.remove(self.engine, "before_cursor_execute", fail_second_insert)

        remaining = self.db.query(KnowledgeChunkManifest).all()
        self.assertEqual([row.chunk_id for row in remaining], [old_chunks[0].chunk_id])

    def test_soft_deleted_document_releases_checksum_for_new_upload(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        delete_task = repo.enqueue_document_delete(document, max_attempts=3)
        claimed = repo.claim_task(worker_id="worker-1", lease_seconds=60)

        self.assertEqual(claimed.task.id, delete_task.id)
        self.assertTrue(
            repo.mark_document_deleted(
                "doc-1",
                task_id=delete_task.id,
                lease_token=claimed.lease_token,
            )
        )
        replacement, replacement_version, replacement_task = self._document_bundle(document_id="doc-2")
        saved, _version, _task = repo.create_document_with_task(
            replacement,
            replacement_version,
            replacement_task,
        )

        self.assertEqual(saved.checksum_sha256, "a" * 64)

    def test_document_transaction_does_not_refresh_after_commit(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()

        with patch.object(self.db, "refresh", side_effect=RuntimeError("post-commit refresh failed")) as refresh:
            saved_document, saved_version, saved_task = repo.create_document_with_task(document, version, task)

        refresh.assert_not_called()
        self.assertEqual(saved_document.id, document.id)
        self.assertEqual(saved_version.id, version.id)
        self.assertEqual(saved_task.id, task.id)
        self.assertEqual(self.db.query(KnowledgeDocument).filter_by(id=document.id).count(), 1)

    def test_retry_document_returns_without_any_post_commit_query(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        document.status = "failed"
        version.status = "failed"
        self.db.commit()
        _unused, retry_version, retry_task = self._document_bundle(
            document_id=document.id,
            version_id="version-retry",
        )
        retry_task.id = "task-retry"
        committed = False

        def mark_committed(_session):
            nonlocal committed
            committed = True

        def reject_post_commit_sql(*_args):
            if committed:
                raise AssertionError("重试事务提交后禁止继续查询")

        self.db.expire_on_commit = True
        event.listen(self.db, "after_commit", mark_committed)
        event.listen(self.engine, "before_cursor_execute", reject_post_commit_sql)
        try:
            saved_task = repo.retry_document(document, retry_version, retry_task)
            self.assertEqual(saved_task.id, "task-retry")
            self.assertEqual(saved_task.status, "pending")
        finally:
            event.remove(self.engine, "before_cursor_execute", reject_post_commit_sql)
            event.remove(self.db, "after_commit", mark_committed)
            self.db.expire_on_commit = False

        with self.Session() as verification_db:
            persisted = verification_db.get(KnowledgeIndexTask, "task-retry")
            self.assertIsNotNone(persisted)
            self.assertEqual(persisted.status, "pending")

    def test_soft_delete_tombstones_fit_database_columns(self):
        repo = KnowledgeRepository(self.db)
        knowledge_base = repo.create_knowledge_base(self._base(normalized="名" * 200))
        document, version, task = self._document_bundle(
            document_id="doc-" + "x" * 120,
            version_id="version-tombstone",
        )
        repo.create_document_with_task(document, version, task)
        document_delete_task = repo.enqueue_document_delete(document, max_attempts=3)
        document_claim = repo.claim_task(worker_id="worker-1", lease_seconds=60)

        self.assertEqual(document_claim.task.id, document_delete_task.id)
        self.assertTrue(
            repo.mark_document_deleted(
                document.id,
                task_id=document_delete_task.id,
                lease_token=document_claim.lease_token,
            )
        )
        self.assertTrue(repo.complete_task(document_delete_task.id, document_claim.lease_token))
        self.db.refresh(document)
        self.assertIn("#deleted#", document.dedupe_key)
        self.assertLessEqual(len(document.dedupe_key), 120)

        base_delete_task = repo.enqueue_knowledge_base_delete(knowledge_base, max_attempts=3)
        base_claim = repo.claim_task(worker_id="worker-1", lease_seconds=60)
        self.assertEqual(base_claim.task.id, base_delete_task.id)
        self.assertTrue(
            repo.mark_knowledge_base_deleted(
                knowledge_base.id,
                task_id=base_delete_task.id,
                lease_token=base_claim.lease_token,
            )
        )
        self.db.refresh(knowledge_base)
        self.assertIn("#deleted#", knowledge_base.name_normalized)
        self.assertLessEqual(len(knowledge_base.name_normalized), 200)

    def test_document_limit_detects_duplicate_checksum_and_cleans_its_generation(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        shared_key = "knowledge/v1/users/user-1/bases/kb-1/objects/" + "a" * 64 + "/doc-1"
        document, version, task = self._document_bundle(storage_key=shared_key)
        repo.create_document_with_task(document, version, task, max_documents=1)

        shared_document, shared_version, shared_task = self._document_bundle(
            document_id="doc-shared",
            checksum="a" * 64,
            storage_key="knowledge/v1/users/user-1/bases/kb-1/objects/" + "a" * 64 + "/doc-shared",
        )
        with self.assertRaises(KnowledgeBaseWriteConflict) as shared_conflict:
            repo.create_document_with_task(
                shared_document,
                shared_version,
                shared_task,
                max_documents=1,
            )
        self.assertTrue(shared_conflict.exception.cleanup_storage)
        self.assertTrue(shared_conflict.exception.duplicate)

        unreferenced_document, unreferenced_version, unreferenced_task = self._document_bundle(
            document_id="doc-unreferenced",
            checksum="b" * 64,
            storage_key="knowledge/v1/users/user-1/bases/kb-1/objects/" + "b" * 64 + "/doc-unreferenced",
        )
        with self.assertRaises(KnowledgeBaseWriteConflict) as unreferenced_conflict:
            repo.create_document_with_task(
                unreferenced_document,
                unreferenced_version,
                unreferenced_task,
                max_documents=1,
            )
        self.assertTrue(unreferenced_conflict.exception.cleanup_storage)
        self.assertFalse(unreferenced_conflict.exception.duplicate)

    def test_create_document_refreshes_locked_base_from_database(self):
        repo = KnowledgeRepository(self.db)
        stale_base = repo.create_knowledge_base(self._base())
        self.assertEqual(stale_base.status, "active")
        with self.Session() as concurrent_db:
            concurrent_db.query(KnowledgeBase).filter_by(id=stale_base.id).update(
                {"status": "deleting"},
                synchronize_session=False,
            )
            concurrent_db.commit()
        self.assertEqual(stale_base.status, "active")
        document, version, task = self._document_bundle()

        with self.assertRaises(KnowledgeBaseWriteConflict):
            repo.create_document_with_task(document, version, task)

        self.assertEqual(repo.count_documents(stale_base.id), 0)

    def test_document_write_atomically_consumes_succeeded_upload_fence(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle(storage_key="knowledge/atomic-fence")
        cleanup_repo = StorageCleanupRepository(self.db)
        registration = cleanup_repo.register_upload_intent(
            storage_backend="local",
            storage_key=document.storage_key,
            hold_seconds=60,
            max_attempts=3,
        )
        self.assertTrue(
            cleanup_repo.record_upload_outcome(
                registration,
                outcome="succeeded",
                hold_seconds=60,
            )
        )

        committed_during_flush = []

        def capture(_connection, _cursor, statement, _parameters, _context, _executemany):
            if statement.upper().startswith("INSERT INTO KNOWLEDGE_DOCUMENTS"):
                committed_during_flush.append(self.db.in_transaction())

        event.listen(self.engine, "before_cursor_execute", capture)
        try:
            saved, _, _ = repo.create_document_with_task(
                document,
                version,
                task,
                upload_registration=registration,
            )
        finally:
            event.remove(self.engine, "before_cursor_execute", capture)

        self.assertEqual(saved.id, document.id)
        cleanup_task = self.db.get(KnowledgeStorageCleanupTask, registration.task_id)
        self.assertEqual(cleanup_task.status, "completed")
        self.assertEqual(self.db.query(KnowledgeStorageUploadIntent).count(), 0)
        self.assertEqual(committed_during_flush, [True])

    def test_document_write_rejects_non_succeeded_upload_fence(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle(storage_key="knowledge/uncertain-fence")
        registration = StorageCleanupRepository(self.db).register_upload_intent(
            storage_backend="local",
            storage_key=document.storage_key,
            hold_seconds=60,
            max_attempts=3,
        )

        with self.assertRaises(StorageUploadFenceConflict):
            repo.create_document_with_task(
                document,
                version,
                task,
                upload_registration=registration,
            )

        self.assertEqual(repo.count_documents("kb-1"), 0)

    def test_document_write_rejects_upload_fence_for_another_storage_generation(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle(storage_key="knowledge/document-generation")
        cleanup_repo = StorageCleanupRepository(self.db)
        registration = cleanup_repo.register_upload_intent(
            storage_backend="local",
            storage_key="knowledge/another-generation",
            hold_seconds=60,
            max_attempts=3,
        )
        self.assertTrue(
            cleanup_repo.record_upload_outcome(
                registration,
                outcome="succeeded",
                hold_seconds=60,
            )
        )

        with self.assertRaises(StorageUploadFenceConflict):
            repo.create_document_with_task(
                document,
                version,
                task,
                upload_registration=registration,
            )

        self.assertEqual(repo.count_documents("kb-1"), 0)

    def test_expired_exhausted_candidates_use_postgres_skip_locked(self):
        repo = KnowledgeRepository(self.db)
        now = utc_now()

        with patch.object(self.engine.dialect, "name", "postgresql"):
            statement = repo._expired_exhausted_tasks_query(now).statement

        sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("KNOWLEDGE_INDEX_TASKS.STATUS =", sql)
        self.assertIn("KNOWLEDGE_INDEX_TASKS.LEASE_EXPIRES_AT <", sql)
        self.assertIn("KNOWLEDGE_INDEX_TASKS.ATTEMPT_COUNT >= KNOWLEDGE_INDEX_TASKS.MAX_ATTEMPTS", sql)

    def test_failed_cleanup_reconciler_uses_postgres_skip_locked(self):
        repo = KnowledgeRepository(self.db)

        with patch.object(self.engine.dialect, "name", "postgresql"):
            statement = repo._failed_index_cleanup_tasks_query().statement

        sql = str(statement.compile(dialect=postgresql.dialect())).upper()
        self.assertIn("FOR UPDATE SKIP LOCKED", sql)
        self.assertIn("KNOWLEDGE_INDEX_TASKS.TASK_TYPE =", sql)
        self.assertIn("KNOWLEDGE_INDEX_TASKS.STATUS =", sql)
        self.assertIn("KNOWLEDGE_INDEX_TASKS.INDEX_VERSION IS NOT NULL", sql)

    def test_expired_lease_is_reclaimed_and_old_token_is_fenced(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        first = repo.claim_task(worker_id="worker-1", lease_seconds=60)
        self.assertIsNotNone(first)
        task_row = self.db.query(KnowledgeIndexTask).filter_by(id=task.id).one()
        task_row.lease_expires_at = utc_now() - timedelta(seconds=1)
        self.db.commit()

        second = repo.claim_task(worker_id="worker-2", lease_seconds=60)

        self.assertIsNotNone(second)
        self.assertNotEqual(first.lease_token, second.lease_token)
        self.assertFalse(repo.complete_task(task.id, first.lease_token))
        self.assertTrue(repo.complete_task(task.id, second.lease_token))

    def test_claim_lease_uses_fresh_clock_after_task_lock(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        query_started_at = utc_now() + timedelta(seconds=1)
        lock_acquired_at = query_started_at + timedelta(seconds=9)

        with patch(
            "app.db.knowledge_repository.utc_now",
            side_effect=(query_started_at, lock_acquired_at),
        ) as clock:
            claimed = repo.claim_task(worker_id="worker-1", lease_seconds=10)

        self.assertIsNotNone(claimed)
        self.assertEqual(clock.call_count, 2)
        self.assertEqual(claimed.task.heartbeat_at, lock_acquired_at)
        self.assertEqual(
            claimed.task.lease_expires_at,
            lock_acquired_at + timedelta(seconds=10),
        )

    def test_claim_returns_committed_token_without_post_commit_refresh(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, _version, task = self._document_bundle()
        repo.create_document_with_task(document, _version, task)

        with patch.object(self.db, "refresh", side_effect=RuntimeError("post-commit read failed")) as refresh:
            claimed = repo.claim_task(worker_id="worker-no-refresh", lease_seconds=60)

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.task.status, "running")
        self.assertEqual(claimed.task.lease_owner, "worker-no-refresh")
        self.assertTrue(claimed.lease_token)
        refresh.assert_not_called()

    def test_index_status_updates_cannot_overwrite_deleting_document(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        claimed = repo.claim_task(worker_id="worker-delete-race", lease_seconds=60)
        document.status = "deleting"
        self.db.commit()

        self.assertTrue(repo.update_task_phase(task.id, claimed.lease_token, "writing"))
        self.db.refresh(document)
        self.assertEqual(document.status, "deleting")

        self.assertTrue(
            repo.fail_task(
                task.id,
                claimed.lease_token,
                error_code="KNOWLEDGE_VECTOR_UNAVAILABLE",
                error_summary="Milvus 暂时不可用",
                retryable=True,
                retry_delay_seconds=1,
            )
        )
        self.db.refresh(document)
        self.assertEqual(document.status, "deleting")

    def test_renew_lease_reads_clock_only_after_task_lock(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        claimed = repo.claim_task(
            worker_id="worker-1",
            lease_seconds=60,
            now=utc_now() + timedelta(seconds=1),
        )
        renewed_at = utc_now() + timedelta(seconds=10)
        task_locked = False
        original_lock_task = repo._lock_task

        def lock_task(task_id):
            nonlocal task_locked
            row = original_lock_task(task_id)
            task_locked = True
            return row

        def locked_clock():
            self.assertTrue(task_locked)
            return renewed_at

        with (
            patch.object(repo, "_lock_task", side_effect=lock_task),
            patch("app.db.knowledge_repository.utc_now", side_effect=locked_clock),
        ):
            self.assertTrue(repo.renew_lease(task.id, claimed.lease_token, lease_seconds=30))

        self.db.refresh(task)
        self.assertEqual(task.heartbeat_at, renewed_at.replace(tzinfo=None))
        self.assertEqual(
            task.lease_expires_at,
            (renewed_at + timedelta(seconds=30)).replace(tzinfo=None),
        )

    def test_expired_index_task_waits_for_external_write_grace_before_reclaim(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        started_at = utc_now() + timedelta(seconds=1)
        first = repo.claim_task(
            worker_id="worker-1",
            lease_seconds=10,
            external_write_grace_seconds=20,
            now=started_at,
        )
        self.assertIsNotNone(first)
        lease_expires_at = first.task.lease_expires_at

        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-2",
                lease_seconds=10,
                external_write_grace_seconds=20,
                now=lease_expires_at + timedelta(seconds=1),
            )
        )
        self.db.refresh(task)
        self.assertEqual(task.status, "running")
        self.assertEqual(task.attempt_count, 1)
        self.assertEqual(task.lease_owner, "worker-1")

        second = repo.claim_task(
            worker_id="worker-2",
            lease_seconds=10,
            external_write_grace_seconds=20,
            now=lease_expires_at + timedelta(seconds=21),
        )

        self.assertIsNotNone(second)
        self.assertEqual(second.task.id, task.id)
        self.assertEqual(second.task.attempt_count, 2)
        self.assertFalse(repo.complete_task(task.id, first.lease_token))

    def test_index_cleanup_waits_for_running_index_grace_and_reclaim_convergence(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, index_task = self._document_bundle()
        repo.create_document_with_task(document, version, index_task)
        started_at = utc_now() + timedelta(seconds=1)
        first = repo.claim_task(
            worker_id="index-worker-1",
            lease_seconds=10,
            external_write_grace_seconds=20,
            now=started_at,
        )
        self.assertIsNotNone(first)
        cleanup_task = KnowledgeIndexTask(
            id="cleanup-task-running-index",
            knowledge_base_id=document.knowledge_base_id,
            document_id=document.id,
            user_id=document.user_id,
            task_type="delete_index_version",
            index_version=version.id,
            status="pending",
            phase="queued",
            max_attempts=3,
            available_at=started_at,
        )
        self.db.add(cleanup_task)
        self.db.commit()
        lease_expires_at = first.task.lease_expires_at

        self.assertIsNone(
            repo.claim_task(
                worker_id="cleanup-worker-early",
                lease_seconds=10,
                external_write_grace_seconds=20,
                now=lease_expires_at + timedelta(seconds=1),
            )
        )

        reclaimed = repo.claim_task(
            worker_id="index-worker-2",
            lease_seconds=10,
            external_write_grace_seconds=20,
            now=lease_expires_at + timedelta(seconds=21),
        )
        self.assertIsNotNone(reclaimed)
        self.assertEqual(reclaimed.task.id, index_task.id)
        self.assertTrue(repo.complete_task(index_task.id, reclaimed.lease_token))

        cleanup = repo.claim_task(
            worker_id="cleanup-worker-late",
            lease_seconds=10,
            external_write_grace_seconds=20,
            now=lease_expires_at + timedelta(seconds=22),
        )
        self.assertIsNotNone(cleanup)
        self.assertEqual(cleanup.task.id, cleanup_task.id)

    def test_failed_index_cleanup_is_periodically_reconciled_with_bounded_retry(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, index_task = self._document_bundle()
        repo.create_document_with_task(document, version, index_task)
        now = utc_now() + timedelta(seconds=1)
        index_task.status = "cancelled"
        index_task.completed_at = now
        version.status = "deleting"
        cleanup_task = KnowledgeIndexTask(
            id="cleanup-task-1",
            knowledge_base_id=document.knowledge_base_id,
            document_id=document.id,
            user_id=document.user_id,
            task_type="delete_index_version",
            index_version=version.id,
            status="failed",
            phase="deleting",
            attempt_count=3,
            max_attempts=3,
            available_at=now,
            completed_at=now,
            error_code="KNOWLEDGE_VECTOR_UNAVAILABLE",
            error_summary="Milvus 暂时不可用",
        )
        self.db.add(cleanup_task)
        self.db.commit()

        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-1",
                lease_seconds=60,
                external_write_grace_seconds=10,
                now=now,
            )
        )
        self.db.refresh(cleanup_task)
        self.assertEqual(cleanup_task.status, "retry")
        self.assertEqual(cleanup_task.attempt_count, 3)
        self.assertEqual(cleanup_task.max_attempts, 4)
        self.assertIsNone(cleanup_task.completed_at)
        retry_delay = cleanup_task.available_at - now.replace(tzinfo=None)
        self.assertGreater(retry_delay, timedelta(0))
        self.assertLessEqual(retry_delay, timedelta(hours=1))

        claimed = repo.claim_task(
            worker_id="worker-2",
            lease_seconds=60,
            external_write_grace_seconds=10,
            now=cleanup_task.available_at + timedelta(seconds=1),
        )
        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.task.id, cleanup_task.id)
        self.assertEqual(claimed.task.attempt_count, 4)
        self.assertTrue(
            repo.fail_task(
                cleanup_task.id,
                claimed.lease_token,
                error_code="KNOWLEDGE_VECTOR_UNAVAILABLE",
                error_summary="Milvus 暂时不可用",
                retryable=True,
                retry_delay_seconds=1,
            )
        )
        self.db.refresh(cleanup_task)
        self.assertEqual(cleanup_task.status, "failed")

        next_reconcile_at = cleanup_task.available_at + timedelta(days=1)
        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-3",
                lease_seconds=60,
                external_write_grace_seconds=10,
                now=next_reconcile_at,
            )
        )
        self.db.refresh(cleanup_task)
        self.assertEqual(cleanup_task.status, "retry")
        self.assertEqual(cleanup_task.attempt_count, 4)
        self.assertEqual(cleanup_task.max_attempts, 5)

    def test_terminal_versions_do_not_starve_failed_cleanup_reconciliation_limit(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, live_version, index_task = self._document_bundle(version_id="z-live-version")
        repo.create_document_with_task(document, live_version, index_task)
        now = utc_now() + timedelta(seconds=1)
        index_task.status = "cancelled"
        index_task.completed_at = now
        live_version.status = "deleting"
        live_cleanup = KnowledgeIndexTask(
            id="z-live-cleanup",
            knowledge_base_id=document.knowledge_base_id,
            document_id=document.id,
            user_id=document.user_id,
            task_type="delete_index_version",
            index_version=live_version.id,
            status="failed",
            phase="deleting",
            attempt_count=3,
            max_attempts=3,
            available_at=now,
            completed_at=now,
        )
        self.db.add(live_cleanup)
        terminal_versions = []
        terminal_tasks = []
        for index in range(100):
            version_id = f"a-terminal-version-{index:03d}"
            terminal_versions.append(
                KnowledgeIndexVersion(
                    id=version_id,
                    knowledge_base_id=document.knowledge_base_id,
                    document_id=document.id,
                    user_id=document.user_id,
                    status="deleted",
                    parser_version="parser-v1",
                    chunker_version="chunker-v2",
                    chunk_size=1200,
                    chunk_overlap=200,
                    embedding_provider="litellm",
                    embedding_model="embed-v1",
                    embedding_revision="embed-r1",
                    embedding_dimension=2,
                    distance_metric="COSINE",
                    collection_name="knowledge_v1_d2",
                    deleted_at=now,
                )
            )
            terminal_tasks.append(
                KnowledgeIndexTask(
                    id=f"a-terminal-cleanup-{index:03d}",
                    knowledge_base_id=document.knowledge_base_id,
                    document_id=document.id,
                    user_id=document.user_id,
                    task_type="delete_index_version",
                    index_version=version_id,
                    status="failed",
                    phase="deleting",
                    attempt_count=3,
                    max_attempts=3,
                    available_at=now,
                    completed_at=now,
                )
            )
        self.db.add_all(terminal_versions)
        self.db.flush()
        self.db.add_all(terminal_tasks)
        self.db.commit()

        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-1",
                lease_seconds=60,
                external_write_grace_seconds=10,
                now=now,
            )
        )

        self.db.refresh(live_cleanup)
        self.assertEqual(live_cleanup.status, "retry")
        self.assertEqual(live_cleanup.max_attempts, 4)

    def test_document_delete_waits_for_running_index_task(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        index_claim = repo.claim_task(worker_id="index-worker", lease_seconds=60)
        self.assertIsNotNone(index_claim)
        delete_task = repo.enqueue_document_delete(document, max_attempts=3)

        self.assertIsNone(repo.claim_task(worker_id="delete-worker", lease_seconds=60))
        self.assertTrue(repo.complete_task(task.id, index_claim.lease_token))

        delete_claim = repo.claim_task(worker_id="delete-worker", lease_seconds=60)
        self.assertIsNotNone(delete_claim)
        self.assertEqual(delete_claim.task.id, delete_task.id)

    def test_old_version_cannot_activate_after_desired_version_changes(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        repo.create_document_with_task(document, version, task)
        claimed = repo.claim_task(worker_id="worker-1", lease_seconds=60)
        document.desired_index_version = "new-version"
        self.db.commit()

        activated = repo.finalize_document_index(
            task_id=task.id,
            lease_token=claimed.lease_token,
            document_id=document.id,
            index_version=version.id,
            chunk_count=2,
            cleanup_max_attempts=3,
        )

        self.assertEqual(activated, "stale")
        self.assertIsNone(document.active_index_version)

    def test_last_expired_attempt_marks_document_and_version_failed(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        document, version, task = self._document_bundle()
        task.max_attempts = 1
        repo.create_document_with_task(document, version, task)
        claimed = repo.claim_task(worker_id="worker-1", lease_seconds=60)
        lease_expires_at = utc_now() - timedelta(seconds=1)
        claimed.task.lease_expires_at = lease_expires_at
        self.db.commit()

        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-2",
                lease_seconds=60,
                external_write_grace_seconds=20,
                now=lease_expires_at + timedelta(seconds=1),
            )
        )
        self.db.refresh(task)
        self.db.refresh(document)
        self.db.refresh(version)
        self.assertEqual(task.status, "running")
        self.assertEqual(document.status, "queued")
        self.assertEqual(version.status, "building")

        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-2",
                lease_seconds=60,
                external_write_grace_seconds=20,
                now=lease_expires_at + timedelta(seconds=21),
            )
        )

        self.db.refresh(task)
        self.db.refresh(document)
        self.db.refresh(version)
        self.assertEqual(task.status, "failed")
        self.assertIsNone(task.lease_token_hash)
        self.assertEqual(document.status, "failed")
        self.assertEqual(version.status, "deleting")
        self.assertEqual(task.error_code, "KNOWLEDGE_TASK_RETRY_EXHAUSTED")
        cleanup = (
            self.db.query(KnowledgeIndexTask)
            .filter_by(task_type="delete_index_version", index_version=version.id)
            .one()
        )
        self.assertEqual(cleanup.status, "pending")


if __name__ == "__main__":
    unittest.main()
