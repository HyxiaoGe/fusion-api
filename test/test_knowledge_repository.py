import unittest
from datetime import timedelta

from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.knowledge_repository import (
    KnowledgeBaseLimitExceeded,
    KnowledgeBaseWriteConflict,
    KnowledgeRepository,
)
from app.db.models import KnowledgeBase, KnowledgeDocument, KnowledgeIndexTask, KnowledgeIndexVersion, User
from app.utils.time import utc_now


class KnowledgeRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
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

    def test_document_limit_cleans_only_unreferenced_content_addressed_object(self):
        repo = KnowledgeRepository(self.db)
        repo.create_knowledge_base(self._base())
        shared_key = "knowledge/v1/users/user-1/bases/kb-1/objects/" + "a" * 64
        document, version, task = self._document_bundle(storage_key=shared_key)
        repo.create_document_with_task(document, version, task, max_documents=1)

        shared_document, shared_version, shared_task = self._document_bundle(
            document_id="doc-shared",
            checksum="a" * 64,
            storage_key=shared_key,
        )
        with self.assertRaises(KnowledgeBaseWriteConflict) as shared_conflict:
            repo.create_document_with_task(
                shared_document,
                shared_version,
                shared_task,
                max_documents=1,
            )
        self.assertFalse(shared_conflict.exception.cleanup_storage)
        self.assertTrue(shared_conflict.exception.duplicate)

        unreferenced_document, unreferenced_version, unreferenced_task = self._document_bundle(
            document_id="doc-unreferenced",
            checksum="b" * 64,
            storage_key="knowledge/v1/users/user-1/bases/kb-1/objects/" + "b" * 64,
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
        claimed.task.lease_expires_at = utc_now() - timedelta(seconds=1)
        self.db.commit()

        self.assertIsNone(
            repo.claim_task(
                worker_id="worker-2",
                lease_seconds=60,
                external_write_grace_seconds=20,
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
