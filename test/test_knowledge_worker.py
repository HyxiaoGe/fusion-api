import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.embeddings.base import EmbeddingError
from app.db.database import Base
from app.db.models import (
    KnowledgeBase,
    KnowledgeChunkManifest,
    KnowledgeDocument,
    KnowledgeIndexTask,
    KnowledgeIndexVersion,
    User,
)
from app.services.knowledge.milvus import KnowledgeVectorError
from app.services.knowledge.worker import KnowledgeWorker


class KnowledgeWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            db.add(User(id="user-1", username="worker-user", email="worker@example.com"))
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
            document = KnowledgeDocument(
                id="doc-1",
                knowledge_base_id="kb-1",
                user_id="user-1",
                original_filename="manual.txt",
                mimetype="text/plain",
                size=12,
                checksum_sha256="",
                dedupe_key="checksum",
                storage_backend="local",
                storage_key="knowledge/doc-1",
                status="queued",
                parser_version="parser-v1",
                chunker_version="chunker-v2",
                embedding_provider="litellm",
                embedding_model="embed-v1",
                embedding_revision="embed-r1",
                embedding_dimension=2,
                distance_metric="COSINE",
                desired_index_version="version-1",
            )
            content = "知识库正文。".encode()
            import hashlib

            document.checksum_sha256 = hashlib.sha256(content).hexdigest()
            version = KnowledgeIndexVersion(
                id="version-1",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                user_id="user-1",
                status="building",
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
            )
            task = KnowledgeIndexTask(
                id="task-1",
                knowledge_base_id="kb-1",
                document_id="doc-1",
                user_id="user-1",
                task_type="index_document",
                index_version="version-1",
                max_attempts=2,
            )
            db.add_all([document, version, task])
            db.commit()
        self.content = content
        self.storage = MagicMock()
        self.storage.download = AsyncMock(return_value=content)
        self.storage.exists = AsyncMock(return_value=True)
        self.storage.delete = AsyncMock(return_value=True)
        self.embedding = MagicMock()
        self.embedding.embed = AsyncMock(side_effect=lambda texts, profile: [[1.0, 0.5] for _ in texts])
        self.vector_store = MagicMock()
        self.vector_store.ensure_collection = AsyncMock(return_value="knowledge_v1_d2")
        self.vector_store.upsert_prepared = AsyncMock()
        self.vector_store.delete_index_version = AsyncMock()

    def tearDown(self):
        self.engine.dispose()

    def _set_document_content(self, content: bytes, *, chunk_size: int = 100, chunk_overlap: int = 0):
        import hashlib

        self.content = content
        self.storage.download.return_value = content
        with self.Session() as db:
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            document.size = len(content)
            document.checksum_sha256 = hashlib.sha256(content).hexdigest()
            version = db.query(KnowledgeIndexVersion).filter_by(id="version-1").one()
            version.chunk_size = chunk_size
            version.chunk_overlap = chunk_overlap
            db.commit()

    async def test_index_task_completes_and_activates_version(self):
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage):
            handled = await worker.run_once()

        self.assertTrue(handled)
        self.vector_store.ensure_collection.assert_awaited_once()
        self.vector_store.upsert_prepared.assert_awaited_once()
        with self.Session() as db:
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            task = db.query(KnowledgeIndexTask).filter_by(id="task-1").one()
            version = db.query(KnowledgeIndexVersion).filter_by(id="version-1").one()
            self.assertEqual(document.status, "ready")
            self.assertEqual(document.active_index_version, "version-1")
            self.assertEqual(task.status, "completed")
            self.assertEqual(version.status, "active")
            self.assertGreater(document.chunk_count, 0)

    async def test_legacy_building_v1_fails_closed_and_schedules_cleanup(self):
        with self.Session() as db:
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            version = db.query(KnowledgeIndexVersion).filter_by(id="version-1").one()
            document.chunker_version = "chunker-v1"
            version.chunker_version = "chunker-v1"
            db.commit()
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage):
            handled = await worker.run_once()

        self.assertTrue(handled)
        self.storage.download.assert_not_awaited()
        self.embedding.embed.assert_not_awaited()
        self.vector_store.ensure_collection.assert_not_awaited()
        self.vector_store.upsert_prepared.assert_not_awaited()
        with self.Session() as db:
            original = db.query(KnowledgeIndexTask).filter_by(id="task-1").one()
            cleanup = db.query(KnowledgeIndexTask).filter_by(task_type="delete_index_version").one()
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            version = db.query(KnowledgeIndexVersion).filter_by(id="version-1").one()
            self.assertEqual(original.status, "failed")
            self.assertEqual(original.error_code, "KNOWLEDGE_INDEX_VERSION_UNSUPPORTED")
            self.assertEqual(document.status, "failed")
            self.assertEqual(version.status, "deleting")
            self.assertIn(cleanup.status, {"pending", "retry"})

    async def test_non_retryable_embedding_contract_failure_is_visible(self):
        self.embedding.embed.side_effect = EmbeddingError(
            "KNOWLEDGE_EMBEDDING_DIMENSION_MISMATCH",
            "Embedding 返回维度不一致",
            retryable=False,
        )
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage):
            handled = await worker.run_once()

        self.assertTrue(handled)
        with self.Session() as db:
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            task = db.query(KnowledgeIndexTask).filter_by(id="task-1").one()
            self.assertEqual(document.status, "failed")
            self.assertEqual(task.status, "failed")
            self.assertEqual(task.error_code, "KNOWLEDGE_EMBEDDING_DIMENSION_MISMATCH")
            version_status = db.query(KnowledgeIndexVersion).filter_by(id="version-1").one().status
            self.assertEqual(version_status, "deleting")

    async def test_retryable_vector_failure_is_requeued_without_duplicate_task(self):
        from app.services.knowledge.milvus import KnowledgeVectorError

        self.vector_store.upsert_prepared.side_effect = KnowledgeVectorError(
            "KNOWLEDGE_VECTOR_UNAVAILABLE",
            "Milvus 暂时不可用",
            retryable=True,
        )
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage):
            await worker.run_once()

        with self.Session() as db:
            tasks = db.query(KnowledgeIndexTask).all()
            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].status, "retry")
            self.assertEqual(tasks[0].attempt_count, 1)
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            self.assertEqual(document.status, "queued")

    async def test_lost_lease_never_deletes_vectors_owned_by_reclaimer(self):
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with (
            patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage),
            patch(
                "app.services.knowledge.worker.KnowledgeRepository.finalize_document_index",
                return_value="lease_lost",
            ),
        ):
            await worker.run_once()

        self.vector_store.ensure_collection.assert_awaited_once()
        self.vector_store.upsert_prepared.assert_awaited_once()
        self.vector_store.delete_index_version.assert_not_awaited()

    async def test_embedding_and_vector_write_alternate_in_bounded_batches(self):
        self._set_document_content(b"x" * 450)
        events = []

        async def embed(texts, _profile):
            events.append(("embed", len(texts)))
            return [[1.0, 0.5] for _ in texts]

        async def ensure(_profile):
            events.append(("ensure", 1))
            return "knowledge_v1_d2"

        async def upsert(_profile, collection, records):
            self.assertEqual(collection, "knowledge_v1_d2")
            events.append(("upsert", len(records)))

        self.embedding.embed.side_effect = embed
        self.vector_store.ensure_collection.side_effect = ensure
        self.vector_store.upsert_prepared.side_effect = upsert
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with (
            patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage),
            patch("app.services.knowledge.worker.settings.KNOWLEDGE_EMBEDDING_BATCH_SIZE", 2),
        ):
            await worker.run_once()

        self.assertEqual(
            events,
            [
                ("ensure", 1),
                ("embed", 2),
                ("upsert", 2),
                ("embed", 2),
                ("upsert", 2),
                ("embed", 1),
                ("upsert", 1),
            ],
        )
        self.vector_store.ensure_collection.assert_awaited_once()
        with self.Session() as db:
            self.assertEqual(db.query(KnowledgeChunkManifest).count(), 5)
            self.assertEqual(db.query(KnowledgeDocument).filter_by(id="doc-1").one().status, "ready")

    async def test_non_retryable_later_vector_batch_failure_schedules_partial_write_cleanup(self):
        self._set_document_content(b"x" * 450)
        calls = 0

        async def fail_second_batch(_profile, _collection, _records):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KnowledgeVectorError(
                    "KNOWLEDGE_VECTOR_WRITE_INCOMPLETE",
                    "Milvus 写入完整性校验失败",
                    retryable=False,
                )

        self.vector_store.upsert_prepared.side_effect = fail_second_batch
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with (
            patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage),
            patch("app.services.knowledge.worker.settings.KNOWLEDGE_EMBEDDING_BATCH_SIZE", 2),
        ):
            await worker.run_once()

        self.assertEqual(calls, 2)
        with self.Session() as db:
            original = db.query(KnowledgeIndexTask).filter_by(id="task-1").one()
            cleanup = db.query(KnowledgeIndexTask).filter_by(task_type="delete_index_version").one()
            self.assertEqual(original.status, "failed")
            self.assertEqual(original.error_code, "KNOWLEDGE_VECTOR_WRITE_INCOMPLETE")
            self.assertEqual(cleanup.index_version, "version-1")
            self.assertIn(cleanup.status, {"pending", "retry"})
            self.assertEqual(db.query(KnowledgeDocument).filter_by(id="doc-1").one().status, "failed")

    async def test_chunk_limit_is_stable_non_retryable_failure(self):
        self._set_document_content(b"x" * 301)
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with (
            patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage),
            patch("app.services.knowledge.worker.settings.KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT", 2),
        ):
            await worker.run_once()

        self.embedding.embed.assert_not_awaited()
        self.vector_store.ensure_collection.assert_not_awaited()
        self.vector_store.upsert_prepared.assert_not_awaited()
        with self.Session() as db:
            task = db.query(KnowledgeIndexTask).filter_by(id="task-1").one()
            self.assertEqual(task.status, "failed")
            self.assertEqual(task.error_code, "KNOWLEDGE_DOCUMENT_CHUNK_LIMIT_EXCEEDED")
            self.assertEqual(db.query(KnowledgeChunkManifest).count(), 0)

    async def test_lost_lease_between_batches_stops_new_writes_without_deleting_version(self):
        self._set_document_content(b"x" * 450)
        worker = KnowledgeWorker(
            self.Session,
            worker_id="worker-1",
            embedding=self.embedding,
            vector_store=self.vector_store,
        )

        with (
            patch("app.services.knowledge.worker.get_storage_for_backend", return_value=self.storage),
            patch("app.services.knowledge.worker.settings.KNOWLEDGE_EMBEDDING_BATCH_SIZE", 2),
            patch.object(worker, "_index_write_state", side_effect=["allowed", "allowed", "lease_lost"]),
        ):
            await worker.run_once()

        self.vector_store.ensure_collection.assert_awaited_once()
        self.assertEqual(self.vector_store.upsert_prepared.await_count, 1)
        self.vector_store.delete_index_version.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
