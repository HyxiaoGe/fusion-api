import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.ai.embeddings.base import EmbeddingError
from app.db.database import Base
from app.db.models import KnowledgeBase, KnowledgeDocument, KnowledgeIndexTask, KnowledgeIndexVersion, User
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
                chunker_version="chunker-v1",
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
        self.vector_store.upsert = AsyncMock()
        self.vector_store.delete_index_version = AsyncMock()

    def tearDown(self):
        self.engine.dispose()

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
        self.vector_store.upsert.assert_awaited_once()
        with self.Session() as db:
            document = db.query(KnowledgeDocument).filter_by(id="doc-1").one()
            task = db.query(KnowledgeIndexTask).filter_by(id="task-1").one()
            version = db.query(KnowledgeIndexVersion).filter_by(id="version-1").one()
            self.assertEqual(document.status, "ready")
            self.assertEqual(document.active_index_version, "version-1")
            self.assertEqual(task.status, "completed")
            self.assertEqual(version.status, "active")
            self.assertGreater(document.chunk_count, 0)

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

        self.vector_store.upsert.side_effect = KnowledgeVectorError(
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

        self.vector_store.upsert.assert_awaited_once()
        self.vector_store.delete_index_version.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
