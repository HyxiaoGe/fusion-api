import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers

from app.db.database import Base
from app.db.models import User
from app.schemas.knowledge import KnowledgeBaseCreate, KnowledgeRetrievalRequest
from app.schemas.response import ApiException
from app.services.knowledge.milvus import KnowledgeVectorHit
from app.services.knowledge.service import KnowledgeService


class KnowledgeServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.db = sessionmaker(bind=self.engine, expire_on_commit=False)()
        self.db.add_all(
            [
                User(id="user-1", username="service-one", email="one-service@example.com"),
                User(id="user-2", username="service-two", email="two-service@example.com"),
            ]
        )
        self.db.commit()
        self.storage = MagicMock()
        self.storage.upload = AsyncMock(return_value="key")
        self.storage.delete = AsyncMock(return_value=True)
        self.embedding = MagicMock()
        self.embedding.embed = AsyncMock(return_value=[[1.0, 0.5]])
        self.vector_store = MagicMock()
        self.vector_store.collection_name.return_value = "knowledge_v1_d2"
        self.vector_store.search = AsyncMock(return_value=[])
        self.service = KnowledgeService(
            self.db,
            storage=self.storage,
            embedding=self.embedding,
            vector_store=self.vector_store,
        )
        self.settings_patchers = [
            patch("app.services.knowledge.service.settings.KNOWLEDGE_BASE_ENABLED", True),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_PROVIDER", "litellm"),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_MODEL", "embed-v1"),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_REVISION", "embed-r1"),
            patch("app.services.knowledge.service.settings.LITELLM_PROXY_URL", "http://litellm-proxy:4000"),
            patch("app.services.knowledge.service.settings.LITELLM_API_KEY", "proxy-key"),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_DIMENSION", 2),
            patch(
                "app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS",
                "2",
            ),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_DISTANCE_METRIC", "COSINE"),
            patch("app.services.knowledge.service.settings.MILVUS_URI", "http://milvus:19530"),
            patch("app.services.knowledge.service.settings.MILVUS_USERNAME", "fusion_knowledge"),
            patch("app.services.knowledge.service.settings.MILVUS_PASSWORD", "secret"),
            patch("app.services.knowledge.service.settings.MILVUS_DATABASE", "fusion_knowledge"),
            patch("app.services.knowledge.service.settings.MILVUS_COLLECTION_PREFIX", "fusion_knowledge"),
            patch(
                "app.services.knowledge.service.settings.KNOWLEDGE_ALLOWED_MIME_TYPES",
                "text/plain",
            ),
        ]
        for patcher in self.settings_patchers:
            patcher.start()

    def tearDown(self):
        for patcher in reversed(self.settings_patchers):
            patcher.stop()
        self.db.close()
        self.engine.dispose()

    def _create_base(self):
        return self.service.create_knowledge_base(
            "user-1",
            KnowledgeBaseCreate(name="  产品 手册  ", description="说明", business_type="product"),
        )

    def test_crud_is_user_scoped_and_name_is_normalized(self):
        created = self._create_base()

        self.assertEqual(created.name, "产品 手册")
        with self.assertRaises(ApiException) as raised:
            self.service.get_knowledge_base("user-2", created.id)
        self.assertEqual(raised.exception.status_code, 404)

        with self.assertRaises(ApiException) as duplicate:
            self.service.create_knowledge_base(
                "user-1",
                KnowledgeBaseCreate(name="产品   手册", description="", business_type="product"),
            )
        self.assertEqual(duplicate.exception.status_code, 409)

    def test_repeated_delete_requeues_after_terminal_failure(self):
        created = self._create_base()
        first = self.service.delete_knowledge_base("user-1", created.id)
        task = self.service._latest_task(
            knowledge_base_id=created.id,
            task_type="delete_knowledge_base",
        )
        task.status = "failed"
        self.db.commit()

        second = self.service.delete_knowledge_base("user-1", created.id)

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(second.status, "pending")

    async def test_upload_persists_task_and_duplicate_returns_stable_conflict(self):
        knowledge_base = self._create_base()

        def upload_file():
            return UploadFile(
                filename="manual.txt",
                file=io.BytesIO(b"knowledge content"),
                headers=Headers({"content-type": "text/plain"}),
            )

        result = await self.service.upload_document("user-1", knowledge_base.id, upload_file())
        self.assertEqual(result.document.status, "queued")
        self.assertEqual(result.task.status, "pending")
        self.storage.upload.assert_awaited_once()

        with self.assertRaises(ApiException) as duplicate:
            await self.service.upload_document("user-1", knowledge_base.id, upload_file())
        self.assertEqual(duplicate.exception.code, "KNOWLEDGE_DOCUMENT_DUPLICATE")
        self.assertEqual(duplicate.exception.status_code, 409)

    async def test_uncertain_database_failure_never_deletes_content_addressed_object(self):
        knowledge_base = self._create_base()
        upload = UploadFile(
            filename="manual.txt",
            file=io.BytesIO(b"knowledge content"),
            headers=Headers({"content-type": "text/plain"}),
        )
        self.service.repo.create_document_with_task = MagicMock(side_effect=RuntimeError("commit outcome unknown"))

        with self.assertRaisesRegex(RuntimeError, "commit outcome unknown"):
            await self.service.upload_document("user-1", knowledge_base.id, upload)

        self.storage.upload.assert_awaited_once()
        self.storage.delete.assert_not_awaited()

    async def test_retrieval_fails_closed_if_any_base_is_not_owned(self):
        owned = self._create_base()
        payload = KnowledgeRetrievalRequest(
            knowledge_base_ids=[owned.id, "other-users-base"],
            query="如何配置？",
            top_k=5,
        )

        with self.assertRaises(ApiException) as raised:
            await self.service.retrieve("user-1", payload)

        self.assertEqual(raised.exception.status_code, 404)
        self.vector_store.search.assert_not_awaited()

    async def test_retrieval_revalidates_postgres_after_milvus_search(self):
        knowledge_base = self._create_base()
        upload = UploadFile(
            filename="manual.txt",
            file=io.BytesIO(b"knowledge content"),
            headers=Headers({"content-type": "text/plain"}),
        )
        created = await self.service.upload_document("user-1", knowledge_base.id, upload)
        document = self.service.repo.get_document(created.document.id, "user-1")
        document.status = "ready"
        document.active_index_version = document.desired_index_version
        self.db.commit()

        async def delete_during_vector_search(**_kwargs):
            document.status = "deleting"
            self.db.commit()
            return [
                KnowledgeVectorHit(
                    chunk_id="chunk-1",
                    document_id=document.id,
                    knowledge_base_id=knowledge_base.id,
                    index_version=document.active_index_version,
                    text="不应越过删除竞态",
                    similarity=0.99,
                    filename="manual.txt",
                    char_start=0,
                    char_end=8,
                    page=None,
                    section=None,
                )
            ]

        self.vector_store.search.side_effect = delete_during_vector_search
        result = await self.service.retrieve(
            "user-1",
            KnowledgeRetrievalRequest(
                knowledge_base_ids=[knowledge_base.id],
                query="配置",
                top_k=5,
            ),
        )

        self.assertEqual(result.hits, [])


if __name__ == "__main__":
    unittest.main()
