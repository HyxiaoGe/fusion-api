import hashlib
import io
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from starlette.datastructures import Headers

from app.db.database import Base
from app.db.knowledge_repository import KnowledgeBaseWriteConflict
from app.db.models import KnowledgeChunkManifest, User
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

    async def _create_ready_document(self, *, base_name: str, content: bytes, embedding_model: str):
        knowledge_base = self.service.create_knowledge_base(
            "user-1",
            KnowledgeBaseCreate(name=base_name, description="说明", business_type="product"),
        )
        upload = UploadFile(
            filename=f"{base_name}.txt",
            file=io.BytesIO(content),
            headers=Headers({"content-type": "text/plain"}),
        )
        created = await self.service.upload_document("user-1", knowledge_base.id, upload)
        document = self.service.repo.get_document(created.document.id, "user-1")
        document.status = "ready"
        document.active_index_version = document.desired_index_version
        version = next(item for item in document.index_versions if item.id == document.active_index_version)
        version.status = "active"
        version.embedding_model = embedding_model
        version.embedding_revision = f"{embedding_model}-r1"
        version.collection_name = f"knowledge_{embedding_model}_d2"
        self.db.commit()
        return knowledge_base, document

    def _add_chunk(self, document, *, chunk_id: str, ordinal: int, text: str):
        self.db.add(
            KnowledgeChunkManifest(
                chunk_id=chunk_id,
                knowledge_base_id=document.knowledge_base_id,
                document_id=document.id,
                user_id=document.user_id,
                index_version=document.active_index_version,
                ordinal=ordinal,
                text=text,
                text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                filename=document.original_filename,
                char_start=ordinal * 10,
                char_end=ordinal * 10 + len(text),
                page=None,
                section=None,
            )
        )

    @staticmethod
    def _vector_hit(document, *, chunk_id: str, similarity: float) -> KnowledgeVectorHit:
        return KnowledgeVectorHit(
            chunk_id=chunk_id,
            document_id=document.id,
            knowledge_base_id=document.knowledge_base_id,
            index_version=document.active_index_version,
            text="Milvus 中的正文不应直接返回",
            similarity=similarity,
            filename=document.original_filename,
            char_start=0,
            char_end=1,
            page=None,
            section=None,
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

    async def test_quota_recheck_duplicate_preserves_stable_error(self):
        knowledge_base = self._create_base()
        upload = UploadFile(
            filename="manual.txt",
            file=io.BytesIO(b"knowledge content"),
            headers=Headers({"content-type": "text/plain"}),
        )
        self.service.repo.create_document_with_task = MagicMock(
            side_effect=KnowledgeBaseWriteConflict(
                "相同内容的活动文档已存在于该知识库",
                cleanup_storage=False,
                duplicate=True,
            )
        )

        with self.assertRaises(ApiException) as duplicate:
            await self.service.upload_document("user-1", knowledge_base.id, upload)

        self.assertEqual(duplicate.exception.code, "KNOWLEDGE_DOCUMENT_DUPLICATE")
        self.assertEqual(duplicate.exception.status_code, 409)
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

    async def test_single_profile_retrieval_preserves_similarity_order(self):
        knowledge_base, document = await self._create_ready_document(
            base_name="单模型手册",
            content=b"single profile",
            embedding_model="embed-a",
        )
        self._add_chunk(document, chunk_id="low-score", ordinal=0, text="低相似度")
        self._add_chunk(document, chunk_id="high-score", ordinal=1, text="高相似度")
        self.db.commit()
        self.vector_store.search.return_value = [
            self._vector_hit(document, chunk_id="low-score", similarity=0.2),
            self._vector_hit(document, chunk_id="high-score", similarity=0.9),
        ]

        result = await self.service.retrieve(
            "user-1",
            KnowledgeRetrievalRequest(
                knowledge_base_ids=[knowledge_base.id],
                query="配置",
                top_k=2,
            ),
        )

        self.assertEqual([hit.chunk_id for hit in result.hits], ["high-score", "low-score"])
        self.assertEqual([hit.similarity for hit in result.hits], [0.9, 0.2])

    async def test_multi_profile_retrieval_uses_deterministic_rank_fusion(self):
        base_a, document_a = await self._create_ready_document(
            base_name="模型 A 手册",
            content=b"profile a",
            embedding_model="embed-a",
        )
        base_z, document_z = await self._create_ready_document(
            base_name="模型 Z 手册",
            content=b"profile z",
            embedding_model="embed-z",
        )
        for document, chunks in (
            (document_a, (("a-rank-1", "A 第一名"), ("a-rank-2", "A 第二名"))),
            (document_z, (("z-rank-1", "Z 第一名"), ("z-rank-2", "Z 第二名"))),
        ):
            for ordinal, (chunk_id, text) in enumerate(chunks):
                self._add_chunk(document, chunk_id=chunk_id, ordinal=ordinal, text=text)
        self.db.commit()

        async def search_by_profile(**kwargs):
            if kwargs["profile"].model == "embed-a":
                return [
                    self._vector_hit(document_a, chunk_id="a-rank-2", similarity=0.19),
                    self._vector_hit(document_a, chunk_id="a-rank-1", similarity=0.2),
                ]
            return [
                self._vector_hit(document_z, chunk_id="z-rank-2", similarity=0.98),
                self._vector_hit(document_z, chunk_id="z-rank-1", similarity=0.99),
            ]

        self.vector_store.search.side_effect = search_by_profile
        result = await self.service.retrieve(
            "user-1",
            KnowledgeRetrievalRequest(
                knowledge_base_ids=[base_z.id, base_a.id],
                query="配置",
                top_k=4,
            ),
        )

        self.assertEqual(
            [hit.chunk_id for hit in result.hits],
            ["a-rank-1", "z-rank-1", "a-rank-2", "z-rank-2"],
        )
        self.assertEqual([hit.similarity for hit in result.hits], [0.2, 0.99, 0.19, 0.98])


if __name__ == "__main__":
    unittest.main()
