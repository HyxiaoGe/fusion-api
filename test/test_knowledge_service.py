import asyncio
import hashlib
import io
import json
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import UploadFile
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.datastructures import Headers

from app.db.database import Base
from app.db.knowledge_repository import KnowledgeBaseWriteConflict
from app.db.models import KnowledgeChunkManifest, KnowledgeStorageCleanupTask, KnowledgeStorageUploadIntent, User
from app.schemas.knowledge import KnowledgeBaseCreate, KnowledgeBaseUpdate, KnowledgeRetrievalRequest
from app.schemas.response import ApiException
from app.services.knowledge.milvus import KnowledgeVectorError, KnowledgeVectorHit
from app.services.knowledge.service import KnowledgeService
from app.services.knowledge.storage_upload_guard import drain_storage_upload_lifecycles


class KnowledgeServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
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
            patch(
                "app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_REVISION_ROUTES",
                json.dumps({"embed-v1@embed-r1": "embed-v1-r1-immutable"}),
            ),
            patch("app.services.knowledge.service.settings.LITELLM_PROXY_URL", "http://litellm-proxy:4000"),
            patch("app.services.knowledge.service.settings.LITELLM_API_KEY", "proxy-key"),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_DIMENSION", 2),
            patch(
                "app.services.knowledge.service.settings.KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS",
                "2",
            ),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_DISTANCE_METRIC", "COSINE"),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_CHUNKER_VERSION", "chunker-v2"),
            patch("app.services.knowledge.service.settings.MILVUS_URI", "http://milvus:19530"),
            patch("app.services.knowledge.service.settings.MILVUS_USERNAME", "fusion_knowledge"),
            patch("app.services.knowledge.service.settings.MILVUS_PASSWORD", "secret"),
            patch("app.services.knowledge.service.settings.MILVUS_DATABASE", "fusion_knowledge"),
            patch("app.services.knowledge.service.settings.MILVUS_COLLECTION_PREFIX", "fusion_knowledge"),
            patch(
                "app.services.knowledge.service.settings.KNOWLEDGE_ALLOWED_MIME_TYPES",
                "text/plain",
            ),
            patch("app.services.knowledge.service.settings.KNOWLEDGE_SEARCH_MAX_PROFILES", 16),
        ]
        for patcher in self.settings_patchers:
            patcher.start()

    async def asyncTearDown(self):
        await drain_storage_upload_lifecycles()

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

    def test_expanded_normalized_name_is_rejected_before_database_write(self):
        expanding_name = "ﬃ" * 120
        create_payload = KnowledgeBaseCreate.model_construct(
            name=expanding_name,
            description="",
            business_type="product",
        )

        with self.assertRaises(ApiException) as create_error:
            self.service.create_knowledge_base("user-1", create_payload)

        self.assertEqual(create_error.exception.code, "INVALID_PARAM")
        self.assertEqual(create_error.exception.status_code, 400)
        knowledge_base = self._create_base()
        update_payload = KnowledgeBaseUpdate.model_construct(name=expanding_name)

        with self.assertRaises(ApiException) as update_error:
            self.service.update_knowledge_base("user-1", knowledge_base.id, update_payload)

        self.assertEqual(update_error.exception.code, "INVALID_PARAM")
        self.assertEqual(update_error.exception.status_code, 400)
        self.assertEqual(self.service.get_knowledge_base("user-1", knowledge_base.id).name, "产品 手册")

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

    async def test_upload_normalizes_mime_before_allowlist_and_persistence(self):
        knowledge_base = self._create_base()
        upload = UploadFile(
            filename="manual.TXT",
            file=io.BytesIO(b"knowledge content"),
            headers=Headers({"content-type": " Text/Plain ; Charset=UTF-8 "}),
        )

        result = await self.service.upload_document("user-1", knowledge_base.id, upload)

        document = self.service.repo.get_document(result.document.id, "user-1")
        self.assertEqual(document.mimetype, "text/plain")
        self.assertEqual(document.original_filename, "manual.txt")
        self.assertEqual(self.storage.upload.await_args.args[2], "text/plain")

    async def test_transient_storage_upload_failure_returns_stable_503_and_keeps_cleanup(self):
        knowledge_base = self._create_base()
        self.storage.upload.side_effect = TimeoutError("temporary storage timeout")
        upload = UploadFile(
            filename="manual.txt",
            file=io.BytesIO(b"knowledge content"),
            headers=Headers({"content-type": "text/plain"}),
        )

        with self.assertRaises(ApiException) as raised:
            await self.service.upload_document("user-1", knowledge_base.id, upload)

        self.assertEqual(raised.exception.code, "KNOWLEDGE_STORAGE_UNAVAILABLE")
        self.assertEqual(raised.exception.status_code, 503)
        await drain_storage_upload_lifecycles()
        cleanup_task = self.db.query(KnowledgeStorageCleanupTask).one()
        upload_intent = self.db.query(KnowledgeStorageUploadIntent).one()
        self.assertEqual(cleanup_task.status, "pending")
        self.assertEqual(upload_intent.outcome, "uncertain")

    async def test_upload_rejects_missing_or_mismatched_suffix_before_storage(self):
        knowledge_base = self._create_base()
        for filename in ("README", "note.md"):
            upload = UploadFile(
                filename=filename,
                file=io.BytesIO(b"knowledge content"),
                headers=Headers({"content-type": "text/plain"}),
            )
            upload.read = AsyncMock(return_value=b"knowledge content")
            with self.subTest(filename=filename), self.assertRaises(ApiException) as raised:
                await self.service.upload_document("user-1", knowledge_base.id, upload)
            self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_TYPE_MISMATCH")
            self.assertEqual(raised.exception.status_code, 400)
            upload.read.assert_not_awaited()
        self.storage.upload.assert_not_awaited()
        self.assertEqual(self.service.repo.count_documents(knowledge_base.id), 0)

    def test_safe_filename_preserves_supported_suffix_with_character_and_byte_limits(self):
        ascii_name = self.service._safe_filename(f"nested/{'a' * 400}.PDF", "application/pdf")
        unicode_name = self.service._safe_filename(
            f"nested\\{'资料' * 200}.DOCX", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        sanitized_name = self.service._safe_filename("nested/💩.PDF", "application/pdf")

        for filename, suffix in ((ascii_name, ".pdf"), (unicode_name, ".docx"), (sanitized_name, ".pdf")):
            self.assertTrue(filename.endswith(suffix))
            self.assertLessEqual(len(filename), 255)
            self.assertLessEqual(len(filename.encode("utf-8")), 500)
            self.assertNotIn("/", filename)
            self.assertNotIn("\\", filename)
        self.assertEqual(sanitized_name, "document.pdf")

    async def test_uncertain_database_failure_leaves_generation_for_guarded_recovery(self):
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
                cleanup_storage=True,
                duplicate=True,
            )
        )

        with self.assertRaises(ApiException) as duplicate:
            await self.service.upload_document("user-1", knowledge_base.id, upload)

        self.assertEqual(duplicate.exception.code, "KNOWLEDGE_DOCUMENT_DUPLICATE")
        self.assertEqual(duplicate.exception.status_code, 409)
        self.storage.delete.assert_awaited_once()

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

    async def test_retrieval_fails_if_base_loses_ready_document_after_milvus_search(self):
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
        with self.assertRaises(ApiException) as raised:
            await self.service.retrieve(
                "user-1",
                KnowledgeRetrievalRequest(
                    knowledge_base_ids=[knowledge_base.id],
                    query="配置",
                    top_k=5,
                ),
            )

        self.assertEqual(raised.exception.code, "KNOWLEDGE_BASE_NOT_READY")
        self.assertEqual(raised.exception.status_code, 409)

    async def test_final_ready_check_uses_one_query_after_expiring_session_state(self):
        knowledge_base_ids = []
        for index in range(6):
            knowledge_base, _document = await self._create_ready_document(
                base_name=f"批量检查手册 {index}",
                content=f"ready document {index}".encode(),
                embedding_model=f"embed-{index}",
            )
            knowledge_base_ids.append(knowledge_base.id)

        statements = []

        def count_selects(_connection, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(self.engine, "before_cursor_execute", count_selects)
        try:
            self.service._ensure_requested_bases_still_ready("user-1", knowledge_base_ids)
        finally:
            event.remove(self.engine, "before_cursor_execute", count_selects)

        self.assertEqual(len(statements), 1)

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

    async def test_active_legacy_v1_index_remains_retrievable(self):
        knowledge_base, document = await self._create_ready_document(
            base_name="旧版切片手册",
            content=b"legacy chunker index",
            embedding_model="embed-a",
        )
        version = next(item for item in document.index_versions if item.id == document.active_index_version)
        version.chunker_version = "chunker-v1"
        self._add_chunk(document, chunk_id="legacy-v1", ordinal=0, text="旧版索引仍可检索")
        self.db.commit()
        self.vector_store.search.return_value = [self._vector_hit(document, chunk_id="legacy-v1", similarity=0.9)]

        result = await self.service.retrieve(
            "user-1",
            KnowledgeRetrievalRequest(
                knowledge_base_ids=[knowledge_base.id],
                query="旧版",
                top_k=1,
            ),
        )

        self.assertEqual([hit.chunk_id for hit in result.hits], ["legacy-v1"])

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

    async def test_multi_profile_embedding_and_vector_queries_run_concurrently(self):
        base_a, _document_a = await self._create_ready_document(
            base_name="并发模型 A",
            content=b"concurrent profile a",
            embedding_model="embed-a",
        )
        base_z, _document_z = await self._create_ready_document(
            base_name="并发模型 Z",
            content=b"concurrent profile z",
            embedding_model="embed-z",
        )
        embedding_started: set[str] = set()
        embedding_release = asyncio.Event()
        vector_started: set[str] = set()
        vector_release = asyncio.Event()

        async def concurrent_embed(_texts, profile):
            embedding_started.add(profile.model)
            if len(embedding_started) == 2:
                embedding_release.set()
            await asyncio.wait_for(embedding_release.wait(), timeout=0.5)
            return [[1.0, 0.5]]

        async def concurrent_search(**kwargs):
            vector_started.add(kwargs["profile"].model)
            if len(vector_started) == 2:
                vector_release.set()
            await asyncio.wait_for(vector_release.wait(), timeout=0.5)
            return []

        self.embedding.embed.side_effect = concurrent_embed
        self.vector_store.search.side_effect = concurrent_search

        result = await self.service.retrieve(
            "user-1",
            KnowledgeRetrievalRequest(
                knowledge_base_ids=[base_z.id, base_a.id],
                query="配置",
                top_k=2,
            ),
        )

        self.assertEqual(result.hits, [])
        self.assertEqual(embedding_started, {"embed-a", "embed-z"})
        self.assertEqual(vector_started, {"embed-a", "embed-z"})

    async def test_multi_profile_dependency_failure_does_not_return_partial_hits(self):
        base_a, document_a = await self._create_ready_document(
            base_name="失败模型 A",
            content=b"failure profile a",
            embedding_model="embed-a",
        )
        base_z, _document_z = await self._create_ready_document(
            base_name="失败模型 Z",
            content=b"failure profile z",
            embedding_model="embed-z",
        )
        self._add_chunk(document_a, chunk_id="partial-hit", ordinal=0, text="不得部分返回")
        self.db.commit()

        async def search_with_failure(**kwargs):
            if kwargs["profile"].model == "embed-z":
                raise KnowledgeVectorError("KNOWLEDGE_VECTOR_UNAVAILABLE", "向量检索失败", retryable=True)
            return [self._vector_hit(document_a, chunk_id="partial-hit", similarity=0.9)]

        self.vector_store.search.side_effect = search_with_failure

        with self.assertRaises(ApiException) as raised:
            await self.service.retrieve(
                "user-1",
                KnowledgeRetrievalRequest(
                    knowledge_base_ids=[base_a.id, base_z.id],
                    query="配置",
                    top_k=2,
                ),
            )

        self.assertEqual(raised.exception.code, "KNOWLEDGE_VECTOR_UNAVAILABLE")
        self.assertEqual(raised.exception.status_code, 503)

    async def test_profile_search_concurrency_is_bounded_and_failure_cancels_waiters(self):
        active = 0
        peak = 0
        started: list[str] = []
        release = asyncio.Event()

        async def bounded_search(**kwargs):
            nonlocal active, peak
            model = kwargs["profile_key"][1]
            started.append(model)
            active += 1
            peak = max(peak, active)
            try:
                if model == "embed-0":
                    await asyncio.sleep(0)
                    raise KnowledgeVectorError("KNOWLEDGE_VECTOR_UNAVAILABLE", "向量检索失败", retryable=True)
                await release.wait()
                return kwargs["profile_key"], []
            finally:
                active -= 1

        documents_by_profile = {
            ("litellm", f"embed-{index}", 2, "COSINE", f"collection-{index}", f"revision-{index}"): []
            for index in range(12)
        }
        self.service._search_profile = bounded_search

        with self.assertRaises(KnowledgeVectorError):
            await self.service._search_profiles(
                user_id="user-1",
                query="配置",
                documents_by_profile=documents_by_profile,
                limit=2,
            )

        self.assertLessEqual(peak, 4)
        self.assertLess(len(started), len(documents_by_profile))
        self.assertEqual(active, 0)

    async def test_profile_search_fails_closed_before_external_calls_when_total_exceeds_limit(self):
        documents_by_profile = {
            ("litellm", f"embed-{index}", 2, "COSINE", f"collection-{index}", f"revision-{index}"): []
            for index in range(5)
        }
        self.service._search_profile = AsyncMock()

        with (
            patch("app.services.knowledge.service.settings.KNOWLEDGE_SEARCH_MAX_PROFILES", 4),
            self.assertRaises(ApiException) as raised,
        ):
            await self.service._search_profiles(
                user_id="user-1",
                query="配置",
                documents_by_profile=documents_by_profile,
                limit=2,
            )

        self.assertEqual(raised.exception.code, "KNOWLEDGE_CONFIG_INVALID")
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("Embedding profile", raised.exception.message)
        self.service._search_profile.assert_not_awaited()

    async def test_retrieval_fails_if_active_version_changes_after_milvus_search(self):
        knowledge_base, document = await self._create_ready_document(
            base_name="版本切换手册",
            content=b"active version",
            embedding_model="embed-a",
        )
        self._add_chunk(document, chunk_id="stale-version", ordinal=0, text="旧版本内容")
        self.db.commit()
        active_version = next(item for item in document.index_versions if item.id == document.active_index_version)

        async def supersede_during_search(**_kwargs):
            active_version.status = "superseded"
            self.db.commit()
            return [self._vector_hit(document, chunk_id="stale-version", similarity=0.99)]

        self.vector_store.search.side_effect = supersede_during_search

        with self.assertRaises(ApiException) as raised:
            await self.service.retrieve(
                "user-1",
                KnowledgeRetrievalRequest(
                    knowledge_base_ids=[knowledge_base.id],
                    query="配置",
                    top_k=1,
                ),
            )

        self.assertEqual(raised.exception.code, "KNOWLEDGE_BASE_NOT_READY")
        self.assertEqual(raised.exception.status_code, 409)

    async def test_retrieval_revalidates_each_hit_after_base_ready_check(self):
        knowledge_base, hit_document = await self._create_ready_document(
            base_name="并发删除手册",
            content=b"hit document",
            embedding_model="embed-a",
        )
        second_upload = UploadFile(
            filename="other.txt",
            file=io.BytesIO(b"other ready document"),
            headers=Headers({"content-type": "text/plain"}),
        )
        created = await self.service.upload_document("user-1", knowledge_base.id, second_upload)
        other_document = self.service.repo.get_document(created.document.id, "user-1")
        other_document.status = "ready"
        other_document.active_index_version = other_document.desired_index_version
        other_version = next(
            item for item in other_document.index_versions if item.id == other_document.active_index_version
        )
        other_version.status = "active"
        self._add_chunk(hit_document, chunk_id="deleted-hit", ordinal=0, text="不得返回的旧正文")
        self.db.commit()
        self.vector_store.search.return_value = [
            self._vector_hit(hit_document, chunk_id="deleted-hit", similarity=0.99)
        ]
        original_ready_check = self.service._ensure_requested_bases_still_ready

        def delete_hit_after_base_check(user_id, knowledge_base_ids):
            original_ready_check(user_id, knowledge_base_ids)
            hit_document.status = "deleting"
            self.db.commit()

        with patch.object(
            self.service,
            "_ensure_requested_bases_still_ready",
            side_effect=delete_hit_after_base_check,
        ):
            result = await self.service.retrieve(
                "user-1",
                KnowledgeRetrievalRequest(
                    knowledge_base_ids=[knowledge_base.id],
                    query="删除竞态",
                    top_k=1,
                ),
            )

        self.assertEqual(result.hits, [])


if __name__ == "__main__":
    unittest.main()
