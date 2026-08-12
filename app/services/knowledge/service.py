from __future__ import annotations

import hashlib
import unicodedata
import uuid
from collections import defaultdict

from fastapi import UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingAdapter, EmbeddingError, EmbeddingProfile
from app.ai.embeddings.litellm_embedding import LiteLLMEmbeddingAdapter
from app.core.config import settings
from app.db.knowledge_repository import (
    KnowledgeBaseLimitExceeded,
    KnowledgeBaseWriteConflict,
    KnowledgeRepository,
)
from app.db.models import KnowledgeBase, KnowledgeDocument, KnowledgeIndexTask, KnowledgeIndexVersion
from app.schemas.knowledge import (
    KnowledgeBaseCreate,
    KnowledgeBasePage,
    KnowledgeBaseUpdate,
    KnowledgeBaseView,
    KnowledgeDocumentPage,
    KnowledgeDocumentUploadResult,
    KnowledgeDocumentView,
    KnowledgeRetrievalHit,
    KnowledgeRetrievalRequest,
    KnowledgeRetrievalResult,
    KnowledgeTaskView,
)
from app.schemas.response import ApiException, ErrorCode
from app.services.knowledge.milvus import KnowledgeVectorError, MilvusKnowledgeStore
from app.services.storage import get_storage


class KnowledgeService:
    """知识库 CRUD、上传入队和 fail-closed 检索。"""

    def __init__(
        self,
        db: Session,
        *,
        embedding: EmbeddingAdapter | None = None,
        vector_store: MilvusKnowledgeStore | None = None,
        storage=None,
    ):
        self.db = db
        self.repo = KnowledgeRepository(db)
        self.embedding = embedding or LiteLLMEmbeddingAdapter()
        self.vector_store = vector_store or MilvusKnowledgeStore()
        self.storage = storage or get_storage()

    def create_knowledge_base(self, user_id: str, payload: KnowledgeBaseCreate) -> KnowledgeBaseView:
        self._require_enabled()
        profile = self._current_profile()
        row = KnowledgeBase(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=payload.name,
            name_normalized=self.normalize_name(payload.name),
            description=payload.description,
            business_type=payload.business_type,
            status="active",
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_revision=profile.revision,
            embedding_dimension=profile.dimension,
            distance_metric=profile.distance_metric,
        )
        try:
            return self._base_view(
                self.repo.create_knowledge_base(
                    row,
                    max_bases=settings.KNOWLEDGE_MAX_BASES_PER_USER,
                )
            )
        except KnowledgeBaseLimitExceeded as exc:
            raise ApiException.bad_request("已达到每用户知识库数量上限") from exc
        except IntegrityError as exc:
            raise ApiException.conflict("同一用户下知识库名称不能重复") from exc

    def list_knowledge_bases(self, user_id: str, *, page: int, page_size: int) -> KnowledgeBasePage:
        self._require_enabled()
        rows, total = self.repo.list_knowledge_bases(user_id=user_id, page=page, page_size=page_size)
        total_pages = (total + page_size - 1) // page_size if total else 0
        return KnowledgeBasePage(
            items=[self._base_view(row) for row in rows],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    def get_knowledge_base(self, user_id: str, knowledge_base_id: str) -> KnowledgeBaseView:
        self._require_enabled()
        return self._base_view(self._require_base(user_id, knowledge_base_id))

    def update_knowledge_base(
        self,
        user_id: str,
        knowledge_base_id: str,
        payload: KnowledgeBaseUpdate,
    ) -> KnowledgeBaseView:
        self._require_enabled()
        row = self._require_base(user_id, knowledge_base_id, lock=True)
        if row.status != "active":
            raise ApiException.conflict("删除中的知识库不能修改")
        updates = payload.model_dump(exclude_unset=True)
        if any(value is None for value in updates.values()):
            raise ApiException.bad_request("知识库更新字段不能为 null")
        if "name" in updates:
            updates["name_normalized"] = self.normalize_name(str(updates["name"]))
        try:
            return self._base_view(self.repo.update_knowledge_base(row, **updates))
        except IntegrityError as exc:
            raise ApiException.conflict("同一用户下知识库名称不能重复") from exc

    def delete_knowledge_base(self, user_id: str, knowledge_base_id: str) -> KnowledgeTaskView:
        self._require_enabled()
        row = self._require_base(user_id, knowledge_base_id, lock=True)
        if row.status == "deleting":
            task = self._latest_task(knowledge_base_id=knowledge_base_id, task_type="delete_knowledge_base")
            if task is not None and task.status in {"pending", "running", "retry"}:
                return self._task_view(task)
        task = self.repo.enqueue_knowledge_base_delete(row, max_attempts=settings.KNOWLEDGE_WORKER_MAX_ATTEMPTS)
        return self._task_view(task)

    async def upload_document(
        self,
        user_id: str,
        knowledge_base_id: str,
        upload: UploadFile,
    ) -> KnowledgeDocumentUploadResult:
        self._require_enabled()
        knowledge_base = self._require_base(user_id, knowledge_base_id)
        if knowledge_base.status != "active":
            raise ApiException.conflict("删除中的知识库不能上传文档")
        if self.repo.count_documents(knowledge_base_id) >= settings.KNOWLEDGE_MAX_DOCUMENTS_PER_BASE:
            raise ApiException.bad_request("已达到每个知识库的文档数量上限")
        filename = self._safe_filename(upload.filename or "document")
        mimetype = upload.content_type or "application/octet-stream"
        if mimetype not in settings.RESOLVED_KNOWLEDGE_ALLOWED_MIME_TYPES:
            raise ApiException(
                ErrorCode.KNOWLEDGE_DOCUMENT_UNSUPPORTED,
                "当前知识库版本不支持该文档格式",
                400,
            )
        content = await upload.read(settings.KNOWLEDGE_MAX_FILE_SIZE + 1)
        if len(content) > settings.KNOWLEDGE_MAX_FILE_SIZE:
            raise ApiException(ErrorCode.FILE_TOO_LARGE, "知识库文档超过文件大小上限", 413)
        checksum = hashlib.sha256(content).hexdigest()
        if self.repo.get_duplicate_document(knowledge_base_id, checksum):
            raise ApiException(
                ErrorCode.KNOWLEDGE_DOCUMENT_DUPLICATE,
                "相同内容的活动文档已存在于该知识库",
                409,
            )
        document_id = str(uuid.uuid4())
        version_id = str(uuid.uuid4())
        # 同一 KB 的对象键按内容寻址。并发重复上传会覆盖为相同字节；唯一约束失败的一方
        # 不得删除该键，否则可能删掉已经提交成功的文档原件。
        storage_key = f"knowledge/v1/users/{user_id}/bases/{knowledge_base_id}/objects/{checksum}"
        await self.storage.upload(storage_key, content, mimetype)
        profile = self._base_profile(knowledge_base)
        document = KnowledgeDocument(
            id=document_id,
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            original_filename=filename,
            mimetype=mimetype,
            size=len(content),
            checksum_sha256=checksum,
            dedupe_key=checksum,
            storage_backend=settings.STORAGE_BACKEND,
            storage_key=storage_key,
            status="queued",
            parser_version=settings.KNOWLEDGE_PARSER_VERSION,
            chunker_version=settings.KNOWLEDGE_CHUNKER_VERSION,
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_revision=profile.revision,
            embedding_dimension=profile.dimension,
            distance_metric=profile.distance_metric,
            desired_index_version=version_id,
        )
        version = KnowledgeIndexVersion(
            id=version_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            user_id=user_id,
            status="building",
            parser_version=settings.KNOWLEDGE_PARSER_VERSION,
            chunker_version=settings.KNOWLEDGE_CHUNKER_VERSION,
            chunk_size=settings.KNOWLEDGE_CHUNK_SIZE,
            chunk_overlap=settings.KNOWLEDGE_CHUNK_OVERLAP,
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_revision=profile.revision,
            embedding_dimension=profile.dimension,
            distance_metric=profile.distance_metric,
            collection_name=self.vector_store.collection_name(profile.dimension),
        )
        task = KnowledgeIndexTask(
            knowledge_base_id=knowledge_base_id,
            document_id=document_id,
            user_id=user_id,
            task_type="index_document",
            index_version=version_id,
            max_attempts=settings.KNOWLEDGE_WORKER_MAX_ATTEMPTS,
        )
        try:
            saved_document, _saved_version, saved_task = self.repo.create_document_with_task(
                document,
                version,
                task,
                max_documents=settings.KNOWLEDGE_MAX_DOCUMENTS_PER_BASE,
            )
        except KnowledgeBaseWriteConflict as exc:
            if exc.cleanup_storage:
                await self._delete_uploaded_object(storage_key)
            raise ApiException.conflict(str(exc)) from exc
        except IntegrityError as exc:
            raise ApiException(
                ErrorCode.KNOWLEDGE_DOCUMENT_DUPLICATE,
                "相同内容的活动文档已存在于该知识库",
                409,
            ) from exc
        return KnowledgeDocumentUploadResult(
            document=self._document_view(saved_document),
            task=self._task_view(saved_task),
        )

    def list_documents(
        self,
        user_id: str,
        knowledge_base_id: str,
        *,
        page: int,
        page_size: int,
    ) -> KnowledgeDocumentPage:
        self._require_enabled()
        self._require_base(user_id, knowledge_base_id)
        rows, total = self.repo.list_documents(
            knowledge_base_id=knowledge_base_id,
            user_id=user_id,
            page=page,
            page_size=page_size,
        )
        total_pages = (total + page_size - 1) // page_size if total else 0
        return KnowledgeDocumentPage(
            items=[self._document_view(row) for row in rows],
            page=page,
            page_size=page_size,
            total=total,
            total_pages=total_pages,
            has_next=page < total_pages,
            has_prev=page > 1,
        )

    def get_document(self, user_id: str, knowledge_base_id: str, document_id: str) -> KnowledgeDocumentView:
        self._require_enabled()
        self._require_base(user_id, knowledge_base_id)
        return self._document_view(self._require_document(user_id, knowledge_base_id, document_id))

    def delete_document(self, user_id: str, knowledge_base_id: str, document_id: str) -> KnowledgeTaskView:
        self._require_enabled()
        self._require_base(user_id, knowledge_base_id)
        document = self._require_document(user_id, knowledge_base_id, document_id, lock=True)
        if document.status == "deleting":
            task = self._latest_task(document_id=document_id, task_type="delete_document")
            if task is not None and task.status in {"pending", "running", "retry"}:
                return self._task_view(task)
        task = self.repo.enqueue_document_delete(document, max_attempts=settings.KNOWLEDGE_WORKER_MAX_ATTEMPTS)
        return self._task_view(task)

    def retry_document(self, user_id: str, knowledge_base_id: str, document_id: str) -> KnowledgeTaskView:
        self._require_enabled()
        knowledge_base = self._require_base(user_id, knowledge_base_id)
        document = self._require_document(user_id, knowledge_base_id, document_id, lock=True)
        if knowledge_base.status != "active" or document.status != "failed":
            raise ApiException(
                ErrorCode.KNOWLEDGE_DOCUMENT_NOT_RETRYABLE,
                "只有活动知识库中的失败文档可以重试",
                409,
            )
        profile = self._base_profile(knowledge_base)
        version_id = str(uuid.uuid4())
        version = KnowledgeIndexVersion(
            id=version_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            user_id=user_id,
            status="building",
            parser_version=settings.KNOWLEDGE_PARSER_VERSION,
            chunker_version=settings.KNOWLEDGE_CHUNKER_VERSION,
            chunk_size=settings.KNOWLEDGE_CHUNK_SIZE,
            chunk_overlap=settings.KNOWLEDGE_CHUNK_OVERLAP,
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_revision=profile.revision,
            embedding_dimension=profile.dimension,
            distance_metric=profile.distance_metric,
            collection_name=self.vector_store.collection_name(profile.dimension),
        )
        task = KnowledgeIndexTask(
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            user_id=user_id,
            task_type="index_document",
            index_version=version_id,
            max_attempts=settings.KNOWLEDGE_WORKER_MAX_ATTEMPTS,
        )
        task = self.repo.retry_document(document, version, task)
        return self._task_view(task)

    def rebuild_document(self, user_id: str, knowledge_base_id: str, document_id: str) -> KnowledgeTaskView:
        self._require_enabled()
        knowledge_base = self._require_base(user_id, knowledge_base_id, lock=True)
        document = self._require_document(user_id, knowledge_base_id, document_id, lock=True)
        if knowledge_base.status != "active" or document.status not in {"ready", "failed"}:
            raise ApiException(
                ErrorCode.KNOWLEDGE_DOCUMENT_NOT_RETRYABLE,
                "只有就绪或失败的文档可以重建",
                409,
            )
        profile = self._current_profile()
        knowledge_base.embedding_provider = profile.provider
        knowledge_base.embedding_model = profile.model
        knowledge_base.embedding_revision = profile.revision
        knowledge_base.embedding_dimension = profile.dimension
        knowledge_base.distance_metric = profile.distance_metric
        version_id = str(uuid.uuid4())
        version = KnowledgeIndexVersion(
            id=version_id,
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            user_id=user_id,
            status="building",
            parser_version=settings.KNOWLEDGE_PARSER_VERSION,
            chunker_version=settings.KNOWLEDGE_CHUNKER_VERSION,
            chunk_size=settings.KNOWLEDGE_CHUNK_SIZE,
            chunk_overlap=settings.KNOWLEDGE_CHUNK_OVERLAP,
            embedding_provider=profile.provider,
            embedding_model=profile.model,
            embedding_revision=profile.revision,
            embedding_dimension=profile.dimension,
            distance_metric=profile.distance_metric,
            collection_name=self.vector_store.collection_name(profile.dimension),
        )
        task = KnowledgeIndexTask(
            knowledge_base_id=knowledge_base_id,
            document_id=document.id,
            user_id=user_id,
            task_type="index_document",
            index_version=version_id,
            max_attempts=settings.KNOWLEDGE_WORKER_MAX_ATTEMPTS,
        )
        return self._task_view(self.repo.retry_document(document, version, task))

    def get_task(self, user_id: str, task_id: str) -> KnowledgeTaskView:
        self._require_enabled()
        task = self.repo.get_task(task_id, user_id)
        if task is None:
            raise ApiException.not_found("任务不存在或无权访问")
        return self._task_view(task)

    async def retrieve(self, user_id: str, payload: KnowledgeRetrievalRequest) -> KnowledgeRetrievalResult:
        self._require_enabled()
        bases = self.repo.get_knowledge_bases_by_ids(user_id, payload.knowledge_base_ids)
        if len(bases) != len(payload.knowledge_base_ids):
            raise ApiException.not_found("知识库不存在或无权访问")
        if any(row.status != "active" for row in bases):
            raise ApiException(ErrorCode.KNOWLEDGE_BASE_NOT_READY, "知识库尚不可检索", 409)
        documents = self.repo.get_ready_documents(user_id, payload.knowledge_base_ids)
        bases_with_ready_documents = {document.knowledge_base_id for document in documents}
        if bases_with_ready_documents != set(payload.knowledge_base_ids):
            raise ApiException(ErrorCode.KNOWLEDGE_BASE_NOT_READY, "至少一个知识库尚无就绪文档", 409)
        documents_by_profile: dict[tuple[str, str, int, str, str, str], list[KnowledgeDocument]] = defaultdict(list)
        for document in documents:
            active_version = next(
                (version for version in document.index_versions if version.id == document.active_index_version),
                None,
            )
            if active_version is None:
                raise ApiException(ErrorCode.KNOWLEDGE_BASE_NOT_READY, "知识库索引版本记录不完整", 503)
            key = (
                active_version.embedding_provider,
                active_version.embedding_model,
                active_version.embedding_dimension,
                active_version.distance_metric,
                active_version.collection_name,
                active_version.embedding_revision,
            )
            documents_by_profile[key].append(document)
        hits = []
        overfetch = min(payload.top_k * 3, 100)
        try:
            for key, profile_documents in documents_by_profile.items():
                profile = EmbeddingProfile(*key)
                query_vector = (await self.embedding.embed([payload.query], profile))[0]
                profile_hits = await self.vector_store.search(
                    profile=profile,
                    query_vector=query_vector,
                    user_id=user_id,
                    knowledge_base_ids=sorted({document.knowledge_base_id for document in profile_documents}),
                    index_versions=sorted({str(document.active_index_version) for document in profile_documents}),
                    limit=overfetch,
                )
                hits.extend(profile_hits)
        except EmbeddingError as exc:
            raise ApiException(exc.code, exc.summary, 503 if exc.retryable else 409) from exc
        except KnowledgeVectorError as exc:
            raise ApiException(exc.code, exc.summary, 503 if exc.retryable else 409) from exc
        revalidated_documents = self.repo.revalidate_retrieval_documents(
            user_id=user_id,
            knowledge_base_ids=payload.knowledge_base_ids,
            document_ids=sorted({hit.document_id for hit in hits}),
            index_versions=sorted({hit.index_version for hit in hits}),
        )
        current_documents = {document.id: document for document in revalidated_documents}
        authorized_chunks = {
            chunk.chunk_id: chunk
            for chunk in self.repo.get_authorized_chunks(
                user_id=user_id,
                knowledge_base_ids=payload.knowledge_base_ids,
                chunk_ids=sorted({hit.chunk_id for hit in hits}),
            )
        }
        validated = []
        for hit in hits:
            document = current_documents.get(hit.document_id)
            chunk = authorized_chunks.get(hit.chunk_id)
            if (
                document is None
                or chunk is None
                or document.knowledge_base_id != hit.knowledge_base_id
                or document.active_index_version != hit.index_version
                or chunk.document_id != hit.document_id
                or chunk.index_version != hit.index_version
            ):
                continue
            validated.append(
                KnowledgeRetrievalHit(
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    knowledge_base_id=hit.knowledge_base_id,
                    index_version=hit.index_version,
                    text=chunk.text,
                    similarity=hit.similarity,
                    filename=document.original_filename,
                    source={
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                        "page": chunk.page,
                        "section": chunk.section,
                    },
                )
            )
        validated.sort(key=lambda item: item.similarity, reverse=True)
        return KnowledgeRetrievalResult(hits=validated[: payload.top_k], query=payload.query, top_k=payload.top_k)

    def _require_base(self, user_id: str, knowledge_base_id: str, *, lock: bool = False) -> KnowledgeBase:
        row = self.repo.get_knowledge_base(knowledge_base_id, user_id, lock=lock)
        if row is None:
            raise ApiException.not_found("知识库不存在或无权访问")
        return row

    def _require_document(
        self,
        user_id: str,
        knowledge_base_id: str,
        document_id: str,
        *,
        lock: bool = False,
    ) -> KnowledgeDocument:
        row = self.repo.get_document(document_id, user_id, lock=lock)
        if row is None or row.knowledge_base_id != knowledge_base_id:
            raise ApiException.not_found("文档不存在或无权访问")
        return row

    def _latest_task(
        self,
        *,
        task_type: str,
        knowledge_base_id: str | None = None,
        document_id: str | None = None,
    ) -> KnowledgeIndexTask | None:
        query = self.db.query(KnowledgeIndexTask).filter(KnowledgeIndexTask.task_type == task_type)
        if knowledge_base_id:
            query = query.filter(KnowledgeIndexTask.knowledge_base_id == knowledge_base_id)
        if document_id:
            query = query.filter(KnowledgeIndexTask.document_id == document_id)
        return query.order_by(KnowledgeIndexTask.created_at.desc()).first()

    def _base_view(self, row: KnowledgeBase) -> KnowledgeBaseView:
        return KnowledgeBaseView(
            id=row.id,
            name=row.name,
            description=row.description,
            business_type=row.business_type,
            status=row.status,
            document_stats=self.repo.document_stats(row.id),
            embedding_provider=row.embedding_provider,
            embedding_model=row.embedding_model,
            embedding_revision=row.embedding_revision,
            embedding_dimension=row.embedding_dimension,
            distance_metric=row.distance_metric,
            created_at=row.created_at,
            updated_at=row.updated_at,
            deleted_at=row.deleted_at,
        )

    @staticmethod
    def _document_view(row: KnowledgeDocument) -> KnowledgeDocumentView:
        return KnowledgeDocumentView(
            id=row.id,
            knowledge_base_id=row.knowledge_base_id,
            filename=row.original_filename,
            mimetype=row.mimetype,
            size=row.size,
            checksum_sha256=row.checksum_sha256,
            status=row.status,
            parser_version=row.parser_version,
            chunker_version=row.chunker_version,
            embedding_provider=row.embedding_provider,
            embedding_model=row.embedding_model,
            embedding_revision=row.embedding_revision,
            embedding_dimension=row.embedding_dimension,
            distance_metric=row.distance_metric,
            desired_index_version=row.desired_index_version,
            active_index_version=row.active_index_version,
            chunk_count=row.chunk_count,
            error_code=row.error_code,
            error_summary=row.error_summary,
            attempt_count=row.attempt_count,
            created_at=row.created_at,
            updated_at=row.updated_at,
            ready_at=row.ready_at,
            deleted_at=row.deleted_at,
        )

    @staticmethod
    def _task_view(row: KnowledgeIndexTask) -> KnowledgeTaskView:
        return KnowledgeTaskView(
            id=row.id,
            task_type=row.task_type,
            status=row.status,
            phase=row.phase,
            attempt_count=row.attempt_count,
            max_attempts=row.max_attempts,
            error_code=row.error_code,
            error_summary=row.error_summary,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    @staticmethod
    def normalize_name(name: str) -> str:
        normalized = unicodedata.normalize("NFKC", name)
        return " ".join(normalized.split()).casefold()

    @staticmethod
    def _safe_filename(filename: str) -> str:
        normalized = unicodedata.normalize("NFKC", filename)
        safe = "".join(character for character in normalized if character.isalnum() or character in "._- ")
        candidate = safe.strip() or "document"
        while len(candidate.encode("utf-8")) > 500:
            candidate = candidate[:-1]
        return candidate[:255]

    @staticmethod
    def _base_profile(row: KnowledgeBase) -> EmbeddingProfile:
        return EmbeddingProfile(
            row.embedding_provider,
            row.embedding_model,
            row.embedding_dimension,
            row.distance_metric,
            None,
            row.embedding_revision,
        )

    @staticmethod
    def _current_profile() -> EmbeddingProfile:
        if settings.KNOWLEDGE_EMBEDDING_PROVIDER != "litellm" or not settings.KNOWLEDGE_EMBEDDING_MODEL:
            raise ApiException.service_unavailable("知识库 Embedding 配置不完整")
        if settings.KNOWLEDGE_EMBEDDING_DIMENSION not in settings.RESOLVED_KNOWLEDGE_EMBEDDING_ALLOWED_DIMENSIONS:
            raise ApiException.service_unavailable("知识库 Embedding 维度未进入允许列表")
        if settings.KNOWLEDGE_DISTANCE_METRIC != "COSINE":
            raise ApiException.service_unavailable("知识库 v1 仅支持 COSINE 距离")
        return EmbeddingProfile(
            settings.KNOWLEDGE_EMBEDDING_PROVIDER,
            settings.KNOWLEDGE_EMBEDDING_MODEL,
            settings.KNOWLEDGE_EMBEDDING_DIMENSION,
            settings.KNOWLEDGE_DISTANCE_METRIC,
            None,
            settings.KNOWLEDGE_EMBEDDING_REVISION,
        )

    @staticmethod
    def _require_enabled() -> None:
        if not settings.KNOWLEDGE_BASE_ENABLED:
            raise ApiException(ErrorCode.KNOWLEDGE_BASE_DISABLED, "知识库功能尚未启用", 503)
        try:
            settings.validate_knowledge_base_configuration()
        except ValueError as exc:
            raise ApiException(
                ErrorCode.KNOWLEDGE_CONFIG_INVALID,
                f"知识库配置无效：{exc}",
                503,
            ) from exc

    async def _delete_uploaded_object(self, storage_key: str) -> None:
        try:
            await self.storage.delete(storage_key)
        except Exception:
            pass
