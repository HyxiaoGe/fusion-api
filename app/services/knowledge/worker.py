from __future__ import annotations

import asyncio
import hashlib
import logging
import uuid
from contextlib import suppress
from dataclasses import dataclass
from typing import Callable

from sqlalchemy.orm import Session

from app.ai.embeddings.base import EmbeddingAdapter, EmbeddingError, EmbeddingProfile
from app.ai.embeddings.litellm_embedding import LiteLLMEmbeddingAdapter
from app.core.config import settings
from app.db.knowledge_repository import ClaimedKnowledgeTask, KnowledgeRepository
from app.db.models import KnowledgeDocument, KnowledgeIndexVersion
from app.services.knowledge.chunker import DeterministicKnowledgeChunker
from app.services.knowledge.milvus import (
    KnowledgeVectorError,
    KnowledgeVectorRecord,
    MilvusKnowledgeStore,
)
from app.services.knowledge.parser import KnowledgeDocumentParser, KnowledgeParseError, parse_document_isolated
from app.services.storage import get_storage_for_backend

logger = logging.getLogger(__name__)


class KnowledgeWorkerError(RuntimeError):
    def __init__(self, code: str, summary: str, *, retryable: bool):
        self.code = code
        self.summary = summary
        self.retryable = retryable
        super().__init__(summary)


@dataclass(frozen=True)
class TaskContext:
    task_id: str
    task_type: str
    knowledge_base_id: str
    document_id: str | None
    user_id: str
    index_version: str | None
    attempt_count: int
    max_attempts: int
    document: KnowledgeDocument | None
    version: KnowledgeIndexVersion | None


class KnowledgeWorker:
    """独立进程使用的知识库任务执行器。"""

    def __init__(
        self,
        session_factory: Callable[[], Session],
        *,
        worker_id: str | None = None,
        parser: KnowledgeDocumentParser | None = None,
        embedding: EmbeddingAdapter | None = None,
        vector_store: MilvusKnowledgeStore | None = None,
    ):
        self.session_factory = session_factory
        self.worker_id = worker_id or f"knowledge-worker-{uuid.uuid4()}"
        self.parser = parser or KnowledgeDocumentParser()
        self.chunker = DeterministicKnowledgeChunker(
            chunk_size=settings.KNOWLEDGE_CHUNK_SIZE,
            overlap=settings.KNOWLEDGE_CHUNK_OVERLAP,
        )
        self.embedding = embedding or LiteLLMEmbeddingAdapter()
        self.vector_store = vector_store or MilvusKnowledgeStore()

    async def run_once(self) -> bool:
        claimed = self._claim()
        if claimed is None:
            return False
        context = TaskContext(
            task_id=claimed.task.id,
            task_type=claimed.task.task_type,
            knowledge_base_id=claimed.task.knowledge_base_id,
            document_id=claimed.task.document_id,
            user_id=claimed.task.user_id,
            index_version=claimed.task.index_version,
            attempt_count=claimed.task.attempt_count,
            max_attempts=claimed.task.max_attempts,
            document=None,
            version=None,
        )
        heartbeat = None
        work = None
        try:
            context = self._load_context(claimed)
            work = asyncio.create_task(self._execute(context, claimed.lease_token))
            heartbeat = asyncio.create_task(self._heartbeat(context.task_id, claimed.lease_token))
            done, _pending = await asyncio.wait((work, heartbeat), return_when=asyncio.FIRST_COMPLETED)
            if heartbeat in done:
                heartbeat.result()
            completed_in_work = await work
            if not completed_in_work:
                self._complete(context.task_id, claimed.lease_token)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = self._classify_error(exc)
            retry_delay = min(
                settings.KNOWLEDGE_WORKER_RETRY_BASE_SECONDS * (2 ** max(context.attempt_count - 1, 0)),
                settings.KNOWLEDGE_WORKER_RETRY_MAX_SECONDS,
            )
            self._fail(
                context.task_id,
                claimed.lease_token,
                error,
                retry_delay_seconds=retry_delay,
            )
            logger.warning(
                "知识库任务失败: knowledge_base_id=%s document_id=%s task_id=%s index_version=%s "
                "attempt=%s phase=%s error_code=%s retryable=%s",
                context.knowledge_base_id,
                context.document_id,
                context.task_id,
                context.index_version,
                context.attempt_count,
                claimed.task.phase,
                error.code,
                error.retryable,
            )
        finally:
            if work is not None and not work.done():
                work.cancel()
                with suppress(asyncio.CancelledError):
                    await work
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat
        return True

    def _claim(self) -> ClaimedKnowledgeTask | None:
        with self.session_factory() as db:
            return KnowledgeRepository(db).claim_task(
                worker_id=self.worker_id,
                lease_seconds=settings.KNOWLEDGE_WORKER_LEASE_SECONDS,
                external_write_grace_seconds=max(1, int(settings.MILVUS_TIMEOUT_SECONDS * 2)),
            )

    def _load_context(self, claimed: ClaimedKnowledgeTask) -> TaskContext:
        with self.session_factory() as db:
            repo = KnowledgeRepository(db)
            document = (
                repo.get_document(claimed.task.document_id, claimed.task.user_id, include_deleted=True)
                if claimed.task.document_id
                else None
            )
            version = repo.get_index_version(claimed.task.index_version) if claimed.task.index_version else None
            if claimed.task.task_type == "index_document" and (document is None or version is None):
                raise KnowledgeWorkerError("KNOWLEDGE_TASK_RESOURCE_MISSING", "索引任务资源不存在", retryable=False)
            return TaskContext(
                task_id=claimed.task.id,
                task_type=claimed.task.task_type,
                knowledge_base_id=claimed.task.knowledge_base_id,
                document_id=claimed.task.document_id,
                user_id=claimed.task.user_id,
                index_version=claimed.task.index_version,
                attempt_count=claimed.task.attempt_count,
                max_attempts=claimed.task.max_attempts,
                document=document,
                version=version,
            )

    async def _execute(self, context: TaskContext, lease_token: str) -> bool:
        if context.task_type == "index_document":
            return await self._index_document(context, lease_token)
        elif context.task_type == "delete_document":
            await self._delete_document(context, lease_token)
        elif context.task_type == "delete_knowledge_base":
            await self._delete_knowledge_base(context, lease_token)
        elif context.task_type == "delete_index_version":
            await self._delete_index_version(context, lease_token)
        else:
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_TYPE_INVALID", "知识库任务类型无效", retryable=False)
        return False

    async def _index_document(self, context: TaskContext, lease_token: str) -> bool:
        document = context.document
        version = context.version
        if document is None or version is None or context.index_version is None:
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_RESOURCE_MISSING", "索引任务资源不存在", retryable=False)
        if version.parser_version != self.parser.VERSION or version.chunker_version != DeterministicKnowledgeChunker.VERSION:
            raise KnowledgeWorkerError(
                "KNOWLEDGE_INDEX_VERSION_UNSUPPORTED",
                "Worker 不支持该解析或切片版本",
                retryable=False,
            )
        if document.desired_index_version != context.index_version or document.status in {"deleting", "deleted"}:
            await self._cleanup_stale_version(context, version, lease_token)
            return False
        self._phase(context.task_id, lease_token, "parsing")
        storage = get_storage_for_backend(document.storage_backend)
        try:
            content = await storage.download(document.storage_key)
        except FileNotFoundError as exc:
            raise KnowledgeWorkerError("KNOWLEDGE_DOCUMENT_MISSING", "原始文档不存在", retryable=False) from exc
        checksum = await asyncio.to_thread(lambda: hashlib.sha256(content).hexdigest())
        if checksum != document.checksum_sha256:
            raise KnowledgeWorkerError("KNOWLEDGE_DOCUMENT_CHECKSUM_MISMATCH", "原始文档校验失败", retryable=False)
        if document.mimetype in {
            "application/pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }:
            sections = await asyncio.to_thread(
                parse_document_isolated,
                content,
                mimetype=document.mimetype,
                filename=document.original_filename,
                timeout_seconds=settings.KNOWLEDGE_PARSE_TIMEOUT_SECONDS,
            )
        else:
            sections = await asyncio.to_thread(
                self.parser.parse,
                content,
                mimetype=document.mimetype,
                filename=document.original_filename,
            )
        self._phase(context.task_id, lease_token, "chunking")
        version_chunker = DeterministicKnowledgeChunker(
            chunk_size=version.chunk_size,
            overlap=version.chunk_overlap,
        )
        chunks = await asyncio.to_thread(
            version_chunker.chunk,
            sections,
            document_id=document.id,
            index_version=context.index_version,
        )
        self._phase(context.task_id, lease_token, "embedding")
        profile = self._version_profile(version)
        vectors = []
        for offset in range(0, len(chunks), settings.KNOWLEDGE_EMBEDDING_BATCH_SIZE):
            batch = chunks[offset : offset + settings.KNOWLEDGE_EMBEDDING_BATCH_SIZE]
            vectors.extend(await self.embedding.embed([chunk.text for chunk in batch], profile))
        self._phase(context.task_id, lease_token, "writing")
        records = [
            KnowledgeVectorRecord(
                chunk=chunk,
                vector=vector,
                user_id=document.user_id,
                knowledge_base_id=document.knowledge_base_id,
                document_id=document.id,
                index_version=context.index_version,
                filename=document.original_filename,
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        write_state = self._index_write_state(context, lease_token)
        if write_state == "stale":
            await self._cleanup_stale_version(context, version, lease_token)
            return False
        if write_state != "allowed":
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)
        with self.session_factory() as db:
            if not KnowledgeRepository(db).replace_chunk_manifest(
                task_id=context.task_id,
                lease_token=lease_token,
                version=version,
                filename=document.original_filename,
                chunks=chunks,
            ):
                raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)
        await self.vector_store.upsert(profile, records)
        with self.session_factory() as db:
            finalized = KnowledgeRepository(db).finalize_document_index(
                task_id=context.task_id,
                lease_token=lease_token,
                document_id=document.id,
                index_version=context.index_version,
                chunk_count=len(chunks),
                cleanup_max_attempts=settings.KNOWLEDGE_WORKER_MAX_ATTEMPTS,
            )
        if finalized == "activated":
            return True
        if finalized == "stale":
            await self._cleanup_stale_version(context, version, lease_token)
            return False
        # lease 被新 Worker 接管时，旧 Worker 不能删除同一 version 的幂等 upsert。
        raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    async def _delete_document(self, context: TaskContext, lease_token: str) -> None:
        self._phase(context.task_id, lease_token, "deleting")
        if context.document_id is None:
            return
        with self.session_factory() as db:
            repo = KnowledgeRepository(db)
            document = repo.get_document(context.document_id, context.user_id, include_deleted=True)
            versions = list(document.index_versions) if document is not None else []
        if document is None or document.status == "deleted":
            return
        for version in versions:
            await self.vector_store.delete_index_version(self._version_profile(version), version.id)
        await self._delete_storage_object(document)
        with self.session_factory() as db:
            deleted = KnowledgeRepository(db).mark_document_deleted(
                document.id,
                task_id=context.task_id,
                lease_token=lease_token,
            )
        if not deleted:
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    async def _delete_knowledge_base(self, context: TaskContext, lease_token: str) -> None:
        self._phase(context.task_id, lease_token, "deleting")
        with self.session_factory() as db:
            documents = KnowledgeRepository(db).documents_for_cleanup(context.knowledge_base_id)
            detached = [(document, list(document.index_versions)) for document in documents]
        for document, versions in detached:
            for version in versions:
                await self.vector_store.delete_index_version(self._version_profile(version), version.id)
            await self._delete_storage_object(document)
        with self.session_factory() as db:
            deleted = KnowledgeRepository(db).mark_knowledge_base_deleted(
                context.knowledge_base_id,
                task_id=context.task_id,
                lease_token=lease_token,
            )
        if not deleted:
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    async def _delete_index_version(self, context: TaskContext, lease_token: str) -> None:
        self._phase(context.task_id, lease_token, "deleting")
        if context.version is None or context.index_version is None:
            return
        await self.vector_store.delete_index_version(self._version_profile(context.version), context.index_version)
        with self.session_factory() as db:
            deleted = KnowledgeRepository(db).mark_index_version_deleted(
                context.index_version,
                task_id=context.task_id,
                lease_token=lease_token,
            )
        if not deleted:
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    async def _cleanup_stale_version(
        self,
        context: TaskContext,
        version: KnowledgeIndexVersion,
        lease_token: str,
    ) -> None:
        await self.vector_store.delete_index_version(self._version_profile(version), version.id)
        with self.session_factory() as db:
            deleted = KnowledgeRepository(db).mark_index_version_deleted(
                version.id,
                task_id=context.task_id,
                lease_token=lease_token,
            )
        if not deleted:
            raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    async def _delete_storage_object(self, document: KnowledgeDocument) -> None:
        storage = get_storage_for_backend(document.storage_backend)
        if not await storage.exists(document.storage_key):
            return
        if not await storage.delete(document.storage_key):
            raise KnowledgeWorkerError("KNOWLEDGE_STORAGE_DELETE_FAILED", "原始文档删除失败", retryable=True)

    async def _heartbeat(self, task_id: str, lease_token: str) -> None:
        while True:
            await asyncio.sleep(settings.KNOWLEDGE_WORKER_HEARTBEAT_SECONDS)
            with self.session_factory() as db:
                renewed = KnowledgeRepository(db).renew_lease(
                    task_id,
                    lease_token,
                    lease_seconds=settings.KNOWLEDGE_WORKER_LEASE_SECONDS,
                )
            if not renewed:
                raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    def _phase(self, task_id: str, lease_token: str, phase: str) -> None:
        with self.session_factory() as db:
            if not KnowledgeRepository(db).update_task_phase(task_id, lease_token, phase):
                raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    def _index_write_state(self, context: TaskContext, lease_token: str) -> str:
        if context.document_id is None or context.index_version is None:
            return "stale"
        with self.session_factory() as db:
            return KnowledgeRepository(db).index_task_write_state(
                task_id=context.task_id,
                lease_token=lease_token,
                document_id=context.document_id,
                index_version=context.index_version,
            )

    def _complete(self, task_id: str, lease_token: str) -> None:
        with self.session_factory() as db:
            if not KnowledgeRepository(db).complete_task(task_id, lease_token):
                raise KnowledgeWorkerError("KNOWLEDGE_TASK_LEASE_LOST", "知识库任务租约已失效", retryable=True)

    def _fail(
        self,
        task_id: str,
        lease_token: str,
        error: KnowledgeWorkerError,
        *,
        retry_delay_seconds: int,
    ) -> None:
        with self.session_factory() as db:
            KnowledgeRepository(db).fail_task(
                task_id,
                lease_token,
                error_code=error.code,
                error_summary=error.summary,
                retryable=error.retryable,
                retry_delay_seconds=retry_delay_seconds,
                external_write_grace_seconds=max(1, int(settings.MILVUS_TIMEOUT_SECONDS * 2)),
            )

    @staticmethod
    def _version_profile(version: KnowledgeIndexVersion) -> EmbeddingProfile:
        return EmbeddingProfile(
            version.embedding_provider,
            version.embedding_model,
            version.embedding_dimension,
            version.distance_metric,
            version.collection_name,
            version.embedding_revision,
        )

    @staticmethod
    def _classify_error(exc: Exception) -> KnowledgeWorkerError:
        if isinstance(exc, KnowledgeWorkerError):
            return exc
        if isinstance(exc, KnowledgeParseError):
            return KnowledgeWorkerError(exc.code, exc.summary, retryable=False)
        if isinstance(exc, EmbeddingError):
            return KnowledgeWorkerError(exc.code, exc.summary, retryable=exc.retryable)
        if isinstance(exc, KnowledgeVectorError):
            return KnowledgeWorkerError(exc.code, exc.summary, retryable=exc.retryable)
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return KnowledgeWorkerError("KNOWLEDGE_DEPENDENCY_UNAVAILABLE", "知识库依赖暂时不可用", retryable=True)
        return KnowledgeWorkerError("KNOWLEDGE_WORKER_INTERNAL_ERROR", "知识库任务处理失败", retryable=True)
