import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.file_token import generate_file_token
from app.core.logger import app_logger as logger
from app.db.repositories import ConversationRepository, FileRepository
from app.processor.file_processor import FileProcessor
from app.processor.image_processor import ImageProcessor
from app.schemas.chat import Conversation
from app.services.storage import get_storage, get_storage_for_backend
from app.services.storage.base import StorageBackend

# 图片 MIME 类型集合
IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
    "image/heic",
    "image/heif",
}


class UpdatedFileView:
    """合并原文件对象与本次更新字段的只读视图。"""

    def __init__(self, base: Any, updates: Dict[str, Any]):
        self._base = base
        self._updates = updates

    def __getattr__(self, name: str) -> Any:
        if name in self._updates:
            return self._updates[name]
        return getattr(self._base, name)


def is_image_mime(mime_type: str) -> bool:
    """判断 MIME 类型是否为图片"""
    return mime_type in IMAGE_MIME_TYPES or mime_type.startswith("image/")


class FileService:
    """文件服务，负责文件上传、存储和管理"""

    def __init__(self, db: Session):
        self.db = db
        self.file_repo = FileRepository(db)
        self.base_path = settings.FILE_STORAGE_PATH
        # 确保存储目录存在（兼容本地模式）
        os.makedirs(self.base_path, exist_ok=True)
        # 初始化文件处理器
        self.file_processor = FileProcessor()
        self.image_processor = ImageProcessor()
        # 获取存储后端
        self.storage: StorageBackend = get_storage()

    @staticmethod
    def _storage_for_file(file_obj: Any) -> StorageBackend:
        """按文件记录选择读写存储后端，兼容切换 OSS 前的历史文件。"""
        return get_storage_for_backend(getattr(file_obj, "storage_backend", None))

    @staticmethod
    def _sign_local_url(url: str, file_id: str, expires: int) -> str:
        """
        为本地存储模式的相对路径 URL 追加签名 token。
        MinIO presigned URL（绝对路径）直接原样返回。
        """
        if not url.startswith("/"):
            return url
        token = generate_file_token(file_id, expires)
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}token={token}"

    def _validate_file(self, file: UploadFile) -> None:
        """验证文件类型和大小"""
        self._validate_file_metadata(file.content_type)

    def _validate_file_metadata(self, mimetype: str, size: Optional[int] = None) -> None:
        """验证文件元信息"""
        if mimetype not in settings.ALLOWED_FILE_TYPES:
            raise ValueError(f"不支持的文件类型: {mimetype}")
        if size is not None and size > settings.MAX_FILE_SIZE:
            raise ValueError(f"文件过大，最大允许{settings.MAX_FILE_SIZE / (1024 * 1024)}MB")

    def _safe_filename(self, filename: str) -> str:
        """生成安全的文件名"""
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        return safe_filename.strip() or "file"

    def _ensure_upload_conversation(self, user_id: str, conversation_id: str, model: str) -> None:
        """确保上传目标会话存在。"""
        conv_repo = ConversationRepository(self.db)
        conversation = conv_repo.get_by_id(conversation_id, user_id)

        if not conversation:
            temp_conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                title="新会话",
                messages=[],
                model_id=model or settings.DEFAULT_MODEL,
                created_at=datetime.now(),
                updated_at=datetime.now(),
            )
            conv_repo.create(temp_conversation)

    async def _ensure_file_quota(self, conversation_id: str, incoming_count: int) -> None:
        """校验会话文件数量上限。"""
        await self._cleanup_stale_uploading_files(conversation_id)
        existing_count = self.file_repo.count_conversation_files(conversation_id)
        if existing_count + incoming_count > 5:
            raise ValueError(f"每个对话最多支持5个文件，当前已有{existing_count}个")

    async def _cleanup_stale_uploading_files(self, conversation_id: str) -> None:
        """清理过期直传占位记录，避免中断上传永久占用配额。"""
        cutoff = datetime.now() - timedelta(seconds=settings.DIRECT_UPLOAD_STALE_SECONDS)
        stale_files = self.file_repo.get_stale_uploading_files(conversation_id, cutoff)
        for stale_file in stale_files:
            try:
                await self._delete_storage_objects(stale_file)
                self.file_repo.delete_file(stale_file.id, stale_file.user_id)
                logger.info("已清理过期直传文件: file_id=%s", stale_file.id)
            except Exception as e:
                logger.warning("清理过期直传文件失败: file_id=%s, error=%s", getattr(stale_file, "id", None), e)

    @staticmethod
    def _file_storage_root(user_id: str, conversation_id: str, file_id: str) -> str:
        prefix_parts = [part for part in settings.FILE_STORAGE_KEY_PREFIX.strip("/").split("/") if part]
        parts = [*prefix_parts, "users", user_id, "conversations", conversation_id, "files", file_id]
        return "/".join(parts)

    @classmethod
    def _original_storage_key(cls, user_id: str, conversation_id: str, file_id: str) -> str:
        """直传原文件对象 key。"""
        return f"{cls._file_storage_root(user_id, conversation_id, file_id)}/original"

    async def upload_files(
        self, files: List[UploadFile], user_id: str, conversation_id: str, provider: str, model: str
    ) -> List[Dict[str, Any]]:
        """
        处理文件上传并关联到对话。

        Returns:
            包含 file_id 和 thumbnail_url 的字典列表
        """
        results = []

        self._ensure_upload_conversation(user_id, conversation_id, model)
        await self._ensure_file_quota(conversation_id, len(files))

        # 确保对话目录存在（兼容本地模式）
        conversation_dir = os.path.join(self.base_path, conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)

        for file in files:
            self._validate_file(file)

            file_id = str(uuid.uuid4())
            safe_filename = self._safe_filename(file.filename)

            try:
                content = await file.read()
                if len(content) > settings.MAX_FILE_SIZE:
                    raise ValueError(f"文件过大，最大允许{settings.MAX_FILE_SIZE / (1024 * 1024)}MB")

                await file.seek(0)

                mime_type = file.content_type
                thumbnail_url = None
                storage_key = None
                thumbnail_key = None
                width = None
                height = None

                if is_image_mime(mime_type):
                    # 图片走预处理管线
                    result = await self._process_and_store_image(content, mime_type, user_id, conversation_id, file_id)
                    storage_key = result["storage_key"]
                    thumbnail_key = result["thumbnail_key"]
                    thumbnail_url = result["thumbnail_url"]
                    mime_type = result["mime_type"]
                    width = result["width"]
                    height = result["height"]
                    file_status = "processed"  # 图片预处理完成即可用
                    file_path = storage_key  # 存储 key 作为路径
                else:
                    # 非图片文件：沿用现有流程写入存储
                    file_path_local = os.path.join(conversation_dir, f"{file_id}_{safe_filename}")
                    storage_key = self._original_storage_key(user_id, conversation_id, file_id)

                    await self.storage.upload(storage_key, content, mime_type)

                    # 兼容本地模式：同时写一份到本地（供 file_processor 读取）
                    if settings.STORAGE_BACKEND == "local":
                        file_path = os.path.join(self.base_path, storage_key)
                    else:
                        # MinIO 模式：仍需本地临时文件供 LLM 解析
                        async with aiofiles.open(file_path_local, "wb") as f:
                            await f.write(content)
                        file_path = file_path_local

                    file_status = "parsing"

                # 创建文件记录
                file_record = {
                    "id": file_id,
                    "user_id": user_id,
                    "filename": os.path.basename(f"{file_id}_{safe_filename}"),
                    "original_filename": file.filename,
                    "mimetype": mime_type,
                    "size": len(content),
                    "path": file_path,
                    "status": file_status,
                    "processing_result": None,
                    "storage_key": storage_key,
                    "thumbnail_key": thumbnail_key,
                    "storage_backend": settings.STORAGE_BACKEND,
                    "width": width,
                    "height": height,
                }

                self.file_repo.create_file(file_record)
                self.file_repo.link_file_to_conversation(conversation_id, file_id)

                result_item = {"file_id": file_id, "thumbnail_url": thumbnail_url}
                results.append(result_item)

                # 非图片文件：启动异步 LLM 解析
                if not is_image_mime(file.content_type):
                    asyncio.create_task(self._parse_file_with_llm(file_id=file_id, file_path=file_path))

                logger.info(f"文件上传成功: {file_id}, 原始文件名: {file.filename}, 类型: {file.content_type}")

            except Exception as e:
                logger.error(f"文件上传失败: {e}, 文件名: {file.filename}")
                raise

        return results

    async def create_direct_upload(
        self,
        user_id: str,
        conversation_id: str,
        provider: str,
        model: str,
        filename: str,
        mimetype: str,
        size: int,
    ) -> Dict[str, Any]:
        """创建 OSS 直传占位记录并签发 PUT URL。"""
        self._validate_file_metadata(mimetype, size)
        self._ensure_upload_conversation(user_id, conversation_id, model)
        await self._ensure_file_quota(conversation_id, 1)

        file_id = str(uuid.uuid4())
        safe_filename = self._safe_filename(filename)
        original_key = self._original_storage_key(user_id, conversation_id, file_id)
        upload = await self.storage.get_upload_url(
            original_key,
            content_type=mimetype,
            expires=settings.MINIO_PRESIGN_EXPIRES,
        )

        file_record = {
            "id": file_id,
            "user_id": user_id,
            "filename": os.path.basename(f"{file_id}_{safe_filename}"),
            "original_filename": filename,
            "mimetype": mimetype,
            "size": size,
            "path": original_key,
            "status": "uploading",
            "processing_result": None,
            "storage_key": original_key,
            "thumbnail_key": None,
            "storage_backend": settings.STORAGE_BACKEND,
            "width": None,
            "height": None,
        }
        self.file_repo.create_file(file_record)
        self.file_repo.link_file_to_conversation(conversation_id, file_id)

        return {
            "file_id": file_id,
            "upload_url": upload["url"],
            "method": upload["method"],
            "headers": upload["headers"],
            "expires_in": upload["expires_in"],
        }

    async def complete_direct_upload(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """确认直传对象已落 OSS，并接入现有图片/文件处理管线。"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            raise FileNotFoundError("文件不存在或无权访问")

        if file.status in {"processed", "parsing"}:
            return await self._build_upload_result_from_file(file)

        original_key = file.path or file.storage_key
        if not original_key:
            raise ValueError("文件尚未上传完成，请稍后重试")
        conversation_id = self._conversation_id_from_key(original_key)
        if not await self._storage_object_exists(original_key):
            raise ValueError("文件尚未上传完成，请稍后重试")

        actual_size = await self.storage.get_size(original_key)
        if actual_size > settings.MAX_FILE_SIZE:
            await self._delete_storage_objects(file)
            self.file_repo.delete_file(file_id, user_id)
            raise ValueError(f"文件过大，最大允许{settings.MAX_FILE_SIZE / (1024 * 1024)}MB")

        content = await self.storage.download(original_key)

        if is_image_mime(file.mimetype):
            result = await self._process_and_store_image(content, file.mimetype, user_id, conversation_id, file_id)
            updates = {
                "status": "processed",
                "mimetype": result["mime_type"],
                "storage_key": result["storage_key"],
                "thumbnail_key": result["thumbnail_key"],
                "width": result["width"],
                "height": result["height"],
                "size": actual_size,
                "processing_result": None,
            }
            self.file_repo.update_file(file_id=file_id, updates=updates)
            file = UpdatedFileView(file, updates)
            return await self._build_upload_result_from_file(file, thumbnail_url=result["thumbnail_url"])

        file_path = await self._write_direct_upload_temp_file(
            conversation_id=conversation_id,
            file_id=file_id,
            safe_filename=self._safe_filename(file.original_filename),
            content=content,
        )
        updates = {
            "status": "parsing",
            "path": file_path,
            "storage_key": original_key,
            "size": actual_size,
            "processing_result": None,
        }
        self.file_repo.update_file(file_id=file_id, updates=updates)
        asyncio.create_task(self._parse_file_with_llm(file_id=file_id, file_path=file_path))
        file = UpdatedFileView(file, updates)
        return await self._build_upload_result_from_file(file)

    @staticmethod
    def _conversation_id_from_key(key: str) -> str:
        """从对象 key 取会话 ID。"""
        prefix_parts = [part for part in settings.FILE_STORAGE_KEY_PREFIX.strip("/").split("/") if part]
        parts = key.split("/")
        offset = len(prefix_parts)
        matches_prefix = parts[:offset] == prefix_parts if offset else True
        has_current_shape = (
            matches_prefix
            and len(parts) >= offset + 7
            and parts[offset] == "users"
            and bool(parts[offset + 1])
            and parts[offset + 2] == "conversations"
            and bool(parts[offset + 3])
            and parts[offset + 4] == "files"
            and bool(parts[offset + 5])
        )
        if not has_current_shape:
            raise ValueError("文件对象 key 不符合当前存储结构，请重新上传")
        return parts[offset + 3]

    async def _write_direct_upload_temp_file(
        self,
        conversation_id: str,
        file_id: str,
        safe_filename: str,
        content: bytes,
    ) -> str:
        """为非图片直传文件落一份本地临时文件，供现有解析器读取。"""
        conversation_dir = os.path.join(self.base_path, conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)
        file_path = os.path.join(conversation_dir, f"{file_id}_{safe_filename}")
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)
        return file_path

    async def _build_upload_result_from_file(
        self,
        file_obj: Any,
        thumbnail_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """构造上传完成响应。"""
        thumbnail_key = getattr(file_obj, "thumbnail_key", None)
        if thumbnail_url is None and thumbnail_key:
            thumbnail_url = await self.get_file_url(file_obj.id, file_obj.user_id, "thumbnail")
        return {
            "file_id": file_obj.id,
            "thumbnail_url": thumbnail_url,
            "status": file_obj.status,
            "filename": file_obj.original_filename,
            "mimetype": file_obj.mimetype,
            "size": file_obj.size,
        }

    async def _process_and_store_image(
        self, content: bytes, mime_type: str, user_id: str, conversation_id: str, file_id: str
    ) -> Dict[str, Any]:
        """图片预处理 + 存储处理图和缩略图"""
        # Pillow 预处理
        processed = await self.image_processor.process(content, mime_type)

        # 确定文件扩展名
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }
        ext = ext_map.get(processed["mime_type"], ".jpg")

        # 存储键
        storage_root = self._file_storage_root(user_id, conversation_id, file_id)
        storage_key = f"{storage_root}/processed{ext}"
        thumbnail_key = f"{storage_root}/thumbnail{ext}"

        # 上传到存储后端
        await self.storage.upload(storage_key, processed["processed"], processed["mime_type"])
        await self.storage.upload(thumbnail_key, processed["thumbnail"], processed["mime_type"])

        # 获取缩略图访问 URL（本地存储模式追加签名 token）
        thumbnail_url = await self.storage.get_url(thumbnail_key, expires=settings.MINIO_PRESIGN_EXPIRES)
        thumbnail_url = self._sign_local_url(thumbnail_url, file_id, settings.MINIO_PRESIGN_EXPIRES)

        return {
            "storage_key": storage_key,
            "thumbnail_key": thumbnail_key,
            "thumbnail_url": thumbnail_url,
            "mime_type": processed["mime_type"],
            "width": processed["width"],
            "height": processed["height"],
        }

    async def get_file_url(self, file_id: str, user_id: str, variant: str = "thumbnail") -> Optional[str]:
        """获取文件访问 URL（MinIO 返回 presigned URL，本地存储返回带签名的代理 URL）"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return None

        key = file.thumbnail_key if variant == "thumbnail" else file.storage_key
        if not key:
            return None

        storage = self._storage_for_file(file)
        if not await self._storage_object_exists(key, storage):
            logger.warning("文件实体缺失，跳过 URL 签发: file_id=%s, variant=%s, key=%s", file_id, variant, key)
            return None

        url = await storage.get_url(key, expires=settings.MINIO_PRESIGN_EXPIRES)
        return self._sign_local_url(url, file_id, settings.MINIO_PRESIGN_EXPIRES)

    async def get_file_content(self, file_id: str, user_id: str, variant: str = "thumbnail") -> Optional[tuple]:
        """
        获取文件内容（用于本地模式代理）。

        Returns:
            (bytes, mime_type) 或 None
        """
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return None

        key = file.thumbnail_key if variant == "thumbnail" else file.storage_key
        if not key:
            return None

        storage = self._storage_for_file(file)
        try:
            data = await storage.download(key)
            return data, file.mimetype
        except FileNotFoundError:
            return None

    async def _parse_file_with_llm(self, file_id: str, file_path: str) -> None:
        """使用LLM模型解析文件内容"""
        try:
            file = self.file_repo.get_file_by_id(file_id)
            if not file:
                logger.warning(f"要解析的文件不存在: {file_id}")
                return

            response = await self.file_processor.process_files(
                file_paths=[file_path],
                query=self._get_file_parsing_prompt(file.mimetype, file.original_filename),
                mime_types=[file.mimetype],
            )

            error_message = response.get("error")
            if error_message:
                raise RuntimeError(error_message)

            parsed_content = self._normalize_parsed_content(response.get("content"))
            if not parsed_content:
                raise ValueError("无法解析文件内容")

            self.file_repo.update_file(
                file_id=file_id,
                updates=self._build_processed_file_updates(parsed_content),
            )

            logger.info(f"文件 {file_id} 解析成功")

        except (RuntimeError, ValueError) as e:
            logger.warning(f"文件解析失败 {file_id}: {e}")
            self._mark_file_error(file_id, str(e))
        except Exception as e:
            logger.exception(f"文件解析异常 {file_id}")
            self._mark_file_error(file_id, str(e))

    def _mark_file_error(self, file_id: str, message: str) -> None:
        """统一记录文件解析失败状态"""
        self.file_repo.update_file(
            file_id=file_id,
            updates={
                "status": "error",
                "processing_result": {"status": "error", "message": message},
            },
        )

    @staticmethod
    def _build_success_processing_result() -> Dict[str, str]:
        """统一构造文件解析成功结果"""
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
        }

    def _build_processed_file_updates(self, parsed_content: str) -> Dict[str, Any]:
        """统一构造文件解析成功后的更新内容"""
        return {
            "status": "processed",
            "parsed_content": parsed_content,
            "processing_result": self._build_success_processing_result(),
        }

    @staticmethod
    def _normalize_parsed_content(content: Any) -> Optional[str]:
        """将解析结果规范化为可持久化文本"""
        if content is None:
            return None
        if isinstance(content, (dict, list)):
            return json.dumps(content, ensure_ascii=False)
        text = str(content).strip()
        return text or None

    def _get_file_parsing_prompt(self, mimetype: str, filename: str) -> str:
        """根据文件类型生成解析提示词"""
        if mimetype.startswith("image/"):
            return f"请详细描述这张图片的内容。图片文件名: {filename}"
        elif mimetype == "application/pdf":
            return f"请分析这个PDF文档(文件名:{filename})的内容并提供详细摘要。"
        elif mimetype.startswith("text/"):
            return f"请分析这个文本文件(文件名:{filename})的内容并提供详细摘要。"
        else:
            return f"请分析这个文件(文件名:{filename})的内容并提供详细摘要。"

    @staticmethod
    def _serialize_created_at(created_at: Any) -> Any:
        """序列化文件创建时间，便于前端直接消费。"""
        if hasattr(created_at, "isoformat"):
            return created_at.isoformat()
        return created_at

    @staticmethod
    def _extract_processing_error_message(file_status: Optional[str], processing_result: Any) -> Optional[str]:
        """从处理结果中提取用户可见错误信息。"""
        if not isinstance(processing_result, dict):
            return None
        if file_status != "error" and processing_result.get("status") != "error":
            return None
        return processing_result.get("message") or processing_result.get("error")

    async def _serialize_file_summary(self, file_obj) -> Dict[str, Any]:
        """统一序列化文件列表摘要"""
        thumbnail_key = getattr(file_obj, "thumbnail_key", None)
        thumbnail_url = None
        if thumbnail_key:
            try:
                storage = self._storage_for_file(file_obj)
                if await self._storage_object_exists(thumbnail_key, storage):
                    thumbnail_url = await storage.get_url(thumbnail_key, expires=settings.MINIO_PRESIGN_EXPIRES)
                    thumbnail_url = self._sign_local_url(thumbnail_url, file_obj.id, settings.MINIO_PRESIGN_EXPIRES)
                else:
                    logger.warning("文件缩略图实体缺失: file_id=%s, key=%s", file_obj.id, thumbnail_key)
            except Exception as e:
                logger.warning(f"文件缩略图 URL 构造失败: file_id={file_obj.id}, error={e}")

        file_status = file_obj.status

        return {
            "id": file_obj.id,
            "filename": file_obj.original_filename,
            "mimetype": file_obj.mimetype,
            "size": file_obj.size,
            "status": file_status,
            "thumbnail_key": thumbnail_key,
            "thumbnail_url": thumbnail_url,
            "width": getattr(file_obj, "width", None),
            "height": getattr(file_obj, "height", None),
            "created_at": self._serialize_created_at(getattr(file_obj, "created_at", None)),
            "error_message": self._extract_processing_error_message(
                file_status,
                getattr(file_obj, "processing_result", None),
            ),
        }

    def get_file_status(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """获取文件状态，并验证用户权限"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return None
        return self._serialize_file_status(file)

    async def get_conversation_files(self, conversation_id: str) -> List[Dict[str, Any]]:
        """获取对话关联的所有文件信息"""
        files = self.file_repo.get_conversation_files(conversation_id)
        return [await self._serialize_file_summary(f.file) for f in files]

    async def get_conversation_files_for_user(
        self, conversation_id: str, user_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """获取用户有权访问的对话文件列表"""
        conv_repo = ConversationRepository(self.db)
        conversation = conv_repo.get_by_id(conversation_id, user_id)
        if not conversation:
            return None
        return await self.get_conversation_files(conversation_id)

    async def get_files_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有文件"""
        files = self.file_repo.get_files_by_user_id(user_id)
        return [await self._serialize_file_summary(file) for file in files]

    @staticmethod
    def _serialize_file_status(file_obj) -> Dict[str, Any]:
        """统一序列化文件状态响应"""
        return {
            "id": file_obj.id,
            "status": file_obj.status,
            "processing_result": file_obj.processing_result,
            "thumbnail_url": None,  # 前端会单独请求 URL
        }

    async def _storage_object_exists(self, key: str, storage: Optional[StorageBackend] = None) -> bool:
        """检查存储实体是否存在，避免给丢失文件签发无效 URL。"""
        try:
            return await (storage or self.storage).exists(key)
        except Exception as e:
            logger.warning("文件实体存在性检查失败: key=%s, error=%s", key, e)
            return False

    async def _delete_storage_objects(self, file_obj: Any) -> None:
        """删除文件记录对应的物理对象，按记录里的 storage_backend 路由。"""
        storage = self._storage_for_file(file_obj)
        deleted_keys = set()
        storage_key = getattr(file_obj, "storage_key", None)
        thumbnail_key = getattr(file_obj, "thumbnail_key", None)
        file_path = getattr(file_obj, "path", None)

        if storage_key:
            await storage.delete(storage_key)
            deleted_keys.add(storage_key)
        if thumbnail_key:
            await storage.delete(thumbnail_key)
            deleted_keys.add(thumbnail_key)
        # path 可能是本地临时文件，也可能是原始对象 key。
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        elif file_path and file_path not in deleted_keys:
            await storage.delete(file_path)

    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """删除文件，并验证用户权限"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return False

        # 删除存储后端中的文件
        try:
            await self._delete_storage_objects(file)
        except Exception as e:
            logger.error(f"删除物理文件失败: {e}")

        return self.file_repo.delete_file(file_id, user_id)
