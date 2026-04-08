import asyncio
import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

import aiofiles
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository, ConversationRepository
from app.processor.file_processor import FileProcessor
from app.processor.image_processor import ImageProcessor
from app.schemas.chat import Conversation
from app.services.storage import get_storage
from app.services.storage.base import StorageBackend


# 图片 MIME 类型集合
IMAGE_MIME_TYPES = {
    "image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp",
    "image/heic", "image/heif",
}


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

    def _validate_file(self, file: UploadFile) -> None:
        """验证文件类型和大小"""
        allowed_mimetypes = settings.ALLOWED_FILE_TYPES
        if file.content_type not in allowed_mimetypes:
            raise ValueError(f"不支持的文件类型: {file.content_type}")

    def _safe_filename(self, filename: str) -> str:
        """生成安全的文件名"""
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        return safe_filename

    async def upload_files(self, files: List[UploadFile], user_id: str, conversation_id: str, provider: str, model: str) -> List[Dict[str, Any]]:
        """
        处理文件上传并关联到对话。

        Returns:
            包含 file_id 和 thumbnail_url 的字典列表
        """
        results = []

        conv_repo = ConversationRepository(self.db)
        conversation = conv_repo.get_by_id(conversation_id, user_id)

        if not conversation:
            model = model or settings.DEFAULT_MODEL
            temp_conversation = Conversation(
                id=conversation_id,
                user_id=user_id,
                title="新会话",
                messages=[],
                model_id=model or settings.DEFAULT_MODEL,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            conv_repo.create(temp_conversation)

        # 检查对话关联的文件数量限制
        existing_count = self.file_repo.count_conversation_files(conversation_id)
        if existing_count + len(files) > 5:
            raise ValueError(f"每个对话最多支持5个文件，当前已有{existing_count}个")

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
                    result = await self._process_and_store_image(
                        content, mime_type, conversation_id, file_id
                    )
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
                    storage_key = f"{conversation_id}/{file_id}/{safe_filename}"

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
                    asyncio.create_task(
                        self._parse_file_with_llm(file_id=file_id, file_path=file_path)
                    )

                logger.info(f"文件上传成功: {file_id}, 原始文件名: {file.filename}, 类型: {file.content_type}")

            except Exception as e:
                logger.error(f"文件上传失败: {e}, 文件名: {file.filename}")
                raise

        return results

    async def _process_and_store_image(
        self, content: bytes, mime_type: str, conversation_id: str, file_id: str
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
        storage_key = f"{conversation_id}/{file_id}/processed{ext}"
        thumbnail_key = f"{conversation_id}/{file_id}/thumbnail{ext}"

        # 上传到存储后端
        await self.storage.upload(storage_key, processed["processed"], processed["mime_type"])
        await self.storage.upload(thumbnail_key, processed["thumbnail"], processed["mime_type"])

        # 获取缩略图访问 URL
        thumbnail_url = await self.storage.get_url(
            thumbnail_key, expires=settings.MINIO_PRESIGN_EXPIRES
        )

        return {
            "storage_key": storage_key,
            "thumbnail_key": thumbnail_key,
            "thumbnail_url": thumbnail_url,
            "mime_type": processed["mime_type"],
            "width": processed["width"],
            "height": processed["height"],
        }

    async def get_file_url(self, file_id: str, user_id: str, variant: str = "thumbnail") -> Optional[str]:
        """获取文件访问 URL"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return None

        key = file.thumbnail_key if variant == "thumbnail" else file.storage_key
        if not key:
            return None

        return await self.storage.get_url(key, expires=settings.MINIO_PRESIGN_EXPIRES)

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

        try:
            data = await self.storage.download(key)
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
                mime_types=[file.mimetype]
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
    def _serialize_file_summary(file_obj) -> Dict[str, Any]:
        """统一序列化文件列表摘要"""
        return {
            "id": file_obj.id,
            "filename": file_obj.original_filename,
            "mimetype": file_obj.mimetype,
            "size": file_obj.size,
            "status": file_obj.status,
            "thumbnail_key": getattr(file_obj, "thumbnail_key", None),
            "width": getattr(file_obj, "width", None),
            "height": getattr(file_obj, "height", None),
        }

    def get_file_status(self, file_id: str, user_id: str) -> Dict[str, Any]:
        """获取文件状态，并验证用户权限"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return None
        return self._serialize_file_status(file)

    def get_conversation_files(self, conversation_id: str) -> List[Dict[str, Any]]:
        """获取对话关联的所有文件信息"""
        files = self.file_repo.get_conversation_files(conversation_id)
        return [self._serialize_file_summary(f.file) for f in files]

    def get_conversation_files_for_user(self, conversation_id: str, user_id: str) -> Optional[List[Dict[str, Any]]]:
        """获取用户有权访问的对话文件列表"""
        conv_repo = ConversationRepository(self.db)
        conversation = conv_repo.get_by_id(conversation_id, user_id)
        if not conversation:
            return None
        return self.get_conversation_files(conversation_id)

    def get_files_by_user(self, user_id: str) -> List[Dict[str, Any]]:
        """获取用户的所有文件"""
        files = self.file_repo.get_files_by_user_id(user_id)
        return [self._serialize_file_summary(file) for file in files]

    @staticmethod
    def _serialize_file_status(file_obj) -> Dict[str, Any]:
        """统一序列化文件状态响应"""
        return {
            "id": file_obj.id,
            "status": file_obj.status,
            "processing_result": file_obj.processing_result,
            "thumbnail_url": None,  # 前端会单独请求 URL
        }

    async def delete_file(self, file_id: str, user_id: str) -> bool:
        """删除文件，并验证用户权限"""
        file = self.file_repo.get_file_by_id(file_id, user_id=user_id)
        if not file:
            return False

        # 删除存储后端中的文件
        try:
            if file.storage_key:
                await self.storage.delete(file.storage_key)
            if file.thumbnail_key:
                await self.storage.delete(file.thumbnail_key)
            # 兼容旧数据：删除本地路径文件
            if file.path and os.path.exists(file.path):
                os.remove(file.path)
        except Exception as e:
            logger.error(f"删除物理文件失败: {e}")

        return self.file_repo.delete_file(file_id, user_id)
