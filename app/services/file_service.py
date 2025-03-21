import os
import uuid
from datetime import datetime
from typing import List, Dict, Any

import aiofiles
from fastapi import UploadFile
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository, ConversationRepository
from app.schemas.chat import Conversation


class FileService:
    """文件服务，负责文件上传、存储和管理"""

    def __init__(self, db: Session):
        self.db = db
        self.file_repo = FileRepository(db)
        self.base_path = settings.FILE_STORAGE_PATH
        # 确保存储目录存在
        os.makedirs(self.base_path, exist_ok=True)

    def _validate_file(self, file: UploadFile) -> None:
        """验证文件类型和大小"""
        # 检查文件类型
        allowed_mimetypes = settings.ALLOWED_FILE_TYPES
        if file.content_type not in allowed_mimetypes:
            raise ValueError(f"不支持的文件类型: {file.content_type}")

        # 文件大小检查会在上传过程中进行

    def _safe_filename(self, filename: str) -> str:
        """生成安全的文件名"""
        # 移除不安全字符
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")
        return safe_filename

    async def upload_files(self, files: List[UploadFile], conversation_id: str) -> List[str]:
        """处理文件上传并关联到对话"""
        file_ids = []

        conv_repo = ConversationRepository(self.db)
        conversation = conv_repo.get_by_id(conversation_id)

        if not conversation:
            temp_conversation = Conversation(
                id=conversation_id,
                title="新会话",
                messages=[],
                model=settings.DEFAULT_MODEL,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            conv_repo.create(temp_conversation)

        # 检查对话关联的文件数量限制
        existing_count = self.file_repo.count_conversation_files(conversation_id)
        if existing_count + len(files) > 5:
            raise ValueError(f"每个对话最多支持5个文件，当前已有{existing_count}个")

        # 确保对话目录存在
        conversation_dir = os.path.join(self.base_path, conversation_id)
        os.makedirs(conversation_dir, exist_ok=True)

        # 处理每个文件
        for file in files:
            # 验证文件类型
            self._validate_file(file)

            # 生成文件ID和存储路径
            file_id = str(uuid.uuid4())
            safe_filename = self._safe_filename(file.filename)
            file_path = os.path.join(conversation_dir, f"{file_id}_{safe_filename}")

            # 保存文件
            try:
                # 读取文件内容，检查大小
                content = await file.read()
                if len(content) > settings.MAX_FILE_SIZE:
                    raise ValueError(f"文件过大，最大允许{settings.MAX_FILE_SIZE / (1024 * 1024)}MB")

                # 写入文件
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(content)

                # 重置文件指针，以便后续可能的操作
                await file.seek(0)

                # 创建文件记录
                file_record = {
                    "id": file_id,
                    "filename": os.path.basename(file_path),
                    "original_filename": file.filename,
                    "mimetype": file.content_type,
                    "size": len(content),
                    "path": file_path,
                    "status": "processed"
                }

                # 保存到数据库
                saved_file = self.file_repo.create_file(file_record)
                self.file_repo.link_file_to_conversation(conversation_id, file_id)

                file_ids.append(file_id)
                logger.info(f"文件上传成功: {file_id}, 原始文件名: {file.filename}")

            except Exception as e:
                logger.error(f"文件上传失败: {e}, 文件名: {file.filename}")
                # 清理可能已创建的文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                raise

        return file_ids

    def get_conversation_files(self, conversation_id: str) -> List[Dict[str, Any]]:
        """获取对话关联的所有文件信息"""
        try:
            conversation_files = self.file_repo.get_conversation_files(conversation_id)
            return [
                {
                    "id": cf.file.id,
                    "filename": cf.file.original_filename,
                    "mimetype": cf.file.mimetype,
                    "size": cf.file.size,
                    "created_at": cf.created_at
                }
                for cf in conversation_files
            ]
        except Exception as e:
            logger.error(f"获取对话文件列表失败: {e}")
            return []

    def delete_file(self, file_id: str) -> bool:
        """删除文件"""
        try:
            # 获取文件信息
            file = self.file_repo.get_file_by_id(file_id)
            if not file:
                logger.warning(f"要删除的文件不存在: {file_id}")
                return False

            # 删除物理文件
            if os.path.exists(file.path):
                os.remove(file.path)

            # 删除数据库记录
            self.file_repo.delete_file(file_id)
            logger.info(f"文件删除成功: {file_id}")
            return True
        except Exception as e:
            logger.error(f"删除文件失败: {e}")
            return False
