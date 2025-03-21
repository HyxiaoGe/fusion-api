import logging
from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.repositories import ConversationRepository
from app.schemas.chat import Conversation

logger = logging.getLogger(__name__)


class MemoryService:
    """
    内存服务 - 管理对话历史和上下文记忆
    使用PostgreSQL数据库实现
    """

    def __init__(self, db: Session):
        self.db = db
        self.repo = ConversationRepository(db)

    def save_conversation(self, conversation: Conversation) -> bool:
        """保存或更新对话"""
        try:
            existing = self.repo.get_by_id(conversation.id)
            if existing:
                self.repo.update(conversation)
            else:
                self.repo.create(conversation)
            return True
        except Exception as e:
            logger.error(f"保存对话失败: {e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取特定对话"""
        try:
            return self.repo.get_by_id(conversation_id)
        except Exception as e:
            logger.error(f"获取对话失败: {e}")
            return None

    def get_all_conversations(self) -> List[Conversation]:
        """获取所有对话"""
        try:
            return self.repo.get_all()
        except Exception as e:
            logger.error(f"获取所有对话失败: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除特定对话"""
        try:
            return self.repo.delete(conversation_id)
        except Exception as e:
            logger.error(f"删除对话失败: {e}")
            return False
