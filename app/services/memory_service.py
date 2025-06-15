import logging
from typing import List, Optional, Dict, Any
import math

from sqlalchemy.orm import Session

from app.db.repositories import ConversationRepository
from app.schemas.chat import Conversation, Message

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

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """获取特定消息"""
        try:
            return self.repo.get_message_by_id(message_id)
        except Exception as e:
            logger.error(f"获取消息失败: {e}")
            return None

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        """更新消息"""
        try:
            return self.repo.update_message(message_id, update_data)
        except Exception as e:
            logger.error(f"更新消息失败: {e}")
            return None

    def get_conversations_paginated(self, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """分页获取对话列表"""
        try:
            conversations, total = self.repo.get_paginated(page, page_size)
            
            # 计算分页信息
            total_pages = math.ceil(total / page_size) if total > 0 else 0
            has_next = page < total_pages
            has_prev = page > 1
            
            return {
                "items": conversations,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }
        except Exception as e:
            logger.error(f"分页获取对话失败: {e}")
            return {
                "items": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            }

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除特定对话"""
        try:
            return self.repo.delete(conversation_id)
        except Exception as e:
            logger.error(f"删除对话失败: {e}")
            return False

    def create_message(self, message: Message, conversation_id: str) -> Message:
        """创建新消息"""
        try:
            return self.repo.create_message(message, conversation_id)
        except Exception as e:
            logger.error(f"创建消息失败: {e}")
            raise
