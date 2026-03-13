import math
from typing import List, Optional, Dict, Any

from sqlalchemy.orm import Session

from app.db.repositories import ConversationRepository
from app.schemas.chat import Conversation, Message


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
        existing = self.repo.get_by_id(conversation.id, conversation.user_id)
        if existing:
            self.repo.update(conversation)
        else:
            self.repo.create(conversation)
        return True

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """获取特定对话"""
        return self.repo.get_by_id(conversation_id, user_id)

    def get_all_conversations(self, user_id: str) -> List[Conversation]:
        """获取指定用户的所有对话"""
        return self.repo.get_all(user_id)

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """获取特定消息"""
        return self.repo.get_message_by_id(message_id)

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        """更新消息"""
        return self.repo.update_message(message_id, update_data)

    def get_conversations_paginated(self, user_id: str, page: int = 1, page_size: int = 20) -> Dict[str, Any]:
        """分页获取对话列表"""
        conversations, total = self.repo.get_paginated(user_id, page, page_size)

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

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """删除特定对话"""
        return self.repo.delete(conversation_id, user_id)

    def create_message(self, message: Message, conversation_id: str) -> Message:
        """创建新消息"""
        return self.repo.create_message(message, conversation_id)
