from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional
from app.schemas.chat import Conversation, Message
from app.ai.vectorstores.chroma_store import ChromaVectorStore
from app.core.logger import app_logger as logger
import threading


class VectorService:
    """向量服务 - 负责消息和对话的向量化处理"""

    # 单例实例
    _instance = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls, db: Optional[Session] = None):
        """单例模式获取向量服务实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = VectorService(db)
        return cls._instance

    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.vector_store = ChromaVectorStore()

    def vectorize_message(self, message: Message, conversation_id: str) -> bool:
        """向量化单条消息"""
        try:
            return self.vector_store.add_message(
                message_id=message.id,
                conversation_id=conversation_id,
                role=message.role,
                content=message.content
            )
        except Exception as e:
            logger.error(f"向量化消息失败: {e}")
            return False

    def vectorize_conversation(self, conversation: Conversation) -> bool:
        """向量化整个对话"""
        try:
            # 先向量化所有消息
            for message in conversation.messages:
                self.vectorize_message(message, conversation.id)

            # 创建对话摘要
            summary = self._generate_conversation_summary(conversation)

            # 向量化对话摘要
            return self.vector_store.add_conversation_summary(
                conversation_id=conversation.id,
                title=conversation.title,
                summary=summary,
                model=conversation.model
            )
        except Exception as e:
            logger.error(f"向量化对话失败: {e}")
            return False

    def _generate_conversation_summary(self, conversation: Conversation) -> str:
        """生成对话摘要"""
        # 提取所有用户消息
        user_messages = [msg.content for msg in conversation.messages if msg.role == "user"]

        # 如果消息太少，直接拼接
        if len(user_messages) <= 3:
            return " ".join(user_messages)

        # 否则使用前两条和最后一条消息组成摘要
        return f"{user_messages[0]} {user_messages[1]} ... {user_messages[-1]}"

    def delete_conversation_vectors(self, conversation_id: str) -> bool:
        """删除特定对话的所有向量数据"""
        try:
            return self.vector_store.delete_conversation_data(conversation_id)
        except Exception as e:
            logger.error(f"删除对话向量数据失败: {e}")
            return False

    def search_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相关对话"""
        try:
            return self.vector_store.search_conversations(query, limit)
        except Exception as e:
            logger.error(f"搜索对话失败: {e}")
            return []

    def search_messages(self, query: str, conversation_id: Optional[str] = None,
                        limit: int = 5) -> List[Dict[str, Any]]:
        """搜索相关消息"""
        try:
            return self.vector_store.search_messages(query, limit, conversation_id)
        except Exception as e:
            logger.error(f"搜索消息失败: {e}")
            return []

    def get_relevant_context(self, query: str, conversation_id: Optional[str] = None,
                             limit: int = 3) -> List[Dict[str, Any]]:
        """获取与查询相关的上下文"""
        try:
            return self.vector_store.get_related_context(query, conversation_id, limit)
        except Exception as e:
            logger.error(f"获取相关上下文失败: {e}")
            return []