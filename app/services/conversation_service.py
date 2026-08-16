import math
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.db.knowledge_repository import KnowledgeRepository
from app.db.repositories import ConversationRepository
from app.schemas.chat import Conversation, ConversationSummary, Message
from app.schemas.response import ApiException, ErrorCode


class ConversationService:
    """会话服务 — 管理对话与消息的持久化"""

    def __init__(self, db: Session):
        self.db = db
        self.repo = ConversationRepository(db)
        self.knowledge_repo = KnowledgeRepository(db)

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
        """分页获取对话列表，返回 ConversationSummary（不含 messages）"""
        conversations, total = self.repo.get_paginated(user_id, page, page_size)

        # 转为轻量 ConversationSummary，不携带 messages
        summaries = [
            ConversationSummary(
                id=conv.id,
                model_id=conv.model_id,
                title=conv.title,
                knowledge_base_ids=conv.knowledge_base_ids,
                created_at=conv.created_at,
                updated_at=conv.updated_at,
            )
            for conv in conversations
        ]

        total_pages = math.ceil(total / page_size) if total > 0 else 0
        has_next = page < total_pages
        has_prev = page > 1

        return {
            "items": summaries,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "has_next": has_next,
            "has_prev": has_prev,
        }

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """删除特定对话"""
        return self.repo.delete(conversation_id, user_id)

    def create_message(self, message: Message, conversation_id: str) -> Message:
        """创建新消息"""
        return self.repo.create_message(message, conversation_id)

    def prepare_message_retry(
        self,
        *,
        conversation_id: str,
        user_id: str,
        user_message_id: str,
        assistant_message_id: str | None,
    ) -> tuple[Message, Message | None]:
        """锁定并校验可重试的最后一轮，返回原 user 与可复用 assistant。"""

        conversation = self.repo.lock_owned_model(conversation_id, user_id)
        if conversation is None:
            raise ApiException.not_found("会话不存在或无权访问")
        rows = self.repo.list_messages_for_retry(conversation_id)
        target_index = next(
            (index for index, row in enumerate(rows) if row.id == user_message_id),
            None,
        )
        if target_index is None or rows[target_index].role != "user":
            raise ApiException.not_found("待重试的用户消息不存在")

        expected_assistant = rows[target_index + 1] if target_index + 1 < len(rows) else None
        if expected_assistant is not None and expected_assistant.role != "assistant":
            raise ApiException.conflict("只能重试会话中的最后一轮消息")
        if assistant_message_id is None:
            if expected_assistant is not None or target_index != len(rows) - 1:
                raise ApiException.conflict("重试目标与会话最后一轮不一致")
        else:
            if expected_assistant is None or expected_assistant.id != assistant_message_id:
                raise ApiException.conflict("重试目标与原回答不一致")
            if target_index + 1 != len(rows) - 1:
                raise ApiException.conflict("只能重试会话中的最后一轮消息")

        user_message = self.repo.convert_message_model(rows[target_index])
        assistant_message = (
            self.repo.convert_message_model(expected_assistant) if expected_assistant is not None else None
        )
        return user_message, assistant_message

    def lock_conversation_for_message_write(self, conversation_id: str, user_id: str) -> Conversation:
        """串行化消息写入，并在锁内重载会话历史供后续模型调用。"""

        conversation = self.repo.lock_owned_model(conversation_id, user_id)
        if conversation is None:
            raise ApiException.not_found("会话不存在或无权访问")
        self.db.expire(conversation, ["messages", "knowledge_base_selections"])
        return self.repo.convert_conversation_model(conversation)

    def claim_assistant_message_generation(
        self,
        *,
        conversation_id: str,
        message_id: str,
        task_id: str,
    ) -> None:
        """在当前事务内切换 assistant 写所有权，同时保留原回答。"""

        self.repo.claim_assistant_message_generation(
            conversation_id=conversation_id,
            message_id=message_id,
            task_id=task_id,
        )

    def claim_unanswered_user_generation(
        self,
        *,
        conversation_id: str,
        message_id: str,
        task_id: str,
    ) -> None:
        """在当前事务内切换未回答 user 的生成所有权。"""

        self.repo.claim_unanswered_user_generation(
            conversation_id=conversation_id,
            message_id=message_id,
            task_id=task_id,
        )

    def replace_assistant_message(
        self,
        message: Message,
        conversation_id: str,
        *,
        generation_task_id: str | None = None,
    ) -> Message:
        """用重试结果原位替换 assistant 消息。"""

        replaced = self.repo.replace_assistant_message(
            message,
            conversation_id,
            generation_task_id=generation_task_id,
        )
        if replaced is None:
            self.db.rollback()
            raise ApiException.conflict("回答已被更新请求取代")
        return replaced

    def create_retry_assistant_message(
        self,
        message: Message,
        conversation_id: str,
        *,
        retry_user_message_id: str,
        generation_task_id: str,
    ) -> Message:
        """为未回答 user 原子创建回答；被新请求取代时拒绝迟到写。"""

        created = self.repo.create_retry_assistant_message(
            message,
            conversation_id,
            retry_user_message_id=retry_user_message_id,
            generation_task_id=generation_task_id,
        )
        if created is None:
            self.db.rollback()
            raise ApiException.conflict("回答已被更新请求取代")
        return created

    def reserve_message_sequence_pair(self) -> tuple[int, int]:
        """为同一轮 user/assistant 消息预留稳定顺序号。"""
        return self.repo.reserve_message_sequence_pair()

    def replace_knowledge_base_selection(
        self,
        *,
        conversation_id: str,
        user_id: str,
        knowledge_base_ids: list[str],
    ) -> None:
        """校验后在当前事务内替换会话知识库集合，不单独提交。"""

        conversation = self.validate_knowledge_base_selection(
            conversation_id=conversation_id,
            user_id=user_id,
            knowledge_base_ids=knowledge_base_ids,
        )
        self.repo.replace_knowledge_base_selection(conversation, knowledge_base_ids)

    def validate_knowledge_base_selection(
        self,
        *,
        conversation_id: str,
        user_id: str,
        knowledge_base_ids: list[str],
    ) -> Any:
        """在当前事务内锁定会话并校验知识库所有权与可检索状态。"""

        if len(knowledge_base_ids) > 5:
            raise ApiException.bad_request("每个会话最多选择 5 个知识库")
        if len(set(knowledge_base_ids)) != len(knowledge_base_ids):
            raise ApiException.bad_request("knowledge_base_ids 不能重复")
        if any("\x00" in knowledge_base_id for knowledge_base_id in knowledge_base_ids):
            raise ApiException.bad_request("knowledge_base_ids 不能包含 NUL 字符")
        conversation = self.repo.lock_owned_model(conversation_id, user_id)
        if conversation is None:
            raise ApiException.not_found("会话不存在或无权访问")
        bases = self.knowledge_repo.get_knowledge_bases_by_ids(user_id, knowledge_base_ids)
        if len(bases) != len(knowledge_base_ids):
            raise ApiException.not_found("知识库不存在或无权访问")
        if any(base.status != "active" for base in bases):
            raise ApiException(ErrorCode.KNOWLEDGE_BASE_NOT_READY, "知识库尚不可用于问答", 409)
        if knowledge_base_ids:
            ready_documents = self.knowledge_repo.get_ready_documents(user_id, knowledge_base_ids)
            ready_base_ids = {row.document.knowledge_base_id for row in ready_documents}
            if ready_base_ids != set(knowledge_base_ids):
                raise ApiException(
                    ErrorCode.KNOWLEDGE_BASE_NOT_READY,
                    "至少一个知识库尚无就绪文档",
                    409,
                )
        return conversation
