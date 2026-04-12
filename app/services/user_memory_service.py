# app/services/user_memory_service.py
"""
用户级记忆服务

提供记忆的 CRUD 管理和 LLM 自动提取功能。
提取使用固定的轻量模型（qwen-max-latest），不跟随对话模型。
"""

import json
from functools import wraps
from typing import Optional

import litellm
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.ai.prompts import prompt_manager
from app.core.logger import app_logger as logger
from app.db.repositories import ConversationRepository, MemoryRepository

# 与 ChatService 保持一致的辅助功能模型
UTILITY_MODEL_ID = "qwen-max-latest"

# 每轮最多提取的新记忆数量
MAX_NEW_MEMORIES_PER_TURN = 3


def transactional(method):
    """Service 方法事务装饰器：成功 commit，异常 rollback"""
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            result = method(self, *args, **kwargs)
            self.db.commit()
            return result
        except Exception:
            self.db.rollback()
            raise
    return wrapper


class UserMemoryService:
    def __init__(self, db: Session):
        self.db = db
        self.repo = MemoryRepository(db)

    # ==================== CRUD ====================

    def get_active_memories(self, user_id: str) -> list:
        """获取用户所有活跃记忆（用于注入 LLM）"""
        return self.repo.get_active(user_id)

    def get_all_memories(self, user_id: str) -> list:
        """获取用户所有记忆（用于管理页面）"""
        return self.repo.get_all(user_id)

    @transactional
    def create_memory(
        self,
        user_id: str,
        content: str,
        source: str = "manual",
        conversation_id: Optional[str] = None,
    ):
        """创建新记忆"""
        return self.repo.create({
            "user_id": user_id,
            "content": content,
            "source": source,
            "conversation_id": conversation_id,
        })

    @transactional
    def update_memory(self, memory_id: str, user_id: str, content: str):
        """更新记忆内容"""
        return self.repo.update_content(memory_id, user_id, content)

    @transactional
    def toggle_memory(self, memory_id: str, user_id: str, is_active: bool):
        """切换记忆启用/停用"""
        return self.repo.toggle_active(memory_id, user_id, is_active)

    @transactional
    def delete_memory(self, memory_id: str, user_id: str) -> bool:
        """软删除记忆"""
        return self.repo.soft_delete(memory_id, user_id)

    # ==================== LLM 自动提取 ====================

    async def extract_memories(self, conversation_id: str, user_id: str) -> None:
        """
        从对话中自动提取记忆。

        在 stream_handler 落库后异步调用，提取失败不影响主流程。
        """
        try:
            # 获取最近对话内容
            conv_repo = ConversationRepository(self.db)
            conversation = conv_repo.get_by_id(conversation_id, user_id)
            if not conversation or len(conversation.messages) < 2:
                return

            recent_dialog = self._build_recent_dialog(conversation)
            if not recent_dialog:
                return

            # 获取已有记忆用于去重
            existing = self.repo.get_active(user_id)
            existing_text = self._format_existing_memories(existing)

            # 调用 LLM 提取
            prompt = prompt_manager.format_prompt(
                "extract_memories",
                existing_memories=existing_text,
                recent_messages=recent_dialog,
            )

            litellm_model, _, litellm_kwargs = self._resolve_utility_model()

            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=500,
                **litellm_kwargs,
            )

            raw = response.choices[0].message.content or ""
            self._apply_extraction_result(raw, user_id, conversation_id, existing)

        except Exception as e:
            logger.warning(f"记忆提取失败 (conv_id={conversation_id}): {e}")

    def _build_recent_dialog(self, conversation) -> str:
        """提取最近一轮用户+助手对话文本"""
        latest_user = ""
        latest_ai = ""

        for msg in reversed(conversation.messages):
            text_parts = [b.text for b in msg.content if b.type == "text"]
            text = "\n".join(text_parts)
            if not text:
                continue
            if not latest_ai and msg.role == "assistant":
                latest_ai = text[:500]  # 截断过长内容
            elif not latest_user and msg.role == "user":
                latest_user = text[:500]
            if latest_user and latest_ai:
                break

        if not latest_user:
            return ""

        lines = [f"用户: {latest_user}"]
        if latest_ai:
            lines.append(f"助手: {latest_ai}")
        return "\n".join(lines)

    def _format_existing_memories(self, memories) -> str:
        """格式化已有记忆列表"""
        if not memories:
            return "（暂无已有记忆）"
        return "\n".join(f"[{m.id}] {m.content}" for m in memories)

    def _apply_extraction_result(
        self, raw: str, user_id: str, conversation_id: str, existing: list
    ) -> None:
        """解析 LLM 返回的 JSON 并执行新增/更新"""
        try:
            # 清理可能的 markdown 代码块标记
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            cleaned = cleaned.strip()

            result = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"记忆提取 JSON 解析失败: {raw[:200]}")
            return

        existing_ids = {m.id for m in existing}

        # 处理新增
        new_memories = result.get("new", [])
        for content in new_memories[:MAX_NEW_MEMORIES_PER_TURN]:
            if isinstance(content, str) and content.strip():
                self.repo.create({
                    "user_id": user_id,
                    "content": content.strip(),
                    "source": "auto",
                    "conversation_id": conversation_id,
                })

        # 处理更新
        updates = result.get("update", [])
        for item in updates:
            if isinstance(item, dict):
                mid = item.get("id", "")
                new_content = item.get("content", "")
                if mid in existing_ids and new_content.strip():
                    self.repo.update_content(mid, user_id, new_content.strip())

        self.db.commit()

    def _resolve_utility_model(self) -> tuple:
        """解析辅助功能模型"""
        try:
            return llm_manager.resolve_model(UTILITY_MODEL_ID, self.db)
        except ValueError:
            raise ValueError("辅助模型不可用，无法提取记忆")
