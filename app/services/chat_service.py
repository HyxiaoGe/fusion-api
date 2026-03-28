# app/services/chat_service.py
import asyncio
import uuid as uuid_mod
from typing import Any, Dict, List, Optional, Union

import litellm
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.ai.prompts import prompt_manager
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.schemas.chat import (
    ChatResponse, Conversation, Message,
    TextBlock, FileBlock, Usage,
)
from app.services.memory_service import MemoryService
from app.services.stream_handler import StreamHandler, stream_redis_as_sse
from app.services.task_manager import register_task
from app.services.chat.utils import ChatUtils


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.memory_service = MemoryService(db)
        self.file_repo = FileRepository(db)
        self.stream_handler = StreamHandler()

    async def process_message(
        self,
        model_id: str,
        message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        stream: bool = True,
        options: Optional[Dict[str, Any]] = None,
        file_ids: Optional[List[str]] = None,
    ) -> Union[StreamingResponse, ChatResponse]:
        """处理用户消息，路由到流式或非流式响应"""
        if options is None:
            options = {}

        # 解析模型调用参数
        litellm_model, provider, litellm_kwargs = llm_manager.resolve_model(model_id, self.db)

        # 获取或创建会话
        conversation = self._get_or_create_conversation(
            conversation_id, user_id, model_id, message
        )
        is_new_conversation = not conversation_id

        # 构造用户消息 content blocks
        user_content = [TextBlock(type="text", text=message)]
        if file_ids:
            for fid in file_ids:
                file_info = self.file_repo.get_file_by_id(fid)
                if file_info:
                    user_content.append(FileBlock(
                        type="file",
                        file_id=fid,
                        filename=file_info.original_filename,
                        mime_type=file_info.mimetype,
                    ))

        user_message = Message(role="user", content=user_content)

        # 持久化用户消息
        if is_new_conversation:
            self.memory_service.save_conversation(conversation)
            self.db.commit()
        self.memory_service.create_message(user_message, conversation.id)
        self.db.commit()

        conversation.messages.append(user_message)

        # 将会话历史转为 LiteLLM 所需的 dict 格式
        lm_messages = self._build_llm_messages(conversation.messages)

        # 注入文件内容到 LLM 上下文（纯文本注入，不影响 content blocks 存储）
        if file_ids:
            file_contents = self.file_repo.get_parsed_file_content(file_ids)
            if file_contents:
                lm_messages = self._inject_file_content(lm_messages, message, file_contents)

        if stream:
            # 预分配 assistant 消息 ID 和 task ID
            assistant_message_id = str(uuid_mod.uuid4())
            task_id = str(uuid_mod.uuid4())

            # 启动后台生成任务（独立于 HTTP 连接生命周期）
            task = asyncio.create_task(
                self.stream_handler.generate_to_redis(
                    conversation_id=conversation.id,
                    user_id=user_id,
                    model_id=model_id,
                    litellm_model=litellm_model,
                    litellm_kwargs=litellm_kwargs,
                    provider=provider,
                    messages=lm_messages,
                    assistant_message_id=assistant_message_id,
                    task_id=task_id,
                    options=options,
                )
            )
            register_task(conversation.id, task, task_id)

            # SSE 从 Redis Stream 读取，不直接调 LLM
            return StreamingResponse(
                stream_redis_as_sse(
                    conversation_id=conversation.id,
                    message_id=assistant_message_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )
        else:
            return await self._handle_non_stream(
                litellm_model, model_id, litellm_kwargs,
                lm_messages, conversation.id, options,
            )

    def _get_or_create_conversation(
        self,
        conversation_id: Optional[str],
        user_id: str,
        model_id: str,
        message: str,
    ) -> Conversation:
        """获取已有会话，或初始化新会话对象"""
        if conversation_id:
            existing = self.memory_service.get_conversation(conversation_id, user_id)
            if existing:
                return existing

        return Conversation(
            id=conversation_id or str(uuid_mod.uuid4()),
            user_id=user_id,
            model_id=model_id,
            title=message[:30] + "..." if len(message) > 30 else message,
            messages=[],
        )

    @staticmethod
    def _build_llm_messages(messages: List[Message]) -> List[dict]:
        """
        将 content blocks 消息列表转为 LLM 可消费的 dict 格式。
        thinking block 不传给 LLM（避免污染上下文）。
        """
        result = []
        for msg in messages:
            text_parts = [
                block.text
                for block in msg.content
                if block.type == "text"
            ]
            text = "\n".join(text_parts)
            if text:
                result.append({"role": msg.role, "content": text})
        return result

    @staticmethod
    def _inject_file_content(
        messages: List[dict],
        original_message: str,
        file_contents: Dict[str, str],
    ) -> List[dict]:
        """将文件内容注入到最后一条用户消息的文本中"""
        if not messages:
            return messages

        combined = "\n\n".join(
            f"文件内容 ({i + 1}):\n{content}"
            for i, content in enumerate(file_contents.values())
        )
        enhanced = f"{original_message}\n\n以下是相关文件内容，请结合这些内容回答：\n{combined}"

        result = messages.copy()
        result[-1] = {"role": "user", "content": enhanced}
        return result

    async def _handle_non_stream(
        self,
        litellm_model: str,
        model_id: str,
        litellm_kwargs: dict,
        messages: List[dict],
        conversation_id: str,
        options: dict,
    ) -> ChatResponse:
        """处理非流式响应"""
        response = await litellm.acompletion(
            model=litellm_model,
            messages=messages,
            stream=False,
            **litellm_kwargs,
        )

        content_text = response.choices[0].message.content or ""
        usage_data = None
        if response.usage:
            usage_data = Usage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )

        assistant_message = Message(
            role="assistant",
            content=[TextBlock(type="text", text=content_text)],
            model_id=model_id,
            usage=usage_data,
        )
        self.memory_service.create_message(assistant_message, conversation_id)
        self.db.commit()

        return ChatResponse(
            conversation_id=conversation_id,
            message=assistant_message,
        )

    async def generate_title(
        self,
        user_id: str,
        conversation_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """基于会话最后一条用户消息生成标题，并写回数据库"""
        conversation = self.memory_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError(f"找不到会话: {conversation_id}")

        # 提取最后一条用户消息文本
        seed_text = ""
        for msg in reversed(conversation.messages):
            if msg.role == "user":
                parts = [b.text for b in msg.content if b.type == "text"]
                seed_text = "\n".join(parts)
                if seed_text:
                    break

        if not seed_text:
            raise ValueError("会话中没有可用的用户消息")

        # 生成失败时的回退标题
        fallback_title = seed_text[:30] + "..." if len(seed_text) > 30 else seed_text

        try:
            prompt = prompt_manager.format_prompt("generate_title", content=seed_text)
            litellm_model, _, litellm_kwargs = llm_manager.resolve_model(
                conversation.model_id, self.db
            )
            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=30,
                **litellm_kwargs,
            )
            raw = response.choices[0].message.content or ""

            # 标题清理：去除引号、常见前缀、控制长度
            title = raw.strip().strip('"').strip("'")
            for prefix in ["标题：", "标题:", "Title:", "Title："]:
                if title.startswith(prefix):
                    title = title[len(prefix):].strip()
            title = title[:30] if len(title) > 30 else title
            title = title or fallback_title

        except Exception as e:
            logger.error(f"生成标题失败，使用回退标题: {e}")
            title = fallback_title

        # 写回数据库
        self.memory_service.repo.update_title(conversation_id, title)
        self.db.commit()

        return title

    async def generate_suggested_questions(
        self,
        user_id: str,
        conversation_id: str,
        options: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """基于会话内容生成推荐问题"""
        conversation = self.memory_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError(f"找不到会话: {conversation_id}")

        # 提取最近一轮对话内容
        dialog_content = self._build_recent_dialog_content(conversation)

        if not dialog_content:
            return [
                "有什么我可以帮您解答的问题吗？",
                "您想了解更多哪方面的信息？",
                "还有其他我能帮助您的事情吗？"
            ]

        try:
            prompt = prompt_manager.format_prompt("generate_suggested_questions", content=dialog_content)
            litellm_model, _, litellm_kwargs = llm_manager.resolve_model(
                conversation.model_id, self.db
            )
            response = await litellm.acompletion(
                model=litellm_model,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
                max_tokens=200,
                **litellm_kwargs,
            )
            raw = response.choices[0].message.content or ""
            questions = ChatUtils.parse_questions(raw)[:3]

            # 写回到最后一条 assistant 消息，刷新后随消息一起返回
            last_msg = self.memory_service.repo.get_last_assistant_message(conversation_id)
            if last_msg and questions:
                self.memory_service.repo.update_message_suggested_questions(last_msg.id, questions)
                self.db.commit()

            return questions

        except Exception as e:
            logger.error(f"生成推荐问题失败: {e}")
            return [
                "您对这个主题还有其他问题吗？",
                "您想了解更多相关信息吗？",
                "您想要探讨这个话题的哪些方面？"
            ]

    def _build_recent_dialog_content(self, conversation: Conversation) -> str:
        """提取最近一轮用户/助手对话内容"""
        latest_user = ""
        latest_ai = ""

        for msg in reversed(conversation.messages):
            text_parts = [b.text for b in msg.content if b.type == "text"]
            text = "\n".join(text_parts)
            if not text:
                continue
            if not latest_ai and msg.role == "assistant":
                latest_ai = text
            elif not latest_user and msg.role == "user":
                latest_user = text
            if latest_user and latest_ai:
                break

        lines = []
        if latest_user:
            lines.append(f"用户: {latest_user}")
        if latest_ai:
            lines.append(f"助手: {latest_ai}")
        return "\n".join(lines)

    # ==================== CRUD 代理方法 ====================

    def get_conversation(self, conversation_id: str, user_id: str):
        return self.memory_service.get_conversation(conversation_id, user_id)

    def get_all_conversations(self, user_id: str):
        return self.memory_service.get_all_conversations(user_id)

    def get_conversations_paginated(self, user_id: str, page: int = 1, page_size: int = 20):
        return self.memory_service.get_conversations_paginated(user_id, page, page_size)

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        updated = self.memory_service.update_message(message_id, update_data)
        if updated:
            self.db.commit()
        return updated

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        return self.memory_service.delete_conversation(conversation_id, user_id)
