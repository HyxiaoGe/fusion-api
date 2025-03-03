import json

from app.ai.llm_manager import llm_manager
from app.schemas.chat import ChatResponse, Message, Conversation
from app.services.memory_service import MemoryService
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime
from fastapi.responses import StreamingResponse
import asyncio


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.memory_service = MemoryService(db)

    async def process_message(
            self,
            model: str,
            message: str,
            conversation_id: Optional[str] = None,
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None
    ) -> ChatResponse:
        """处理用户消息并获取AI响应"""
        # 获取或创建会话
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id)

        if not conversation:
            conversation_id = str(uuid.uuid4())
            conversation = Conversation(
                id=conversation_id,
                title=message[:30] + "..." if len(message) > 30 else message,
                model=model,
                messages=[]
            )

        # 记录用户消息
        user_message = Message(
            role="user",
            content=message
        )
        conversation.messages.append(user_message)

        # 获取AI模型并生成响应
        llm = llm_manager.get_model(model)

        # 准备聊天历史
        chat_history = []
        for msg in conversation.messages:
            chat_history.append({"role": msg.role, "content": msg.content})

        # 生成响应
        ai_response = ""
        if stream:
            ai_response = await self.generate_stream_response(model, self._prepare_chat_messages(chat_history))
        else:
            # 从聊天历史中提取过去的消息
            messages = self._prepare_chat_messages(chat_history)

            # 调用LLM获取响应
            response = llm.invoke(messages)
            if hasattr(response, 'content'):  # ChatModel返回的响应
                ai_response = response.content
            else:  # 普通LLM返回的响应
                ai_response = response

        # 记录AI响应消息
        ai_message = Message(
            role="assistant",
            content=ai_response
        )
        conversation.messages.append(ai_message)
        conversation.updated_at = datetime.now()

        # 保存到数据库
        self.memory_service.save_conversation(conversation)

        # 构建并返回响应
        return ChatResponse(
            id=str(uuid.uuid4()),
            model=model,
            message=ai_message,
            conversation_id=conversation_id
        )

    async def generate_stream_response(self, model, messages):
        """生成流式响应"""
        llm = llm_manager.get_model(model)

        async def stream_generator():
            for chunk in llm.stream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = chunk

                if content:
                    yield f"data: {json.dumps({'content': content})}\n\n"
                await asyncio.sleep(0.01)
            yield f"data: [DONE]\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )

    def _prepare_chat_messages(self, chat_history):
        """准备发送给LLM的消息格式"""
        from langchain.schema import HumanMessage, AIMessage, SystemMessage

        messages = []
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))

        return messages

    def get_all_conversations(self) -> List[Conversation]:
        """获取所有对话"""
        return self.memory_service.get_all_conversations()

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取特定对话"""
        return self.memory_service.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除特定对话"""
        return self.memory_service.delete_conversation(conversation_id)