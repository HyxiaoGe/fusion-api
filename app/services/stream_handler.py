import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Callable

from app.ai.llm_manager import llm_manager
from app.schemas.chat import Message, Conversation
from app.services.message_processor import MessageProcessor


class StreamHandler:
    def __init__(self, db, memory_service):
        self.db = db
        self.memory_service = memory_service
        self.message_processor = MessageProcessor(db)

    async def generate_normal_stream(self, provider, model, messages, conversation_id) -> AsyncGenerator:
        """生成常规流式响应（无推理模式）"""
        llm = llm_manager.get_model(provider=provider, model=model)
        full_response = ""

        # 流式响应处理
        for chunk in llm.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else chunk

            if content:
                full_response += content
                yield f"data: {json.dumps({'content': content, 'conversation_id': conversation_id})}\n\n"
                await asyncio.sleep(0.01)

        # 流结束后，将完整响应保存到对话历史
        await self.save_stream_response(conversation_id, full_response)
        yield f"data: {json.dumps({'content': '[DONE]', 'conversation_id': conversation_id})}\n\n"

    async def save_stream_response(self, conversation_id, response_text):
        """保存流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 创建并添加AI响应消息
                ai_message = Message(
                    role="assistant",
                    content=response_text
                )
                conversation.messages.append(ai_message)
                conversation.updated_at = datetime.now()

                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logging.error(f"保存流式响应失败: {str(e)}")

    async def generate_reasoning_stream(self, provider, model, messages, conversation_id) -> AsyncGenerator:
        """生成带推理功能的流式响应，适用于支持推理能力的模型"""
        
        # 构造发送事件的辅助函数
        async def send_event(event_type, content=None):
            data = {"type": event_type, "conversation_id": conversation_id}
            if content is not None:
                data["content"] = content
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield await send_event("reasoning_start")
        
        # 获取模型
        llm = llm_manager.get_model(provider=provider, model=model)
        reasoning_result = ""
        answer_result = ""
        
        # 状态跟踪
        in_reasoning_phase = True
        reasoning_completed = False
        answering_started = False

        # 根据不同模型准备参数
        stream_kwargs = {}
        if provider == "deepseek" and model == "deepseek-reasoner":
            stream_kwargs["reasoning_effort"] = "medium"
        
        # 流式获取响应
        for chunk in llm.stream(messages, **stream_kwargs):
            has_reasoning = False
            has_answer = False
            
            # 处理思考过程
            if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
                reasoning_content = chunk.additional_kwargs['reasoning_content']
                if reasoning_content and reasoning_content.strip():
                    reasoning_result += reasoning_content
                    yield await send_event("reasoning_content", reasoning_content)
                    has_reasoning = True
            
            # 处理最终答案
            content = chunk.content if hasattr(chunk, 'content') else chunk
            
            # Deepseek模型需要额外检查content不等于reasoning_result
            if provider == "deepseek" and model == "deepseek-reasoner":
                if content and content.strip() and content != reasoning_result:
                    # 如果推理阶段结束但还没发送完成信号
                    if in_reasoning_phase and not reasoning_completed and not has_reasoning:
                        in_reasoning_phase = False
                        reasoning_completed = True
                        yield await send_event("reasoning_complete")
                    
                    if not answering_started:
                        answering_started = True
                        yield await send_event("answering_start")
                    
                    answer_result += content
                    yield await send_event("answering_content", content)
                    has_answer = True
            else:
                if content and content.strip():
                    # 如果开始接收到答案内容，但还没有结束推理阶段
                    if in_reasoning_phase and not reasoning_completed and not has_reasoning:
                        in_reasoning_phase = False
                        reasoning_completed = True
                        yield await send_event("reasoning_complete")
                    
                    if not answering_started:
                        answering_started = True
                        yield await send_event("answering_start")
                    
                    answer_result += content
                    yield await send_event("answering_content", content)
                    has_answer = True
        
        # 确保所有阶段正确结束
        if in_reasoning_phase and not reasoning_completed:
            yield await send_event("reasoning_complete")
        
        if not answering_started:
            yield await send_event("answering_start")
        
        yield await send_event("answering_complete")

        # 保存到数据库的最终结果
        await self.save_stream_response_with_reasoning(
            conversation_id=conversation_id,
            response_text=answer_result,
            reasoning_text=reasoning_result
        )

        # 完成标志
        yield await send_event("done")

    async def save_stream_response_with_reasoning(self, conversation_id, response_text, reasoning_text):
        """保存流式响应和推理过程到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 如果有推理内容，创建推理消息
                if reasoning_text:
                    reasoning_message = Message(
                        role="reasoning",
                        content=reasoning_text
                    )
                    conversation.messages.append(reasoning_message)

                # 创建并添加AI响应消息
                ai_message = Message(
                    role="assistant",
                    content=response_text
                )
                conversation.messages.append(ai_message)
                conversation.updated_at = datetime.now()

                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logging.error(f"保存推理流式响应失败: {str(e)}") 