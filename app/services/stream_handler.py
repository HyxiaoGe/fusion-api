import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, AsyncGenerator, Callable, Optional

from app.ai.llm_manager import llm_manager
from app.schemas.chat import Message, Conversation
from app.services.message_processor import MessageProcessor
from app.core.logger import app_logger as logger
from openai import AsyncOpenAI
import inspect
from app.constants import MessageRoles, MessageTypes

class StreamHandler:
    def __init__(self, db, memory_service):
        self.db = db
        self.memory_service = memory_service
        self.message_processor = MessageProcessor(db)

    async def _create_placeholder_message(self, conversation_id: str, message_type: str, turn_id: Optional[str]) -> Message:
        """在数据库中创建一个空的占位消息并返回"""
        placeholder_message = Message(
            role=MessageRoles.ASSISTANT,
            type=message_type,
            content="",  # 内容为空
            turn_id=turn_id
        )
        # 直接创建消息，而不是修改整个会话
        return self.memory_service.create_message(placeholder_message, conversation_id)

    async def generate_normal_stream(self, provider, model, messages, conversation_id, options=None, turn_id=None) -> AsyncGenerator:
        """生成常规流式响应（无推理模式）"""
        if options is None:
            options = {}
            
        # 1. 创建占位消息
        assistant_message = await self._create_placeholder_message(conversation_id, MessageTypes.ASSISTANT_CONTENT, turn_id)
        assistant_message_id = assistant_message.id

        llm = llm_manager.get_model(provider=provider, model=model, options=options)
        full_response = ""

        # 2. 流式响应处理，并在事件中加入message_id
        for chunk in llm.stream(messages):
            content = chunk.content if hasattr(chunk, 'content') else chunk

            if content:
                full_response += content
                event_data = {
                    "content": content,
                    "conversation_id": conversation_id,
                    "message_id": assistant_message_id
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(0.01)

        # 3. 流结束后，更新占位消息
        await self.update_stream_response(assistant_message_id, full_response)
        
        # 4. 发送结束信号
        done_data = {
            "content": "[DONE]",
            "conversation_id": conversation_id,
            "message_id": assistant_message_id
        }
        yield f"data: {json.dumps(done_data)}\n\n"
        

    async def update_stream_response(self, message_id: str, response_text: str):
        """更新流式响应到数据库"""
        try:
            update_data = {"content": response_text}
            self.memory_service.update_message(message_id, update_data)
            self.db.commit()  # 提交事务，确保数据持久化
        except Exception as e:
            logging.error(f"更新流式响应失败: {str(e)}")

    async def save_stream_response(self, conversation_id, response_text, turn_id=None):
        """保存流式响应到对话历史 - 此方法将在后续被废弃"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 创建并添加AI响应消息
                ai_message = Message(
                    role=MessageRoles.ASSISTANT,
                    type=MessageTypes.ASSISTANT_CONTENT,
                    content=response_text,
                    turn_id=turn_id
                )
                conversation.messages.append(ai_message)
                conversation.updated_at = datetime.now()

                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logging.error(f"保存流式响应失败: {str(e)}")

    async def generate_reasoning_stream(self, provider, model, messages, conversation_id, options=None, turn_id=None) -> AsyncGenerator:
        """生成带推理功能的流式响应，适用于支持推理能力的模型"""
        if options is None:
            options = {}
        
        # 1. 创建占位消息
        reasoning_message = await self._create_placeholder_message(conversation_id, MessageTypes.REASONING_CONTENT, turn_id)
        assistant_message = await self._create_placeholder_message(conversation_id, MessageTypes.ASSISTANT_CONTENT, turn_id)

        # 构造发送事件的辅助函数
        async def send_event(event_type, content=None, message_id=None):
            data = {"type": event_type, "conversation_id": conversation_id}
            if content is not None:
                data["content"] = content
            if message_id:
                data["message_id"] = message_id
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        yield await send_event("reasoning_start", message_id=reasoning_message.id)
        
        # 获取模型
        llm = llm_manager.get_model(provider=provider, model=model, options=options)
        reasoning_result = ""
        answer_result = ""
        
        # 状态跟踪
        in_reasoning_phase = True
        reasoning_completed = False
        answering_started = False

        # 根据不同模型准备参数
        stream_kwargs = {}
        if provider == "deepseek":
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
                    yield await send_event("reasoning_content", reasoning_content, message_id=reasoning_message.id)
                    has_reasoning = True
            
            # 处理最终答案
            content = chunk.content if hasattr(chunk, 'content') else chunk
            
            # Deepseek模型需要额外检查content不等于reasoning_result
            if provider == "deepseek":
                if content and content.strip() and content != reasoning_result:
                    # 如果推理阶段结束但还没发送完成信号
                    if in_reasoning_phase and not reasoning_completed and not has_reasoning:
                        in_reasoning_phase = False
                        reasoning_completed = True
                        yield await send_event("reasoning_complete", message_id=reasoning_message.id)
                    
                    if not answering_started:
                        answering_started = True
                        yield await send_event("answering_start", message_id=assistant_message.id)
                    
                    answer_result += content
                    yield await send_event("answering_content", content, message_id=assistant_message.id)
                    has_answer = True
            else:
                if content and content.strip():
                    # 如果开始接收到答案内容，但还没有结束推理阶段
                    if in_reasoning_phase and not reasoning_completed and not has_reasoning:
                        in_reasoning_phase = False
                        reasoning_completed = True
                        yield await send_event("reasoning_complete", message_id=reasoning_message.id)
                    
                    if not answering_started:
                        answering_started = True
                        yield await send_event("answering_start", message_id=assistant_message.id)
                    
                    answer_result += content
                    yield await send_event("answering_content", content, message_id=assistant_message.id)
                    has_answer = True
        
        # 确保所有阶段正确结束
        if in_reasoning_phase and not reasoning_completed:
            yield await send_event("reasoning_complete", message_id=reasoning_message.id)
        
        if not answering_started:
            yield await send_event("answering_start", message_id=assistant_message.id)
        
        yield await send_event("answering_complete", message_id=assistant_message.id)

        # 更新数据库中的占位消息
        if reasoning_result:
            await self.update_stream_response(reasoning_message.id, reasoning_result)
        if answer_result:
            await self.update_stream_response(assistant_message.id, answer_result)

        # 完成标志
        done_data = {
            "type": "done",
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "reasoning_message_id": reasoning_message.id,
        }
        yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

    async def save_stream_response_with_reasoning(self, conversation_id, response_text, reasoning_text, turn_id=None):
        """保存流式响应和推理过程到对话历史 - 此方法将在后续被废弃"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 如果有推理内容，创建推理消息
                if reasoning_text:
                    reasoning_message = Message(
                        role=MessageRoles.ASSISTANT,
                        type=MessageTypes.REASONING_CONTENT,
                        content=reasoning_text,
                        turn_id=turn_id
                    )
                    conversation.messages.append(reasoning_message)

                # 创建并添加AI响应消息
                ai_message = Message(
                    role=MessageRoles.ASSISTANT,
                    type=MessageTypes.ASSISTANT_CONTENT,
                    content=response_text,
                    turn_id=turn_id
                )
                conversation.messages.append(ai_message)
                conversation.updated_at = datetime.now()

                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logging.error(f"保存推理流式响应失败: {str(e)}")

    async def direct_reasoning_stream(self, provider, model, messages, conversation_id, options=None, turn_id=None) -> AsyncGenerator:
        """直接使用OpenAI客户端生成带推理功能的流式响应，绕过LangChain"""
        from openai import AsyncOpenAI
        
        if options is None:
            options = {}
        
        # 如果消息是单个消息而非列表，则转换为列表
        if not isinstance(messages, list):
            messages = [messages]
        
        # 1. 创建占位消息
        reasoning_message = await self._create_placeholder_message(conversation_id, MessageTypes.REASONING_CONTENT, turn_id)
        assistant_message = await self._create_placeholder_message(conversation_id, MessageTypes.ASSISTANT_CONTENT, turn_id)

        # 构造发送事件的辅助函数
        async def send_event(event_type, content=None, message_id=None):
            data = {"type": event_type, "conversation_id": conversation_id}
            if content is not None:
                data["content"] = content
            if message_id:
                data["message_id"] = message_id
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

        # 获取API凭证
        credentials = llm_manager._get_model_credentials(provider, model)
        if not credentials:
            raise ValueError(f"未找到{provider}的API凭证")
        
        # 初始化OpenAI客户端
        client = AsyncOpenAI(
            api_key=credentials.get("api_key"),
            base_url=credentials.get("base_url"),
            timeout=60 * 30  # 30分钟超时
        )
        
        # 准备推理和回答内容
        reasoning_result = ""
        answer_result = ""
        
        # 状态跟踪
        in_reasoning_phase = True
        reasoning_completed = False
        answering_started = False
        
        # 开始推理流
        yield await send_event("reasoning_start", message_id=reasoning_message.id)
        
        # 准备发送的消息
        openai_messages = []
        
        for msg in messages:
            # 检查是否是LangChain消息类型
            msg_dict = None
            
            try:
                # 尝试获取消息的字典表示
                if hasattr(msg, 'dict') and callable(msg.dict):
                    msg_dict = msg.dict()
                elif hasattr(msg, 'model_dump') and callable(msg.model_dump):
                    msg_dict = msg.model_dump()
                
                if msg_dict and 'type' in msg_dict:
                    msg_type = msg_dict.get('type', '')
                    msg_content = msg_dict.get('content', '')
                    
                    if msg_type == 'human':
                        openai_messages.append({"role": "user", "content": msg_content})
                    elif msg_type == 'ai':
                        openai_messages.append({"role": "assistant", "content": msg_content})
                    elif msg_type == 'system':
                        openai_messages.append({"role": "system", "content": msg_content})
                    continue
            except Exception as e:
                logger.error(f"尝试获取消息字典时出错: {e}")
            
            # 通过类名判断
            class_name = msg.__class__.__name__
            if class_name == 'HumanMessage':
                openai_messages.append({"role": "user", "content": msg.content})
            elif class_name == 'AIMessage':
                openai_messages.append({"role": "assistant", "content": msg.content})
            elif class_name == 'SystemMessage':
                openai_messages.append({"role": "system", "content": msg.content})
            # 处理自定义消息对象
            elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                openai_messages.append({"role": msg.role, "content": msg.content})
            # 如果是已经格式化好的字典
            elif isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                openai_messages.append(msg)
            else:
                logger.warning(f"无法识别的消息格式: {type(msg)}")
        
        if not openai_messages:
            raise ValueError("无有效消息可发送，请检查消息格式")
        
        try:
            # 创建流式请求
            stream = await client.chat.completions.create(
                model=model,
                messages=openai_messages,
                stream=True
            )
            
            # 处理流式响应
            async for chunk in stream:
                has_reasoning = False
                
                # 提取推理内容
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'reasoning_content'):
                    reasoning_content = chunk.choices[0].delta.reasoning_content
                    if reasoning_content:
                        reasoning_result += reasoning_content
                        yield await send_event("reasoning_content", reasoning_content, message_id=reasoning_message.id)
                        has_reasoning = True
                
                # 提取内容
                if len(chunk.choices) > 0 and hasattr(chunk.choices[0].delta, 'content'):
                    content = chunk.choices[0].delta.content
                    if content:
                        # 如果仍在推理阶段且没有接收到推理内容，则推理阶段结束
                        if in_reasoning_phase and not reasoning_completed and not has_reasoning:
                            in_reasoning_phase = False
                            reasoning_completed = True
                            yield await send_event("reasoning_complete", message_id=reasoning_message.id)
                        
                        # 如果尚未开始回答阶段，则开始回答阶段
                        if not answering_started:
                            answering_started = True
                            yield await send_event("answering_start", message_id=assistant_message.id)
                        
                        # 累积回答内容并发送事件
                        answer_result += content
                        yield await send_event("answering_content", content, message_id=assistant_message.id)
            
            # 确保所有阶段正确结束
            if in_reasoning_phase and not reasoning_completed:
                yield await send_event("reasoning_complete", message_id=reasoning_message.id)
            
            if not answering_started:
                yield await send_event("answering_start", message_id=assistant_message.id)
            
            yield await send_event("answering_complete", message_id=assistant_message.id)
            
        except Exception as e:
            logger.error(f"直接API调用出错: {str(e)}")
            yield await send_event("error", f"生成出错: {str(e)}")
        
        # 保存到数据库
        await self.save_stream_response_with_reasoning(
            conversation_id=conversation_id,
            response_text=answer_result,
            reasoning_text=reasoning_result,
            turn_id=turn_id
        )
        
        # 完成标志
        done_data = {
            "type": "done",
            "conversation_id": conversation_id,
            "message_id": assistant_message.id,
            "reasoning_message_id": reasoning_message.id,
        }
        yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n" 