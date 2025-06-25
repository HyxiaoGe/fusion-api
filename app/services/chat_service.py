import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from app.core.function_manager import function_registry, function_adapter
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.ai.prompts import prompt_manager
from app.ai.prompts.templates import FUNCTION_CALL_BEHAVIOR_PROMPT, SYNTHESIZE_TOOL_RESULT_PROMPT
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.processor.file_processor import FileProcessor
from app.schemas.chat import ChatResponse, Message, Conversation
from app.services.file_content_service import FileContentService
from app.services.memory_service import MemoryService
from app.services.message_processor import MessageProcessor
from app.services.model_strategies import ModelStrategyFactory
from app.services.stream_handler import StreamHandler
from app.services.web_search_service import WebSearchService
from app.constants import MessageRoles, EventTypes, FunctionNames, MessageTexts, FUNCTION_DESCRIPTIONS, USER_FRIENDLY_FUNCTION_DESCRIPTIONS, MessageTypes
from app.services.chat.stream_processor import ReasoningState, StreamProcessor
from app.services.chat.utils import ChatUtils
from app.services.chat.function_call_processor import FunctionCallProcessor
from app.services.chat.search_processor import SearchProcessor


# ==================== ChatService 类 ====================

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        # 初始化各种服务
        self.memory_service = MemoryService(db)
        self.file_processor = FileProcessor()
        
        self.message_processor = MessageProcessor(db)
        self.stream_handler = StreamHandler(db, self.memory_service)
        self.file_service = FileContentService(db)
        
        # 初始化新的处理器
        self.function_call_processor = FunctionCallProcessor(db, self.memory_service)
        self.search_processor = SearchProcessor(db, self.memory_service)

    def _create_event_sender(self, conversation_id: str):
        """创建事件发送器函数"""
        return ChatUtils.create_event_sender(conversation_id)

    def _extract_user_message_from_messages(self, messages: List) -> str:
        """从消息列表中提取最后一条用户消息"""
        return ChatUtils.extract_user_message_from_messages(messages)

    def _parse_function_arguments(self, function_args: Union[str, dict]) -> dict:
        """解析函数参数，确保返回有效的字典"""
        return ChatUtils.parse_function_arguments(function_args)

    async def _generate_search_query(self, user_message: str, llm) -> str:
        """生成优化后的搜索查询"""
        return await ChatUtils.generate_search_query(user_message, llm)

    def _extract_original_user_query(self, messages: List) -> str:
        """从消息列表中提取用户原始查询"""
        return ChatUtils.extract_original_user_query(messages)

    def _validate_and_process_function_arguments(self, function_call_data: dict) -> str:
        """验证和处理函数参数，确保返回有效的JSON字符串"""
        return ChatUtils.validate_and_process_function_arguments(function_call_data)

    async def process_message(
            self,
            user_id: str,
            provider: str,
            model: str,
            message: str,
            conversation_id: Optional[str] = None,
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None,
            file_ids: Optional[List[str]] = None,
            topic_info: Optional[Dict[str, Any]] = None,
    ) -> Union[StreamingResponse, ChatResponse]:
        """处理用户消息并获取AI响应"""
        # 初始化options
        if options is None:
            options = {}
        # 获取或创建会话
        conversation = self._get_or_create_conversation(conversation_id, user_id, provider, model, message)

        # 记录用户消息（保持用户原始完整消息，如："请帮我分析以下热点话题： xxx"）
        user_message = Message(
            role=MessageRoles.USER, 
            type=MessageTypes.USER_QUERY,
            content=message  # 保持用户原始完整消息内容
        )
        
        # 使用用户消息的ID作为turn_id
        turn_id = user_message.id
        user_message.turn_id = turn_id
        
        conversation.messages.append(user_message)
        
        # 处理话题信息：如果有话题信息，生成扩展的话题分析提示词发送给LLM
        llm_message = message  # 默认使用用户原始消息
        if topic_info:
            # 使用提示词管理器生成包含话题详细信息的分析提示词
            llm_message = prompt_manager.format_prompt(
                "hot_topic_analysis",
                title=topic_info.get("title", ""),
                description=topic_info.get("description", ""),
                additional_content=topic_info.get("additional_content", "")
            )
        
        # 准备聊天历史
        chat_history = []
        for msg in conversation.messages[:-1]:  # 排除刚添加的用户消息
            chat_history.append({"role": msg.role, "content": msg.content})
        
        # 添加用于LLM处理的消息（可能是原始消息或扩展后的话题分析提示词）
        chat_history.append({"role": MessageRoles.USER, "content": llm_message})

        # 从聊天历史中提取消息
        messages = self.message_processor.prepare_chat_messages(chat_history)

        # 保存会话和用户消息，确保在流式处理前它们已存在于数据库中
        conversation.updated_at = datetime.now()
        self.memory_service.save_conversation(conversation)
        self.db.commit()

        # 处理文件内容
        if file_ids and len(file_ids) > 0:
            # 检查文件状态
            status_response = self.file_service.check_files_status(file_ids, provider, model, conversation.id)
            if status_response:
                return status_response
                
            # 获取文件内容并增强消息
            file_contents = self.file_service.get_files_content(file_ids)
            if file_contents:
                messages = self.message_processor.enhance_with_file_content(messages, llm_message, file_contents)

        # 根据是否为流式响应分别处理
        if stream:
            return await self._handle_stream_response(provider, model, messages, conversation.id, user_id, options, turn_id)
        else:
            return await self._handle_normal_response(provider, model, messages, conversation.id, user_id, options, turn_id)

    def _get_or_create_conversation(self, conversation_id: Optional[str], user_id: str, provider: str, model: str, message: str) -> Conversation:
        """获取或创建会话"""
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id, user_id)
            if conversation:
                print(f"conversation1: {conversation}")
                return conversation
    
    
        print(f"user_id: {user_id}")
        # 创建新对话
        return Conversation(
            id=conversation_id or str(uuid.uuid4()),
            user_id=user_id,
            title=message[:30] + "..." if len(message) > 30 else message,
            provider=provider,
            model=model,
            messages=[]
        )

    async def _handle_stream_response(self, provider, model, messages, conversation_id, user_id: str, options=None, turn_id=None):
        """处理流式响应"""
        if options is None:
            options = {}
        # 将user_id注入到options中，以便下游处理器可以访问
        options["user_id"] = user_id
            
        # 检查是否启用函数调用
        use_function_calls = options.get("use_function_calls", False)
        use_reasoning = options.get("use_reasoning", False)
        
        if use_function_calls:
            return StreamingResponse(
                self.function_call_processor.generate_function_call_stream(provider, model, messages, conversation_id, options, turn_id),
                media_type="text/event-stream"
            )
        elif provider == "volcengine": 
            return StreamingResponse(
                self.stream_handler.direct_reasoning_stream(provider, model, messages, conversation_id, options, turn_id),
                media_type="text/event-stream"
            )
        elif use_reasoning:
            # 只有当显式启用推理时才使用推理流
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id, options, turn_id),
                media_type="text/event-stream"
            )
        elif use_reasoning is None and provider in ("deepseek", "qwen", "xai"):
            # 只有当use_reasoning未明确设置时，才根据provider自动判断
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id, options, turn_id),
                media_type="text/event-stream"
            )
        else:
            return StreamingResponse(
                self.stream_handler.generate_normal_stream(provider, model, messages, conversation_id, options, turn_id),
                media_type="text/event-stream"
            )

    async def _handle_normal_response(self, provider, model, messages, conversation_id, user_id: str, options=None, turn_id=None):
        """处理非流式响应"""
        # 默认options
        if options is None:
            options = {}
        
        # 获取适合的模型处理策略
        strategy = ModelStrategyFactory.get_strategy(provider, model, options)
        
        try:
            # 使用策略处理请求
            ai_message, reasoning_message = await strategy.process(provider, model, messages, conversation_id, self.memory_service, options, turn_id)
            
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id, user_id)
            
            # 如果有推理内容，添加到会话
            if reasoning_message:
                conversation.messages.append(reasoning_message)
            
            # 添加AI响应到会话
            conversation.messages.append(ai_message)
            
            # 更新并保存会话
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            
            # 返回响应
            reasoning_content = reasoning_message.content if reasoning_message else ""
            return ChatResponse(
                id=str(uuid.uuid4()),
                provider=provider,
                model=model,
                message=ai_message,
                conversation_id=conversation.id,
                reasoning=reasoning_content
            )
        except Exception as e:
            logger.error(f"模型处理失败: {e}")
            raise

    def get_all_conversations(self, user_id: str) -> List[Conversation]:
        """获取指定用户的所有对话"""
        return self.memory_service.get_all_conversations(user_id)

    def get_conversations_paginated(self, user_id: str, page: int = 1, page_size: int = 20):
        """分页获取指定用户的对话列表"""
        return self.memory_service.get_conversations_paginated(user_id, page, page_size)

    def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """获取特定对话的详细信息，并验证用户权限"""
        conversation = self.memory_service.get_conversation(conversation_id, user_id)
        if conversation and conversation.user_id == user_id:
            return conversation
        return None

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        """更新消息"""
        return self.memory_service.update_message(message_id, update_data)

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """删除特定对话，并验证用户权限"""
        conversation = self.memory_service.get_conversation(conversation_id, user_id)
        if not conversation:
            return False  # 对话不存在或用户无权访问
        return self.memory_service.delete_conversation(conversation_id, user_id)

    async def generate_title(
            self,
            user_id: str,
            message: Optional[str] = None,
            conversation_id: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成与消息或会话相关的标题"""
        # 如果提供了会话ID，获取会话
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id, user_id)
            if not conversation:
                raise ValueError(f"找不到会话ID: {conversation_id}")

            # 使用会话的最后一次对话（用户和助手的消息）作为输入
            if not message and conversation.messages:
                # 获取会话中最后的用户和助手消息
                user_message = None
                assistant_message = None
                
                # 从后向前查找最近的一组对话
                for i in range(len(conversation.messages) - 1, -1, -1):
                    msg = conversation.messages[i]
                    if not assistant_message and msg.role == MessageRoles.ASSISTANT:
                        assistant_message = msg.content
                    if not user_message and msg.role == MessageRoles.USER:
                        user_message = msg.content
                    if user_message and assistant_message:
                        break
                
                # 组合用户和助手的消息
                dialog_messages = []
                if user_message:
                    dialog_messages.append(f"用户: {user_message}")
                if assistant_message:
                    dialog_messages.append(f"助手: {assistant_message}")
                
                if dialog_messages:
                    message = "\n".join(dialog_messages)
                else:
                    # 如果没有找到对话，回退到之前的逻辑
                    user_messages = []
                    for msg in conversation.messages:
                        if msg.role == MessageRoles.USER:
                            user_messages.append(msg.content)
                            if len(user_messages) >= 3:
                                break
                    
                    if user_messages:
                        message = "\n".join(user_messages)

        if not message:
            raise ValueError("必须提供消息内容或有效的会话ID")

        # 使用提示词管理器获取并格式化提示词
        prompt = prompt_manager.format_prompt("generate_title", content=message)

        try:
            # 获取AI模型并生成标题
            llm = llm_manager.get_default_model()
            response = llm.invoke([HumanMessage(content=prompt)])

            if hasattr(response, 'content'):  # ChatModel返回的响应
                title = response.content
            else:  # 普通LLM返回的响应
                title = response

            # 清理标题（去除多余的引号、空白和解释性文字）
            title = title.strip().strip('"\'')

            # 如果标题中包含"标题："等前缀，去除
            prefixes = ["标题：", "标题:", "主题：", "主题:"]
            for prefix in prefixes:
                if title.startswith(prefix):
                    title = title[len(prefix):].strip()

            # 限制标题长度
            if len(title) > 30:
                title = title[:30] + "..."

            # 如果提供了会话ID，更新会话标题
            if conversation_id and conversation:
                conversation.title = title
                conversation.updated_at = datetime.now()
                self.memory_service.save_conversation(conversation)

            return title
        except Exception as e:
            logging.error(f"生成标题时发生错误: {str(e)}")
            # 如果生成失败，返回一个默认标题
            if conversation_id:
                return f"对话 {conversation_id[:8]}..."
            else:
                return "新对话"

    async def generate_suggested_questions(
        self,
        user_id: str,
        conversation_id: str,
        latest_only: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """生成与当前对话轮次相关的推荐问题"""
        # 获取会话
        conversation = self.memory_service.get_conversation(conversation_id, user_id)
        if not conversation:
            raise ValueError(f"找不到会话ID: {conversation_id}")

        # 准备对话内容 - 只取最近一轮对话(最新的用户问题和AI回答)
        latest_user_msg = None
        latest_ai_msg = None
        
        # 从后向前查找最近的用户消息和AI回答
        for i in range(len(conversation.messages) - 1, -1, -1):
            msg = conversation.messages[i]
            if not latest_ai_msg and msg.role == MessageRoles.ASSISTANT:
                latest_ai_msg = msg.content
            elif not latest_user_msg and msg.role == MessageRoles.USER:
                latest_user_msg = msg.content
            if latest_user_msg and latest_ai_msg:
                break
        
        # 组合最近一轮对话
        dialog_content = ""
        if latest_user_msg:
            dialog_content += f"用户: {latest_user_msg}\n"
        if latest_ai_msg:
            dialog_content += f"助手: {latest_ai_msg}"
        
        if not dialog_content:
            # 如果没有对话内容，返回默认问题
            return [
                "有什么我可以帮您解答的问题吗？",
                "您想了解更多哪方面的信息？",
                "还有其他我能帮助您的事情吗？"
            ]

        # 使用提示词管理器获取并格式化提示词
        prompt = prompt_manager.format_prompt("generate_suggested_questions", content=dialog_content)

        try:
            # 获取AI模型并生成问题
            llm = llm_manager.get_default_model()
            response = llm.invoke([HumanMessage(content=prompt)])

            if hasattr(response, 'content'):  # ChatModel返回的响应
                response_text = response.content
            else:  # 普通LLM返回的响应
                response_text = response

            # 解析响应文本，提取问题
            questions = self._parse_questions(response_text)
            
            return questions[:3]  # 确保只返回3个问题
        except Exception as e:
            logger.error(f"生成推荐问题时发生错误: {str(e)}")
            # 如果生成失败，返回默认问题
            return [
                "您对这个主题还有其他问题吗？",
                "您想了解更多相关信息吗？",
                "您想要探讨这个话题的哪些方面？"
            ]

    def _parse_questions(self, response_text: str) -> List[str]:
        """从响应文本中解析出问题列表"""
        return ChatUtils.parse_questions(response_text)
        
    async def handle_function_calls(self, provider, model, messages, conversation_id, user_id: str, options=None):
        """
        处理函数调用流程
        
        参数:
            provider: 模型提供商
            model: 模型名称
            messages: 聊天消息列表
            conversation_id: 会话ID
            user_id: 用户ID
            options: 其他选项
            
        返回:
            Chat响应对象
        """
        # 默认options
        if options is None:
            options = {}
            
        # 准备上下文
        context = {
            "db": self.db,
            "conversation_id": conversation_id
        }
        
        # 获取AI模型
        llm = llm_manager.get_model(provider=provider, model=model)
        
        # 绑定工具（函数）
        llm_with_tools = llm.bind_tools(function_adapter)
        
        # 构建消息链
        chain = llm_with_tools
        
        # 添加用于合成工具结果的提示词
        synthesize_prompt = SystemMessage(content=SYNTHESIZE_TOOL_RESULT_PROMPT)
        
        response = None
        
        # 初始调用
        try:
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id, user_id)
            if not conversation:
                logger.error(f"在函数调用中未找到会话: {conversation_id}")
                return ChatResponse(
                    message=Message(role="assistant", type="assistant_content", content="会话丢失，请重试。"),
                    conversation_id=conversation_id,
                    provider=provider,
                    model=model
                )

            # 获取最后的用户消息
            last_user_message = self._extract_user_message_from_messages(messages)

            # 如果找到 function_call，说明是第二次进入
            if any(isinstance(m, AIMessage) and m.tool_calls for m in messages):
                 # 添加合成提示词，并调用LLM进行最终回答
                final_messages = messages + [synthesize_prompt]
                response = chain.invoke(final_messages)
            else:
                # 第一次进入，正常调用
                response = chain.invoke(messages)

            # 保存AI的响应（可能是函数调用或最终回答）
            if isinstance(response, AIMessage):
                ai_message_schema = Message(
                    role="assistant",
                    type="function_call" if response.tool_calls else "assistant_content",
                    content=response.content or "",
                    turn_id=conversation.messages[-1].turn_id if conversation.messages else None
                )
                
                # 如果是函数调用，需要特殊处理
                if response.tool_calls:
                    # 将 tool_calls 转换为JSON字符串存储
                    ai_message_schema.content = json.dumps([
                        {"id": tc["id"], "name": tc["name"], "args": tc["args"]}
                        for tc in response.tool_calls
                    ])

                conversation.messages.append(ai_message_schema)
            
            # 更新会话
            self.memory_service.save_conversation(conversation)

        except Exception as e:
            logger.error(f"处理函数调用时发生错误: {e}")
            # 返回错误信息给用户
            return ChatResponse(
                message=Message(role="assistant", type="assistant_content", content=f"处理函数调用时发生错误: {e}"),
                conversation_id=conversation_id,
                provider=provider,
                model=model
            )

        # 如果没有函数调用，直接返回响应
        if not isinstance(response, AIMessage) or not response.tool_calls:
            return ChatResponse(
                message=Message(
                    role="assistant",
                    type="assistant_content",
                    content=response.content if hasattr(response, "content") else str(response)
                ),
                conversation_id=conversation_id,
                provider=provider,
                model=model
            )

        # 执行函数调用
        tool_outputs = []
        for tool_call in response.tool_calls:
            function_name = tool_call["name"]
            function_args_raw = tool_call["args"]
            
            # 解析函数参数
            function_args = self._parse_function_arguments(function_args_raw)
            
            # 执行函数
            try:
                # 再次获取会话以确保数据最新
                conversation = self.memory_service.get_conversation(conversation_id, user_id)
                if not conversation:
                    raise Exception("会话在函数执行期间丢失")
                
                # 更新上下文信息
                context["conversation"] = conversation
                
                # 动态执行函数
                selected_function = function_registry.get(function_name)
                if not selected_function:
                    raise Exception(f"未找到函数: {function_name}")
                
                # 执行函数并获取结果
                function_result = await selected_function(**function_args, context=context)
                
                tool_outputs.append(
                    ToolMessage(content=str(function_result), tool_call_id=tool_call["id"])
                )
            except Exception as e:
                logger.error(f"执行函数 {function_name} 失败: {e}")
                tool_outputs.append(
                    ToolMessage(content=f"执行函数时出错: {e}", tool_call_id=tool_call["id"])
                )

        # 将函数执行结果添加到消息历史中
        messages.extend(tool_outputs)
        
        # 再次调用，让模型根据函数结果生成最终回答
        return await self.handle_function_calls(provider, model, messages, conversation_id, user_id, options)

    async def save_stream_response(self, conversation_id: str, user_id: str, response_content: str):
        """保存流式响应的最终结果"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id, user_id)
            if not conversation:
                logger.error(f"保存流式响应时未找到会d话: {conversation_id}")
                return

            # 更新最后一条助手消息的内容
            if conversation.messages and conversation.messages[-1].role == MessageRoles.ASSISTANT:
                conversation.messages[-1].content = response_content
            else:
                # 如果最后一条不是助手消息，则创建一条新的
                assistant_message = Message(
                    role=MessageRoles.ASSISTANT,
                    type=MessageTypes.ASSISTANT_CONTENT,
                    content=response_content
                )
                conversation.messages.append(assistant_message)
            
            # 更新会话
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            
        except Exception as e:
            logger.error(f"保存流式响应失败: {e}")

    def get_latest_user_message(self, conversation_id: str, user_id: str) -> Optional[Message]:
        """获取最新的用户消息"""
        conversation = self.memory_service.get_conversation(conversation_id, user_id)
        if not conversation:
            return None
        
        for i in range(len(conversation.messages) - 1, -1, -1):
            if conversation.messages[i].role == MessageRoles.USER:
                return conversation.messages[i]
        
        return None



