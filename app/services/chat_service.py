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
from app.constants import MessageRoles, EventTypes, FunctionNames, MessageTexts, FUNCTION_DESCRIPTIONS, USER_FRIENDLY_FUNCTION_DESCRIPTIONS
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
            provider: str,
            model: str,
            message: str,
            conversation_id: Optional[str] = None,
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None,
            file_ids: Optional[List[str]] = None,
    ) -> Union[StreamingResponse, ChatResponse]:
        """处理用户消息并获取AI响应"""
        # 初始化options
        if options is None:
            options = {}
        
        # 获取或创建会话
        conversation = self._get_or_create_conversation(conversation_id, provider, model, message)

        # 记录用户消息
        user_message = Message(role=MessageRoles.USER, content=message)
        conversation.messages.append(user_message)
        
        # 准备聊天历史
        chat_history = []
        for msg in conversation.messages:
            chat_history.append({"role": msg.role, "content": msg.content})

        # 从聊天历史中提取消息
        messages = self.message_processor.prepare_chat_messages(chat_history)

        # 保存会话（先保存用户消息）
        conversation.updated_at = datetime.now()
        self.memory_service.save_conversation(conversation)

        # 处理文件内容
        if file_ids and len(file_ids) > 0:
            # 检查文件状态
            status_response = self.file_service.check_files_status(file_ids, provider, model, conversation.id)
            if status_response:
                return status_response
                
            # 获取文件内容并增强消息
            file_contents = self.file_service.get_files_content(file_ids)
            if file_contents:
                messages = self.message_processor.enhance_with_file_content(messages, message, file_contents)

        # 根据是否为流式响应分别处理
        if stream:
            return await self._handle_stream_response(provider, model, messages, conversation.id, options)
        else:
            return await self._handle_normal_response(provider, model, messages, conversation.id, options)

    def _get_or_create_conversation(self, conversation_id, provider, model, message):
        """获取或创建会话"""
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id)

        if not conversation:
            # 创建新对话
            conversation = Conversation(
                id=conversation_id or str(uuid.uuid4()),
                title=message[:30] + "..." if len(message) > 30 else message,
                provider=provider,
                model=model,
                messages=[]
            )
            
        return conversation

    async def _handle_stream_response(self, provider, model, messages, conversation_id, options=None):
        """处理流式响应"""
        if options is None:
            options = {}
            
        user_wants_web_search = options.get("use_web_search", False)
        ai_can_use_functions = options.get("use_function_call", False)
        use_reasoning = options.get("use_reasoning", False)

        if user_wants_web_search:
            return StreamingResponse(
                self.search_processor.generate_user_prioritized_web_search_stream(provider, model, messages, conversation_id, options),
                media_type="text/event-stream"
            )
        elif ai_can_use_functions:
            return StreamingResponse(
                self.function_call_processor.generate_function_call_stream(provider, model, messages, conversation_id, options),
                media_type="text/event-stream"
            )
        elif provider == "volcengine": 
            return StreamingResponse(
                self.stream_handler.direct_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
        elif use_reasoning or provider in ("deepseek", "qwen"): 
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
        else:
            return StreamingResponse(
                self.stream_handler.generate_normal_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )

    async def _handle_normal_response(self, provider, model, messages, conversation_id, options=None):
        """处理非流式响应"""
        # 默认options
        if options is None:
            options = {}
        
        # 获取适合的模型处理策略
        strategy = ModelStrategyFactory.get_strategy(provider, model, options)
        
        try:
            # 使用策略处理请求
            ai_message, reasoning_message = await strategy.process(provider, model, messages, conversation_id, self.memory_service, options)
            
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id)
            
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













    def get_all_conversations(self) -> List[Conversation]:
        """获取所有对话"""
        return self.memory_service.get_all_conversations()

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取特定对话"""
        return self.memory_service.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除特定对话"""
        try:
            # 然后删除数据库记录
            return self.memory_service.delete_conversation(conversation_id)
        except Exception as e:
            logging.error(f"删除对话失败: {e}")
            return False

    async def generate_title(
            self,
            message: Optional[str] = None,
            conversation_id: Optional[str] = None,
            options: Optional[Dict[str, Any]] = None
    ) -> str:
        """生成与消息或会话相关的标题"""
        # 如果提供了会话ID，获取会话
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id)
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
        conversation_id: str,
        latest_only: bool = True,
        options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """生成与当前对话轮次相关的推荐问题"""
        # 获取会话
        conversation = self.memory_service.get_conversation(conversation_id)
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
        
    async def handle_function_calls(self, provider, model, messages, conversation_id, options=None):
        """
        处理函数调用流程
        
        参数:
            provider: 模型提供商
            model: 模型名称
            messages: 聊天消息列表
            conversation_id: 会话ID
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
        
        # 准备函数定义
        functions_kwargs = function_adapter.prepare_functions_for_model(provider, model)
        
        try:
            # 调用模型，让其决定是否需要调用函数
            response = await llm.ainvoke(messages, **functions_kwargs)
            
            # 提取函数调用信息
            function_call, tool_call_id = function_adapter.extract_function_call(provider, response)
            
            # 如果没有函数调用，直接创建回复消息
            if not function_call:
                # 创建AI消息
                ai_message = Message(
                    role=MessageRoles.ASSISTANT,
                    content=response.content if hasattr(response, 'content') else str(response)
                )
                
                # 获取会话
                conversation = self.memory_service.get_conversation(conversation_id)
                
                # 添加AI响应到会话
                conversation.messages.append(ai_message)
                
                # 更新并保存会话
                conversation.updated_at = datetime.now()
                self.memory_service.save_conversation(conversation)
                
                # 返回响应
                return ChatResponse(
                    id=str(uuid.uuid4()),
                    provider=provider,
                    model=model,
                    message=ai_message,
                    conversation_id=conversation.id
                )
            
            # 记录函数调用
            logger.info(f"模型选择调用函数: {function_call.get('name')}")
            
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id)
            
            # 添加AI选择调用函数的消息
            function_name = function_call.get('name')
            user_friendly_description = USER_FRIENDLY_FUNCTION_DESCRIPTIONS.get(
                function_name, 
                "我需要调用工具获取更多信息..."
            )
            ai_function_message = Message(
                role=MessageRoles.ASSISTANT,
                content=response.content if hasattr(response, 'content') and response.content else user_friendly_description
            )
            conversation.messages.append(ai_function_message)
            
            # 处理函数调用
            function_result = await function_adapter.process_function_call(provider, function_call, context)
            
            # 准备工具消息
            tool_message_data = function_adapter.prepare_tool_message(
                provider, function_name, function_result, tool_call_id
            )
            
            # 创建工具消息
            tool_message = Message(
                role=tool_message_data["role"],
                content=tool_message_data["content"]
            )
            conversation.messages.append(tool_message)
            
            # 复制原始消息并添加函数调用结果
            full_messages = list(messages)
            full_messages.append(response)
            full_messages.append(tool_message_data)
            
            # 再次调用模型生成最终回答
            final_response = await llm.ainvoke(full_messages)
            
            # 创建最终AI消息
            final_ai_message = Message(
                role=MessageRoles.ASSISTANT,
                content=final_response.content if hasattr(final_response, 'content') else str(final_response)
            )
            
            # 添加最终消息到会话
            conversation.messages.append(final_ai_message)
            
            # 更新并保存会话
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            
            # 返回响应
            return ChatResponse(
                id=str(uuid.uuid4()),
                provider=provider,
                model=model,
                message=final_ai_message,
                conversation_id=conversation.id
            )
        except Exception as e:
            logger.error(f"函数调用处理失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # 创建错误消息
            error_message = Message(
                role=MessageRoles.ASSISTANT,
                content=f"在处理函数调用时出现错误: {str(e)}"
            )
            
            # 返回错误响应
            return ChatResponse(
                id=str(uuid.uuid4()),
                provider=provider,
                model=model,
                message=error_message,
                conversation_id=conversation_id
            )

    async def save_stream_response(self, conversation_id, response_content):
        """保存普通流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 创建AI响应消息
                ai_message = Message(
                    role=MessageRoles.ASSISTANT,
                    content=response_content
                )
                
                # 添加AI响应到会话
                conversation.messages.append(ai_message)
                
                # 更新会话时间
                conversation.updated_at = datetime.now()
                
                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logger.error(f"保存流式响应失败: {e}")



