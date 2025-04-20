import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.ai.prompts import prompt_manager
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

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        # 初始化各种服务
        self.memory_service = MemoryService(db)
        self.file_processor = FileProcessor()
        
        self.message_processor = MessageProcessor(db)
        self.stream_handler = StreamHandler(db, self.memory_service)
        self.file_service = FileContentService(db)

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
        user_message = Message(role="user", content=message)
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
        # 默认options
        if options is None:
            options = {}
            
        # 判断是否使用推理模式
        use_reasoning = options.get("use_reasoning", False)
        
        # 火山引擎特殊处理 - 直接使用OpenAI客户端访问API
        if provider == "volcengine":
            return StreamingResponse(
                self.stream_handler.direct_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
            # 根据模型名称判断使用推理模式（向后兼容）
        elif provider in ("deepseek", "qwen") and use_reasoning:
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
        # 根据options判断使用推理模式
        elif use_reasoning:
            return StreamingResponse(
                self.stream_handler.generate_reasoning_stream(provider, model, messages, conversation_id),
                media_type="text/event-stream"
            )
        
        # 默认使用常规流式响应
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
                    if not assistant_message and msg.role == "assistant":
                        assistant_message = msg.content
                    if not user_message and msg.role == "user":
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
                        if msg.role == "user":
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
            if not latest_ai_msg and msg.role == "assistant":
                latest_ai_msg = msg.content
            elif not latest_user_msg and msg.role == "user":
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
        questions = []
        
        # 尝试不同的解析方法
        # 1. 尝试按数字列表解析
        import re
        numbered_questions = re.findall(r'\d+[\.\)]\s*(.*?)(?=\n\d+[\.\)]|\n*$)', response_text, re.DOTALL)
        if numbered_questions and len(numbered_questions) >= 3:
            return [q.strip() for q in numbered_questions]
        
        # 2. 按行分割
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        for line in lines:
            # 移除行首的数字、点、括号等
            cleaned_line = re.sub(r'^\d+[\.\)]\s*', '', line).strip()
            if cleaned_line:
                questions.append(cleaned_line)
        
        # 如果没有找到足够的问题，返回原始文本分成的前三行
        if len(questions) < 3:
            questions = lines[:3] if len(lines) >= 3 else lines
        
        return questions
        
    async def handle_function_calls(self, provider: str, model: str, message: str, conversation_id: str):
        """处理函数调用"""

        functions = [
            {
                "name": "web_search",
                "description": "在互联网上搜索最新信息",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "搜索查询文本"},
                        "limit": {"type": "integer", "description": "返回结果数量", "default": 5}
                    },
                    "required": ["query"]
                }
            }
        ]

        # 获取AI模型
        llm = llm_manager.get_model(provider=provider, model=model)

        # 调用模型，让其觉得是否需要调用函数
        response = llm.invoke(
            [HumanMessage(content=message)],
            functions=functions
        )

        # 检查是否需要调用函数
        function_call = response.additional_kwargs.get("function_call", None)

        if function_call:
            # 解析函数调用参数
            function_name = function_call.get("name")
            function_args = json.loads(function_call.get("arguments", "{}"))

            # 处理web_search函数调用
            if function_name == "web_search":
                # 调用web_search服务
                search_service = WebSearchService()
                search_results = await search_service.search(
                    function_args.get("query"),
                    function_args.get("limit", 5)
                )

                # 将搜索结果返回给模型，让它生成最终回答
                search_result_response = llm.invoke([
                    HumanMessage(content=message),
                    AIMessage(content=response.content),
                    {"role": "function", "name": "web_search", "content": json.dumps(search_results)}
                ])

                return search_result_response.content

        return response.content
                
            
