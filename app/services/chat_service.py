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
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.processor.file_processor import FileProcessor
from app.schemas.chat import ChatResponse, Message, Conversation
from app.services.context_service import ContextEnhancer
from app.services.file_content_service import FileContentService
from app.services.memory_service import MemoryService
from app.services.message_processor import MessageProcessor
from app.services.model_strategies import ModelStrategyFactory
from app.services.stream_handler import StreamHandler
from app.services.vector_service import VectorService
from app.services.web_search_service import WebSearchService

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        # 初始化各种服务
        self.memory_service = MemoryService(db)
        self.vector_service = VectorService.get_instance(db)
        self.context_enhancer = ContextEnhancer(db)
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

        # 应用上下文增强
        messages = await self._apply_context_enhancement(messages, message, conversation_id)

        # 保存会话（先保存用户消息）
        conversation.updated_at = datetime.now()
        self.memory_service.save_conversation(conversation)

        # 异步向量化用户消息
        asyncio.create_task(self.message_processor.vectorize_message_async(user_message, conversation_id))

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
            return await self._handle_stream_response(provider, model, messages, conversation.id)
        else:
            return await self._handle_normal_response(provider, model, messages, conversation.id)

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

    async def _apply_context_enhancement(self, messages, message, conversation_id):
        """应用上下文增强"""
        use_enhancement = False
        logger.info(f"是否使用上下文增强: {use_enhancement}")

        if use_enhancement:
            # 获取增强提示
            enhancement = self.context_enhancer.enhance_prompt(
                query=message,
                conversation_id=conversation_id
            )

            logger.info(f"增强结果: has_enhancement={enhancement['has_enhancement']}")

            if enhancement["has_enhancement"]:
                # 如果有增强，用增强后的提示替换最后一条用户消息
                messages[-1].content = enhancement["enhanced_prompt"]
                logger.info("已应用增强提示")
                
        return messages

    async def _handle_stream_response(self, provider, model, messages, conversation_id):
        """处理流式响应"""
        # 使用支持推理的模型
        if (provider == "deepseek" and model == "deepseek-reasoner") or (provider == "qwen" and "qwq" in model.lower()):
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

    async def _handle_normal_response(self, provider, model, messages, conversation_id):
        """处理非流式响应"""
        # 获取适合的模型处理策略
        strategy = ModelStrategyFactory.get_strategy(provider, model)
        
        try:
            # 使用策略处理请求
            ai_message, reasoning_message = await strategy.process(provider, model, messages, conversation_id, self.memory_service)
            
            # 获取会话
            conversation = self.memory_service.get_conversation(conversation_id)
            
            # 如果有推理内容，添加到会话
            if reasoning_message:
                conversation.messages.append(reasoning_message)
                # 异步向量化推理消息
                asyncio.create_task(self.message_processor.vectorize_message_async(reasoning_message, conversation.id))
            
            # 添加AI响应到会话
            conversation.messages.append(ai_message)
            
            # 更新并保存会话
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            
            # 异步向量化AI消息
            asyncio.create_task(self.message_processor.vectorize_message_async(ai_message, conversation.id))
            
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
            # 首先删除向量数据
            self.vector_service.delete_conversation_vectors(conversation_id)
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

            # 使用会话的消息作为输入
            if not message and conversation.messages:
                # 获取前3条用户消息，提供更好的上下文
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

        # 准备给LLM的提示
        prompt = f"""请为以下对话内容生成一个简短、具体且有描述性的标题。
        要求：
        1. 返回的内容不允许超过15个字
        2. 直接给出标题，不要包含引号或其他解释性文字
        3. 避免使用"关于"、"讨论"等过于宽泛的词语
        4. 标题应该明确反映对话的核心主题
        5. 如果对话内容没有实质性内容，则直接返回原文

        对话内容：
        {message}"""

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
                
            
