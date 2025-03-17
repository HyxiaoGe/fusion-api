import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.ai.adapters.file_adapters import get_file_adapter
from app.ai.llm_manager import llm_manager
from app.db.repositories import FileRepository
from app.schemas.chat import ChatResponse, Message, Conversation
from app.services.memory_service import MemoryService
from app.services.vector_service import VectorService
from app.services.context_service import ContextEnhancer
from app.core.logger import app_logger as logger

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.memory_service = MemoryService(db)
        # 初始化向量服务和上下文增强器
        self.vector_service = VectorService.get_instance(db)
        self.context_enhancer = ContextEnhancer(db)
        self.file_repo = FileRepository(db)

    async def process_message(
            self,
            model: str,
            message: str,
            conversation_id: Optional[str] = None,
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None,
            file_ids: Optional[List[str]] = None
    ) -> StreamingResponse | ChatResponse:
        """处理用户消息并获取AI响应"""
        # 获取或创建会话
        conversation = None
        if conversation_id:
            conversation = self.memory_service.get_conversation(conversation_id)

        if not conversation:
            # 创建新对话
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

        # 准备聊天历史
        chat_history = []
        for msg in conversation.messages:
            chat_history.append({"role": msg.role, "content": msg.content})

        # 从聊天历史中提取过去的消息
        messages = self._prepare_chat_messages(chat_history)

        # 应用上下文增强
        # 判断是否需要使用上下文增强
        use_enhancement = options.get("use_enhancement", True) if options else True

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

        file_paths = []
        if file_ids and len(file_ids) > 0:
            file_paths = self.file_repo.get_file_paths(file_ids)
            logger.info(f"处理带文件的请求, 文件数量: {len(file_paths)}")

        file_adapter = get_file_adapter(model)

        # 根据是否为流式响应分别处理
        if stream:
            # 保存会话（先保存用户消息）
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)

            # 异步向量化用户消息
            asyncio.create_task(self._vectorize_message_async(user_message, conversation_id))

            return await self.generate_stream_response(model, messages, conversation_id, file_paths)
        else:
            # 获取AI模型
            llm = llm_manager.get_model(model)

            # 如果有文件，使用文件适配器准备请求
            if file_paths:
                try:
                    # 准备带文件的请求
                    file_request = file_adapter.prepare_file_for_request(file_paths, message)

                    # 调用LLM获取响应
                    response = await self._call_model_with_files(model, file_request)
                    ai_response = response.get("content", "无法处理文件请求")
                except Exception as e:
                    logger.error(f"处理文件请求失败: {e}")
                    ai_response = f"[process_message] 处理文件时出错: {str(e)}"
            else:
                # 常规调用LLM获取响应
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

        # 异步向量化消息
        asyncio.create_task(self._vectorize_messages_async(
            [user_message, ai_message],
            conversation_id,
            conversation
        ))

        # 构建并返回响应
        return ChatResponse(
            id=str(uuid.uuid4()),
            model=model,
            message=ai_message,
            conversation_id=conversation_id
        )

    async def generate_stream_response(self, model, messages, conversation_id, file_paths=None):
        """生成流式响应"""
        llm = llm_manager.get_model(model)
        full_response = ""

        async def stream_generator():
            nonlocal full_response

            # 如果有文件，使用文件适配器处理
            if file_paths and len(file_paths) > 0:
                try:
                    # 获取文件适配器
                    file_adapter = get_file_adapter(model)
                    # 准备带文件的请求
                    file_request = file_adapter.prepare_file_for_request(
                        file_paths,
                        messages[-1].content if messages else ""
                    )

                    # 调用带文件的API
                    response = await self._call_model_with_files(model, file_request)
                    content = response.get("content", "")

                    # 返回完整响应
                    full_response = content
                    yield f"data: {json.dumps({'content': content, 'conversation_id': conversation_id})}\n\n"
                    yield f"data: {json.dumps({'content': '[DONE]', 'conversation_id': conversation_id})}\n\n"

                    # 流结束后，将完整响应保存到对话历史
                    await self._save_stream_response(model, conversation_id, full_response)
                    return
                except Exception as e:
                    logger.error(f"[generate_stream_response] 处理文件流式响应失败: {e}")
                    error_msg = f"处理文件时出错: {str(e)}"
                    yield f"data: {json.dumps({'content': error_msg, 'conversation_id': conversation_id})}\n\n"
                    yield f"data: {json.dumps({'content': '[DONE]', 'conversation_id': conversation_id})}\n\n"

                    # 保存错误信息到对话历史
                    await self._save_stream_response(model, conversation_id, error_msg)
                    return

            for chunk in llm.stream(messages):
                content = ""
                if hasattr(chunk, 'content'):
                    content = chunk.content
                else:
                    content = chunk

                if content:
                    full_response += content
                    yield f"data: {json.dumps({'content': content, 'conversation_id': conversation_id})}\n\n"

                await asyncio.sleep(0.01)

            # 流结束后，将完整响应保存到对话历史
            await self._save_stream_response(model, conversation_id, full_response)
            yield f"data: {json.dumps({'content': '[DONE]', 'conversation_id': conversation_id})}\n\n"

        return StreamingResponse(
            stream_generator(),
            media_type="text/event-stream"
        )

    async def _call_model_with_files(self, model: str, file_request: Dict[str, Any]) -> Dict[str, Any]:
        """调用支持文件的模型API"""
        # 这里需要根据不同模型实现具体的API调用
        # 以下是一个示例实现框架
        try:
            if model == "wenxin":
                # 调用文心一言API
                return await self._call_wenxin_with_files(file_request)
            elif model == "qwen":
                # 调用通义千问API
                return await self._call_qwen_with_files(file_request)
            else:
                # 不支持文件的模型
                raise ValueError(f"模型 {model} 不支持文件处理")
        except Exception as e:
            logger.error(f"调用带文件的模型API失败: {e}")
            raise

    # 具体的模型API调用实现
    async def _call_wenxin_with_files(self, file_request: Dict[str, Any]) -> Dict[str, Any]:
        """调用文心一言的文件API"""
        # 实际实现需要根据文心一言的API文档
        pass

    async def _call_qwen_with_files(self, file_request: Dict[str, Any]) -> Dict[str, Any]:
        """调用通义千问的文件API"""
        # 实际实现需要根据通义千问的API文档
        pass

    async def _save_stream_response(self, model, conversation_id, response_text):
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
            model: str,
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
        1. 不超过15个字
        2. 直接给出标题，不要包含引号或其他解释性文字
        3. 避免使用"关于"、"讨论"等过于宽泛的词语
        4. 标题应该明确反映对话的核心主题

        对话内容：
        {message}"""

        try:
            # 获取AI模型并生成标题
            llm = llm_manager.get_model(model)
            from langchain.schema import HumanMessage
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

    async def _vectorize_message_async(self, message: Message, conversation_id: str):
        """异步向量化单条消息"""
        try:
            self.vector_service.vectorize_message(message, conversation_id)
        except Exception as e:
            logging.error(f"异步向量化消息失败: {e}")

    async def _vectorize_messages_async(self, messages: List[Message], conversation_id: str,
                                        conversation: Optional[Conversation] = None):
        """异步向量化多条消息和对话"""
        try:
            for message in messages:
                self.vector_service.vectorize_message(message, conversation_id)

            if conversation:
                self.vector_service.vectorize_conversation(conversation)
        except Exception as e:
            logging.error(f"异步向量化失败: {e}")