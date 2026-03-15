import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Union

from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.ai.prompts import prompt_manager
from sqlalchemy.orm import Session

from app.ai.llm_manager import llm_manager
from app.core.logger import app_logger as logger
from app.db.repositories import FileRepository
from app.schemas.chat import ChatResponse, Message, Conversation
from app.services.memory_service import MemoryService
from app.services.stream_handler import StreamHandler
from app.constants import MessageRoles, MessageTypes
from app.services.chat.utils import ChatUtils
from app.services.chat.function_call_processor import FunctionCallProcessor


# ==================== ChatService 类 ====================

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.memory_service = MemoryService(db)
        self.file_repo = FileRepository(db)
        self.stream_handler = StreamHandler(db, self.memory_service)
        self.function_call_processor = FunctionCallProcessor(db, self.memory_service)

    def _persist_conversation(self, conversation: Conversation):
        """统一保存会话并提交事务。"""
        conversation.updated_at = datetime.now()
        self.memory_service.save_conversation(conversation)
        self.db.commit()

    @staticmethod
    def _get_response_text(response: Any) -> Any:
        """统一提取模型响应正文。"""
        return response.content if hasattr(response, "content") else response

    def _prepare_chat_messages(self, chat_history: List[Dict[str, str]]) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        """将会话历史转换成 LLM 可消费的消息列表。"""
        messages = []
        last_role = None

        for msg in chat_history:
            current_role = msg["role"]

            if current_role == "reasoning":
                continue

            if last_role == current_role:
                if current_role == MessageRoles.USER:
                    messages[-1].content += "\n" + msg["content"]
                    continue
                if current_role == MessageRoles.ASSISTANT:
                    continue

            if current_role == MessageRoles.USER:
                messages.append(HumanMessage(content=msg["content"]))
            elif current_role == MessageRoles.ASSISTANT:
                messages.append(AIMessage(content=msg["content"]))
            elif current_role == MessageRoles.SYSTEM:
                messages.append(SystemMessage(content=msg["content"]))

            last_role = current_role

        return messages

    def _enhance_with_file_content(self, messages, message: str, file_contents: Dict[str, str]):
        """将文件内容拼进最后一条用户消息。"""
        if not file_contents:
            return messages

        file_content_text = "\n\n".join(
            f"文件内容 ({i + 1}):\n{content}"
            for i, content in enumerate(file_contents.values())
        )
        enhanced_message = prompt_manager.format_prompt(
            "file_content_enhancement",
            query=message,
            file_content=file_content_text,
        )
        messages[-1] = HumanMessage(content=enhanced_message)
        return messages

    def _get_files_content(self, file_ids: List[str]) -> Dict[str, str]:
        """读取已经完成解析的文件内容。"""
        if not file_ids:
            return {}
        return self.file_repo.get_parsed_file_content(file_ids)

    def _check_files_status(self, file_ids: List[str], provider: str, model: str, conversation_id: str) -> Optional[ChatResponse]:
        """如果有文件尚未就绪，返回可直接响应给聊天接口的消息。"""
        if not file_ids:
            return None

        file_contents = self.file_repo.get_parsed_file_content(file_ids)
        for file_id in file_ids:
            if file_id in file_contents:
                continue

            file = self.file_repo.get_file_by_id(file_id)
            if file and file.status == "parsing":
                return ChatResponse(
                    id="file_parsing",
                    provider=provider,
                    model=model,
                    message=Message(
                        role=MessageRoles.ASSISTANT,
                        type=MessageTypes.ASSISTANT_CONTENT,
                        content="文件正在解析中，请稍后再试...",
                    ),
                    conversation_id=conversation_id,
                )
            if file and file.status == "error":
                return ChatResponse(
                    id="file_error",
                    provider=provider,
                    model=model,
                    message=Message(
                        role=MessageRoles.ASSISTANT,
                        type=MessageTypes.ASSISTANT_CONTENT,
                        content=f"文件处理出错: {file.processing_result.get('message', '未知错误')}",
                    ),
                    conversation_id=conversation_id,
                )
        return None

    def _should_use_reasoning_strategy(self, provider: str, model: str, options: Optional[Dict[str, Any]] = None) -> bool:
        """决定非流式请求是否走 reasoning 解析。"""
        if options is None:
            options = {}

        use_reasoning = options.get("use_reasoning")
        if use_reasoning is True:
            return True
        if use_reasoning is False:
            return False

        model_name = model.lower()
        if provider == "volcengine" and ("thinking" in model_name or "deepseek-r1" in model_name):
            return True
        if provider == "deepseek" and model == "deepseek-reasoner":
            return True
        if provider == "qwen" and ("qwq" in model_name or "qwen3" in model_name):
            return True
        return False

    async def _invoke_non_stream_model(self, provider: str, model: str, messages, options: Optional[Dict[str, Any]] = None):
        """执行一次非流式模型调用。"""
        if options is None:
            options = {}

        llm = llm_manager.get_model(provider=provider, model=model, options=options)
        return llm.invoke(messages)

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
    ) -> Union[StreamingResponse, ChatResponse]:
        """处理用户消息并获取AI响应"""
        # 初始化options
        if options is None:
            options = {}
        # 获取或创建会话
        conversation = self._get_or_create_conversation(conversation_id, user_id, provider, model, message)

        # 记录用户消息
        user_message = Message(
            role=MessageRoles.USER, 
            type=MessageTypes.USER_QUERY,
            content=message
        )
        
        # 使用用户消息的ID作为turn_id
        turn_id = user_message.id
        user_message.turn_id = turn_id
        
        conversation.messages.append(user_message)
        
        # 准备聊天历史
        chat_history = []
        for msg in conversation.messages[:-1]:  # 排除刚添加的用户消息
            chat_history.append({"role": msg.role, "content": msg.content})
        
        chat_history.append({"role": MessageRoles.USER, "content": message})

        # 从聊天历史中提取消息
        messages = self._prepare_chat_messages(chat_history)

        # 保存会话和用户消息，确保在流式处理前它们已存在于数据库中
        self._persist_conversation(conversation)

        # 处理文件内容
        if file_ids and len(file_ids) > 0:
            # 检查文件状态
            status_response = self._check_files_status(file_ids, provider, model, conversation.id)
            if status_response:
                return status_response
                
            # 获取文件内容并增强消息
            file_contents = self._get_files_content(file_ids)
            if file_contents:
                messages = self._enhance_with_file_content(messages, message, file_contents)

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
                return conversation

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
        if options is None:
            options = {}

        try:
            response = await self._invoke_non_stream_model(provider, model, messages, options)
            reasoning_message = None
            if self._should_use_reasoning_strategy(provider, model, options):
                reasoning_content = ""
                if hasattr(response, "reasoning_content"):
                    reasoning_content = response.reasoning_content
                elif hasattr(response, "additional_kwargs") and "reasoning_content" in response.additional_kwargs:
                    reasoning_content = response.additional_kwargs["reasoning_content"]
                if reasoning_content:
                    reasoning_message = Message(
                        role=MessageRoles.ASSISTANT,
                        type=MessageTypes.REASONING_CONTENT,
                        content=reasoning_content,
                        turn_id=turn_id,
                    )

            ai_content = self._get_response_text(response)
            ai_message = Message(
                role=MessageRoles.ASSISTANT,
                type=MessageTypes.ASSISTANT_CONTENT,
                content=ai_content,
                turn_id=turn_id,
            )

            conversation = self.memory_service.get_conversation(conversation_id, user_id)
            if reasoning_message:
                conversation.messages.append(reasoning_message)
            conversation.messages.append(ai_message)
            self._persist_conversation(conversation)

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
            logger.exception(f"模型处理失败: {e}")
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

    def _build_recent_dialog_content(self, conversation: Conversation, fallback_user_limit: int = 3) -> str:
        """提取最近一轮用户/助手对话，必要时回退到最近几条用户消息。"""
        latest_user_msg = None
        latest_ai_msg = None

        for msg in reversed(conversation.messages):
            if not latest_ai_msg and msg.role == MessageRoles.ASSISTANT:
                latest_ai_msg = msg.content
            elif not latest_user_msg and msg.role == MessageRoles.USER:
                latest_user_msg = msg.content
            if latest_user_msg and latest_ai_msg:
                break

        dialog_lines = []
        if latest_user_msg:
            dialog_lines.append(f"用户: {latest_user_msg}")
        if latest_ai_msg:
            dialog_lines.append(f"助手: {latest_ai_msg}")

        if dialog_lines:
            return "\n".join(dialog_lines)

        fallback_user_messages = []
        for msg in conversation.messages:
            if msg.role == MessageRoles.USER:
                fallback_user_messages.append(msg.content)
                if len(fallback_user_messages) >= fallback_user_limit:
                    break

        return "\n".join(fallback_user_messages)

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        """更新消息"""
        updated_message = self.memory_service.update_message(message_id, update_data)
        if updated_message:
            self.db.commit()
        return updated_message

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
                message = self._build_recent_dialog_content(conversation)

        if not message:
            raise ValueError("必须提供消息内容或有效的会话ID")

        # 使用提示词管理器获取并格式化提示词
        prompt = prompt_manager.format_prompt("generate_title", content=message)

        try:
            if conversation:
                response = await self._invoke_non_stream_model(
                    conversation.provider,
                    conversation.model,
                    [HumanMessage(content=prompt)],
                    {"use_reasoning": False},
                )
            else:
                llm = llm_manager.get_default_model()
                response = llm.invoke([HumanMessage(content=prompt)])

            # 清理标题（去除多余的引号、空白和解释性文字）
            title = ChatUtils.clean_model_text(self._get_response_text(response))

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
                self._persist_conversation(conversation)

            return title
        except Exception as e:
            logger.exception(f"生成标题时发生错误: {str(e)}")
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

        dialog_content = self._build_recent_dialog_content(conversation)
        
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
            response = await self._invoke_non_stream_model(
                conversation.provider,
                conversation.model,
                [HumanMessage(content=prompt)],
                {"use_reasoning": False},
            )

            response_text = self._get_response_text(response)

            # 解析响应文本，提取问题
            questions = self._parse_questions(response_text)
            
            return questions[:3]  # 确保只返回3个问题
        except Exception as e:
            logger.exception(f"生成推荐问题时发生错误: {str(e)}")
            # 如果生成失败，返回默认问题
            return [
                "您对这个主题还有其他问题吗？",
                "您想了解更多相关信息吗？",
                "您想要探讨这个话题的哪些方面？"
            ]

    def _parse_questions(self, response_text: str) -> List[str]:
        """从响应文本中解析出问题列表"""
        return ChatUtils.parse_questions(response_text)
