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
from app.services.memory_service import MemoryService
from app.services.vector_service import VectorService
from app.services.web_search_service import WebSearchService

class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.memory_service = MemoryService(db)
        # 初始化向量服务和上下文增强器
        self.vector_service = VectorService.get_instance(db)
        self.context_enhancer = ContextEnhancer(db)
        self.file_repo = FileRepository(db)
        self.file_processor = FileProcessor()

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

        file_paths = []
        file_mimetypes = []
        if file_ids and len(file_ids) > 0:
            # 获取文件信息
            files_info = self.file_repo.get_files_info(file_ids)
            for file_info in files_info:
                if file_info and file_info.path:
                    file_paths.append(file_info.path)
                    file_mimetypes.append(file_info.mimetype)
            logger.info(f"处理带文件的请求, 文件数量: {len(file_paths)}")

        # 保存会话（先保存用户消息）
        conversation.updated_at = datetime.now()
        self.memory_service.save_conversation(conversation)

        # 异步向量化用户消息
        asyncio.create_task(self._vectorize_message_async(user_message, conversation_id))

        file_contents = {}
        if file_ids and len(file_ids) > 0:
            # 获取文件解析结果
            file_contents = self.file_repo.get_parsed_file_content(file_ids)

            # 检查是否所有文件都已解析
            all_files_processed = True
            for file_id in file_ids:
                if file_id not in file_contents:
                    all_files_processed = False
                    file = self.file_repo.get_file_by_id(file_id)
                    if file and file.status == "parsing":
                        # 文件仍在解析中
                        return ChatResponse(
                            id=str(uuid.uuid4()),
                            provider=provider,
                            model=model,
                            message=Message(
                                role="assistant",
                                content="文件正在解析中，请稍后再试..."
                            ),
                            conversation_id=conversation.id
                        )
                    elif file and file.status == "error":
                        # 文件解析出错
                        return ChatResponse(
                            id=str(uuid.uuid4()),
                            provider=provider,
                            model=model,
                            message=Message(
                                role="assistant",
                                content=f"文件处理出错: {file.processing_result.get('message', '未知错误')}"
                            ),
                            conversation_id=conversation.id
                        )

            # 如果有文件解析结果，增强用户消息
            if file_contents:
                # 将文件内容添加到消息中
                file_content_text = "\n\n".join([
                    f"文件内容 ({i + 1}):\n{content}"
                    for i, content in enumerate(file_contents.values())
                ])

                enhanced_message = f"""用户问题: {message}\n\n参考以下文件内容:\n{file_content_text}"""
                # 替换最后一条消息内容
                messages[-1] = HumanMessage(content=enhanced_message)

        # 根据是否为流式响应分别处理
        if stream:
            # 使用支持推理的模型
            if (provider == "deepseek" and model == "deepseek-reasoner") or (provider == "qwen" and "qwq" in model.lower()):
                return StreamingResponse(
                    self._generate_reasoning_stream(provider, model, messages, conversation.id),
                    media_type="text/event-stream"
                )
            # 默认使用常规流式响应
            else:
                return StreamingResponse(
                    self._generate_normal_stream(provider, model, messages, conversation.id),
                    media_type="text/event-stream"
                )
        else:
            # 非流式响应处理
            # 使用支持推理的模型
            if (provider == "deepseek" and model == "deepseek-reasoner") or (provider == "qwen" and "qwq" in model.lower()):
                try:
                    # 获取AI模型
                    llm = llm_manager.get_model(provider=provider, model=model)

                    # 调用模型
                    response = llm.invoke(messages)
                    
                    # 从响应中提取 reasoning_content 和 content
                    reasoning_content = getattr(response, 'reasoning_content', '')
                    ai_content = response.content if hasattr(response, 'content') else response

                    # 记录推理过程
                    reasoning_message = Message(
                        role="reasoning",
                        content=reasoning_content
                    )

                    # 记录最终答案
                    ai_message = Message(
                        role="assistant",
                        content=ai_content
                    )

                    # 添加到会话
                    conversation.messages.append(reasoning_message)
                    conversation.messages.append(ai_message)

                    # 更新并保存会话
                    conversation.updated_at = datetime.now()
                    self.memory_service.save_conversation(conversation)

                    # 向量化消息
                    asyncio.create_task(self._vectorize_message_async(reasoning_message, conversation.id))
                    asyncio.create_task(self._vectorize_message_async(ai_message, conversation.id))

                    # 返回响应
                    return ChatResponse(
                        id=str(uuid.uuid4()),
                        provider=provider,
                        model=model,
                        message=ai_message,
                        conversation_id=conversation.id,
                        reasoning=reasoning_content
                    )
                except Exception as e:
                    logger.error(f"推理模型处理失败: {e}")
                    logger.info("尝试使用常规模式处理...")

            # 常规模式（无文件）
            else:
                try:
                    # 获取AI模型
                    llm = llm_manager.get_model(provider=provider, model=model)

                    # 调用模型
                    response = llm.invoke(messages)
                    
                    # 从响应中提取推理内容
                    reasoning_content = ''
                    if provider == "deepseek" and model == "deepseek-reasoner":
                        reasoning_content = getattr(response, 'reasoning_content', '')
                    elif hasattr(response, 'additional_kwargs') and 'reasoning_content' in response.additional_kwargs:
                        reasoning_content = response.additional_kwargs['reasoning_content']
                    
                    # 获取最终答案
                    ai_content = response.content if hasattr(response, 'content') else response

                    # 如果有推理内容，记录推理过程
                    if reasoning_content:
                        reasoning_message = Message(
                            role="reasoning",
                            content=reasoning_content
                        )
                        conversation.messages.append(reasoning_message)
                        # 异步向量化推理消息
                        asyncio.create_task(self._vectorize_message_async(reasoning_message, conversation.id))

                    # 记录最终答案
                    ai_message = Message(
                        role="assistant",
                        content=ai_content
                    )
                    conversation.messages.append(ai_message)

                    # 更新并保存会话
                    conversation.updated_at = datetime.now()
                    self.memory_service.save_conversation(conversation)

                    # 向量化答案消息
                    asyncio.create_task(self._vectorize_message_async(ai_message, conversation.id))

                    # 返回响应
                    return ChatResponse(
                        id=str(uuid.uuid4()),
                        provider=provider,
                        model=model,
                        message=ai_message,
                        conversation_id=conversation.id,
                        reasoning=reasoning_content
                    )
                except Exception as e:
                    logger.error(f"常规模式处理失败: {e}")
                    raise

    async def _generate_normal_stream(self, provider, model, messages, conversation_id):
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
        await self._save_stream_response(conversation_id, full_response)
        yield f"data: {json.dumps({'content': '[DONE]', 'conversation_id': conversation_id})}\n\n"

    async def _save_stream_response(self, conversation_id, response_text):
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

                # 异步向量化消息
                asyncio.create_task(self._vectorize_message_async(ai_message, conversation_id))
        except Exception as e:
            logging.error(f"保存流式响应失败: {str(e)}")

    def _prepare_chat_messages(self, chat_history):
        """准备发送给LLM的消息格式"""
        messages = []
        last_role = None
        
        for msg in chat_history:
            current_role = msg["role"]
            
            # 如果是 Deepseek Reasoner 模型，跳过 reasoning 角色的消息
            if current_role == "reasoning":
                continue
                
            # 检查是否有连续的角色
            if last_role and last_role == current_role:
                # 如果是连续的用户消息，合并内容
                if current_role == "user":
                    messages[-1].content += "\n" + msg["content"]
                    continue
                # 如果是连续的助手消息，跳过
                elif current_role == "assistant":
                    continue
            
            # 添加消息
            if current_role == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif current_role == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif current_role == "system":
                messages.append(SystemMessage(content=msg["content"]))
            
            last_role = current_role

        return messages

    async def _vectorize_message_async(self, message: Message, conversation_id: str):
        """异步向量化单条消息"""
        # 为了提高回复速度，暂时关闭
        return
        # try:
        #     # 线程池执行CPU密集型向量化操作
        #     loop = asyncio.get_event_loop()
        #     await loop.run_in_executor(None, self.vector_service.vectorize_message, message, conversation_id)
        # except Exception as e:
        #     logging.error(f"异步向量化消息失败: {e}")

    async def _vectorize_messages_async(self, messages: List[Message], conversation_id: str):
        """异步向量化多条消息"""
        try:
            for message in messages:
                await self._vectorize_message_async(message, conversation_id)
        except Exception as e:
            logging.error(f"异步向量化多条消息失败: {e}")

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

    async def _generate_reasoning_stream(self, provider, model, messages, conversation_id):
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
        await self._save_stream_response_with_reasoning(
            conversation_id=conversation_id,
            response_text=answer_result,
            reasoning_text=reasoning_result
        )

        # 完成标志
        yield await send_event("done")

    async def _save_stream_response_with_reasoning(self, conversation_id, response_text, reasoning_text):
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

                # 异步向量化消息
                if reasoning_text:
                    asyncio.create_task(self._vectorize_message_async(
                        reasoning_message, conversation_id
                    ))
                asyncio.create_task(self._vectorize_message_async(
                    ai_message, conversation_id
                ))
        except Exception as e:
            logging.error(f"保存推理流式响应失败: {str(e)}")

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
                
            
