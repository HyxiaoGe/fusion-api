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

        # 判断是否启用推理模式
        use_reasoning = options.get("use_reasoning", True) if options else True
        logger.info(f"是否使用推理模式: {use_reasoning}")

        # 应用上下文增强
        use_enhancement = False
        # use_enhancement = options.get("use_enhancement", True) if options else True
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
            # 如果启用推理模式，使用二阶段流式推理
            if use_reasoning:
                return StreamingResponse(
                    self.generate_stream_with_reasoning(
                        provider=provider,
                        model=model,
                        message=message,
                        file_contents=file_contents,
                        conversation_id=conversation.id,
                        options=options
                    ),
                    media_type="text/event-stream"
                )
            else:
                return StreamingResponse(
                    self._generate_normal_stream(provider, model, messages, conversation.id),
                    media_type="text/event-stream"
                )
        else:
            # 非流式响应处理
            # 如果有文件
            if file_paths:
                try:
                    # 使用统一的文件处理器处理文件
                    file_response = await self.file_processor.process_files(
                        file_paths=file_paths,
                        query=message,
                        mimetypes=file_mimetypes
                    )

                    ai_content = file_response.get("content", "无法处理文件请求")
                except Exception as e:
                    logger.error(f"处理文件请求失败: {e}")
                    ai_content = f"处理文件时出错: {str(e)}"

                # 创建AI响应消息
                ai_message = Message(
                    role="assistant",
                    content=ai_content
                )
                conversation.messages.append(ai_message)

                # 更新并保存会话
                conversation.updated_at = datetime.now()
                self.memory_service.save_conversation(conversation)

                # 向量化消息
                asyncio.create_task(self._vectorize_message_async(ai_message, conversation.id))

                # 返回响应
                return ChatResponse(
                    id=str(uuid.uuid4()),
                    provider=provider,
                    model=model,
                    message=ai_message,
                    conversation_id=conversation.id
                )

            # 如果启用推理模式
            elif use_reasoning:
                try:
                    # 使用二阶段推理处理
                    reasoning_result = await self.process_message_with_reasoning(
                        provider=provider,
                        model=model,
                        message=message,
                        conversation_id=conversation.id
                    )

                    # 记录推理过程
                    reasoning_message = Message(
                        role="reasoning",
                        content=reasoning_result["reasoning"]
                    )

                    # 记录最终答案
                    ai_message = Message(
                        role="assistant",
                        content=reasoning_result["answer"]
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
                        reasoning=reasoning_result["reasoning"]
                    )

                except Exception as e:
                    logger.error(f"推理模式处理失败: {e}")
                    # 推理失败时继续使用常规模式
                    use_reasoning = False

            # 常规模式（无推理，无文件）
            if not use_reasoning:
                try:
                    # 获取AI模型
                    llm = llm_manager.get_model(provider=provider, model=model)

                    # 调用模型
                    response = llm.invoke(messages)
                    ai_content = response.content if hasattr(response, 'content') else response

                    # 创建AI响应消息
                    ai_message = Message(
                        role="assistant",
                        content=ai_content
                    )

                    # 添加到会话
                    conversation.messages.append(ai_message)

                    # 更新并保存会话
                    conversation.updated_at = datetime.now()
                    self.memory_service.save_conversation(conversation)

                    # 向量化消息
                    asyncio.create_task(self._vectorize_message_async(ai_message, conversation.id))

                    # 返回响应
                    return ChatResponse(
                        id=str(uuid.uuid4()),
                        provider=provider,
                        model=model,
                        message=ai_message,
                        conversation_id=conversation.id
                    )

                except Exception as e:
                    logger.error(f"常规模式处理失败: {e}")
                    raise

    async def process_message_with_reasoning(
            self, provider, model, message, conversation_id,
            options=None, file_contents=None
    ):
        """使用双调用方式处理模型推理的消息，支持文件内容"""
        # 准备推理提示，如果有文件内容则包含文件内容
        reasoning_base_prompt = "请针对以下问题进行深入思考和分析，详细说明你的思考过程。仅提供思考过程，不要给出最终答案。"

        if file_contents and len(file_contents) > 0:
            # 整合文件内容
            file_content_text = "\n\n".join([
                f"文件内容 ({i + 1}):\n{content}"
                for i, content in enumerate(file_contents.values())
            ])

            reasoning_prompt = f"""{reasoning_base_prompt}

            用户问题: {message}

            参考以下文件内容:
            {file_content_text}

            请基于文件内容和用户问题，详细分析这个问题，考虑各种角度和可能性。"""
        else:
            reasoning_prompt = f"""{reasoning_base_prompt}

            用户问题: {message}

            请详细分析这个问题，考虑各种角度和可能性。"""

        # 推理调用
        reasoning_llm = llm_manager.get_model(provider=provider, model=model)
        reasoning_response = reasoning_llm.invoke([HumanMessage(content=reasoning_prompt)])
        reasoning_content = reasoning_response.content if hasattr(reasoning_response, 'content') else reasoning_response

        # 第二步：获取最终答案 - 使用推理结果作为上下文，但要求只返回答案
        answer_base_prompt = "根据以下思考过程，请直接回答原始问题。仅提供答案，不要重复思考过程。"

        if file_contents and len(file_contents) > 0:
            answer_prompt = f"""{answer_base_prompt}

            原始问题: {message}

            参考文件内容: {file_content_text}

            思考过程: {reasoning_content}

            基于以上思考和文件内容，你的答案是:"""
        else:
            answer_prompt = f"""{answer_base_prompt}

            原始问题: {message}

            思考过程: {reasoning_content}

            基于以上思考，你的答案是:"""

        # 答案调用
        answer_llm = llm_manager.get_model(provider=provider, model=model)
        answer_response = answer_llm.invoke([HumanMessage(content=answer_prompt)])
        answer_content = answer_response.content if hasattr(answer_response, 'content') else answer_response

        return {
            "reasoning": reasoning_content.strip(),
            "answer": answer_content.strip(),
            "conversation_id": conversation_id
        }

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

    async def generate_stream_with_reasoning(
            self, provider, model, message, conversation_id,
            options=None, file_contents=None
    ):
        """使用双阶段流式处理推理和回答，支持文件内容"""

        # 构造发送事件的辅助函数
        async def send_event(event_type, content=None):
            data = {"type": event_type, "conversation_id": conversation_id}
            if content is not None:
                data["content"] = content
            return f"data: {json.dumps(data)}\n\n"

        # 阶段1：先获取并流式输出推理过程
        yield await send_event("reasoning_start")

        # 准备推理提示，加入文件内容
        if file_contents and len(file_contents) > 0:
            file_content_text = "\n\n".join([
                f"文件内容 ({i + 1}):\n{content}"
                for i, content in enumerate(file_contents.values())
            ])

            reasoning_prompt = f"""请针对以下问题进行深入思考和分析，详细说明你的思考过程。仅提供思考过程，不要给出最终答案。
                用户问题: {message}

                参考以下文件内容:
                {file_content_text}"""
        else:
            reasoning_prompt = f"""请针对以下问题进行深入思考和分析，详细说明你的思考过程。仅提供思考过程，不要给出最终答案。
                用户问题: {message}"""

        reasoning_llm = llm_manager.get_model(provider=provider, model=model)
        reasoning_result = ""

        # 流式获取推理过程
        for chunk in reasoning_llm.stream([HumanMessage(content=reasoning_prompt)]):
            content = chunk.content if hasattr(chunk, 'content') else chunk
            reasoning_result += content
            yield await send_event("reasoning_content", content)

        yield await send_event("reasoning_complete")

        # 阶段2：再获取并流式输出最终答案
        yield await send_event("answering_start")

        # 准备回答提示，加入文件内容
        if file_contents and len(file_contents) > 0:
            file_content_text = "\n\n".join([
                f"文件内容 ({i + 1}):\n{content}"
                for i, content in enumerate(file_contents.values())
            ])

            answer_prompt = f"""根据以下思考过程，请直接回答原始问题。仅提供答案，不要重复思考过程。
            原始问题: {message}

            参考文件内容: {file_content_text}

            思考过程: {reasoning_result}

            基于以上思考和文件内容，你的答案是:"""
        else:
            answer_prompt = f"""根据以下思考过程，请直接回答原始问题。仅提供答案，不要重复思考过程。
            原始问题: {message}

            思考过程: {reasoning_result}

            基于以上思考，你的答案是:"""

        answer_llm = llm_manager.get_model(provider=provider, model=model)
        answer_result = ""

        # 流式获取最终答案
        for chunk in answer_llm.stream([HumanMessage(content=answer_prompt)]):
            content = chunk.content if hasattr(chunk, 'content') else chunk
            answer_result += content
            yield await send_event("answering_content", content)

        # 完成所有处理
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
        for msg in chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system":
                messages.append(SystemMessage(content=msg["content"]))

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
        1. 不超过15个字
        2. 直接给出标题，不要包含引号或其他解释性文字
        3. 避免使用"关于"、"讨论"等过于宽泛的词语
        4. 标题应该明确反映对话的核心主题

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
