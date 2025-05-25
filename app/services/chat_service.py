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


# ==================== 常量定义 ====================

# 消息角色常量
class MessageRoles:
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    TOOL = "tool"


# 事件类型常量
class EventTypes:
    FUNCTION_STREAM_START = "function_stream_start"
    REASONING_START = "reasoning_start"
    REASONING_CONTENT = "reasoning_content"
    REASONING_COMPLETE = "reasoning_complete"
    CONTENT = "content"
    DONE = "done"
    ERROR = "error"
    FUNCTION_CALL_DETECTED = "function_call_detected"
    FUNCTION_RESULT = "function_result"
    GENERATING_QUERY = "generating_query"
    QUERY_GENERATED = "query_generated"
    USER_SEARCH_START = "user_search_start"
    PERFORMING_SEARCH = "performing_search"
    SYNTHESIZING_ANSWER = "synthesizing_answer"


# 函数名称常量
class FunctionNames:
    WEB_SEARCH = "web_search"
    HOT_TOPICS = "hot_topics"


# 函数描述常量
FUNCTION_DESCRIPTIONS = {
    FunctionNames.WEB_SEARCH: "我需要搜索网络获取更多信息...",
    FunctionNames.HOT_TOPICS: "我将查询最新的热点话题...",
}

# 消息文本常量
class MessageTexts:
    OPTIMIZING_SEARCH_QUERY = "正在优化搜索查询..."
    SEARCH_QUERY_PREFIX = "搜索查询: "
    SYNTHESIZING_ANSWER = "正在结合搜索结果生成回答..."
    USER_PREVIOUS_QUESTION = "用户的先前问题"
    NO_USER_QUERY_ERROR = "无法找到用户原始提问以执行联网搜索。"
    PROCESSING_ERROR_PREFIX = "处理出错: "
    USER_PRIORITIZED_SEARCH_ERROR_PREFIX = "处理用户优先搜索出错: "
    FUNCTION_CALL_ERROR_PREFIX = "函数调用流处理出错: "
    FUNCTION_CALL_NEED_PREFIX = "需要调用函数: "
    FUNCTION_CALL_GENERIC_PREFIX = "我需要调用 {} 函数获取信息..."


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

    def _create_event_sender(self, conversation_id: str):
        """
        创建事件发送器函数
        
        Args:
            conversation_id: 会话ID
            
        Returns:
            async function: 异步事件发送函数，接受event_type和可选的content参数
        """
        async def send_event(event_type: str, content=None):
            data = {"type": event_type, "conversation_id": conversation_id}
            if content is not None:
                data["content"] = content
            return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        return send_event

    def _extract_user_message_from_messages(self, messages: List) -> str:
        """
        从消息列表中提取最后一条用户消息
        
        Args:
            messages: 消息列表，可能包含不同类型的消息对象
            
        Returns:
            str: 最后一条用户消息的内容，如果没有找到则返回空字符串
        """
        for msg in reversed(messages):
            # 处理不同类型的消息对象
            if hasattr(msg, "type") and msg.type == "human":
                return msg.content
            elif hasattr(msg, "role") and msg.role == MessageRoles.USER:
                return msg.content
            elif isinstance(msg, dict) and msg.get("role") == MessageRoles.USER:
                return msg.get("content", "")
        return ""

    def _parse_function_arguments(self, function_args: Union[str, dict]) -> dict:
        """
        解析函数参数，确保返回有效的字典
        
        Args:
            function_args: 函数参数，可能是JSON字符串或字典
            
        Returns:
            dict: 解析后的参数字典，解析失败时返回空字典
        """
        try:
            if isinstance(function_args, str):
                return json.loads(function_args) if function_args.strip() else {}
            else:
                return function_args
        except:
            return {}

    async def _generate_search_query(self, user_message: str, llm) -> str:
        """
        生成优化后的搜索查询
        
        Args:
            user_message: 用户原始消息
            llm: 语言模型实例
            
        Returns:
            str: 优化后的搜索查询字符串
        """
        current_date = datetime.now().strftime("%Y年%m月%d日")
        search_query_prompt = f"为以下用户问题生成一个简洁明确的搜索查询: '{user_message}'。如果问题中包含与当前时间相关的指代（例如'今天'、'目前'），请以 {current_date} 作为当前日期进行理解。请仅返回搜索查询文本本身，不要附加任何解释或说明。"
        search_query_msgs = [{"role": MessageRoles.USER, "content": search_query_prompt}]
        
        # 使用现有模型生成查询
        search_query_response = await llm.ainvoke(search_query_msgs)
        search_query = search_query_response.content if hasattr(search_query_response, 'content') else str(search_query_response)
        
        # 清理搜索查询（去除引号等）
        return search_query.strip().strip('"\'')

    def _extract_original_user_query(self, messages: List) -> str:
        """
        从消息列表中提取用户原始查询
        
        Args:
            messages: 消息列表，包含各种类型的消息对象
            
        Returns:
            str: 用户原始查询内容，如果没有找到则返回默认文本
        """
        original_user_query = MessageTexts.USER_PREVIOUS_QUESTION
        if messages:
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                content_to_check = None
                is_user_role = False
                if isinstance(msg, dict):
                    if msg.get("role") == MessageRoles.USER:
                        content_to_check = msg.get("content")
                        is_user_role = True
                elif hasattr(msg, 'type') and msg.type == 'human' and hasattr(msg, 'content'):
                    content_to_check = msg.content
                    is_user_role = True
                
                if is_user_role and content_to_check:
                    original_user_query = content_to_check
                    break
            
            if original_user_query == MessageTexts.USER_PREVIOUS_QUESTION and messages:
                last_msg_obj = messages[-1]
                if isinstance(last_msg_obj, dict) and last_msg_obj.get("role") == MessageRoles.USER:
                    original_user_query = last_msg_obj.get("content", original_user_query)
                elif hasattr(last_msg_obj, 'type') and last_msg_obj.type == 'human' and hasattr(last_msg_obj, 'content'):
                     original_user_query = last_msg_obj.content
        
        return original_user_query

    def _validate_and_process_function_arguments(self, function_call_data: dict) -> str:
        """
        验证和处理函数参数，确保返回有效的JSON字符串
        
        Args:
            function_call_data: 函数调用数据字典
            
        Returns:
            str: 有效的JSON字符串，验证失败时返回空对象字符串"{}"
        """
        original_arguments_str = function_call_data["function"].get("arguments", "{}")
        try:
            json.loads(original_arguments_str)
            return original_arguments_str if original_arguments_str.strip() else "{}"
        except json.JSONDecodeError:
            return "{}"

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
                self.generate_user_prioritized_web_search_stream(provider, model, messages, conversation_id, options),
                media_type="text/event-stream"
            )
        elif ai_can_use_functions:
            return StreamingResponse(
                self.generate_function_call_stream(provider, model, messages, conversation_id, options),
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

    async def generate_function_call_stream(self, provider, model, messages, conversation_id, options=None):
        """
        生成支持函数调用的流式响应
        
        Args:
            provider: 模型提供商
            model: 模型名称
            messages: 消息列表
            conversation_id: 会话ID
            options: 可选参数
        """
        if options is None:
            options = {}
        
        # 初始化基础组件
        send_event = self._create_event_sender(conversation_id)
        llm = llm_manager.get_model(provider=provider, model=model)
        context = {"db": self.db, "conversation_id": conversation_id}
        
        # 准备消息和函数定义
        processed_messages = self._prepare_function_call_messages(messages)
        functions_kwargs = function_adapter.prepare_functions_for_model(provider, model)
        
        try:
            yield await send_event(EventTypes.FUNCTION_STREAM_START)
            
            # 处理函数调用流程
            async for event in self._execute_function_call_flow(
                provider, llm, processed_messages, functions_kwargs, 
                messages, context, send_event, options
            ):
                yield event
                
        except Exception as e:
            logger.error(f"{MessageTexts.FUNCTION_CALL_ERROR_PREFIX}{e}")
            import traceback
            logger.error(traceback.format_exc())
            yield await send_event(EventTypes.ERROR, f"{MessageTexts.PROCESSING_ERROR_PREFIX}{str(e)}")

    def _prepare_function_call_messages(self, messages):
        """
        准备函数调用的消息列表
        
        Args:
            messages: 原始消息列表
            
        Returns:
            list: 处理后的消息列表
        """
        new_system_prompt_dict = {
            "role": MessageRoles.SYSTEM,
            "content": FUNCTION_CALL_BEHAVIOR_PROMPT
        }
        
        # 移除原有系统消息，添加新的系统提示
        return [new_system_prompt_dict] + [
            msg for msg in messages if not (isinstance(msg, dict) and msg.get("role") == MessageRoles.SYSTEM)
        ]

    async def _execute_function_call_flow(self, provider, llm, processed_messages, functions_kwargs, 
                                         original_messages, context, send_event, options):
        """
        执行完整的函数调用流程
        
        Args:
            provider: 模型提供商
            llm: 语言模型实例
            processed_messages: 处理后的消息列表
            functions_kwargs: 函数调用参数
            original_messages: 原始消息列表
            context: 上下文信息
            send_event: 事件发送函数
            options: 可选参数
        """
        use_reasoning = options.get("use_reasoning", False)
        
        # 第一阶段：检测函数调用
        async for result in self._process_first_llm_stream(llm, processed_messages, functions_kwargs, send_event, use_reasoning):
            if isinstance(result, tuple):
                function_call_detected, function_call_data, full_response, reasoning_start_sent = result
                break
            else:
                yield result
        
        # 如果没有检测到函数调用
        if not function_call_detected:
            if reasoning_start_sent: 
                yield await send_event(EventTypes.REASONING_COMPLETE)
            await self.save_stream_response(context["conversation_id"], full_response) 
            yield await send_event(EventTypes.DONE)
            return
        
        # 第二阶段：处理函数调用
        function_call_data["first_llm_thought"] = full_response
        function_name = function_call_data["function"].get("name", "")
        
        # 处理查询生成（如果需要）
        async for result in self._handle_web_search_query_generation(function_call_data, original_messages, llm, send_event):
            if isinstance(result, dict):
                function_call_data = result
                break
            else:
                yield result
        
        # 执行函数调用
        function_result = await function_adapter.process_function_call(
            provider, function_call_data["function"], context
        )
        
        # 第三阶段：处理函数结果
        function_handlers = {
            FunctionNames.WEB_SEARCH: self._handle_web_search_function,
            FunctionNames.HOT_TOPICS: self._handle_hot_topics_function,
        }
        
        handler = function_handlers.get(function_name)
        if handler:
            # 使用专门的处理器
            async for event in handler(
                send_event, function_call_data, function_result, 
                context["conversation_id"], llm, original_messages, options
            ):
                yield event
        else:
            # 使用默认处理流程
            async for result in self._handle_default_function_processing(
                original_messages, function_call_data, function_result, llm, send_event, provider
            ):
                if isinstance(result, str):
                    final_response = result
                    break
                else:
                    yield result
            
            # 保存对话历史
            await self.save_function_call_stream_response(
                conversation_id=context["conversation_id"],
                function_name=function_name,
                function_args=function_call_data["function"].get("arguments", "{}"),
                function_result=function_result,
                final_response=final_response
            )
            
            yield await send_event(EventTypes.DONE)

    async def _process_first_llm_stream(self, llm, processed_messages, functions_kwargs, 
                                       send_event, use_reasoning):
        """
        处理第一次LLM流式调用，检测函数调用
        
        Args:
            llm: 语言模型实例
            processed_messages: 处理后的消息列表
            functions_kwargs: 函数调用参数
            send_event: 事件发送函数
            use_reasoning: 是否使用推理模式
            
        Returns:
            tuple: (function_call_detected, function_call_data, full_response)
        """
        full_response = ""
        function_call_detected = False
        function_call_data = {}
        reasoning_state = ReasoningState()
        
        for chunk in llm.stream(processed_messages, **functions_kwargs):
            # 处理推理内容
            reasoning_events = await StreamProcessor.handle_reasoning_content_with_events(
                chunk, send_event, reasoning_state, use_reasoning
            )
            # 如果有推理事件，yield它们
            for event in reasoning_events:
                yield event

            # 检查流中是否有函数调用
            has_new_reasoning = len(reasoning_events) > 0
            if not function_call_detected:
                function_call_detected, function_call_data = function_adapter.detect_function_call_in_stream(chunk)
                
                if function_call_detected:
                    function_name = function_call_data['function'].get('name')
                    yield await send_event(EventTypes.FUNCTION_CALL_DETECTED, {
                        "function_type": function_name,
                        "description": f"{MessageTexts.FUNCTION_CALL_NEED_PREFIX}{function_name}"
                    })
            
            # 累积第一次LLM的思考过程
            content_chunk_text = StreamProcessor.extract_content(chunk)
            if content_chunk_text:
                full_response += content_chunk_text
                if not function_call_detected:
                     yield await send_event(EventTypes.CONTENT, content_chunk_text)
            
            # 如果已检测到函数调用，退出循环
            if function_call_detected:
                break
        
        yield (function_call_detected, function_call_data, full_response, reasoning_state.reasoning_start_sent)

    async def _handle_web_search_query_generation(self, function_call_data, messages, llm, send_event):
        """
        处理web_search函数的查询生成
        
        Args:
            function_call_data: 函数调用数据
            messages: 原始消息列表
            llm: 语言模型实例
            send_event: 事件发送函数
            
        Returns:
            dict: 更新后的函数调用数据
        """
        function_name = function_call_data["function"].get("name", "")
        function_args = function_call_data["function"].get("arguments", "{}")
        args_dict = self._parse_function_arguments(function_args)
            
        # 如果是web_search函数但没有query参数，使用LLM生成搜索查询
        if function_name == FunctionNames.WEB_SEARCH and not args_dict.get("query"):
            yield await send_event(EventTypes.GENERATING_QUERY, MessageTexts.OPTIMIZING_SEARCH_QUERY)
            
            user_message = self._extract_user_message_from_messages(messages)
            
            if user_message:
                search_query = await self._generate_search_query(user_message, llm)
                args_dict["query"] = search_query
                function_call_data["function"]["arguments"] = json.dumps(args_dict)
                yield await send_event(EventTypes.QUERY_GENERATED, f"{MessageTexts.SEARCH_QUERY_PREFIX}{search_query}")
        
        yield function_call_data

    async def _handle_default_function_processing(self, messages, function_call_data, function_result, 
                                                 llm, send_event, provider):
        """
        处理默认的函数调用流程（没有专门处理器的函数）
        
        Args:
            messages: 原始消息列表
            function_call_data: 函数调用数据
            function_result: 函数执行结果
            llm: 语言模型实例
            send_event: 事件发送函数
            provider: 模型提供商
            
        Returns:
            str: 最终响应内容
        """
        # 复制原始消息并添加函数调用结果
        full_messages = list(messages)
        
        # 添加LLM的函数调用响应
        if provider in ["deepseek", "openai", "anthropic", "qwen", "volcengine"]:
            # 使用tool_calls格式
            full_messages.append({
                "role": MessageRoles.ASSISTANT,
                "content": None,
                "tool_calls": [{
                    "type": "function",
                    "function": function_call_data["function"],
                    "id": function_call_data.get("tool_call_id", "call_1")
                }]
            })
        else:
            # 使用传统function_call格式
            full_messages.append({
                "role": MessageRoles.ASSISTANT,
                "content": "",
                "function_call": function_call_data["function"]
            })
        
        # 添加函数执行结果
        full_messages.append(function_result)
        
        # 处理最终回答的流式响应
        final_response = ""
        for chunk in llm.stream(full_messages):
            content = StreamProcessor.extract_content(chunk)
            if content:
                final_response += content
                yield await send_event(EventTypes.CONTENT, content)
        
        yield final_response

    async def _handle_web_search_function(self, send_event, function_call_data, function_result, 
                                          conversation_id, llm, messages, options):
        """处理web_search函数的专门处理器"""
        function_name = FunctionNames.WEB_SEARCH
        use_reasoning = options.get("use_reasoning", False)
        
        # 1. 先将 function_result 以 function_result 类型消息返回前端
        yield await send_event(EventTypes.FUNCTION_RESULT, {
            "function_type": function_name,
            "result": function_result
        })
        
        # --- 开始为第二次LLM调用构建新的消息列表 ---

        # 1. 获取用户原始提问 (Simplified: take the last user/human message)
        original_user_query = self._extract_original_user_query(messages)

        # 2. 构建第二次LLM调用的消息列表
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            original_user_query, function_name, function_result
        )

        original_tool_call_id = function_call_data.get("tool_call_id", f"call_{uuid.uuid4()}")
        valid_arguments_str = self._validate_and_process_function_arguments(function_call_data)
        first_llm_thought_content = function_call_data.get("first_llm_thought", None)

        assistant_tool_call_dict = {
            "id": original_tool_call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": valid_arguments_str
            }
        }
        
        ai_message_content = first_llm_thought_content if first_llm_thought_content and first_llm_thought_content.strip() else None

        current_assistant_message_dict = {
            "role": MessageRoles.ASSISTANT,
            "content": ai_message_content,
            "tool_calls": [assistant_tool_call_dict]
        }
        second_llm_messages.append(current_assistant_message_dict)

        tool_message_content = json.dumps(function_result, ensure_ascii=False)
        second_llm_messages.append(ToolMessage(content=tool_message_content, tool_call_id=original_tool_call_id).dict())


        # 4. 流式返回 LLM 的最终回复
        logger.info(f"{function_name}_handler: Preparing for second LLM stream to generate final answer.")
        final_response = ""
        async for result in StreamProcessor.process_llm_stream_with_reasoning(
            llm, second_llm_messages, send_event, use_reasoning
        ):
            # 传递所有事件给前端
            yield result
            # 如果是最终的完整响应（不是事件字符串），保存它
            if isinstance(result, str) and not result.startswith("data: "):
                final_response = result

        # 5. 保存完整对话历史
        await self.save_function_call_stream_response(
            conversation_id=conversation_id,
            function_name=function_name,
            function_args=valid_arguments_str,
            function_result=function_result,
            final_response=final_response
        )

        # 6. 完成标志
        yield await send_event(EventTypes.DONE)

    async def _handle_hot_topics_function(self, send_event, function_call_data, function_result, 
                                          conversation_id, llm, messages, options):
        """处理hot_topics函数的专门处理器"""
        function_name = FunctionNames.HOT_TOPICS
        use_reasoning = options.get("use_reasoning", False)
        
        # 1. 先将 function_result 以 function_result 类型消息返回前端
        yield await send_event(EventTypes.FUNCTION_RESULT, {
            "function_type": function_name,
            "result": function_result
        })

        # --- 开始为第二次LLM调用构建新的消息列表 (Similar to _handle_web_search_function) ---

        # 1. 获取用户原始提问
        original_user_query = self._extract_original_user_query(messages)

        # 2. 构建第二次LLM调用的消息列表
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            original_user_query, function_name, function_result
        )

        original_tool_call_id = function_call_data.get("tool_call_id", f"call_{uuid.uuid4()}")
        valid_arguments_str = self._validate_and_process_function_arguments(function_call_data)
        first_llm_thought_content = function_call_data.get("first_llm_thought", None)

        assistant_tool_call_dict = {
            "id": original_tool_call_id,
            "type": "function",
            "function": {
                "name": function_name,
                "arguments": valid_arguments_str
            }
        }
        
        ai_message_content = first_llm_thought_content if first_llm_thought_content and first_llm_thought_content.strip() else None
        
        current_assistant_message_dict = {
            "role": MessageRoles.ASSISTANT,
            "content": ai_message_content,
            "tool_calls": [assistant_tool_call_dict]
        }
        second_llm_messages.append(current_assistant_message_dict)

        tool_message_content = json.dumps(function_result, ensure_ascii=False)
        second_llm_messages.append(ToolMessage(content=tool_message_content, tool_call_id=original_tool_call_id).dict())

        # 4. 流式返回 LLM 的最终回复
        final_response = ""
        async for result in StreamProcessor.process_llm_stream_with_reasoning(
            llm, second_llm_messages, send_event, use_reasoning
        ):
            # 传递所有事件给前端
            yield result
            # 如果是最终的完整响应（不是事件字符串），保存它
            if isinstance(result, str) and not result.startswith("data: "):
                final_response = result

        # 5. 保存完整对话历史
        await self.save_function_call_stream_response(
            conversation_id=conversation_id,
            function_name=function_name,
            function_args=valid_arguments_str,
            function_result=function_result,
            final_response=final_response
        )

        # 6. 完成标志
        yield await send_event(EventTypes.DONE)

    async def save_function_call_stream_response(self, conversation_id, function_name, 
                                           function_args, function_result, final_response):
        """保存函数调用流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                # 创建函数调用请求消息，根据函数类型提供不同描述
                function_desc = FUNCTION_DESCRIPTIONS.get(function_name, f"我需要调用 {function_name} 函数获取信息...")
                
                # 创建函数调用消息
                function_call_message = Message(
                    role=MessageRoles.ASSISTANT,
                    content=function_desc
                )
                
                # 创建函数结果消息
                function_result_message = Message(
                    role=MessageRoles.FUNCTION,
                    content=json.dumps(function_result, ensure_ascii=False)
                )
                
                # 创建最终AI响应消息
                ai_message = Message(
                    role=MessageRoles.ASSISTANT,
                    content=final_response
                )
                
                # 添加所有消息到会话
                conversation.messages.append(function_call_message)
                conversation.messages.append(function_result_message)
                conversation.messages.append(ai_message)
                
                # 更新会话时间
                conversation.updated_at = datetime.now()
                
                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logger.error(f"保存函数调用流式响应失败: {e}")

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
            ai_function_message = Message(
                role=MessageRoles.ASSISTANT,
                content=response.content if hasattr(response, 'content') and response.content else f"我需要调用 {function_call.get('name')} 函数获取更多信息..."
            )
            conversation.messages.append(ai_function_message)
            
            # 处理函数调用
            function_result = await function_adapter.process_function_call(provider, function_call, context)
            
            # 准备工具消息
            function_name = function_call.get("name", "")
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

    async def generate_user_prioritized_web_search_stream(self, provider, model, messages, conversation_id, options):
        """
        生成用户优先的联网搜索流式响应
        
        Args:
            provider: 模型提供商
            model: 模型名称
            messages: 消息列表
            conversation_id: 会话ID
            options: 可选参数
        """
        if options is None:
            options = {}
        
        # 初始化基础组件
        llm = llm_manager.get_model(provider=provider, model=model)
        send_event = self._create_event_sender(conversation_id)
        
        # 验证用户查询
        original_user_query_content = self._extract_user_message_from_messages(messages)
        if not original_user_query_content:
            logger.warning(f"User-prioritized search: Could not extract original user query from messages: {messages}")
            yield await send_event(EventTypes.ERROR, MessageTexts.NO_USER_QUERY_ERROR)
            yield await send_event(EventTypes.DONE)
            return

        try:
            # 执行用户优先搜索流程
            async for event in self._execute_user_search_flow(
                provider, llm, original_user_query_content, 
                conversation_id, send_event, options
            ):
                yield event
                
        except Exception as e:
            logger.error(f"{MessageTexts.USER_PRIORITIZED_SEARCH_ERROR_PREFIX}{e}")
            import traceback
            logger.error(traceback.format_exc())
            yield await send_event(EventTypes.ERROR, f"{MessageTexts.USER_PRIORITIZED_SEARCH_ERROR_PREFIX}{str(e)}")
            yield await send_event(EventTypes.DONE)

    async def _execute_user_search_flow(self, provider, llm, user_query, conversation_id, send_event, options):
        """
        执行用户优先搜索的完整流程
        
        Args:
            provider: 模型提供商
            llm: 语言模型实例
            user_query: 用户查询内容
            conversation_id: 会话ID
            send_event: 事件发送函数
            options: 可选参数
        """
        use_reasoning = options.get("use_reasoning", False)
        
        yield await send_event(EventTypes.USER_SEARCH_START)

        # 第一阶段：生成搜索查询
        yield await send_event(EventTypes.GENERATING_QUERY, MessageTexts.OPTIMIZING_SEARCH_QUERY)
        generated_search_query = await self._generate_search_query(user_query, llm)
        yield await send_event(EventTypes.QUERY_GENERATED, f"{MessageTexts.SEARCH_QUERY_PREFIX}{generated_search_query}")

        # 第二阶段：执行搜索
        yield await send_event(EventTypes.PERFORMING_SEARCH, {"query": generated_search_query})
        search_result_data = await self._perform_web_search(provider, generated_search_query, conversation_id)
        yield await send_event(EventTypes.FUNCTION_RESULT, {
            "function_type": FunctionNames.WEB_SEARCH,
            "result": search_result_data
        })

        # 第三阶段：合成答案
        yield await send_event(EventTypes.SYNTHESIZING_ANSWER, MessageTexts.SYNTHESIZING_ANSWER)
        final_response_content = ""
        
        async for result in self._synthesize_search_answer(llm, user_query, search_result_data, send_event, use_reasoning):
            if isinstance(result, str):
                final_response_content = result
                break
            else:
                yield result

        # 保存搜索结果
        await self.save_user_prioritized_web_search_stream_response(
            conversation_id, user_query, generated_search_query, search_result_data, final_response_content
        )
        yield await send_event(EventTypes.DONE)

    async def _perform_web_search(self, provider, search_query, conversation_id):
        """
        执行网络搜索
        
        Args:
            provider: 模型提供商
            search_query: 搜索查询
            conversation_id: 会话ID
            
        Returns:
            dict: 搜索结果数据
        """
        context_for_tool = {"db": self.db, "conversation_id": conversation_id}
        web_search_function_call_payload = {
            "name": FunctionNames.WEB_SEARCH,
            "arguments": json.dumps({"query": search_query})
        }
        return await function_adapter.process_function_call(
            provider, 
            web_search_function_call_payload,
            context_for_tool
        )

    async def _synthesize_search_answer(self, llm, user_query, search_result_data, send_event, use_reasoning):
        """
        合成搜索结果的答案
        
        Args:
            llm: 语言模型实例
            user_query: 用户原始查询
            search_result_data: 搜索结果数据
            send_event: 事件发送函数
            use_reasoning: 是否使用推理模式
            
        Returns:
            str: 最终响应内容
        """
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            user_query, FunctionNames.WEB_SEARCH, search_result_data
        )

        final_response_content = ""
        async for result in StreamProcessor.process_llm_stream_with_reasoning(
            llm, second_llm_messages, send_event, use_reasoning
        ):
            # 传递所有事件给前端
            yield result
            # 如果是最终的完整响应（不是事件字符串），保存它
            if isinstance(result, str) and not result.startswith("data: "):
                final_response_content = result

        yield final_response_content

    async def save_user_prioritized_web_search_stream_response(self, conversation_id: str, 
                                               original_user_query_content: str,
                                               generated_search_query: str, 
                                               search_result: Any, 
                                               final_response: str):
        """保存用户优先的联网搜索流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if not conversation:
                logger.error(f"保存用户优先搜索流时未找到会话: {conversation_id}")
                return

            user_prioritized_tool_call_id = f"user_search_{uuid.uuid4()}"
            
            assistant_tool_calling_dict = {
                "id": user_prioritized_tool_call_id,
                "type": "function",
                "function": {
                    "name": FunctionNames.WEB_SEARCH,
                    "arguments": json.dumps({"query": generated_search_query})
                }
            }
            db_assistant_action_message = Message(
                role=MessageRoles.ASSISTANT,
                content="",
                tool_calls=[assistant_tool_calling_dict]
            )

            db_tool_response_message = Message(
                role=MessageRoles.TOOL, 
                tool_call_id=user_prioritized_tool_call_id,
                content=json.dumps(search_result, ensure_ascii=False)
            )

            db_final_ai_message = Message(
                role=MessageRoles.ASSISTANT,
                content=final_response
            )
            
            conversation.messages.append(db_assistant_action_message)
            conversation.messages.append(db_tool_response_message)
            conversation.messages.append(db_final_ai_message)
            
            conversation.updated_at = datetime.now()
            self.memory_service.save_conversation(conversation)
            logger.info(f"用户优先搜索流响应已保存到会话 {conversation_id}")

        except Exception as e:
            logger.error(f"保存用户优先搜索流响应失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

class ReasoningState:
    """推理状态管理类"""
    
    def __init__(self):
        self.reasoning_start_sent = False
        self.reasoning_complete_sent = False
    
    def reset(self):
        """重置推理状态"""
        self.reasoning_start_sent = False
        self.reasoning_complete_sent = False

class StreamProcessor:
    """流式处理辅助类"""
    
    @staticmethod
    def extract_content(chunk):
        """
        从chunk中提取内容
        
        Args:
            chunk: 流式响应块
            
        Returns:
            str: 提取的内容，如果没有则返回None
        """
        return chunk.content if hasattr(chunk, 'content') else None
    
    @staticmethod
    async def handle_reasoning_content(chunk, send_event, reasoning_state, use_reasoning=True):
        """
        处理推理内容
        
        Args:
            chunk: 流式响应块
            send_event: 事件发送函数
            reasoning_state: 推理状态对象
            use_reasoning: 是否启用推理模式
            
        Returns:
            bool: 是否在此块中处理了新的推理内容
        """
        if not use_reasoning:
            return False
            
        if not (hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs):
            return False
            
        reasoning_content = chunk.additional_kwargs['reasoning_content']
        if not (reasoning_content and reasoning_content.strip()):
            return False
            
        if not reasoning_state.reasoning_start_sent:
            await send_event(EventTypes.REASONING_START)
            reasoning_state.reasoning_start_sent = True
            
        await send_event(EventTypes.REASONING_CONTENT, reasoning_content)
        return True

    @staticmethod
    async def handle_reasoning_content_with_events(chunk, send_event, reasoning_state, use_reasoning=True):
        """
        处理推理内容并返回事件列表
        
        Args:
            chunk: 流式响应块
            send_event: 事件发送函数
            reasoning_state: 推理状态对象
            use_reasoning: 是否启用推理模式
            
        Returns:
            list: 推理事件列表
        """
        events = []
        
        if not use_reasoning:
            return events
            
        if not (hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs):
            return events
            
        reasoning_content = chunk.additional_kwargs['reasoning_content']
        if not (reasoning_content and reasoning_content.strip()):
            return events
            
        if not reasoning_state.reasoning_start_sent:
            events.append(await send_event(EventTypes.REASONING_START))
            reasoning_state.reasoning_start_sent = True
            
        events.append(await send_event(EventTypes.REASONING_CONTENT, reasoning_content))
        return events
    
    @staticmethod
    async def handle_content_and_reasoning_complete(content_chunk_text, send_event, reasoning_state, has_new_reasoning):
        """
        处理内容并在适当时发送推理完成事件
        
        Args:
            content_chunk_text: 内容文本
            send_event: 事件发送函数
            reasoning_state: 推理状态对象
            has_new_reasoning: 当前块是否有新的推理内容
        """
        if content_chunk_text is not None and content_chunk_text.strip():
            # 如果开始有实际内容且推理已开始但未完成，发送推理完成事件
            if (reasoning_state.reasoning_start_sent and 
                not reasoning_state.reasoning_complete_sent and 
                not has_new_reasoning):
                await send_event(EventTypes.REASONING_COMPLETE)
                reasoning_state.reasoning_complete_sent = True
        
        if content_chunk_text is not None:
            await send_event(EventTypes.CONTENT, content_chunk_text)
    
    @staticmethod
    async def finalize_reasoning(send_event, reasoning_state):
        """
        完成推理处理，确保推理完成事件被发送
        
        Args:
            send_event: 事件发送函数
            reasoning_state: 推理状态对象
        """
        if reasoning_state.reasoning_start_sent and not reasoning_state.reasoning_complete_sent:
            await send_event(EventTypes.REASONING_COMPLETE)
            reasoning_state.reasoning_complete_sent = True

    @staticmethod
    def create_tool_synthesis_messages(original_user_query, tool_name, tool_result):
        """
        创建工具结果合成的消息列表
        
        Args:
            original_user_query: 用户原始查询
            tool_name: 工具名称
            tool_result: 工具执行结果
            
        Returns:
            list: 用于合成的消息列表
        """
        system_prompt_content = SYNTHESIZE_TOOL_RESULT_PROMPT.format(
            original_user_query=original_user_query,
            tool_name=tool_name,
            tool_results_json=json.dumps(tool_result, ensure_ascii=False)
        )
        return [SystemMessage(content=system_prompt_content)]

    @staticmethod
    async def process_llm_stream_with_reasoning(llm, messages, send_event, use_reasoning=True):
        """
        处理LLM流式响应，包含推理处理
        
        Args:
            llm: 语言模型实例
            messages: 消息列表
            send_event: 事件发送函数
            use_reasoning: 是否启用推理模式
            
        Yields:
            str: 流式事件或最终响应内容
        """
        reasoning_state = ReasoningState()
        final_response = ""

        for chunk in llm.stream(messages):
            # 处理推理内容
            reasoning_events = await StreamProcessor.handle_reasoning_content_with_events(
                chunk, send_event, reasoning_state, use_reasoning
            )
            # 如果有推理事件，yield它们
            for event in reasoning_events:
                yield event
            has_new_reasoning = len(reasoning_events) > 0
            
            # 处理内容
            content_chunk_text = StreamProcessor.extract_content(chunk)
            if content_chunk_text is not None and content_chunk_text.strip():
                # 如果开始有实际内容且推理已开始但未完成，发送推理完成事件
                if (reasoning_state.reasoning_start_sent and 
                    not reasoning_state.reasoning_complete_sent and 
                    not has_new_reasoning):
                    yield await send_event(EventTypes.REASONING_COMPLETE)
                    reasoning_state.reasoning_complete_sent = True
            
            if content_chunk_text is not None:
                yield await send_event(EventTypes.CONTENT, content_chunk_text)
                final_response += content_chunk_text

        # 确保推理完成事件被发送
        if reasoning_state.reasoning_start_sent and not reasoning_state.reasoning_complete_sent:
            yield await send_event(EventTypes.REASONING_COMPLETE)
        
        yield final_response