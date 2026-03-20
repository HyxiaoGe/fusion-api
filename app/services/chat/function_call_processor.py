"""
函数调用处理器模块

负责处理所有函数调用相关的逻辑，包括函数检测、执行和结果处理
"""

import json
import uuid
from typing import Dict, Any, List, Optional

from langchain_core.messages import ToolMessage

from app.ai.llm_manager import llm_manager
from app.ai.prompts.templates import FUNCTION_CALL_BEHAVIOR_PROMPT, FUNCTION_CALL_BEHAVIOR_PROMPT_FOR_REASONING
from app.core.function_manager import function_adapter
from app.core.logger import app_logger as logger
from app.schemas.chat import Message
from app.constants import MessageRoles, EventTypes, FunctionNames, MessageTexts, FUNCTION_DESCRIPTIONS, USER_FRIENDLY_FUNCTION_DESCRIPTIONS, MessageTypes
from app.services.chat.stream_processor import ReasoningState, StreamProcessor
from app.services.chat.utils import ChatUtils


class FunctionCallProcessor:
    """函数调用处理器类"""
    TOOL_CALL_PROVIDERS = {"deepseek", "openai", "anthropic", "qwen", "volcengine", "google", "xai"}
    
    def __init__(self, db, memory_service):
        self.db = db
        self.memory_service = memory_service

    @staticmethod
    def _resolve_tool_call_id(tool_call_id: Optional[str], prefix: str = "call") -> str:
        """保证工具调用 ID 始终有稳定可用的值。"""
        if tool_call_id:
            return tool_call_id
        if prefix == "call_1":
            return "call_1"
        return f"{prefix}_{uuid.uuid4()}"

    def _persist_conversation(self, conversation):
        """统一保存函数调用产生的会话变更。"""
        self.memory_service.save_conversation(conversation)
        self.db.commit()

    def _get_persistable_conversation(self, conversation_id: str, user_id: Optional[str], error_message: str):
        """统一校验持久化上下文，返回可写入的会话。"""
        if not user_id:
            logger.error(error_message)
            return None
        return self.memory_service.get_conversation(conversation_id, user_id)

    def _append_and_persist_messages(self, conversation, messages: List[Message]) -> None:
        """向会话追加消息并落库。"""
        for message in messages:
            conversation.messages.append(message)
        self._persist_conversation(conversation)

    @staticmethod
    def _build_assistant_content_message(content: str, turn_id: Optional[str] = None) -> Message:
        """统一构造 assistant 正文消息。"""
        return Message(
            role=MessageRoles.ASSISTANT,
            type=MessageTypes.ASSISTANT_CONTENT,
            content=content,
            turn_id=turn_id,
        )

    @staticmethod
    def _resolve_function_call_description(
        function_name: str,
        first_llm_thought: Optional[str],
    ) -> str:
        """统一决定函数调用描述文本。"""
        if first_llm_thought and first_llm_thought.strip():
            return first_llm_thought
        return FUNCTION_DESCRIPTIONS.get(function_name, f"我需要调用 {function_name} 函数获取信息...")

    def _build_function_call_history_messages(
        self,
        function_name: str,
        function_result,
        final_response: str,
        turn_id: Optional[str] = None,
        first_llm_thought: Optional[str] = None,
    ) -> List[Message]:
        """统一构造函数调用落库所需的消息列表。"""
        function_call_message = Message(
            role=MessageRoles.ASSISTANT,
            type=MessageTypes.FUNCTION_CALL,
            content=self._resolve_function_call_description(function_name, first_llm_thought),
            turn_id=turn_id,
        )
        function_result_message = Message(
            role=MessageRoles.SYSTEM,
            type=MessageTypes.FUNCTION_RESULT,
            content=json.dumps(function_result, ensure_ascii=False),
            turn_id=turn_id,
        )
        ai_message = self._build_assistant_content_message(final_response, turn_id)
        return [function_call_message, function_result_message, ai_message]

    @staticmethod
    def _get_function_name(function_call_data: Dict[str, Any]) -> str:
        """统一读取函数名。"""
        return function_call_data.get("function", {}).get("name", "")

    @staticmethod
    def _get_user_friendly_function_description(function_name: str) -> str:
        """统一读取用户友好的函数描述。"""
        return USER_FRIENDLY_FUNCTION_DESCRIPTIONS.get(
            function_name,
            "我需要调用工具获取更多信息...",
        )

    def _build_function_detected_event_data(
        self,
        function_call_data: Dict[str, Any],
    ) -> Dict[str, str]:
        """统一构造函数检测事件 payload。"""
        function_name = self._get_function_name(function_call_data)
        return {
            "function_type": function_name,
            "description": self._get_user_friendly_function_description(function_name),
        }

    @staticmethod
    def _build_function_result_event_data(function_name: str, function_result) -> Dict[str, Any]:
        """统一构造函数结果事件 payload。"""
        return {
            "function_type": function_name,
            "result": function_result,
        }

    @staticmethod
    def _extract_final_stream_response(results: List[Any]) -> str:
        """统一提取流式处理结束后的最终文本结果。"""
        if results and isinstance(results[-1], str) and not results[-1].startswith("data: "):
            return results[-1]
        return ""

    @staticmethod
    def _update_function_arguments(
        function_call_data: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """统一回写函数参数。"""
        function_call_data["function"]["arguments"] = json.dumps(arguments, ensure_ascii=False)
        return function_call_data

    async def _send_function_detected_event(
        self,
        send_event,
        function_call_data: Dict[str, Any],
        function_call_sent: bool,
    ) -> bool:
        """统一发送函数检测事件，避免重复发送。"""
        if function_call_sent:
            return True
        await send_event(
            EventTypes.FUNCTION_CALL_DETECTED,
            self._build_function_detected_event_data(function_call_data),
        )
        return True

    def _finalize_first_pass_response(
        self,
        function_call_detected: bool,
        function_call_data: Dict[str, Any],
        full_response: str,
    ) -> str:
        """统一补全第一段调用缺失的默认响应文本。"""
        if function_call_detected and not full_response.strip():
            return self._get_user_friendly_function_description(
                self._get_function_name(function_call_data)
            )
        return full_response

    @classmethod
    def _provider_uses_tool_calls(cls, provider: str) -> bool:
        """判断 provider 是否使用 tool_calls 格式。"""
        return provider in cls.TOOL_CALL_PROVIDERS

    def _build_assistant_function_call_message(
        self,
        provider: str,
        function_call_data: Dict[str, Any],
        content: Optional[str] = None,
    ) -> tuple[Dict[str, Any], str]:
        """统一构造 assistant 的函数调用消息。"""
        tool_call_id = self._resolve_tool_call_id(function_call_data.get("tool_call_id"), "call_1")
        function_payload = function_call_data["function"]

        if self._provider_uses_tool_calls(provider):
            return {
                "role": MessageRoles.ASSISTANT,
                "content": content,
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": function_payload.get("name", ""),
                        "arguments": ChatUtils.stringify_function_arguments(
                            function_payload.get("arguments", "{}")
                        ),
                    },
                    "id": tool_call_id,
                }],
            }, tool_call_id

        return {
            "role": MessageRoles.ASSISTANT,
            "content": content if content is not None else "",
            "function_call": function_payload,
        }, tool_call_id

    @staticmethod
    def _build_tool_result_message(function_result, tool_call_id: str) -> Dict[str, Any]:
        """统一构造 tool 执行结果消息。"""
        return ToolMessage(
            content=json.dumps(function_result, ensure_ascii=False),
            tool_call_id=tool_call_id,
        ).model_dump()

    def _build_web_search_followup_messages(
        self,
        messages,
        function_call_data: Dict[str, Any],
        function_result,
        provider: str,
    ) -> List[Dict[str, Any]]:
        """统一构造 web_search 第二次 LLM 调用的消息列表。"""
        original_user_query = ChatUtils.extract_latest_user_content(
            messages,
            MessageTexts.USER_PREVIOUS_QUESTION,
        )
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            original_user_query,
            FunctionNames.WEB_SEARCH,
            function_result,
            provider,
        )
        assistant_message, tool_call_id = self._build_assistant_function_call_message(
            provider,
            function_call_data,
            function_call_data.get("first_llm_thought") or None,
        )
        second_llm_messages.append(assistant_message)
        second_llm_messages.append(
            self._build_tool_result_message(function_result, tool_call_id)
        )
        return second_llm_messages
    
    async def generate_function_call_stream(self, provider, model, messages, conversation_id, options=None, turn_id=None):
        """
        生成支持函数调用的流式响应
        
        Args:
            provider: 模型提供商
            model: 模型名称
            messages: 消息列表
            conversation_id: 会话ID
            options: 可选参数
            turn_id: 对话轮次ID
        """
        if options is None:
            options = {}
        
        # 初始化基础组件
        send_event = ChatUtils.create_event_sender(conversation_id)
        llm = llm_manager.get_model(provider=provider, model=model, options=options)
        context = {"db": self.db, "conversation_id": conversation_id}
        
        # 准备消息和函数定义
        processed_messages = self._prepare_function_call_messages(messages, options)
        functions_kwargs = function_adapter.prepare_functions_for_model(provider, model)
        
        try:
            yield await send_event(EventTypes.FUNCTION_STREAM_START)
            
            # 处理函数调用流程
            async for event in self._execute_function_call_flow(
                provider, model, llm, processed_messages, functions_kwargs, 
                messages, context, send_event, options, turn_id
            ):
                yield event
                
        except Exception as e:
            logger.error(f"{MessageTexts.FUNCTION_CALL_ERROR_PREFIX}{e}")
            logger.exception("函数调用流处理异常详情")
            yield await send_event(EventTypes.ERROR, f"{MessageTexts.PROCESSING_ERROR_PREFIX}{str(e)}")

    def _prepare_function_call_messages(self, messages, options=None):
        """
        准备函数调用的消息列表
        
        Args:
            messages: 原始消息列表
            options: 选项参数（可选）
            
        Returns:
            list: 处理后的消息列表
        """
        if options is None:
            options = {}
            
        # 根据options中的use_reasoning参数选择合适的提示词
        use_reasoning = options.get("use_reasoning", False)
        
        if use_reasoning:
            prompt_content = FUNCTION_CALL_BEHAVIOR_PROMPT_FOR_REASONING
        else:
            prompt_content = FUNCTION_CALL_BEHAVIOR_PROMPT
            
        new_system_prompt_dict = {
            "role": MessageRoles.SYSTEM,
            "content": prompt_content
        }
        
        # 移除原有系统消息，添加新的系统提示
        return [new_system_prompt_dict] + [
            msg for msg in messages if not (isinstance(msg, dict) and msg.get("role") == MessageRoles.SYSTEM)
        ]

    async def _execute_function_call_flow(self, provider, model, llm, processed_messages, functions_kwargs, 
                                         original_messages, context, send_event, options, turn_id=None):
        """
        执行完整的函数调用流程
        
        Args:
            provider: 模型提供商
            model: 模型名称
            llm: 语言模型实例
            processed_messages: 处理后的消息列表
            functions_kwargs: 函数调用参数
            original_messages: 原始消息列表
            context: 上下文信息
            send_event: 事件发送函数
            options: 可选参数
            turn_id: 对话轮次ID
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
            await self._save_stream_response(context["conversation_id"], full_response, options.get("user_id")) 
            yield await send_event(EventTypes.DONE)
            return
        
        # 如果检测到函数调用，不要在这里发送reasoning_complete，因为还有第二段思考过程
        # reasoning_complete将在第二段思考过程结束后发送
        
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
        if function_name == FunctionNames.WEB_SEARCH:
            # 使用专门的处理器
            async for event in self._handle_web_search_function(
                send_event, function_call_data, function_result, 
                context["conversation_id"], llm, original_messages, options, provider, model, turn_id
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
            await self._save_function_call_stream_response(
                conversation_id=context["conversation_id"],
                function_name=function_name,
                function_result=function_result,
                final_response=final_response,
                turn_id=turn_id,
                user_id=options.get("user_id"),
                first_llm_thought=function_call_data.get("first_llm_thought")
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
        # 设置function call上下文标志，避免在第一段思考过程结束后发送reasoning_complete
        reasoning_state.function_call_context = True
        function_call_sent = False  # 标记是否已发送函数调用检测事件
        
        for chunk in llm.stream(processed_messages, **functions_kwargs):
            # 处理推理内容
            reasoning_events = await StreamProcessor.handle_reasoning_content_with_events(
                chunk, send_event, reasoning_state, use_reasoning
            )
            # 如果有推理事件，yield它们
            for event in reasoning_events:
                yield event

            # 检查流中是否有函数调用
            if not function_call_detected:
                function_call_detected, function_call_data = function_adapter.detect_function_call_in_stream(chunk)
                
                if function_call_detected and not function_call_sent:
                    function_call_sent = await self._send_function_detected_event(
                        send_event,
                        function_call_data,
                        function_call_sent,
                    )
            
            # 累积第一次LLM的思考过程
            content_chunk_text = StreamProcessor.extract_content(chunk)
            if content_chunk_text:
                full_response += content_chunk_text
                # 如果已经检测到函数调用，仍然可以继续接收和处理文本内容
                # 某些模型（如Google）可能在工具调用之后还有文本内容
                if not function_call_detected:
                     yield await send_event(EventTypes.CONTENT, content_chunk_text)
        
        # 对于Google模型，即使没有文本内容也可能有有效的函数调用
        # 如果检测到函数调用但没有文本内容，提供一个默认的说明
        full_response = self._finalize_first_pass_response(
            function_call_detected,
            function_call_data,
            full_response,
        )
        
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
        function_name = self._get_function_name(function_call_data)
        function_args = function_call_data["function"].get("arguments", "{}")
        args_dict = ChatUtils.parse_function_arguments(function_args)
            
        # 如果是web_search函数但没有query参数，使用LLM生成搜索查询
        if function_name == FunctionNames.WEB_SEARCH and not args_dict.get("query"):
            yield await send_event(EventTypes.GENERATING_QUERY, MessageTexts.OPTIMIZING_SEARCH_QUERY)
            
            user_message = ChatUtils.extract_latest_user_content(messages)
            
            if user_message:
                search_query = await ChatUtils.generate_search_query(user_message, llm)
                args_dict["query"] = search_query
                self._update_function_arguments(function_call_data, args_dict)
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

        assistant_function_message, _ = self._build_assistant_function_call_message(
            provider,
            function_call_data,
        )
        full_messages.append(assistant_function_message)

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
                                          conversation_id, llm, messages, options, provider, model, turn_id):
        """处理web_search函数的专门处理器"""
        function_name = FunctionNames.WEB_SEARCH
        use_reasoning = options.get("use_reasoning", False)
        
        # 1. 先将 function_result 以 function_result 类型消息返回前端
        yield await send_event(
            EventTypes.FUNCTION_RESULT,
            self._build_function_result_event_data(function_name, function_result),
        )
        
        # --- 开始为第二次LLM调用构建新的消息列表 ---

        # 1. 获取用户原始提问 (Simplified: take the last user/human message)
        second_llm_messages = self._build_web_search_followup_messages(
            messages,
            function_call_data,
            function_result,
            provider,
        )

        # 4. 流式返回 LLM 的最终回复
        final_response = ""
        results = []
        async for result in StreamProcessor.process_llm_stream_with_reasoning(
            llm, second_llm_messages, send_event, use_reasoning
        ):
            results.append(result)
            # 传递所有事件给前端
            yield result
        final_response = self._extract_final_stream_response(results)

        # 5. 保存完整对话历史
        await self._save_function_call_stream_response(
            conversation_id=conversation_id,
            function_name=function_name,
            function_result=function_result,
            final_response=final_response,
            turn_id=turn_id,
            user_id=options.get("user_id"),
            first_llm_thought=function_call_data.get("first_llm_thought")
        )

        # 6. 完成标志
        yield await send_event(EventTypes.DONE)

    async def _save_function_call_stream_response(self, conversation_id, function_name,
                                           function_result, final_response, turn_id=None, user_id=None, first_llm_thought=None):
        """保存函数调用流式响应到对话历史"""
        try:
            conversation = self._get_persistable_conversation(
                conversation_id,
                user_id,
                "保存函数调用响应时缺少 user_id",
            )
            if conversation:
                self._append_and_persist_messages(
                    conversation,
                    self._build_function_call_history_messages(
                        function_name,
                        function_result,
                        final_response,
                        turn_id,
                        first_llm_thought,
                    ),
                )
        except Exception as e:
            logger.error(f"保存函数调用流式响应失败: {e}")
            logger.exception("保存函数调用响应异常详情")

    async def _save_stream_response(self, conversation_id, response_content, user_id=None):
        """保存普通流式响应到对话历史"""
        try:
            conversation = self._get_persistable_conversation(
                conversation_id,
                user_id,
                "保存流式响应时缺少 user_id",
            )
            if conversation:
                self._append_and_persist_messages(
                    conversation,
                    [self._build_assistant_content_message(response_content)],
                )
        except Exception as e:
            logger.error(f"保存流式响应失败: {e}")
            logger.exception("保存普通流式响应异常详情")
