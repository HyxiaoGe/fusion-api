"""
函数调用处理器模块

负责处理所有函数调用相关的逻辑，包括函数检测、执行和结果处理
"""

import json
import uuid
from typing import Dict, Any, List, Union
from langchain_core.messages import ToolMessage

from app.ai.llm_manager import llm_manager
from app.ai.prompts.templates import FUNCTION_CALL_BEHAVIOR_PROMPT
from app.core.function_manager import function_adapter
from app.core.logger import app_logger as logger
from app.constants import MessageRoles, EventTypes, FunctionNames, MessageTexts, FUNCTION_DESCRIPTIONS
from app.services.chat.stream_processor import ReasoningState, StreamProcessor
from app.services.chat.utils import ChatUtils


class FunctionCallProcessor:
    """函数调用处理器类"""
    
    def __init__(self, db, memory_service):
        self.db = db
        self.memory_service = memory_service
    
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
        send_event = ChatUtils.create_event_sender(conversation_id)
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
            await self._save_stream_response(context["conversation_id"], full_response) 
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
            await self._save_function_call_stream_response(
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
        args_dict = ChatUtils.parse_function_arguments(function_args)
            
        # 如果是web_search函数但没有query参数，使用LLM生成搜索查询
        if function_name == FunctionNames.WEB_SEARCH and not args_dict.get("query"):
            yield await send_event(EventTypes.GENERATING_QUERY, MessageTexts.OPTIMIZING_SEARCH_QUERY)
            
            user_message = ChatUtils.extract_user_message_from_messages(messages)
            
            if user_message:
                search_query = await ChatUtils.generate_search_query(user_message, llm)
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
        original_user_query = ChatUtils.extract_original_user_query(messages)

        # 2. 构建第二次LLM调用的消息列表
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            original_user_query, function_name, function_result
        )

        original_tool_call_id = function_call_data.get("tool_call_id", f"call_{uuid.uuid4()}")
        valid_arguments_str = ChatUtils.validate_and_process_function_arguments(function_call_data)
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
        await self._save_function_call_stream_response(
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
        original_user_query = ChatUtils.extract_original_user_query(messages)

        # 2. 构建第二次LLM调用的消息列表
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            original_user_query, function_name, function_result
        )

        original_tool_call_id = function_call_data.get("tool_call_id", f"call_{uuid.uuid4()}")
        valid_arguments_str = ChatUtils.validate_and_process_function_arguments(function_call_data)
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
        await self._save_function_call_stream_response(
            conversation_id=conversation_id,
            function_name=function_name,
            function_args=valid_arguments_str,
            function_result=function_result,
            final_response=final_response
        )

        # 6. 完成标志
        yield await send_event(EventTypes.DONE)

    async def _save_function_call_stream_response(self, conversation_id, function_name, 
                                           function_args, function_result, final_response):
        """保存函数调用流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                from app.schemas.chat import Message
                
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
                from datetime import datetime
                conversation.updated_at = datetime.now()
                
                # 保存到数据库
                self.memory_service.save_conversation(conversation)
        except Exception as e:
            logger.error(f"保存函数调用流式响应失败: {e}")

    async def _save_stream_response(self, conversation_id, response_content):
        """保存普通流式响应到对话历史"""
        try:
            conversation = self.memory_service.get_conversation(conversation_id)
            if conversation:
                from app.schemas.chat import Message
                from datetime import datetime
                
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