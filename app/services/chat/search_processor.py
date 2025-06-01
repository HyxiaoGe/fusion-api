"""
搜索处理器模块

专门处理web搜索和用户优先搜索相关的逻辑
"""

import json
import uuid
from typing import Any

from app.ai.llm_manager import llm_manager
from app.core.function_manager import function_adapter
from app.core.logger import app_logger as logger
from app.constants import EventTypes, FunctionNames, MessageTexts, MessageRoles, MessageTypes
from app.services.chat.stream_processor import StreamProcessor
from app.services.chat.utils import ChatUtils


class SearchProcessor:
    """搜索处理器类"""
    
    def __init__(self, db, memory_service):
        self.db = db
        self.memory_service = memory_service
    
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
        send_event = ChatUtils.create_event_sender(conversation_id)
        
        # 验证用户查询
        original_user_query_content = ChatUtils.extract_user_message_from_messages(messages)
        if not original_user_query_content:
            yield await send_event(EventTypes.ERROR, MessageTexts.NO_USER_QUERY_ERROR)
            yield await send_event(EventTypes.DONE)
            return

        try:
            # 执行用户优先搜索流程
            async for event in self._execute_user_search_flow(
                provider, model, llm, original_user_query_content, 
                conversation_id, send_event, options
            ):
                yield event
                
        except Exception as e:
            logger.error(f"{MessageTexts.USER_PRIORITIZED_SEARCH_ERROR_PREFIX}{e}")
            import traceback
            logger.error(traceback.format_exc())
            yield await send_event(EventTypes.ERROR, f"{MessageTexts.USER_PRIORITIZED_SEARCH_ERROR_PREFIX}{str(e)}")
            yield await send_event(EventTypes.DONE)

    async def _execute_user_search_flow(self, provider, model, llm, user_query, conversation_id, send_event, options):
        """
        执行用户优先搜索的完整流程
        
        Args:
            provider: 模型提供商
            model: 模型名称
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
        generated_search_query = await ChatUtils.generate_search_query(user_query, llm)
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
        
        async for result in self._synthesize_search_answer(llm, user_query, search_result_data, send_event, use_reasoning, provider, model):
            # 传递所有事件给前端
            yield result
            # 如果是最终的完整响应（不是事件字符串），保存它
            if isinstance(result, str) and not result.startswith("data: "):
                final_response_content = result

        # 保存搜索结果
        await self._save_user_prioritized_web_search_stream_response(
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

    async def _synthesize_search_answer(self, llm, user_query, search_result_data, send_event, use_reasoning, provider=None, model=None):
        """
        合成搜索结果的答案
        
        Args:
            llm: 语言模型实例
            user_query: 用户原始查询
            search_result_data: 搜索结果数据
            send_event: 事件发送函数
            use_reasoning: 是否使用推理模式
            provider: 模型提供商（可选）
            model: 模型名称（可选）
            
        Returns:
            str: 最终响应内容
        """
        second_llm_messages = StreamProcessor.create_tool_synthesis_messages(
            user_query, FunctionNames.WEB_SEARCH, search_result_data, provider, model, use_reasoning
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

    async def _save_user_prioritized_web_search_stream_response(self, conversation_id: str, 
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

            from app.schemas.chat import Message
            from datetime import datetime

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
                type=MessageTypes.WEB_SEARCH,
                content="",
                tool_calls=[assistant_tool_calling_dict]
            )

            db_tool_response_message = Message(
                role=MessageRoles.SYSTEM, 
                type=MessageTypes.FUNCTION_RESULT,
                tool_call_id=user_prioritized_tool_call_id,
                content=json.dumps(search_result, ensure_ascii=False)
            )

            db_final_ai_message = Message(
                role=MessageRoles.ASSISTANT,
                type=MessageTypes.ASSISTANT_CONTENT,
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