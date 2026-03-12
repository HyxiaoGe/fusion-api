"""
流式处理模块

包含流式响应处理和推理状态管理相关功能
"""

import json
from langchain_core.messages import SystemMessage
from app.ai.prompts.templates import SYNTHESIZE_TOOL_RESULT_PROMPT, SYNTHESIZE_TOOL_RESULT_PROMPT_FOR_REASONING
from app.constants import EventTypes


class ReasoningState:
    """推理状态管理类"""
    
    def __init__(self):
        self.reasoning_start_sent = False
        self.reasoning_complete_sent = False
        self.last_reasoning_chunk = False
        self.function_call_context = False  # 标记是否在function call流程中


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
            
        # 检查是否有推理内容
        has_reasoning_content = (hasattr(chunk, 'additional_kwargs') and 
                               'reasoning_content' in chunk.additional_kwargs)
        
        if has_reasoning_content:
            reasoning_content = chunk.additional_kwargs['reasoning_content']
            if reasoning_content and reasoning_content.strip():
                if not reasoning_state.reasoning_start_sent:
                    events.append(await send_event(EventTypes.REASONING_START))
                    reasoning_state.reasoning_start_sent = True
                    
                events.append(await send_event(EventTypes.REASONING_CONTENT, reasoning_content))
                # 重置推理结束检测标志
                reasoning_state.last_reasoning_chunk = True
                return events
        
        # 如果当前chunk没有推理内容，但之前有推理内容，且还没发送完成事件
        # 这表明推理阶段可能已经结束
        # 但是如果在function call流程中，不要在第一段思考过程结束后就发送reasoning_complete
        if (reasoning_state.reasoning_start_sent and 
            not reasoning_state.reasoning_complete_sent and
            not has_reasoning_content and
            getattr(reasoning_state, 'last_reasoning_chunk', False) and
            not reasoning_state.function_call_context):  # 只有在非function call流程中才发送
            
            events.append(await send_event(EventTypes.REASONING_COMPLETE))
            reasoning_state.reasoning_complete_sent = True
            reasoning_state.last_reasoning_chunk = False
            
        return events

    @staticmethod
    def create_tool_synthesis_messages(original_user_query, tool_name, tool_result, provider=None, model=None, use_reasoning=False):
        """
        创建工具结果合成的消息列表
        
        Args:
            original_user_query: 用户原始查询
            tool_name: 工具名称
            tool_result: 工具执行结果
            provider: 模型提供商（可选）
            model: 模型名称（可选）
            use_reasoning: 是否使用推理模式
            
        Returns:
            list: 用于合成的消息列表
        """
        # 为Google模型创建正确的消息序列
        if provider == "google":
            # Google需要：System -> User 的序列来开始对话
            system_prompt_content = SYNTHESIZE_TOOL_RESULT_PROMPT.format(
                original_user_query=original_user_query,
                tool_name=tool_name,
                tool_results_json=json.dumps(tool_result, ensure_ascii=False)
            )
            return [
                SystemMessage(content=system_prompt_content),
                {"role": "user", "content": original_user_query}
            ]
        else:
            # 其他模型保持原有逻辑
            system_prompt_content = SYNTHESIZE_TOOL_RESULT_PROMPT.format(
                original_user_query=original_user_query,
                tool_name=tool_name,
                tool_results_json=json.dumps(tool_result, ensure_ascii=False)
            )
            return [SystemMessage(content=system_prompt_content)]

    @staticmethod
    async def process_llm_stream_with_reasoning(llm, messages, send_event, use_reasoning=True, is_function_call_second_stage=False):
        """
        处理LLM流式响应，包含推理处理
        
        Args:
            llm: 语言模型实例
            messages: 消息列表
            send_event: 事件发送函数
            use_reasoning: 是否启用推理模式
            is_function_call_second_stage: 是否是function call的第二阶段
            
        Yields:
            str: 流式事件或最终响应内容
        """
        reasoning_state = ReasoningState()
        # 正常情况下不设置function_call_context，允许正常的推理流程
        final_response = ""

        for chunk in llm.stream(messages):
            # 处理推理内容
            reasoning_events = await StreamProcessor.handle_reasoning_content_with_events(
                chunk, send_event, reasoning_state, use_reasoning
            )
            # 如果有推理事件，yield它们
            for event in reasoning_events:
                yield event
            
            # 处理内容
            content_chunk_text = StreamProcessor.extract_content(chunk)
            if content_chunk_text is not None:
                yield await send_event(EventTypes.CONTENT, content_chunk_text)
                final_response += content_chunk_text

        # 确保推理完成事件被发送
        if reasoning_state.reasoning_start_sent and not reasoning_state.reasoning_complete_sent:
            yield await send_event(EventTypes.REASONING_COMPLETE)
        
        yield final_response 
