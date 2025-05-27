"""
流式处理模块

包含流式响应处理和推理状态管理相关功能
"""

import json
from langchain_core.messages import SystemMessage, ToolMessage
from app.ai.prompts.templates import SYNTHESIZE_TOOL_RESULT_PROMPT
from app.constants import EventTypes


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