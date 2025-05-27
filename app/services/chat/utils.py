"""
聊天服务工具方法模块

包含各种辅助方法和工具函数
"""

import json
import re
from datetime import datetime
from typing import List, Union
from app.constants import MessageRoles, MessageTexts


class ChatUtils:
    """聊天服务工具类"""
    
    @staticmethod
    def create_event_sender(conversation_id: str):
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

    @staticmethod
    def extract_user_message_from_messages(messages: List) -> str:
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

    @staticmethod
    def parse_function_arguments(function_args: Union[str, dict]) -> dict:
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

    @staticmethod
    async def generate_search_query(user_message: str, llm) -> str:
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

    @staticmethod
    def extract_original_user_query(messages: List) -> str:
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

    @staticmethod
    def validate_and_process_function_arguments(function_call_data: dict) -> str:
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

    @staticmethod
    def parse_questions(response_text: str) -> List[str]:
        """
        从响应文本中解析出问题列表
        
        Args:
            response_text: 响应文本
            
        Returns:
            List[str]: 解析出的问题列表
        """
        questions = []
        
        # 尝试不同的解析方法
        # 1. 尝试按数字列表解析
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