"""
聊天服务工具方法模块

包含各种辅助方法和工具函数
"""

import json
import re
from datetime import datetime
from typing import Any, List, Union

from app.constants import MessageRoles, MessageTexts


class ChatUtils:
    """聊天服务工具类"""

    @staticmethod
    def get_response_text(response: Any) -> str:
        """统一提取模型返回的正文文本。"""
        return response.content if hasattr(response, "content") else str(response)

    @staticmethod
    def clean_model_text(text: str) -> str:
        """清理模型文本输出的外围空白和引号。"""
        return text.strip().strip("\"'")
    
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
    def _extract_user_content(message: Any) -> str:
        """从单条消息对象中提取用户文本。"""
        if hasattr(message, "type") and message.type == "human":
            return getattr(message, "content", "") or ""
        if hasattr(message, "role") and message.role == MessageRoles.USER:
            return getattr(message, "content", "") or ""
        if isinstance(message, dict) and message.get("role") == MessageRoles.USER:
            return message.get("content", "") or ""
        return ""

    @staticmethod
    def extract_latest_user_content(messages: List, default: str = "") -> str:
        """从消息列表中提取最后一条用户消息内容。"""
        for message in reversed(messages):
            content = ChatUtils._extract_user_content(message)
            if content:
                return content
        return default

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
        except json.JSONDecodeError:
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
        return ChatUtils.clean_model_text(ChatUtils.get_response_text(search_query_response))

    @staticmethod
    def stringify_function_arguments(function_args: Union[str, dict]) -> str:
        """将函数参数标准化为稳定的 JSON 字符串。"""
        arguments = ChatUtils.parse_function_arguments(function_args)
        return json.dumps(arguments, ensure_ascii=False) if arguments else "{}"

    @staticmethod
    def _strip_question_prefix(line: str) -> str:
        """去掉问题列表行首的编号前缀。"""
        return re.sub(r'^\d+[\.\)]\s*', '', line).strip()

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
            cleaned_line = ChatUtils._strip_question_prefix(line)
            if cleaned_line:
                questions.append(cleaned_line)
        
        # 如果没有找到足够的问题，返回原始文本分成的前三行
        if len(questions) < 3:
            questions = lines[:3] if len(lines) >= 3 else lines
        
        return questions 
