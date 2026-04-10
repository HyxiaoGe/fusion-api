"""
聊天服务工具方法模块

包含各种辅助方法和工具函数
"""

import json
import re
from typing import Any, List, Union


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
    def stringify_function_arguments(function_args: Union[str, dict]) -> str:
        """将函数参数标准化为稳定的 JSON 字符串。"""
        arguments = ChatUtils.parse_function_arguments(function_args)
        return json.dumps(arguments, ensure_ascii=False) if arguments else "{}"

    @staticmethod
    def _strip_question_prefix(line: str) -> str:
        """去掉问题列表行首的编号前缀。"""
        return re.sub(r"^\d+[\.\)]\s*", "", line).strip()

    @staticmethod
    def _split_non_empty_lines(text: str) -> List[str]:
        """按行切分并去掉空行。"""
        return [line.strip() for line in text.split("\n") if line.strip()]

    @staticmethod
    def _extract_numbered_questions(response_text: str) -> List[str]:
        """提取按编号组织的问题列表。"""
        numbered_questions = re.findall(
            r"\d+[\.\)]\s*(.*?)(?=\n\d+[\.\)]|\n*$)",
            response_text,
            re.DOTALL,
        )
        return [q.strip() for q in numbered_questions]

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

        # 1. 尝试按数字列表解析
        numbered_questions = ChatUtils._extract_numbered_questions(response_text)
        if numbered_questions and len(numbered_questions) >= 3:
            return numbered_questions

        # 2. 按行分割
        lines = ChatUtils._split_non_empty_lines(response_text)
        for line in lines:
            cleaned_line = ChatUtils._strip_question_prefix(line)
            if cleaned_line:
                questions.append(cleaned_line)

        # 如果没有找到足够的问题，返回原始文本分成的前三行
        if len(questions) < 3:
            questions = lines[:3] if len(lines) >= 3 else lines

        return questions
