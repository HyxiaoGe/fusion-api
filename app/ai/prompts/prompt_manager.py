"""
提示词管理器
负责提供各种提示词模板，并支持模板变量替换
"""
from typing import Dict, Any

from app.ai.prompts.templates import (
    GENERATE_TITLE_PROMPT,
    GENERATE_SUGGESTED_QUESTIONS_PROMPT,
    FILE_ANALYSIS_PROMPT,
    FILE_CONTENT_ENHANCEMENT_PROMPT,
    HOT_TOPIC_ANALYSIS_PROMPT,
    WEB_SEARCH_RESULTS_PROMPT
)


class PromptManager:
    """提示词管理器"""
    
    def __init__(self):
        # 内置提示词模板映射
        self._templates = {
            "generate_title": GENERATE_TITLE_PROMPT,
            "generate_suggested_questions": GENERATE_SUGGESTED_QUESTIONS_PROMPT,
            "file_analysis": FILE_ANALYSIS_PROMPT,
            "file_content_enhancement": FILE_CONTENT_ENHANCEMENT_PROMPT,
            "hot_topic_analysis": HOT_TOPIC_ANALYSIS_PROMPT,
            "web_search_results": WEB_SEARCH_RESULTS_PROMPT
        }
    
    def get_template(self, template_name: str) -> str:
        """获取指定名称的提示词模板"""
        if template_name not in self._templates:
            raise ValueError(f"未找到提示词模板: {template_name}")
        return self._templates[template_name]
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """使用提供的参数格式化提示词模板"""
        template = self.get_template(template_name)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"格式化提示词模板时缺少参数: {e}")
    
    def add_template(self, name: str, template: str) -> None:
        """添加或更新提示词模板"""
        self._templates[name] = template


# 创建全局提示词管理器实例
prompt_manager = PromptManager() 