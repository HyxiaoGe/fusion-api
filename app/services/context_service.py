from typing import List, Dict, Any, Optional
from app.services.vector_service import VectorService
from app.core.logger import app_logger as logger


class ContextEnhancer:
    """上下文增强器 - 用于增强对话提示"""

    def __init__(self, db=None):
        self.vector_service = VectorService.get_instance(db)

    def enhance_prompt(self, query: str, conversation_id: Optional[str] = None,
                       current_context: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        增强用户查询的提示

        Args:
            query: 用户查询
            conversation_id: 当前对话ID
            current_context: 当前对话上下文

        Returns:
            包含原始查询和增强提示的字典
        """
        try:
            # 1. 获取相关上下文
            related_context = self.vector_service.get_relevant_context(
                query=query,
                conversation_id=conversation_id
            )

            # 2. 如果没有找到相关上下文，返回原始查询
            if not related_context:
                return {
                    "original_query": query,
                    "enhanced_prompt": query,
                    "has_enhancement": False,
                    "context_used": []
                }

            # 3. 构建增强提示
            context_text = self._format_context(related_context)

            # 优化系统提示词
            enhanced_prompt = f"""我将回答一个问题，但需要考虑以下相关的历史信息：

{context_text}

考虑上述历史信息，但不要明确提及它是来自历史信息，请回答用户的问题: {query}
"""

            # 4. 准备返回结果
            return {
                "original_query": query,
                "enhanced_prompt": enhanced_prompt,
                "has_enhancement": True,
                "context_used": related_context
            }

        except Exception as e:
            logger.error(f"构建增强提示失败: {e}")
            # 出错时返回原始查询
            return {
                "original_query": query,
                "enhanced_prompt": query,
                "has_enhancement": False,
                "context_used": []
            }

    def _format_context(self, context_items: List[Dict[str, Any]]) -> str:
        """格式化上下文信息"""
        formatted_items = []

        for idx, item in enumerate(context_items):
            role_display = "用户" if item["role"] == "user" else "AI助手"
            formatted_items.append(f"信息 {idx + 1}:\n{role_display}: {item['content']}")

        return "\n\n".join(formatted_items)