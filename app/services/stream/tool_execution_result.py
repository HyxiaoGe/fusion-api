"""工具执行结果记录。

本模块只承载单次工具执行的结果形态和轻量格式化行为，不做 Redis、DB、
LLM 调用或事件发送。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.schemas.chat import ContentBlock
    from app.services.tool_handlers.base import BaseToolHandler, ToolResult

TOOL_RESULT_UNAVAILABLE_CONTEXT = "工具未取得可用结果，不能把该工具结果作为依据。"
TOOL_RESULT_REUSED_CONTEXT = (
    "该工具的同名同参查询已在本轮成功执行。请复用上一条成功结果并直接回答，"
    "不要再次调用相同工具，也不要重复生成结果卡片。"
)


@dataclass
class ToolExecutionRecord:
    """单次工具执行记录，避免调用方依赖裸 tuple 位置。"""

    tool_call: dict
    result: ToolResult
    handler: BaseToolHandler | None
    block_id: str
    log_id: str
    reused: bool = False

    @property
    def tool_name(self) -> str:
        """返回 LLM tool_call 中声明的工具名，保持 step 统计语义不变。"""
        return str(self.tool_call.get("name", ""))

    def format_llm_context(self, *, citation_numbers: list[int] | None = None) -> str:
        """格式化注入下一轮 LLM 的工具上下文。"""
        if self.reused:
            return TOOL_RESULT_REUSED_CONTEXT
        if self.handler is None:
            return TOOL_RESULT_UNAVAILABLE_CONTEXT
        if citation_numbers is None or not getattr(self.handler, "supports_run_level_citations", False):
            return self.handler.format_llm_context(self.result)
        return self.handler.format_llm_context(self.result, citation_numbers=citation_numbers)

    def build_content_block(self) -> ContentBlock | None:
        """构造可落库的工具结果 content block。"""
        if self.reused or self.handler is None:
            return None
        return self.handler.build_content_block(self.result, self.block_id, self.log_id)
