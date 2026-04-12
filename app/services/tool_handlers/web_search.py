"""
WebSearchHandler — 网络搜索工具处理器
从 stream_handler.py 提取，行为保持不变
"""

import time
from typing import List

from app.core.logger import app_logger as logger
from app.schemas.chat import SearchBlock, SearchSource, SearchSourceSummary
from app.services.search_client import search_web
from app.services.tool_handlers.base import BaseToolHandler, ToolResult


class WebSearchHandler(BaseToolHandler):
    @property
    def tool_name(self) -> str:
        return "web_search"

    @property
    def sse_event_prefix(self) -> str:
        return "search"

    async def execute(self, args: dict) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(
                status="degraded",
                error_message="query 为空",
                data={"query": query},
            )

        start = time.monotonic()
        try:
            sources = await search_web(query, count=5)
            duration_ms = int((time.monotonic() - start) * 1000)

            if not sources:
                return ToolResult(
                    status="degraded",
                    duration_ms=duration_ms,
                    error_message="搜索返回空结果",
                    data={"query": query, "sources": [], "result_count": 0},
                )

            return ToolResult(
                status="success",
                duration_ms=duration_ms,
                data={
                    "query": query,
                    "sources": sources,
                    "result_count": len(sources),
                },
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                status="failed",
                duration_ms=duration_ms,
                error_message=str(e),
                data={"query": query, "sources": [], "result_count": 0},
            )

    def build_content_block(self, result: ToolResult, block_id: str, log_id: str) -> SearchBlock:
        sources: List[SearchSource] = result.data.get("sources", [])
        return SearchBlock(
            type="search",
            id=block_id,
            query=result.data.get("query", ""),
            tool_call_log_id=log_id,
            sources=[
                SearchSourceSummary(title=s.title, url=s.url, favicon=s.favicon)
                for s in sources
            ],
        )

    def format_llm_context(self, result: ToolResult) -> str:
        sources: List[SearchSource] = result.data.get("sources", [])
        if not sources:
            return "搜索未返回结果。请基于你的知识回答用户的问题。"

        parts = ["以下是从网络搜索获取的参考信息，请结合这些信息回答用户的问题。"]
        parts.append("如果引用了某条信息，请在相关内容后标注来源编号，格式为 [1]、[2] 等。\n")

        for i, source in enumerate(sources, 1):
            parts.append(f"[{i}] {source.title}")
            parts.append(f"    来源: {source.url}")
            if source.content:
                parts.append(f"    正文: {source.content[:1000]}")
            else:
                parts.append(f"    摘要: {source.description}")
            parts.append("")

        parts.append("注意：")
        parts.append("- 优先使用搜索结果中的信息回答")
        parts.append("- 如果搜索结果不足以回答，可以结合自身知识补充")
        parts.append("- 引用时使用 [n] 格式标注来源编号")
        parts.append("- 直接回答问题，不要再发起搜索或输出任何工具调用指令")

        return "\n".join(parts)
