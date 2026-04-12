"""
UrlReadHandler — 网页读取工具处理器
"""

import time

from app.schemas.chat import UrlBlock
from app.services.reader_client import read_url
from app.services.tool_handlers.base import BaseToolHandler, ToolResult

# 注入 LLM 上下文时的最大字符数（约 4000 token）
MAX_CONTENT_CHARS = 8000


class UrlReadHandler(BaseToolHandler):
    @property
    def tool_name(self) -> str:
        return "url_read"

    @property
    def sse_event_prefix(self) -> str:
        return "url_read"

    async def execute(self, args: dict) -> ToolResult:
        url = args.get("url", "").strip()
        if not url:
            return ToolResult(
                status="degraded",
                error_message="url 为空",
                data={"url": url},
            )

        start = time.monotonic()
        try:
            result = await read_url(url, timeout=5.0)
            duration_ms = int((time.monotonic() - start) * 1000)

            if result is None:
                return ToolResult(
                    status="failed",
                    duration_ms=duration_ms,
                    error_message="reader-service 返回空结果",
                    data={"url": url},
                )

            return ToolResult(
                status="success",
                duration_ms=duration_ms,
                data={
                    "url": result.url,
                    "title": result.title,
                    "content": result.content,
                    "favicon": result.favicon,
                    "content_length": result.content_length,
                },
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                status="failed",
                duration_ms=duration_ms,
                error_message=str(e),
                data={"url": url},
            )

    def build_content_block(self, result: ToolResult, block_id: str, log_id: str) -> UrlBlock:
        return UrlBlock(
            type="url_read",
            id=block_id,
            url=result.data.get("url", ""),
            title=result.data.get("title"),
            favicon=result.data.get("favicon"),
            tool_call_log_id=log_id,
        )

    def format_llm_context(self, result: ToolResult) -> str:
        url = result.data.get("url", "")
        title = result.data.get("title", "")
        content = result.data.get("content", "")

        if not content:
            return f"无法读取网页内容: {url}。请基于你的知识回答用户的问题。"

        # 截断过长的内容
        truncated = False
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS]
            truncated = True

        parts = [f"以下是网页 {url} 的内容："]
        if title:
            parts.append(f"标题：{title}")
        parts.append("")
        parts.append(content)

        if truncated:
            parts.append("\n（内容已截断，仅展示前部分）")

        parts.append("\n请基于以上网页内容回答用户的问题。")

        return "\n".join(parts)
