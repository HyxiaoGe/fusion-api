"""
UrlReadHandler — 网页读取工具处理器
"""

import time

from app.core.config import settings
from app.schemas.chat import SourceReference, UrlBlock
from app.services.external.reader_client import read_url_with_diagnostics
from app.services.security.url_policy import evaluate_url_policy
from app.services.source_context import UntrustedSourceContext, format_untrusted_source_context
from app.services.tool_handlers.base import BaseToolHandler, ToolResult

# 注入 LLM 上下文时的最大字符数（约 4000 token）
MAX_CONTENT_CHARS = 8000
MAX_REASON_CHARS = 160


class UrlReadHandler(BaseToolHandler):
    @property
    def tool_name(self) -> str:
        return "url_read"

    @property
    def sse_event_prefix(self) -> str:
        return "url_read"

    def sanitize_input_params_for_log(self, input_params: dict) -> dict:
        safe_params = dict(input_params or {})
        url = safe_params.get("url")
        if isinstance(url, str):
            try:
                policy = evaluate_url_policy(url)
            except Exception:
                safe_params["url"] = ""
                safe_params["url_policy_reason"] = "invalid_url"
                return safe_params
            safe_params["url"] = policy.safe_log_url or ""
            if not policy.allowed:
                safe_params["url_policy_reason"] = policy.reason
        return safe_params

    async def execute(self, args: dict) -> ToolResult:
        url = args.get("url", "").strip()
        reason = _normalize_reason(args.get("reason"))
        if not url:
            return ToolResult(
                status="degraded",
                error_message="url 为空",
                data={"url": url, "reason": reason},
            )
        policy = evaluate_url_policy(url)
        if not policy.allowed:
            return ToolResult(
                status="degraded",
                error_message=policy.user_visible_message or "URL 不允许读取",
                data={
                    "url": policy.safe_log_url or "",
                    "safe_log_url": policy.safe_log_url,
                    "degraded_reason": policy.reason,
                    "reason": reason,
                },
            )

        start = time.monotonic()
        try:
            response = await read_url_with_diagnostics(
                policy.normalized_url or url,
                timeout=settings.READER_SERVICE_TIMEOUT,
            )
            duration_ms = int((time.monotonic() - start) * 1000)
            result = response.result

            if result is None:
                failure = response.failure
                return ToolResult(
                    status="degraded",
                    duration_ms=duration_ms,
                    error_message=failure.message if failure else "reader-service 暂时未返回内容，已跳过网页读取",
                    data={
                        "url": policy.safe_log_url or policy.normalized_url or "",
                        "safe_log_url": policy.safe_log_url,
                        "reason": reason,
                        "failure_kind": failure.kind if failure else "empty_result",
                        "failure_detail": failure.detail if failure else None,
                    },
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
                    "reader_fetch_ms": result.fetch_ms,
                    "reason": reason,
                },
            )
        except Exception as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            return ToolResult(
                status="failed",
                duration_ms=duration_ms,
                error_message=str(e),
                data={"url": url, "reason": reason},
            )

    def build_content_block(self, result: ToolResult, block_id: str, log_id: str) -> UrlBlock:
        url = result.data.get("url", "")
        source_refs = []
        if result.status == "success" and url:
            source_refs.append(
                SourceReference(
                    kind="url_read",
                    title=result.data.get("title") or "",
                    url=url,
                    favicon=result.data.get("favicon"),
                    status=result.status,
                    tool_call_log_id=log_id,
                )
            )
        return UrlBlock(
            type="url_read",
            id=block_id,
            url=url,
            title=result.data.get("title"),
            favicon=result.data.get("favicon"),
            tool_call_log_id=log_id,
            status=result.status,
            error_message=result.error_message,
            source_count=len(source_refs),
            source_refs=source_refs,
            reason=result.data.get("reason"),
        )

    def format_llm_context(self, result: ToolResult) -> str:
        url = result.data.get("url", "")
        title = result.data.get("title", "")
        content = result.data.get("content", "")

        if not content:
            unavailable_message = (
                "网页未读取成功，不能把该网页作为依据；"
                "如需回答，请说明该来源不可用，或仅基于其他可用信息回答。"
            )
            return format_untrusted_source_context(
                UntrustedSourceContext(
                    source_id="U1",
                    source_type="url_read",
                    title=title or "网页未读取成功",
                    url=url,
                    content=unavailable_message,
                    provider="web",
                ),
                max_chars=MAX_CONTENT_CHARS + 100,
            )

        # 截断过长的内容
        truncated = False
        if len(content) > MAX_CONTENT_CHARS:
            content = content[:MAX_CONTENT_CHARS]
            truncated = True

        if truncated:
            content = f"{content}\n（内容已截断，仅展示前部分）"

        return format_untrusted_source_context(
            UntrustedSourceContext(
                source_id="U1",
                source_type="url_read",
                title=title or "未知",
                url=url,
                content=content,
                provider="web",
            ),
            max_chars=MAX_CONTENT_CHARS + 100,
        )

    def _build_result_summary(self, result: ToolResult) -> dict:
        """URL 读取轻量摘要：title + favicon。

        不返回 count 字段：url_read 单次只读 1 个 URL，count 没有 web_search
        那种"命中数"的有意义语义；硬编码 1 会让 FE "找到 N 条"显示矛盾。
        emitter.tool_call_completed 内部还会经 cap_and_truncate(1024) 兜底。
        """
        if result.status != "success":
            return {"kind": "url_read", "truncated": False}
        data = result.data or {}
        return {
            "kind": "url_read",
            "title": data.get("title", ""),
            "favicon": data.get("favicon"),
            "truncated": False,
        }


def _normalize_reason(value) -> str | None:
    if not isinstance(value, str):
        return None
    reason = value.strip()
    if not reason:
        return None
    return reason[:MAX_REASON_CHARS]
