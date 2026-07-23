"""
Tool Handler 抽象基类 — 所有工具处理器的共性接口和共享能力
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from app.core.logger import app_logger as logger
from app.schemas.chat import ContentBlock
from app.services.agent_logger import log_tool_call

if TYPE_CHECKING:
    from app.services.agent.emitter import AgentEventEmitter


@dataclass
class ToolContext:
    """传递给 handler 的上下文，包含处理 tool_call 所需的全部信息"""

    db: Any
    conversation_id: str
    user_id: str
    model_id: str
    litellm_model: str
    litellm_kwargs: dict
    provider: str
    messages: list
    assistant_message_id: str
    task_id: str
    tool_call_id: str
    tool_call_args: str
    should_use_reasoning: bool
    thinking_block_id: str
    text_block_id: str
    first_round_reasoning: str = ""


@dataclass
class ToolResult:
    """工具执行结果"""

    status: str  # "success" | "failed" | "degraded"
    data: dict = field(default_factory=dict)
    error_message: Optional[str] = None
    duration_ms: Optional[int] = None


def _serialize_for_json(obj):
    """递归序列化，确保所有 Pydantic 对象转为 dict"""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, list):
        return [_serialize_for_json(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    return obj


def _task_done_callback(task: asyncio.Task):
    """asyncio.create_task 异常回调，确保日志写入失败可见"""
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error(f"日志写入异步任务异常: {exc}", exc_info=exc)


class BaseToolHandler(ABC):
    """所有 tool handler 的抽象基类"""

    supports_run_level_citations = False
    supports_automatic_retry = True

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """工具名称，用于日志和注册"""

    @property
    @abstractmethod
    def sse_event_prefix(self) -> str:
        """SSE 事件前缀，如 'search' → search_start/search_complete"""

    @abstractmethod
    async def execute(self, args: dict) -> ToolResult:
        """执行工具核心逻辑"""

    @abstractmethod
    def build_content_block(self, result: ToolResult, block_id: str, log_id: str) -> ContentBlock | None:
        """构造落库用的 content block"""

    @abstractmethod
    def format_llm_context(
        self,
        result: ToolResult,
        *,
        citation_numbers: list[int] | None = None,
    ) -> str:
        """格式化注入第二轮 LLM 的上下文文本"""

    # ---- 共享能力 ----

    async def log(
        self,
        log_id: str,
        conversation_id: str,
        user_id: str,
        model_id: str,
        provider: str,
        result: ToolResult,
        input_params: dict,
        trace_id: str = None,
        step_number: int = None,
        message_id: str | None = None,
    ) -> None:
        """异步记录 ToolCallLog"""
        safe_input_params = self.sanitize_input_params_for_log(input_params)
        safe_output_data = self.sanitize_output_data_for_log(result)
        task = asyncio.create_task(
            log_tool_call(
                log_id=log_id,
                conversation_id=conversation_id,
                message_id=message_id,
                user_id=user_id,
                tool_name=self.tool_name,
                status=result.status,
                duration_ms=result.duration_ms,
                model_id=model_id,
                provider=provider,
                input_params=safe_input_params,
                output_data=_serialize_for_json(safe_output_data),
                error_message=result.error_message,
                trace_id=trace_id,
                step_number=step_number,
            )
        )
        task.add_done_callback(_task_done_callback)

    def sanitize_input_params_for_log(self, input_params: dict) -> dict:
        """子类可覆盖以清理即将持久化的工具入参。"""
        return input_params

    def sanitize_input_params_for_event(self, input_params: dict) -> dict:
        """子类可覆盖以清理进度事件里的工具入参。"""
        return input_params

    def sanitize_output_data_for_log(self, result: ToolResult) -> dict:
        """子类可覆盖以限制即将持久化的工具输出。"""
        return result.data

    def build_successful_call_signature(self, input_params: dict) -> str | None:
        """为可安全复用的只读调用生成运行内签名。

        默认禁用，避免未来有副作用的工具被错误去重。只有能保证“同名 + 归一化参数”
        成功结果可在同一 Agent run 内复用的 handler 才应覆盖。
        """

        return None

    async def execute_with_emitter(
        self,
        *,
        args: dict,
        emitter: "AgentEventEmitter",
        tool_call_id: str,
    ) -> "ToolResult":
        """统一包装：发 tool_call_started → execute → tool_call_completed。

        强保证：tool_call_completed 必发，即使 execute 抛异常（包括 CancelledError）
        也会先 emit failed 事件再 re-raise，避免 tool_call 永远卡在 running。

        所有失败路径（execute 返回 failed/degraded、execute 抛异常）的 result_summary
        都走子类 _build_result_summary，保证 kind 字段在 FE 看起来一致（不会出现
        同一工具的两条失败 chunk 一个 kind="search" 另一个 kind="web_search"）。

        包含计时和 result_summary 自动构造。子类只需实现 execute(args) +
        可选覆盖 _build_result_summary 返回轻量 summary。
        本方法不再推送旧式工具实时 chunk；统一由 AgentEventEmitter 输出事件。
        """
        await emitter.tool_call_started(
            tool_call_id=tool_call_id,
            tool_name=self.tool_name,
            arguments=self.sanitize_input_params_for_event(args),
        )
        start = time.monotonic()
        try:
            result = await self.execute(args)
        except BaseException as exc:  # noqa: BLE001 — 必须在 re-raise 前发 completed
            duration_ms = int((time.monotonic() - start) * 1000)
            # 用合成 failed result 走子类 _build_result_summary，保证 kind 与失败路径一致
            synthetic_failed = ToolResult(
                status="failed",
                data={},
                error_message=f"{type(exc).__name__}: {exc}",
            )
            await emitter.tool_call_completed(
                tool_call_id=tool_call_id,
                tool_name=self.tool_name,
                status="failed",
                duration_ms=duration_ms,
                result_summary=self._build_result_summary(synthetic_failed),
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        duration_ms = int((time.monotonic() - start) * 1000)
        await emitter.tool_call_completed(
            tool_call_id=tool_call_id,
            tool_name=self.tool_name,
            status=result.status,
            duration_ms=duration_ms,
            result_summary=self._build_result_summary(result),
            error=result.error_message if result.status != "success" else None,
        )
        return result

    def _build_result_summary(self, result: "ToolResult") -> dict:
        """子类可覆盖返回轻量摘要（如搜索命中数 / favicon）。

        默认返回最小 {kind, truncated}。

        注意：返回值会被 emitter.tool_call_completed 内部的
        cap_and_truncate(max_bytes=1024) 兜底截断（含递归嵌套），
        子类无需自己截断字符串字段。
        """
        return {"kind": self.tool_name, "truncated": False}
