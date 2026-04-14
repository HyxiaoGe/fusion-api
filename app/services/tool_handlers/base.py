"""
Tool Handler 抽象基类 — 所有工具处理器的共性接口和共享能力
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from app.schemas.chat import ContentBlock
from app.services.stream_state_service import append_chunk
from app.core.logger import app_logger as logger
from app.services.agent_logger import log_tool_call


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
    def build_content_block(self, result: ToolResult, block_id: str, log_id: str) -> ContentBlock:
        """构造落库用的 content block"""

    @abstractmethod
    def format_llm_context(self, result: ToolResult) -> str:
        """格式化注入第二轮 LLM 的上下文文本"""

    # ---- 共享能力 ----

    async def push_sse_start(self, conversation_id: str, block_id: str, data: dict) -> None:
        """推送 xxx_start SSE 事件"""
        await append_chunk(
            conversation_id,
            f"{self.sse_event_prefix}_start",
            json.dumps(data, ensure_ascii=False),
            block_id,
        )

    async def push_sse_complete(self, conversation_id: str, block_id: str, data: dict) -> None:
        """推送 xxx_complete SSE 事件"""
        await append_chunk(
            conversation_id,
            f"{self.sse_event_prefix}_complete",
            json.dumps(data, ensure_ascii=False),
            block_id,
        )

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
    ) -> None:
        """异步记录 ToolCallLog"""
        task = asyncio.create_task(
            log_tool_call(
                log_id=log_id,
                conversation_id=conversation_id,
                message_id=None,
                user_id=user_id,
                tool_name=self.tool_name,
                status=result.status,
                duration_ms=result.duration_ms,
                model_id=model_id,
                provider=provider,
                input_params=input_params,
                output_data=_serialize_for_json(result.data),
                error_message=result.error_message,
                trace_id=trace_id,
                step_number=step_number,
            )
        )
        task.add_done_callback(_task_done_callback)
