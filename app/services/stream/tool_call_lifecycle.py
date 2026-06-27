"""tool_call 生命周期事件与执行结果状态转换。

本模块只负责 tool_call_started/tool_call_completed 事件，以及执行异常到
ToolResult 的状态映射；handler 查找、预算、日志和 ToolExecutionRecord 仍由
tool_executor 负责。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from app.services.tool_handlers.base import ToolResult


class ToolCallEmitter(Protocol):
    async def tool_call_started(self, *, tool_call_id: str, tool_name: str, arguments: dict) -> None:
        """发送工具调用开始事件。"""

    async def tool_call_completed(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        status: str,
        duration_ms: int,
        result_summary: dict,
        error: str | None,
    ) -> None:
        """发送工具调用完成事件。"""


ToolExecutorFn = Callable[[Any, dict], Awaitable[ToolResult]]
ResultSummaryBuilder = Callable[[ToolResult], dict]


async def emit_tool_call_started(
    emitter: ToolCallEmitter | None,
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict,
) -> None:
    if emitter is None:
        return
    await emitter.tool_call_started(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
    )


async def emit_tool_call_result(
    emitter: ToolCallEmitter | None,
    *,
    tool_call_id: str,
    tool_name: str,
    result: ToolResult,
    duration_ms: int | None,
    result_summary_builder: ResultSummaryBuilder,
) -> None:
    if emitter is None:
        return
    await emitter.tool_call_completed(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        status=result.status,
        duration_ms=duration_ms if duration_ms is not None else 0,
        result_summary=result_summary_builder(result),
        error=result.error_message if result.status != "success" else None,
    )


async def execute_tool_with_lifecycle(
    *,
    tool_call_id: str,
    tool_name: str,
    args: dict,
    target: Any,
    execute: ToolExecutorFn,
    result_summary_builder: ResultSummaryBuilder,
    emitter: ToolCallEmitter | None,
) -> ToolResult:
    if emitter is None:
        return await execute(target, args)

    await emit_tool_call_started(
        emitter,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=args,
    )
    start_mono = time.monotonic()
    try:
        result = await execute(target, args)
    except asyncio.CancelledError as exc:
        duration_ms = int((time.monotonic() - start_mono) * 1000)
        failed_result = _build_failed_result(exc)
        await emit_tool_call_result(
            emitter,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=failed_result,
            duration_ms=duration_ms,
            result_summary_builder=result_summary_builder,
        )
        raise
    except Exception as exc:
        duration_ms = int((time.monotonic() - start_mono) * 1000)
        failed_result = _build_failed_result(exc)
        await emit_tool_call_result(
            emitter,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            result=failed_result,
            duration_ms=duration_ms,
            result_summary_builder=result_summary_builder,
        )
        return failed_result

    duration_ms = int((time.monotonic() - start_mono) * 1000)
    if result.duration_ms is None:
        result.duration_ms = duration_ms
    await emit_tool_call_result(
        emitter,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=result,
        duration_ms=duration_ms,
        result_summary_builder=result_summary_builder,
    )
    return result


def _build_failed_result(exc: BaseException) -> ToolResult:
    return ToolResult(
        status="failed",
        error_message=f"{type(exc).__name__}: {exc}",
    )
