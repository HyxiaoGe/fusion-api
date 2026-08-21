"""tool_call 生命周期事件与执行结果状态转换。

本模块只负责 tool_call_started/tool_call_completed 事件，以及执行异常到
ToolResult 的状态映射；handler 查找、预算、日志和 ToolExecutionRecord 仍由
tool_executor 负责。
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

from app.services.tool_handlers.base import ToolResult


class ToolCallEmitter(Protocol):
    async def tool_call_started(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        arguments: dict,
        plan_item_id: str | None = None,
    ) -> None:
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
        plan_item_id: str | None = None,
    ) -> None:
        """发送工具调用完成事件。"""


ToolExecutorFn = Callable[[Any, dict], Awaitable[ToolResult]]
ResultSummaryBuilder = Callable[[ToolResult], dict]


class ToolLifecycleControlPlaneError(RuntimeError):
    """让嵌套 attempt 的 emitter 故障穿透 handler 异常降级层。"""

    def __init__(self, error: Exception):
        self.error = error
        super().__init__(type(error).__name__)


@dataclass(frozen=True)
class ToolLifecycleAttempt:
    result: ToolResult
    duration_ms: int
    cancelled_error: asyncio.CancelledError | None = None
    from_exception: bool = False


async def emit_tool_call_started(
    emitter: ToolCallEmitter | None,
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: dict,
    plan_item_id: str | None = None,
) -> None:
    if emitter is None:
        return
    kwargs = dict(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=arguments,
    )
    if plan_item_id is not None:
        kwargs["plan_item_id"] = plan_item_id
    await emitter.tool_call_started(**kwargs)


async def emit_tool_call_result(
    emitter: ToolCallEmitter | None,
    *,
    tool_call_id: str,
    tool_name: str,
    result: ToolResult,
    duration_ms: int | None,
    result_summary_builder: ResultSummaryBuilder,
    plan_item_id: str | None = None,
) -> None:
    if emitter is None:
        return
    data = getattr(result, "data", None)
    repair = data.get("repair") if isinstance(data, dict) else None
    event_status = "degraded" if isinstance(repair, dict) and repair.get("retryable") is True else result.status
    result_summary = _build_event_result_summary(
        result_summary_builder(result),
        data=data,
        repair=repair,
    )
    kwargs = dict(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        status=event_status,
        duration_ms=duration_ms if duration_ms is not None else 0,
        result_summary=result_summary,
        error=(
            None
            if isinstance(repair, dict) and repair.get("retryable") is True
            else result.error_message
            if result.status != "success"
            else None
        ),
    )
    if plan_item_id is not None:
        kwargs["plan_item_id"] = plan_item_id
    await emitter.tool_call_completed(**kwargs)


def _build_event_result_summary(
    raw_summary: Any,
    *,
    data: dict[str, Any] | None,
    repair: Any,
) -> dict[str, Any]:
    summary = dict(raw_summary) if isinstance(raw_summary, dict) else {"kind": "tool", "truncated": True}
    if isinstance(repair, dict):
        repair_id = _safe_repair_id(repair.get("repair_id"))
        summary["repair_state"] = (
            "retrying"
            if repair.get("retryable") is True
            else "requires_user_input"
            if repair.get("requires_user_input") is True
            else "exhausted"
        )
        if repair_id:
            summary["repair_id"] = repair_id
    elif isinstance(data, dict):
        resolves_repair_id = _safe_repair_id(data.get("resolves_repair_id"))
        if resolves_repair_id:
            summary["repair_state"] = "resolved"
            summary["resolves_repair_id"] = resolves_repair_id
    return summary


def _safe_repair_id(value: Any) -> str | None:
    return value if isinstance(value, str) and re.fullmatch(r"repair_[a-f0-9]{16}", value) else None


def measure_duration_ms(start_mono: float) -> int:
    return int((time.monotonic() - start_mono) * 1000)


async def run_tool_attempt(*, target: Any, args: dict, execute: ToolExecutorFn) -> ToolLifecycleAttempt:
    start_mono = time.monotonic()
    try:
        result = await execute(target, args)
    except ToolLifecycleControlPlaneError as exc:
        raise exc.error from None
    except asyncio.CancelledError as exc:
        return ToolLifecycleAttempt(
            result=_build_failed_result(exc),
            duration_ms=measure_duration_ms(start_mono),
            cancelled_error=exc,
            from_exception=True,
        )
    except Exception as exc:
        return ToolLifecycleAttempt(
            result=_build_failed_result(exc),
            duration_ms=measure_duration_ms(start_mono),
            from_exception=True,
        )
    return ToolLifecycleAttempt(result=result, duration_ms=measure_duration_ms(start_mono))


async def complete_tool_lifecycle(
    *,
    emitter: ToolCallEmitter,
    tool_call_id: str,
    tool_name: str,
    result: ToolResult,
    duration_ms: int,
    result_summary_builder: ResultSummaryBuilder,
    set_result_duration: bool = True,
    plan_item_id: str | None = None,
) -> None:
    if set_result_duration and result.duration_ms is None:
        result.duration_ms = duration_ms
    await emit_tool_call_result(
        emitter,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=result,
        duration_ms=duration_ms,
        result_summary_builder=result_summary_builder,
        plan_item_id=plan_item_id,
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
    plan_item_id: str | None = None,
) -> ToolResult:
    if emitter is None:
        return await execute(target, args)

    await emit_tool_call_started(
        emitter,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        arguments=args,
        plan_item_id=plan_item_id,
    )
    attempt = await run_tool_attempt(target=target, args=args, execute=execute)
    await complete_tool_lifecycle(
        emitter=emitter,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        result=attempt.result,
        duration_ms=attempt.duration_ms,
        result_summary_builder=result_summary_builder,
        plan_item_id=plan_item_id,
        set_result_duration=not attempt.from_exception,
    )
    if attempt.cancelled_error:
        raise attempt.cancelled_error
    return attempt.result


def _build_failed_result(exc: BaseException) -> ToolResult:
    error_code = "tool_cancelled" if isinstance(exc, asyncio.CancelledError) else "tool_execution_failed"
    return ToolResult(
        status="failed",
        data={"error_code": error_code, "retryable": False},
        error_message="工具执行未完成",
    )
