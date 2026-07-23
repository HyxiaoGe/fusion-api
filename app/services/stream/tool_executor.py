"""并行工具执行 + emitter→Redis 适配器。

spec §4.3。execute_tools_parallel 走 emitter 的
tool_call_started/completed 协议，并把 execute_tool_with_retry
（瞬时重试 + 30s timeout）夹在 start/completed 之间。
"""

import asyncio
import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import backoff

from app.core.logger import app_logger as logger
from app.services.agent.emitter import AgentEventEmitter
from app.services.agent.progress_digest import build_evidence_items, build_tool_result_digest
from app.services.stream.tool_call_lifecycle import (
    emit_tool_call_result,
    emit_tool_call_started,
    execute_tool_with_lifecycle,
)
from app.services.stream.tool_execution_result import ToolExecutionRecord
from app.services.stream_state_service import StreamWriteTerminalError, append_chunk

if TYPE_CHECKING:
    from app.services.stream.network_budget import NetworkToolBudget

# 单次工具调用超时
AGENT_TOOL_TIMEOUT = 30
# 瞬时故障重试次数
AGENT_TOOL_MAX_RETRIES = 1

# 永久性错误关键字（不重试）
_TOOL_PERMANENT_KEYWORDS = ("not_found", "invalid", "rate_limit", "400", "401", "403", "404")
_INTERNAL_TOOL_ARG_KEYS = {"budget_decision"}
_AMAP_PRODUCT_EXECUTION_PRIORITY = {
    "route_compare": 0,
    "local_place_search": 1,
}


@dataclass(frozen=True)
class ToolExecutionBatchRequest:
    conversation_id: str
    user_id: str
    model_id: str
    provider: str
    trace_id: str | None = None
    step_number: int | None = None
    message_id: str | None = None
    emitter: Optional[AgentEventEmitter] = None
    network_budget: "NetworkToolBudget | None" = None
    tool_handlers: Mapping[str, Any] | None = None
    runtime_context: Any = None
    successful_tool_call_signatures: set[str] | None = None


@dataclass(frozen=True)
class ToolExecutionIds:
    block_id: str
    log_id: str


def _should_retry_tool_result(result) -> bool:
    """决定 ToolResult 是否应该再试一次（True = 重试，False = 接受当前结果）。"""
    if result.status in ("success", "degraded"):
        return False
    err = (result.error_message or "").lower()
    is_permanent = any(kw in err for kw in _TOOL_PERMANENT_KEYWORDS)
    return not is_permanent


class AgentEventRedisWriter:
    """把 emitter 的 (conv_id, chunk_type, payload:dict) 调用转成
    stream_state_service.append_chunk(conv_id, chunk_type, content, block_id, task_id=task_id)。

    Task 9 引入的 adapter — emitter 不直接知道 stream_state_service 的接口形态，
    通过本 adapter 桥接：payload JSON 序列化进 content 字段，block_id 留空。
    """

    async def append_chunk(self, conversation_id: str, task_id: str, chunk_type: str, payload: dict) -> None:
        await append_chunk(
            conversation_id,
            chunk_type,
            json.dumps(payload, ensure_ascii=False),
            "",  # block_id 不适用于 agent_event chunk
            task_id=task_id,
        )


class AgentEventCompositeWriter:
    """agent_event 双写 adapter：先写 Redis，再旁路记录 progress snapshot。"""

    def __init__(self, *, redis_writer: AgentEventRedisWriter, recorder=None) -> None:
        self.redis_writer = redis_writer
        self.recorder = recorder

    async def append_chunk(self, conversation_id: str, task_id: str, chunk_type: str, payload: dict) -> None:
        await self.redis_writer.append_chunk(conversation_id, task_id, chunk_type, payload)
        if self.recorder is not None:
            self.recorder.record_chunk(conversation_id, chunk_type, payload)


async def _execute_handler(handler, args: dict, runtime_context: Any = None):
    execute_with_runtime_context = getattr(handler, "execute_with_runtime_context", None)
    if runtime_context is not None and callable(execute_with_runtime_context):
        return await execute_with_runtime_context(args, runtime_context)
    return await handler.execute(args)


@backoff.on_predicate(
    backoff.constant,
    predicate=_should_retry_tool_result,
    max_tries=AGENT_TOOL_MAX_RETRIES + 1,
    interval=1,
    on_backoff=lambda d: logger.warning(
        f"工具 {d['args'][0].tool_name} 执行失败（第 {d['tries']} 次），"
        f"{d['wait']:.0f}s 后重试: {d['value'].error_message}"
    ),
)
async def execute_tool_with_retry(handler, args: dict, runtime_context: Any = None):
    """带重试的工具执行（仅瞬时故障重试），返回 ToolResult。

    永久性错误（not_found / invalid / 401 / 403 / 404 / 400 / rate_limit）不重试。
    超时：单次 AGENT_TOOL_TIMEOUT 秒，超时被视为可重试失败。
    重试逻辑由 @backoff.on_predicate 装饰器实现。
    """
    from app.services.tool_handlers import ToolResult

    try:
        return await asyncio.wait_for(
            _execute_handler(handler, args, runtime_context),
            timeout=AGENT_TOOL_TIMEOUT,
        )
    except asyncio.TimeoutError:
        return ToolResult(status="failed", error_message="工具调用超时")


async def execute_tool_once(handler, args: dict, runtime_context: Any = None):
    """有副作用或不具备幂等保证的工具只执行一次，但仍保留统一超时。"""
    from app.services.tool_handlers import ToolResult

    try:
        return await asyncio.wait_for(_execute_handler(handler, args, runtime_context), timeout=AGENT_TOOL_TIMEOUT)
    except asyncio.TimeoutError:
        return ToolResult(status="failed", error_message="工具调用超时")


def new_tool_execution_ids() -> ToolExecutionIds:
    return ToolExecutionIds(
        block_id=f"blk_{uuid.uuid4().hex[:12]}",
        log_id=str(uuid.uuid4()),
    )


def resolve_tool_handler(tool_name: str, tool_handlers: Mapping[str, Any] | None = None):
    if tool_handlers and tool_name in tool_handlers:
        return tool_handlers[tool_name]
    from app.services.tool_handlers import get_handler as _get_handler

    return _get_handler(tool_name)


def parse_tool_arguments(tool_call: dict) -> dict:
    arguments = tool_call["arguments"]
    if not isinstance(arguments, str):
        return arguments
    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {}


def build_successful_call_signature(
    tool_call: dict,
    tool_handlers: Mapping[str, Any] | None = None,
) -> str | None:
    """返回 handler 显式声明的运行内成功调用签名。"""

    handler = resolve_tool_handler(str(tool_call.get("name", "")), tool_handlers)
    if handler is None:
        return None
    signature_builder = getattr(type(handler), "build_successful_call_signature", None)
    if not callable(signature_builder):
        return None
    args = parse_tool_arguments(tool_call)
    if not isinstance(args, dict):
        return None
    try:
        signature = signature_builder(handler, args)
    except Exception:  # noqa: BLE001 — 去重签名失败必须回退真实执行
        logger.warning("工具成功调用签名生成失败: tool=%s", tool_call.get("name"), exc_info=True)
        return None
    return signature if isinstance(signature, str) and signature else None


def prepare_tool_arguments(
    *,
    tool_name: str,
    args: dict,
    network_budget: "NetworkToolBudget | None",
):
    if network_budget is None:
        return args, None
    if tool_name == "web_search":
        return network_budget.prepare_web_search_args(args)
    if tool_name == "url_read":
        return network_budget.prepare_url_read_args(args)
    return args, None


def strip_internal_tool_arguments(args: dict) -> dict:
    if not any(key in args for key in _INTERNAL_TOOL_ARG_KEYS):
        return args
    return {key: value for key, value in args.items() if key not in _INTERNAL_TOOL_ARG_KEYS}


def attach_internal_tool_metadata(result, args: dict) -> None:
    budget_decision = args.get("budget_decision")
    if isinstance(budget_decision, dict):
        result.data.setdefault("budget_decision", budget_decision)


def build_tool_execution_record(
    *,
    tool_call: dict,
    result,
    handler,
    ids: ToolExecutionIds,
    reused: bool = False,
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        tool_call=tool_call,
        result=result,
        handler=handler,
        block_id=ids.block_id,
        log_id=ids.log_id,
        reused=reused,
    )


def build_unknown_tool_record(*, tool_call: dict, ids: ToolExecutionIds) -> ToolExecutionRecord:
    from app.services.tool_handlers import ToolResult

    logger.warning(f"未知的 tool_call: {tool_call['name']}")
    return build_tool_execution_record(
        tool_call=tool_call,
        result=ToolResult(status="failed", error_message=f"未知工具: {tool_call['name']}"),
        handler=None,
        ids=ids,
    )


def build_reused_tool_record(
    *,
    tool_call: dict,
    handler,
    ids: ToolExecutionIds,
) -> ToolExecutionRecord:
    from app.services.tool_handlers import ToolResult

    return build_tool_execution_record(
        tool_call=tool_call,
        result=ToolResult(
            status="success",
            data={"reused_successful_result": True},
            duration_ms=0,
        ),
        handler=handler,
        ids=ids,
        reused=True,
    )


async def emit_budget_result(
    *,
    request: ToolExecutionBatchRequest,
    tool_call: dict,
    handler,
    args: dict,
    result,
) -> None:
    await emit_tool_call_started(
        request.emitter,
        tool_call_id=tool_call["id"],
        tool_name=handler.tool_name,
        arguments=_sanitize_event_arguments(handler, args),
    )
    await emit_tool_call_result(
        request.emitter,
        tool_call_id=tool_call["id"],
        tool_name=handler.tool_name,
        result=result,
        duration_ms=result.duration_ms or 0,
        result_summary_builder=handler._build_result_summary,
    )


async def execute_tool_handler(*, request: ToolExecutionBatchRequest, tool_call: dict, handler, args: dict):
    executor = (
        execute_tool_once if getattr(handler, "supports_automatic_retry", True) is False else execute_tool_with_retry
    )
    if request.emitter is None:
        return await executor(handler, args, request.runtime_context)
    return await execute_tool_with_lifecycle(
        tool_call_id=tool_call["id"],
        tool_name=handler.tool_name,
        args=_sanitize_event_arguments(handler, args),
        target=handler,
        execute=lambda target, _event_arguments: executor(target, args, request.runtime_context),
        result_summary_builder=handler._build_result_summary,
        emitter=request.emitter,
    )


def _sanitize_event_arguments(handler, args: dict) -> dict:
    """只调用真实 handler 类型显式实现的方法，避免无 spec 的 Mock 伪造属性。"""

    sanitizer = getattr(type(handler), "sanitize_input_params_for_event", None)
    return sanitizer(handler, args) if callable(sanitizer) else args


async def log_tool_execution(
    *,
    request: ToolExecutionBatchRequest,
    handler,
    ids: ToolExecutionIds,
    result,
    args: dict,
) -> None:
    await handler.log(
        log_id=ids.log_id,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        model_id=request.model_id,
        provider=request.provider,
        result=result,
        input_params=args,
        trace_id=request.trace_id,
        step_number=request.step_number,
        message_id=request.message_id,
    )


async def emit_progress_digest_events(
    *,
    request: ToolExecutionBatchRequest,
    record: ToolExecutionRecord,
) -> None:
    if request.emitter is None:
        return
    try:
        await request.emitter.tool_result_digest(**build_tool_result_digest(record))
        for evidence in build_evidence_items(record):
            await request.emitter.evidence_item_upserted(
                tool_call_id=str(record.tool_call.get("id", "")),
                evidence=evidence,
            )
    except StreamWriteTerminalError:
        raise
    except Exception as error:  # noqa: BLE001 — v2 非关键进度事件失败不能中断工具结果主链路
        logger.warning(f"工具 digest 事件发送失败: tool={record.tool_name}, error={error}")


async def execute_one_tool_call(request: ToolExecutionBatchRequest, tool_call: dict) -> ToolExecutionRecord:
    handler = resolve_tool_handler(tool_call["name"], request.tool_handlers)
    ids = new_tool_execution_ids()
    if not handler:
        return build_unknown_tool_record(tool_call=tool_call, ids=ids)

    successful_signature = build_successful_call_signature(tool_call, request.tool_handlers)
    if (
        successful_signature is not None
        and request.successful_tool_call_signatures is not None
        and successful_signature in request.successful_tool_call_signatures
    ):
        return build_reused_tool_record(tool_call=tool_call, handler=handler, ids=ids)

    args = parse_tool_arguments(tool_call)
    args, budget_result = prepare_tool_arguments(
        tool_name=tool_call["name"],
        args=args,
        network_budget=request.network_budget,
    )
    executable_args = strip_internal_tool_arguments(args)
    if budget_result is not None:
        result = budget_result
        await emit_budget_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            args=executable_args,
            result=result,
        )
    else:
        result = await execute_tool_handler(
            request=request,
            tool_call=tool_call,
            handler=handler,
            args=executable_args,
        )
    attach_internal_tool_metadata(result, args)
    if (
        result.status == "success"
        and successful_signature is not None
        and request.successful_tool_call_signatures is not None
    ):
        request.successful_tool_call_signatures.add(successful_signature)

    await log_tool_execution(request=request, handler=handler, ids=ids, result=result, args=executable_args)
    record = build_tool_execution_record(tool_call=tool_call, result=result, handler=handler, ids=ids)
    await emit_progress_digest_events(request=request, record=record)
    return record


async def execute_tool_batch(
    request: ToolExecutionBatchRequest,
    tool_calls: list[dict],
) -> list[ToolExecutionRecord]:
    indexed_calls = list(enumerate(tool_calls))
    amap_product_calls = [
        (index, tool_call)
        for index, tool_call in indexed_calls
        if tool_call.get("name") in _AMAP_PRODUCT_EXECUTION_PRIORITY
    ]
    parallel_calls = [
        (index, tool_call)
        for index, tool_call in indexed_calls
        if tool_call.get("name") not in _AMAP_PRODUCT_EXECUTION_PRIORITY
    ]
    reusable_groups: dict[str, list[tuple[int, dict]]] = {}
    ungrouped_parallel_calls: list[tuple[int, dict]] = []
    for index, tool_call in parallel_calls:
        signature = (
            build_successful_call_signature(tool_call, request.tool_handlers)
            if request.successful_tool_call_signatures is not None
            else None
        )
        if signature is None:
            ungrouped_parallel_calls.append((index, tool_call))
        else:
            reusable_groups.setdefault(signature, []).append((index, tool_call))

    async def execute_indexed(index: int, tool_call: dict) -> tuple[int, ToolExecutionRecord]:
        return index, await execute_one_tool_call(request, tool_call)

    async def execute_amap_products() -> list[tuple[int, ToolExecutionRecord]]:
        records = []
        for index, tool_call in sorted(
            amap_product_calls,
            key=lambda item: (_AMAP_PRODUCT_EXECUTION_PRIORITY[item[1]["name"]], item[0]),
        ):
            records.append(await execute_indexed(index, tool_call))
        return records

    async def execute_reusable_group(
        grouped_calls: list[tuple[int, dict]],
    ) -> list[tuple[int, ToolExecutionRecord]]:
        records = []
        for index, tool_call in grouped_calls:
            records.append(await execute_indexed(index, tool_call))
        return records

    pending = [execute_indexed(index, tool_call) for index, tool_call in ungrouped_parallel_calls]
    pending.extend(execute_reusable_group(grouped_calls) for grouped_calls in reusable_groups.values())
    if amap_product_calls:
        pending.append(execute_amap_products())
    batches = await asyncio.gather(*pending)

    indexed_results: list[tuple[int, ToolExecutionRecord]] = []
    for batch in batches:
        if isinstance(batch, list):
            indexed_results.extend(batch)
        else:
            indexed_results.append(batch)
    indexed_results.sort(key=lambda item: item[0])
    return [record for _, record in indexed_results]


async def execute_tools_parallel(
    tool_calls: list[dict],
    conversation_id: str,
    user_id: str,
    model_id: str,
    provider: str,
    trace_id: str = None,
    step_number: int = None,
    message_id: str | None = None,
    emitter: Optional[AgentEventEmitter] = None,
    network_budget: "NetworkToolBudget | None" = None,
    tool_handlers: Mapping[str, Any] | None = None,
    runtime_context: Any = None,
    successful_tool_call_signatures: set[str] | None = None,
) -> list[ToolExecutionRecord]:
    """
    并行执行所有 tool_calls。

    统一走 tool_call 生命周期协议（tool_call_started / completed agent_event），
    但中间塞入 execute_tool_with_retry（瞬时重试 + 30s timeout）。
    tool_call_logs 仍通过 handler.log 写入。

    返回 ToolExecutionRecord 列表，调用方不再依赖裸 tuple 位置。
    """
    request = ToolExecutionBatchRequest(
        conversation_id=conversation_id,
        user_id=user_id,
        model_id=model_id,
        provider=provider,
        trace_id=trace_id,
        step_number=step_number,
        message_id=message_id,
        emitter=emitter,
        network_budget=network_budget,
        tool_handlers=tool_handlers,
        runtime_context=runtime_context,
        successful_tool_call_signatures=successful_tool_call_signatures,
    )
    return await execute_tool_batch(request, tool_calls)
