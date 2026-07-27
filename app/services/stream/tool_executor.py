"""并行工具执行 + emitter→Redis 适配器。

spec §4.3。execute_tools_parallel 走 emitter 的
tool_call_started/completed 协议，并把 execute_tool_with_retry
（瞬时重试 + 30s timeout）夹在 start/completed 之间。
"""

import asyncio
import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Optional

import backoff

from app.core.logger import app_logger as logger
from app.services.agent.emitter import AgentEventEmitter
from app.services.agent.progress_digest import build_evidence_items, build_tool_result_digest
from app.services.mcp.tool_contract import (
    MAX_TOOL_ARGUMENT_JSON_BYTES,
    validate_tool_argument_resource_limits,
)
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

_RETRYABLE_TOOL_ERROR_CODES = frozenset(
    {
        "search_unavailable",
        "tool_timeout",
        "upstream_timeout",
        "upstream_unavailable",
    }
)
_INTERNAL_TOOL_ARG_KEYS = {"budget_decision"}
_AMAP_PRODUCT_EXECUTION_PRIORITY = {
    "route_compare": 0,
    "local_place_search": 1,
    "weather_forecast": 2,
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
    remote_argument_repair_disabled_tools: frozenset[str] = frozenset()
    deferred_route_call_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ToolExecutionIds:
    block_id: str
    log_id: str


def _should_retry_tool_result(result) -> bool:
    """决定 ToolResult 是否应该再试一次（True = 重试，False = 接受当前结果）。"""
    if result.status in ("success", "degraded"):
        return False
    data = result.data if isinstance(result.data, dict) else {}
    if isinstance(data.get("retryable"), bool):
        return data["retryable"]
    error_code = data.get("error_code")
    return isinstance(error_code, str) and error_code in _RETRYABLE_TOOL_ERROR_CODES


def _log_tool_retry(details: dict[str, Any]) -> None:
    handler = details["args"][0]
    result = details.get("value")
    data = getattr(result, "data", None)
    error_code = data.get("error_code") if isinstance(data, dict) else None
    logger.warning(
        "工具执行失败后重试: tool=%s attempt=%s wait_seconds=%s error_code=%s",
        getattr(handler, "tool_name", ""),
        details.get("tries"),
        int(details.get("wait", 0)),
        error_code if isinstance(error_code, str) else "unknown",
    )


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
    on_backoff=_log_tool_retry,
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
        return ToolResult(
            status="failed",
            error_message="工具调用超时",
            data={"error_code": "tool_timeout", "retryable": True},
        )


async def execute_tool_once(handler, args: dict, runtime_context: Any = None):
    """有副作用或不具备幂等保证的工具只执行一次，但仍保留统一超时。"""
    from app.services.tool_handlers import ToolResult

    try:
        return await asyncio.wait_for(_execute_handler(handler, args, runtime_context), timeout=AGENT_TOOL_TIMEOUT)
    except asyncio.TimeoutError:
        return ToolResult(
            status="failed",
            error_message="工具调用超时",
            data={"error_code": "tool_timeout", "retryable": False},
        )


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
    arguments = tool_call.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    if not isinstance(arguments, str):
        raise ToolArgumentsParseError
    if len(arguments.encode("utf-8")) > MAX_TOOL_ARGUMENT_JSON_BYTES:
        raise ToolArgumentsLimitError("max_bytes")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError:
        raise ToolArgumentsParseError from None
    if not isinstance(parsed, dict):
        raise ToolArgumentsParseError
    return parsed


class ToolArgumentsParseError(ValueError):
    pass


class ToolArgumentsLimitError(ToolArgumentsParseError):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


_REPAIR_FIELD_PATTERN = re.compile(
    r"^(?:\$|[A-Za-z_][A-Za-z0-9_:-]{0,63})"
    r"(?:(?:\[\])|(?:\.\*)|(?:\.[A-Za-z_][A-Za-z0-9_:-]{0,63}))*$"
)


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
    try:
        args = parse_tool_arguments(tool_call)
    except ToolArgumentsParseError:
        return None
    if not isinstance(args, dict):
        return None
    try:
        signature = signature_builder(handler, args)
    except Exception as error:  # noqa: BLE001 — 去重签名失败必须回退真实执行
        logger.warning(
            "工具成功调用签名生成失败: tool=%s error_type=%s",
            getattr(handler, "tool_name", ""),
            type(error).__name__,
        )
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
        plan_item_id=tool_call.get("plan_item_id"),
    )
    await emit_tool_call_result(
        request.emitter,
        tool_call_id=tool_call["id"],
        tool_name=handler.tool_name,
        result=result,
        duration_ms=result.duration_ms or 0,
        result_summary_builder=handler._build_result_summary,
        plan_item_id=tool_call.get("plan_item_id"),
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
        plan_item_id=tool_call.get("plan_item_id"),
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
        digest = build_tool_result_digest(record)
        plan_item_id = record.tool_call.get("plan_item_id")
        if isinstance(plan_item_id, str):
            digest["plan_item_id"] = plan_item_id
        await request.emitter.tool_result_digest(**digest)
        for evidence in build_evidence_items(record):
            await request.emitter.evidence_item_upserted(
                tool_call_id=str(record.tool_call.get("id", "")),
                evidence=evidence,
            )
    except StreamWriteTerminalError:
        raise
    except Exception as error:  # noqa: BLE001 — v2 非关键进度事件失败不能中断工具结果主链路
        logger.warning(
            "工具 digest 事件发送失败: tool=%s error_type=%s",
            record.tool_name,
            type(error).__name__,
        )


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

    try:
        args = parse_tool_arguments(tool_call)
    except ToolArgumentsLimitError as error:
        args = {}
        result = _build_argument_preflight_result(
            request,
            handler.tool_name,
            error_code="arguments_schema_invalid",
            validation_errors=[f"$:{error.code}"],
        )
        return await _complete_preflight_tool_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            ids=ids,
            args=args,
            result=result,
        )
    except ToolArgumentsParseError:
        args = {}
        result = _build_argument_preflight_result(
            request,
            handler.tool_name,
            error_code="arguments_json_invalid",
        )
        return await _complete_preflight_tool_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            ids=ids,
            args=args,
            result=result,
        )
    validation_errors = _validate_handler_arguments(handler, args)
    if validation_errors:
        result = _build_argument_preflight_result(
            request,
            handler.tool_name,
            error_code="arguments_schema_invalid",
            validation_errors=validation_errors,
        )
        return await _complete_preflight_tool_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            ids=ids,
            args={},
            result=result,
        )
    if str(tool_call.get("id", "")) in request.deferred_route_call_ids:
        result = _build_argument_preflight_result(
            request,
            handler.tool_name,
            error_code="route_requires_prior_travel_results",
        )
        return await _complete_preflight_tool_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            ids=ids,
            args={},
            result=result,
        )
    runtime_repair = _build_runtime_argument_repair_result(request, handler, args)
    if runtime_repair is not None:
        return await _complete_preflight_tool_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            ids=ids,
            args={},
            result=runtime_repair,
        )
    repair_guard, resolves_repair_id, args = _authorize_repaired_arguments(request, handler.tool_name, args)
    if repair_guard is not None:
        return await _complete_preflight_tool_result(
            request=request,
            tool_call=tool_call,
            handler=handler,
            ids=ids,
            args={},
            result=repair_guard,
        )
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
    result = _normalize_remote_argument_failure(
        request,
        handler.tool_name,
        result,
    )
    if (
        resolves_repair_id
        and result.status in {"success", "degraded"}
        and not isinstance(result.data.get("repair"), dict)
    ):
        _mark_argument_repair_resolved(request, handler.tool_name)
        result.data["resolves_repair_id"] = resolves_repair_id
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


async def _complete_preflight_tool_result(
    *,
    request: ToolExecutionBatchRequest,
    tool_call: dict,
    handler,
    ids: ToolExecutionIds,
    args: dict,
    result,
) -> ToolExecutionRecord:
    await emit_budget_result(
        request=request,
        tool_call=tool_call,
        handler=handler,
        args=args,
        result=result,
    )
    await log_tool_execution(request=request, handler=handler, ids=ids, result=result, args=args)
    record = build_tool_execution_record(
        tool_call=tool_call,
        result=result,
        handler=handler,
        ids=ids,
    )
    await emit_progress_digest_events(request=request, record=record)
    return record


def _validate_handler_arguments(
    handler: Any,
    args: dict,
) -> list[str]:
    raw_errors = validate_tool_argument_resource_limits(args)
    validator = getattr(type(handler), "validate_arguments", None)
    try:
        if not raw_errors:
            raw_errors = validator(handler, args) if callable(validator) else []
    except Exception as error:  # noqa: BLE001 — 本地预校验异常必须回退为不可执行，不能触网
        logger.warning(
            "工具参数本地预校验异常: tool=%s error_type=%s",
            getattr(handler, "tool_name", ""),
            type(error).__name__,
        )
        return ["request:validator_error"]
    if not isinstance(raw_errors, list):
        return []
    errors: list[str] = []
    for item in raw_errors:
        if isinstance(item, dict):
            field = item.get("field")
            error_type = item.get("code")
            separator = ":" if isinstance(field, str) and isinstance(error_type, str) else ""
        elif isinstance(item, str) and len(item) <= 128:
            field, separator, error_type = item.partition(":")
        else:
            continue
        if (
            separator
            and isinstance(field, str)
            and isinstance(error_type, str)
            and _REPAIR_FIELD_PATTERN.fullmatch(field)
            and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_type)
        ):
            errors.append(f"{field}:{error_type}")
    return list(dict.fromkeys(errors))[:8]


def _argument_repair_key(tool_name: str) -> str:
    return f"tool_arguments:{tool_name}"


def _new_argument_repair_id() -> str:
    return f"repair_{uuid.uuid4().hex[:16]}"


def _build_argument_preflight_result(
    request: ToolExecutionBatchRequest,
    tool_name: str,
    *,
    error_code: str,
    validation_errors: list[str] | None = None,
    local_preflight: bool = True,
    repair_action: str = "rebuild_arguments",
    allowed_values: dict[str, list[str]] | None = None,
    repair_values_verified: bool = False,
    original_arguments: dict[str, Any] | None = None,
):
    from app.services.tool_handlers import ToolResult

    required_fields = sorted(
        {item.partition(":")[0] for item in validation_errors or [] if item.endswith(":required")}
    )[:8]
    projected_allowed_values: dict[str, list[str]] = {}
    if repair_values_verified and isinstance(allowed_values, dict):
        for field in required_fields:
            raw_values = allowed_values.get(field)
            if not isinstance(raw_values, list):
                continue
            values = [
                value.strip()
                for value in raw_values[:4]
                if isinstance(value, str) and value.strip() and len(value.strip()) <= 120
            ]
            if values:
                projected_allowed_values[field] = list(dict.fromkeys(values))
    constraints: dict[str, Any] | None = None
    if (
        repair_action == "provide_argument"
        and repair_values_verified
        and isinstance(original_arguments, dict)
        and required_fields
        and set(projected_allowed_values) == set(required_fields)
    ):
        constraints = {
            "immutable_arguments_sha256": _argument_repair_fingerprint(
                original_arguments,
                excluded_fields=required_fields,
            ),
            "required_fields": tuple(required_fields),
            "allowed_values": {field: tuple(projected_allowed_values[field]) for field in required_fields},
            "allowed_argument_keys": tuple(dict.fromkeys([*original_arguments, *required_fields])),
        }

    state = getattr(request.runtime_context, "argument_repair_state", None)
    key = _argument_repair_key(tool_name)
    current_step = getattr(request.runtime_context, "step_number", 0)
    repair_id = _new_argument_repair_id()
    retryable = False
    if isinstance(state, dict):
        existing = state.get(key)
        if existing is None:
            retryable = _claim_argument_repair_slot(request)
            state[key] = {
                "step_number": current_step,
                "status": "pending" if retryable else "consumed",
                "repair_id": repair_id,
                **({"constraints": constraints} if constraints is not None else {}),
            }
        elif existing.get("status") == "pending" and current_step <= existing.get("step_number", 0):
            return ToolResult(
                status="failed",
                data={
                    "error_code": "argument_repair_coalesced",
                    "local_preflight": local_preflight,
                },
            )
        elif existing.get("status") == "resolved":
            state[key] = {
                "step_number": current_step,
                "status": "consumed",
                "repair_id": repair_id,
            }
        else:
            repair_id = existing.get("repair_id") if isinstance(existing.get("repair_id"), str) else repair_id
            retryable = False
            existing["status"] = "consumed"
        if existing is not None and existing.get("status") != "resolved" and isinstance(existing.get("repair_id"), str):
            repair_id = existing["repair_id"]
    else:
        retryable = _claim_argument_repair_slot(request)
    return ToolResult(
        status="failed",
        data={
            "error_code": error_code,
            "local_preflight": local_preflight,
            "repair_values_verified": repair_values_verified,
            "repair": {
                "action": repair_action,
                "repair_id": repair_id,
                "required_fields": required_fields,
                "allowed_values": projected_allowed_values,
                "retryable": retryable,
                "requires_user_input": False,
                "retry_exhausted": not retryable,
                "max_attempts": 1,
                "candidate_count": 0,
            },
        },
    )


def _build_runtime_argument_repair_result(
    request: ToolExecutionBatchRequest,
    handler: Any,
    args: dict,
):
    builder = getattr(type(handler), "build_runtime_argument_repair", None)
    if not callable(builder):
        return None
    try:
        spec = builder(handler, args, request.runtime_context)
    except Exception as error:  # noqa: BLE001 — 修参门禁异常必须拒绝触网
        logger.warning(
            "工具运行时修参检查异常: tool=%s error_type=%s",
            getattr(handler, "tool_name", ""),
            type(error).__name__,
        )
        return _build_runtime_argument_guard_failure()
    if not isinstance(spec, dict):
        return None
    error_code = spec.get("error_code")
    required_fields = spec.get("required_fields")
    allowed_values = spec.get("allowed_values")
    if not isinstance(error_code, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", error_code):
        return _build_runtime_argument_guard_failure()
    if not isinstance(required_fields, list) or not isinstance(allowed_values, dict):
        return _build_runtime_argument_guard_failure()
    validated_fields = [
        field for field in required_fields[:8] if isinstance(field, str) and _REPAIR_FIELD_PATTERN.fullmatch(field)
    ]
    if not validated_fields:
        return _build_runtime_argument_guard_failure()
    if any(
        not isinstance(allowed_values.get(field), list)
        or not any(isinstance(value, str) and value.strip() for value in allowed_values[field])
        for field in validated_fields
    ):
        return _build_runtime_argument_guard_failure()
    return _build_argument_preflight_result(
        request,
        handler.tool_name,
        error_code=error_code,
        validation_errors=[f"{field}:required" for field in validated_fields],
        repair_action="provide_argument",
        allowed_values=allowed_values,
        repair_values_verified=True,
        original_arguments=args,
    )


def _build_runtime_argument_guard_failure():
    from app.services.tool_handlers import ToolResult

    return ToolResult(
        status="failed",
        data={
            "error_code": "runtime_argument_guard_failed",
            "local_preflight": True,
        },
    )


def _argument_repair_fingerprint(
    arguments: dict[str, Any],
    *,
    excluded_fields: list[str] | tuple[str, ...],
) -> str:
    excluded = set(excluded_fields)
    immutable_arguments = {key: value for key, value in arguments.items() if key not in excluded}
    canonical = json.dumps(
        immutable_arguments,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _claim_argument_repair_slot(request: ToolExecutionBatchRequest) -> bool:
    """为新的独立修参原子领取一个下一轮真实调用槽。"""

    context = request.runtime_context
    remaining_steps = getattr(context, "remaining_agent_steps", None)
    if isinstance(remaining_steps, int) and remaining_steps < 1:
        return False

    remaining_slots = getattr(context, "remaining_argument_repair_slots", None)
    if not isinstance(remaining_slots, int) or isinstance(remaining_slots, bool):
        remaining_slots = getattr(context, "remaining_agent_tool_calls_after_batch", None)
        if not isinstance(remaining_slots, int) or isinstance(remaining_slots, bool):
            remaining_slots = getattr(context, "remaining_agent_tool_calls_before_round", None)
        if not isinstance(remaining_slots, int) or isinstance(remaining_slots, bool):
            return True
    if remaining_slots < 1:
        return False
    context.remaining_argument_repair_slots = remaining_slots - 1
    return True


def _normalize_remote_argument_failure(
    request: ToolExecutionBatchRequest,
    tool_name: str,
    result,
):
    data = getattr(result, "data", None)
    if not isinstance(data, dict) or data.get("error_code") != "remote_invalid_arguments":
        return result
    if tool_name in request.remote_argument_repair_disabled_tools:
        return result
    validation_errors = [item for item in data.get("validation_errors", []) if isinstance(item, str)][:8]
    if not validation_errors:
        return result
    repair_result = _build_argument_preflight_result(
        request,
        tool_name,
        error_code="remote_arguments_invalid",
        validation_errors=validation_errors,
        local_preflight=False,
    )
    repair_result.duration_ms = result.duration_ms
    return repair_result


def _authorize_repaired_arguments(
    request: ToolExecutionBatchRequest,
    tool_name: str,
    args: dict[str, Any],
):
    from app.services.tool_handlers import ToolResult

    state = getattr(request.runtime_context, "argument_repair_state", None)
    if not isinstance(state, dict):
        return None, None, args
    key = _argument_repair_key(tool_name)
    pending = state.get(key)
    if not isinstance(pending, dict):
        return None, None, args
    if pending.get("status") == "resolved":
        return None, None, args
    repair_id = pending.get("repair_id") if isinstance(pending.get("repair_id"), str) else _new_argument_repair_id()
    pending["repair_id"] = repair_id
    current_step = getattr(request.runtime_context, "step_number", 0)
    pending_step = pending.get("step_number", 0)
    if pending.get("status") == "pending" and current_step > pending_step:
        pending["status"] = "consumed"
        constraints = pending.get("constraints")
        authorized_args = _project_repaired_arguments(args, constraints)
        if isinstance(constraints, dict) and not _arguments_satisfy_repair_constraints(
            authorized_args,
            constraints,
        ):
            required_fields = [field for field in constraints.get("required_fields", ()) if isinstance(field, str)][:8]
            allowed_values = {
                field: [
                    value for value in constraints.get("allowed_values", {}).get(field, ()) if isinstance(value, str)
                ][:4]
                for field in required_fields
            }
            return (
                ToolResult(
                    status="failed",
                    data={
                        "error_code": "argument_repair_constraint_violation",
                        "local_preflight": True,
                        "repair_values_verified": True,
                        "repair": {
                            "action": "provide_argument",
                            "repair_id": repair_id,
                            "required_fields": required_fields,
                            "allowed_values": allowed_values,
                            "retryable": False,
                            "requires_user_input": False,
                            "retry_exhausted": True,
                            "max_attempts": 1,
                            "candidate_count": 0,
                        },
                    },
                ),
                None,
                args,
            )
        return None, repair_id, authorized_args
    retryable = pending.get("status") == "pending"
    return (
        ToolResult(
            status="failed",
            data={
                "error_code": ("argument_repair_deferred_to_next_round" if retryable else "argument_repair_exhausted"),
                "local_preflight": True,
                "repair": {
                    "action": "rebuild_arguments",
                    "repair_id": repair_id,
                    "required_fields": [],
                    "allowed_values": {},
                    "retryable": retryable,
                    "requires_user_input": False,
                    "retry_exhausted": not retryable,
                    "max_attempts": 1,
                    "candidate_count": 0,
                },
            },
        ),
        None,
        args,
    )


def _project_repaired_arguments(
    args: dict[str, Any],
    constraints: Any,
) -> dict[str, Any]:
    """只保留原调用字段和本轮允许补齐的字段，丢弃模型额外添加的未授权参数。"""

    if not isinstance(constraints, dict):
        return args
    allowed_keys = constraints.get("allowed_argument_keys")
    if not isinstance(allowed_keys, tuple) or not allowed_keys:
        return args
    allowed = {key for key in allowed_keys if isinstance(key, str)}
    return {key: value for key, value in args.items() if key in allowed}


def _arguments_satisfy_repair_constraints(
    args: dict[str, Any],
    constraints: dict[str, Any],
) -> bool:
    required_fields = constraints.get("required_fields")
    allowed_values = constraints.get("allowed_values")
    expected_fingerprint = constraints.get("immutable_arguments_sha256")
    if (
        not isinstance(required_fields, tuple)
        or not required_fields
        or not isinstance(allowed_values, dict)
        or not isinstance(expected_fingerprint, str)
    ):
        return False
    if (
        _argument_repair_fingerprint(
            args,
            excluded_fields=required_fields,
        )
        != expected_fingerprint
    ):
        return False
    return all(
        isinstance(args.get(field), str) and args[field].strip() in allowed_values.get(field, ())
        for field in required_fields
    )


def _mark_argument_repair_resolved(
    request: ToolExecutionBatchRequest,
    tool_name: str,
) -> None:
    state = getattr(request.runtime_context, "argument_repair_state", None)
    if not isinstance(state, dict):
        return
    pending = state.get(_argument_repair_key(tool_name))
    if isinstance(pending, dict):
        pending["status"] = "resolved"


async def execute_tool_batch(
    request: ToolExecutionBatchRequest,
    tool_calls: list[dict],
) -> list[ToolExecutionRecord]:
    indexed_calls = list(enumerate(tool_calls))
    tool_name_counts: dict[str, int] = {}
    for _, tool_call in indexed_calls:
        tool_name = str(tool_call.get("name", ""))
        tool_name_counts[tool_name] = tool_name_counts.get(tool_name, 0) + 1
    request = replace(
        request,
        remote_argument_repair_disabled_tools=frozenset(
            tool_name for tool_name, count in tool_name_counts.items() if tool_name and count > 1
        ),
        deferred_route_call_ids=_deferred_route_call_ids(request, indexed_calls),
    )
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
    invalid_preflight_indexes = {
        index
        for index, tool_call in indexed_calls
        if is_local_argument_preflight_failure(
            tool_call,
            request.tool_handlers,
        )
    }
    repair_sensitive_names = {
        str(tool_call.get("name", "")) for index, tool_call in parallel_calls if index in invalid_preflight_indexes
    }
    _set_remaining_tool_calls_after_batch(
        request,
        indexed_calls=indexed_calls,
        invalid_preflight_indexes=invalid_preflight_indexes,
    )
    argument_repair_groups: dict[str, list[tuple[int, dict]]] = {}
    regular_parallel_calls: list[tuple[int, dict]] = []
    for item in parallel_calls:
        tool_name = str(item[1].get("name", ""))
        if tool_name in repair_sensitive_names:
            argument_repair_groups.setdefault(tool_name, []).append(item)
        else:
            regular_parallel_calls.append(item)
    reusable_groups: dict[str, list[tuple[int, dict]]] = {}
    ungrouped_parallel_calls: list[tuple[int, dict]] = []
    for index, tool_call in regular_parallel_calls:
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
            key=lambda item: (
                _AMAP_PRODUCT_EXECUTION_PRIORITY[item[1]["name"]],
                item[0] not in invalid_preflight_indexes,
                item[0],
            ),
        ):
            records.append(await execute_indexed(index, tool_call))
        return records

    async def execute_reusable_group(
        grouped_calls: list[tuple[int, dict]],
    ) -> list[tuple[int, ToolExecutionRecord]]:
        records = []
        for index, tool_call in sorted(
            grouped_calls,
            key=lambda item: (item[0] not in invalid_preflight_indexes, item[0]),
        ):
            records.append(await execute_indexed(index, tool_call))
        return records

    pending = [execute_indexed(index, tool_call) for index, tool_call in ungrouped_parallel_calls]
    pending.extend(execute_reusable_group(grouped_calls) for grouped_calls in reusable_groups.values())
    pending.extend(execute_reusable_group(grouped_calls) for grouped_calls in argument_repair_groups.values())
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


def _deferred_route_call_ids(
    request: ToolExecutionBatchRequest,
    indexed_calls: list[tuple[int, dict]],
) -> frozenset[str]:
    """同轮先查班次时，拒绝模型凭常识猜接驳点；下一轮只能使用真实班次站点。"""

    if not any(tool_call.get("name") in {"search_flights", "search_trains"} for _, tool_call in indexed_calls):
        return frozenset()
    trusted_hints = getattr(request.runtime_context, "route_city_hints", None)
    trusted_origins = set(trusted_hints) if isinstance(trusted_hints, dict) else set()
    deferred: set[str] = set()
    for _, tool_call in indexed_calls:
        if tool_call.get("name") != "route_compare":
            continue
        try:
            args = parse_tool_arguments(tool_call)
        except ToolArgumentsParseError:
            continue
        origin = args.get("origin")
        if args.get("origin_source") == "named" and isinstance(origin, str) and origin.strip() in trusted_origins:
            continue
        tool_call_id = str(tool_call.get("id", ""))
        if tool_call_id:
            deferred.add(tool_call_id)
    return frozenset(deferred)


def is_local_argument_preflight_failure(
    tool_call: dict,
    tool_handlers: Mapping[str, Any] | None = None,
) -> bool:
    handler = resolve_tool_handler(str(tool_call.get("name", "")), tool_handlers)
    if handler is None:
        return False
    try:
        args = parse_tool_arguments(tool_call)
    except ToolArgumentsParseError:
        return True
    return bool(_validate_handler_arguments(handler, args))


def _set_remaining_tool_calls_after_batch(
    request: ToolExecutionBatchRequest,
    *,
    indexed_calls: list[tuple[int, dict]],
    invalid_preflight_indexes: set[int],
) -> None:
    context = request.runtime_context
    if context is None:
        return
    remaining_before = getattr(context, "remaining_agent_tool_calls_before_round", None)
    if not isinstance(remaining_before, int):
        return
    invalid_tool_names = {
        str(tool_call.get("name", "")) for index, tool_call in indexed_calls if index in invalid_preflight_indexes
    }
    expected_actual_calls = 0
    for index, tool_call in indexed_calls:
        if index in invalid_preflight_indexes:
            continue
        if str(tool_call.get("name", "")) in invalid_tool_names:
            continue
        if (
            request.successful_tool_call_signatures is not None
            and (
                signature := build_successful_call_signature(
                    tool_call,
                    request.tool_handlers,
                )
            )
            and signature in request.successful_tool_call_signatures
        ):
            continue
        expected_actual_calls += 1
    remaining_after_batch = max(
        0,
        remaining_before - expected_actual_calls,
    )
    context.remaining_agent_tool_calls_after_batch = remaining_after_batch
    remaining_steps = getattr(context, "remaining_agent_steps", None)
    context.remaining_argument_repair_slots = (
        0 if isinstance(remaining_steps, int) and remaining_steps < 1 else remaining_after_batch
    )


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
