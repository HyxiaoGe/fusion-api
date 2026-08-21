"""LLM SSE 消费 + 调用重试。

spec §4.2。stream_round 把 litellm streaming response 消费成
(reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)；
每个 delta 通过 append_chunk 写 Redis Stream，每 LOCK_CHECK_INTERVAL
个 chunk 检查锁所有权（被踢则提前返回 finish_reason="cancelled"）。
"""

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Optional

import backoff
import httpx
import litellm

from app.ai.llm_observability import merge_litellm_kwargs
from app.core.logger import app_logger as logger
from app.schemas.chat import Usage
from app.services.stream.reasoning_transport import (
    ReasoningTransportMode,
    initial_reasoning_transport_mode,
)
from app.services.stream_state_service import append_chunk, check_lock_owner
from app.utils.user_visible_content import sanitize_internal_tool_names, sanitize_user_visible_reasoning

# 每 N 个 chunk 检查一次锁状态
LOCK_CHECK_INTERVAL = 20

# LLM 调用重试次数
AGENT_LLM_MAX_RETRIES = 1

_LLM_RETRYABLE_STATUS_CODES = frozenset({429, 502, 503, 504})
_LLM_RETRYABLE_ERROR_CODES = frozenset(
    {
        "429",
        "502",
        "503",
        "504",
        "api_connection_error",
        "api_timeout",
        "connection_error",
        "connection_timeout",
        "overloaded_error",
        "rate_limit_error",
        "rate_limit_exceeded",
        "rate_limited",
        "request_timeout",
        "service_unavailable",
        "timeout",
        "too_many_requests",
        "upstream_unavailable",
    }
)
_SAFE_ERROR_CODE_RE = re.compile(r"[A-Za-z0-9_]{1,64}")
_LITELLM_RETRYABLE_TYPES = (
    (litellm.RateLimitError, "rate_limit"),
    (litellm.ServiceUnavailableError, "service_unavailable"),
    (litellm.APIConnectionError, "connection_error"),
    (litellm.Timeout, "timeout"),
)
_NETWORK_RETRYABLE_TYPES = (
    asyncio.TimeoutError,
    TimeoutError,
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.ProtocolError,
)

_OPEN_THINK_TAG_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_CLOSE_THINK_TAG_RE = re.compile(r"</think\s*>", re.IGNORECASE)

_DSML_TOOL_CALLS_OPEN = "<｜｜DSML｜｜tool_calls>"
_DSML_TOOL_CALLS_CLOSE = "</｜｜DSML｜｜tool_calls>"
_DSML_DISTINCT_PREFIX = "<｜｜DSML"
_GENERIC_TOOL_CALL_OPEN = "<tool_call"
_GENERIC_TOOL_CALL_PREFIX_RE = re.compile(r"<tool_call\b", re.IGNORECASE)
_GENERIC_FUNCTION_OPEN_PREFIX = "<function="
_GENERIC_FUNCTION_OPEN_RE = re.compile(r"<function=[A-Za-z0-9_.:-]+>", re.IGNORECASE)
_DSML_INVOKE_RE = re.compile(
    r'<｜｜DSML｜｜invoke\s+name="(?P<name>[A-Za-z0-9_.:-]+)">'
    r"(?P<body>.*?)"
    r"</｜｜DSML｜｜invoke>",
    re.DOTALL,
)
_DSML_PARAMETER_RE = re.compile(
    r'<｜｜DSML｜｜parameter\s+name="(?P<name>[^"]+)"\s+string="(?P<is_string>true|false)">'
    r"(?P<value>.*?)"
    r"</｜｜DSML｜｜parameter>",
    re.DOTALL | re.IGNORECASE,
)
_REASONING_SNAPSHOT_PROBE_CHUNKS = 4
_REASONING_SNAPSHOT_MIN_LENGTH = 16


@dataclass(frozen=True)
class LLMStreamRequest:
    conversation_id: str
    task_id: str
    should_use_reasoning: bool
    thinking_block_id: str
    text_block_id: str
    provider: Optional[str] = None
    model_id: Optional[str] = None
    reasoning_transport_mode: Optional[ReasoningTransportMode] = None
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    defer_output: bool = False
    allow_deferred_reasoning_output: bool = False
    partial_output: dict[str, str] | None = None
    on_visible_output: Callable[[str], Awaitable[None]] | None = None
    on_output_candidate: Callable[[str, float | None], None] | None = None
    capture_output_candidate_time: Callable[[], float | None] | None = None


@dataclass
class LLMStreamState:
    reasoning_buf: str = ""
    raw_reasoning_buf: str = ""
    content_buf: str = ""
    raw_content_buf: str = ""
    raw_tag_reasoning_buf: str = ""
    usage_data: Optional[Usage] = None
    chunk_count: int = 0
    finish_reason: str = "stop"
    tool_calls_acc: dict[int, dict] = field(default_factory=dict)
    reasoning_transport_mode: ReasoningTransportMode = "delta"
    reasoning_probe_chunks: list[str] = field(default_factory=list)
    reasoning_snapshot_revision_logged: bool = False
    pending_reasoning_candidate_time: float | None = None


@dataclass(frozen=True)
class LLMStreamOutcome:
    reasoning_buf: str
    raw_reasoning_buf: str
    content_buf: str
    raw_content_buf: str
    tool_calls: list[dict]
    finish_reason: str
    usage_data: Optional[Usage]


class StreamRoundTuple(tuple):
    """保持旧五元组契约，并携带只供模型下一轮回放的原始响应。"""

    protocol_reasoning_buf: str
    protocol_content_buf: str

    def __new__(
        cls,
        reasoning_buf: str,
        content_buf: str,
        tool_calls: list[dict],
        finish_reason: str,
        usage_data: Optional[Usage],
        *,
        protocol_reasoning_buf: str,
        protocol_content_buf: str,
    ):
        instance = super().__new__(cls, (reasoning_buf, content_buf, tool_calls, finish_reason, usage_data))
        instance.protocol_reasoning_buf = protocol_reasoning_buf
        instance.protocol_content_buf = protocol_content_buf
        return instance


def _iter_exception_chain(exc: Exception):
    """遍历包装异常，但不读取异常文本。"""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _structured_status_code(exc: BaseException) -> int | None:
    for attribute in ("status_code", "http_status", "status"):
        value = getattr(exc, attribute, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value

    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def _structured_error_code(exc: BaseException) -> str | None:
    for attribute in ("error_code", "code"):
        value = getattr(exc, attribute, None)
        if isinstance(value, str) and _SAFE_ERROR_CODE_RE.fullmatch(value):
            return value.lower()
    return None


def _llm_retry_reason(exc: Exception) -> str | None:
    for current in _iter_exception_chain(exc):
        for error_type, reason in _LITELLM_RETRYABLE_TYPES:
            if isinstance(current, error_type):
                return reason
        if isinstance(current, _NETWORK_RETRYABLE_TYPES):
            return "network_error"

        status_code = _structured_status_code(current)
        if status_code in _LLM_RETRYABLE_STATUS_CODES:
            return f"http_{status_code}"

        error_code = _structured_error_code(current)
        if error_code in _LLM_RETRYABLE_ERROR_CODES:
            return error_code
    return None


def _is_llm_error_retryable(exc: Exception) -> bool:
    """只按结构化状态、错误码和已知异常类型判断是否重试。"""
    return _llm_retry_reason(exc) is not None


def _log_llm_retry(details: dict) -> None:
    """记录可运维字段，禁止把上游异常文本写入日志。"""
    exception = details.get("exception")
    error_type = type(exception).__name__ if isinstance(exception, BaseException) else "UnknownError"
    retry_reason = _llm_retry_reason(exception) if isinstance(exception, Exception) else None
    logger.warning(
        "LLM 调用失败后重试: attempt=%s wait_seconds=%.0f error_type=%s retry_reason=%s",
        details.get("tries", 0),
        details.get("wait", 0),
        error_type,
        retry_reason or "unclassified",
    )


def _usage_value(usage, key: str):
    if isinstance(usage, dict):
        return usage.get(key)
    return getattr(usage, key, None)


def _non_negative_int(*values) -> int | None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, float) and not value.is_integer():
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed >= 0:
            return parsed
    return None


def extract_usage(chunk) -> Optional[Usage]:
    usage = getattr(chunk, "usage", None)
    if not usage:
        return None
    prompt_details = _usage_value(usage, "prompt_tokens_details")
    return Usage(
        input_tokens=_non_negative_int(_usage_value(usage, "prompt_tokens")) or 0,
        output_tokens=_non_negative_int(_usage_value(usage, "completion_tokens")) or 0,
        cache_read_tokens=_non_negative_int(
            _usage_value(usage, "cache_read_tokens"),
            _usage_value(usage, "cache_read_input_tokens"),
            _usage_value(prompt_details, "cached_tokens"),
        ),
        cache_write_tokens=_non_negative_int(
            _usage_value(usage, "cache_write_tokens"),
            _usage_value(usage, "cache_creation_input_tokens"),
        ),
    )


def get_first_choice(chunk):
    return chunk.choices[0] if chunk.choices else None


def has_tool_call_delta(delta) -> bool:
    return bool(hasattr(delta, "tool_calls") and delta.tool_calls)


def accumulate_tool_calls(tool_calls_acc: dict[int, dict], delta) -> None:
    if not has_tool_call_delta(delta):
        return
    for tool_call in delta.tool_calls:
        idx = tool_call.index if hasattr(tool_call, "index") and tool_call.index is not None else 0
        if idx not in tool_calls_acc:
            tool_calls_acc[idx] = {"id": None, "name": None, "arguments": ""}
        if tool_call.id:
            tool_calls_acc[idx]["id"] = tool_call.id
        if tool_call.function and tool_call.function.name:
            tool_calls_acc[idx]["name"] = tool_call.function.name
        if tool_call.function and tool_call.function.arguments:
            tool_calls_acc[idx]["arguments"] += tool_call.function.arguments


def extract_reasoning_delta(delta, should_use_reasoning: bool) -> str:
    if not should_use_reasoning:
        return ""
    reasoning_delta = getattr(delta, "reasoning_content", None) or ""
    if not reasoning_delta and hasattr(delta, "model_extra") and delta.model_extra:
        reasoning_delta = delta.model_extra.get("reasoning_content", "") or ""
    return reasoning_delta


def extract_content_delta(delta, reasoning_delta: str) -> str:
    content_delta = delta.content or ""
    if reasoning_delta and content_delta == reasoning_delta:
        return ""
    return content_delta


def _pending_open_think_tag_start(text: str) -> int | None:
    search_end = len(text)
    while True:
        index = text.rfind("<", 0, search_end)
        if index < 0:
            return None
        tail = text[index:].lower()
        if "<think".startswith(tail) or (tail.startswith("<think") and ">" not in tail):
            return index
        search_end = index


def _pending_close_think_tag_start(text: str) -> int | None:
    lowered = text.lower()
    prefix = "</think>"
    max_prefix_length = min(len(lowered), len(prefix) - 1)
    for prefix_length in range(max_prefix_length, 0, -1):
        if lowered.endswith(prefix[:prefix_length]):
            return len(text) - prefix_length
    return None


def split_reasoning_tag_blocks(text: str) -> tuple[str, str]:
    """把正文通道里的 `<think>` 内容拆到推理通道，并隐藏跨 chunk 半截标签。"""

    reasoning: list[str] = []
    visible: list[str] = []
    cursor = 0
    while cursor < len(text):
        open_match = _OPEN_THINK_TAG_RE.search(text, cursor)
        if not open_match:
            remainder = text[cursor:]
            pending_start = _pending_open_think_tag_start(remainder)
            visible.append(remainder if pending_start is None else remainder[:pending_start])
            break

        visible.append(text[cursor : open_match.start()])
        close_match = _CLOSE_THINK_TAG_RE.search(text, open_match.end())
        if not close_match:
            pending_reasoning = text[open_match.end() :]
            pending_close_start = _pending_close_think_tag_start(pending_reasoning)
            reasoning.append(
                pending_reasoning if pending_close_start is None else pending_reasoning[:pending_close_start]
            )
            break
        reasoning.append(text[open_match.end() : close_match.start()])
        cursor = close_match.end()
    return "".join(reasoning), "".join(visible)


def strip_reasoning_tag_blocks(text: str) -> str:
    """移除被错误写入正文通道的 <think>...</think> 片段。"""
    return split_reasoning_tag_blocks(text)[1]


def strip_pending_dsml_tool_protocol(text: str, *, final: bool = False) -> str:
    """隐藏正文工具协议及末尾尚未收全的跨 chunk 前缀。"""
    hidden_starts: list[int] = []
    dsml_marker_index = text.find(_DSML_TOOL_CALLS_OPEN)
    if dsml_marker_index >= 0:
        hidden_starts.append(dsml_marker_index)
    elif (dsml_pending_start := _pending_dsml_tool_protocol_start(text)) is not None:
        pending_candidate = text[dsml_pending_start:]
        if not final or pending_candidate.startswith(_DSML_DISTINCT_PREFIX):
            hidden_starts.append(dsml_pending_start)

    generic_candidate = _generic_tool_protocol_candidate(text)
    if generic_candidate is not None:
        generic_start, generic_status = generic_candidate
        if not final or _generic_protocol_is_malformed(generic_status):
            hidden_starts.append(generic_start)

    return text if not hidden_starts else text[: min(hidden_starts)]


def sanitize_internal_mcp_aliases(text: str, *, final: bool = False) -> str:
    """兼容旧调用名；正文只隐藏运行级 alias，不改写普通技术文本。"""

    return sanitize_internal_tool_names(text, final=final, include_named_tools=False)


def _visible_delta(visible_text: str, emitted_text: str, *, channel: str) -> str:
    if visible_text.startswith(emitted_text):
        return visible_text[len(emitted_text) :]

    logger.warning(f"{channel} 内容过滤出现非单调输出，已忽略不可追加部分")
    return ""


def _common_prefix_length(left: str, right: str) -> int:
    length = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        length += 1
    return length


def _looks_like_reasoning_snapshot_transition(previous: str, incoming: str) -> bool:
    if not previous or not incoming:
        return False
    if incoming == previous or incoming.startswith(previous):
        return True

    shorter_length = min(len(previous), len(incoming))
    if shorter_length < _REASONING_SNAPSHOT_MIN_LENGTH:
        return False
    return _common_prefix_length(previous, incoming) / shorter_length >= 0.8


def _resolve_reasoning_probe(state: LLMStreamState, *, final: bool = False) -> None:
    chunks = state.reasoning_probe_chunks
    if not chunks:
        return

    transitions_are_snapshots = len(chunks) >= 2 and all(
        _looks_like_reasoning_snapshot_transition(previous, incoming) for previous, incoming in zip(chunks, chunks[1:])
    )
    enough_snapshot_evidence = (
        transitions_are_snapshots
        and len(chunks) >= (_REASONING_SNAPSHOT_PROBE_CHUNKS if not final else 3)
        and len(chunks[-1]) >= _REASONING_SNAPSHOT_MIN_LENGTH
    )
    if enough_snapshot_evidence:
        state.reasoning_transport_mode = "snapshot"
        state.raw_reasoning_buf = chunks[-1]
    else:
        state.reasoning_transport_mode = "delta"
        state.raw_reasoning_buf = "".join(chunks)
    chunks.clear()


def _merge_reasoning_delta(
    state: LLMStreamState,
    incoming: str,
) -> None:
    if state.reasoning_transport_mode == "delta":
        state.raw_reasoning_buf += incoming
        return
    if state.reasoning_transport_mode == "snapshot":
        if (
            state.raw_reasoning_buf
            and not incoming.startswith(state.raw_reasoning_buf)
            and not state.reasoning_snapshot_revision_logged
        ):
            logger.warning("reasoning 累计快照发生修订，已更新原文并保持已发送内容单调")
            state.reasoning_snapshot_revision_logged = True
        state.raw_reasoning_buf = incoming
        return

    state.reasoning_probe_chunks.append(incoming)
    chunks = state.reasoning_probe_chunks
    if len(chunks) >= 2 and not _looks_like_reasoning_snapshot_transition(chunks[-2], chunks[-1]):
        _resolve_reasoning_probe(state)
    elif len(chunks) >= _REASONING_SNAPSHOT_PROBE_CHUNKS:
        _resolve_reasoning_probe(state)


def _pending_dsml_tool_protocol_start(text: str) -> int | None:
    return _pending_protocol_prefix_start(text, _DSML_TOOL_CALLS_OPEN)


def _pending_protocol_prefix_start(text: str, prefix: str) -> int | None:
    max_prefix_length = min(len(text), len(prefix) - 1)
    for prefix_length in range(max_prefix_length, 0, -1):
        if text.endswith(prefix[:prefix_length]):
            return len(text) - prefix_length
    return None


def _generic_tool_protocol_candidate(text: str) -> tuple[int, str] | None:
    search_start = 0
    while (match := _GENERIC_TOOL_CALL_PREFIX_RE.search(text, search_start)) is not None:
        marker_start = match.start()
        if _is_markdown_code_position(text, marker_start):
            search_start = match.end()
            continue

        tag_end = text.find(">", match.end())
        if tag_end < 0:
            return marker_start, "pending_tag"

        body = text[tag_end + 1 :].lstrip()
        if not body:
            return marker_start, "pending_body"
        lowered_body = body.lower()
        if _GENERIC_FUNCTION_OPEN_PREFIX.startswith(lowered_body):
            return marker_start, "pending_function"
        if lowered_body.startswith(_GENERIC_FUNCTION_OPEN_PREFIX):
            status = "confirmed" if _GENERIC_FUNCTION_OPEN_RE.match(body) else "pending_function"
            return marker_start, status
        search_start = match.end()

    partial_start = _pending_protocol_prefix_start(text.lower(), _GENERIC_TOOL_CALL_OPEN)
    if partial_start is not None and not _is_markdown_code_position(text, partial_start):
        return partial_start, "partial_prefix"
    return None


def _generic_protocol_is_malformed(status: str) -> bool:
    return status in {"confirmed", "pending_body", "pending_function", "pending_tag", "partial_prefix"}


def _is_markdown_code_position(text: str, index: int) -> bool:
    before = text[:index]
    if before.count("```") % 2 == 1:
        return True
    without_fences = before.replace("```", "")
    return without_fences.count("`") % 2 == 1


def _decode_dsml_parameter(value: str, *, is_string: bool):
    if is_string:
        return value
    try:
        return json.loads(value.strip())
    except (json.JSONDecodeError, TypeError):
        return value.strip()


def parse_dsml_tool_calls(text: str, *, id_prefix: str) -> list[dict]:
    """把兼容模型误写进正文通道的 DSML 工具协议还原为标准 tool_calls。"""
    candidate = strip_reasoning_tag_blocks(text).lstrip()
    if not candidate.startswith(_DSML_TOOL_CALLS_OPEN):
        return []

    close_index = candidate.find(_DSML_TOOL_CALLS_CLOSE, len(_DSML_TOOL_CALLS_OPEN))
    if close_index < 0:
        return []
    protocol_end = close_index + len(_DSML_TOOL_CALLS_CLOSE)
    if candidate[protocol_end:].strip():
        return []
    protocol_body = candidate[len(_DSML_TOOL_CALLS_OPEN) : close_index]
    calls: list[dict] = []
    invoke_cursor = 0
    for call_index, invoke_match in enumerate(_DSML_INVOKE_RE.finditer(protocol_body), 1):
        if protocol_body[invoke_cursor : invoke_match.start()].strip():
            return []
        arguments = {}
        parameter_cursor = 0
        for parameter_match in _DSML_PARAMETER_RE.finditer(invoke_match.group("body")):
            if invoke_match.group("body")[parameter_cursor : parameter_match.start()].strip():
                return []
            parameter_name = parameter_match.group("name")
            if parameter_name in arguments:
                return []
            arguments[parameter_name] = _decode_dsml_parameter(
                parameter_match.group("value"),
                is_string=parameter_match.group("is_string").lower() == "true",
            )
            parameter_cursor = parameter_match.end()
        if invoke_match.group("body")[parameter_cursor:].strip():
            return []
        calls.append(
            {
                "id": f"dsml-{id_prefix}-{call_index}",
                "name": invoke_match.group("name"),
                "arguments": json.dumps(arguments, ensure_ascii=False),
            }
        )
        invoke_cursor = invoke_match.end()
    if protocol_body[invoke_cursor:].strip():
        return []
    return calls


def filter_reasoning_tag_content_delta(state: LLMStreamState, content_delta: str) -> tuple[str, str]:
    """基于完整原始正文重算标签推理和可见正文。"""
    if not content_delta:
        return "", ""
    state.raw_content_buf += content_delta
    tag_reasoning, visible_content = split_reasoning_tag_blocks(state.raw_content_buf)
    if tag_reasoning.startswith(state.raw_tag_reasoning_buf):
        tag_reasoning_delta = tag_reasoning[len(state.raw_tag_reasoning_buf) :]
        state.raw_tag_reasoning_buf = tag_reasoning
    else:
        logger.warning("thinking 标签内容出现非单调输出，已忽略不可追加部分")
        tag_reasoning_delta = ""
    visible_content = strip_pending_dsml_tool_protocol(visible_content)
    visible_content = sanitize_internal_mcp_aliases(visible_content)
    return (
        tag_reasoning_delta,
        _visible_delta(visible_content, state.content_buf, channel="answering"),
    )


def filter_internal_mcp_reasoning_delta(
    state: LLMStreamState,
    reasoning_delta: str,
) -> str:
    if not reasoning_delta:
        return ""
    _merge_reasoning_delta(state, reasoning_delta)
    visible_reasoning = strip_pending_dsml_tool_protocol(state.raw_reasoning_buf)
    visible_reasoning = sanitize_user_visible_reasoning(
        visible_reasoning,
        buffer_trailing_paragraph=state.reasoning_transport_mode == "probing",
    )
    return _visible_delta(visible_reasoning, state.reasoning_buf, channel="reasoning")


def _final_reasoning_candidate(state: LLMStreamState) -> str:
    raw_reasoning = (
        "".join(state.reasoning_probe_chunks)
        if state.reasoning_transport_mode == "probing"
        else state.raw_reasoning_buf
    )
    candidate = strip_pending_dsml_tool_protocol(raw_reasoning, final=True)
    return sanitize_user_visible_reasoning(candidate, final=True)


async def flush_pending_internal_mcp_aliases(*, request: LLMStreamRequest, state: LLMStreamState) -> None:
    if state.reasoning_transport_mode == "probing":
        _resolve_reasoning_probe(state, final=True)
    final_reasoning = strip_pending_dsml_tool_protocol(state.raw_reasoning_buf, final=True)
    final_reasoning = sanitize_user_visible_reasoning(final_reasoning, final=True)
    final_content = strip_reasoning_tag_blocks(state.raw_content_buf)
    final_content = strip_pending_dsml_tool_protocol(final_content, final=True)
    final_content = sanitize_internal_mcp_aliases(final_content, final=True)
    await append_reasoning_and_content(
        request=request,
        state=state,
        reasoning_delta=_visible_delta(final_reasoning, state.reasoning_buf, channel="reasoning"),
        content_delta=_visible_delta(final_content, state.content_buf, channel="answering"),
        reasoning_candidate_time=state.pending_reasoning_candidate_time,
        content_candidate_time=(
            request.capture_output_candidate_time()
            if request.capture_output_candidate_time is not None
            else None
        ),
    )


async def append_stream_delta(
    *,
    request: LLMStreamRequest,
    chunk_type: str,
    content: str,
    block_id: str,
) -> None:
    await append_chunk(
        request.conversation_id,
        chunk_type,
        content,
        block_id,
        task_id=request.task_id,
        run_id=request.run_id,
        step_id=request.step_id,
    )


async def append_reasoning_and_content(
    *,
    request: LLMStreamRequest,
    state: LLMStreamState,
    reasoning_delta: str,
    content_delta: str,
    reasoning_candidate_time: float | None = None,
    content_candidate_time: float | None = None,
) -> None:
    if reasoning_delta:
        if request.on_output_candidate is not None:
            request.on_output_candidate("reasoning", reasoning_candidate_time)
        state.reasoning_buf += reasoning_delta
        deferred_reasoning_is_allowed = request.allow_deferred_reasoning_output
        if not request.defer_output or deferred_reasoning_is_allowed:
            await append_stream_delta(
                request=request,
                chunk_type="reasoning",
                content=reasoning_delta,
                block_id=request.thinking_block_id,
            )
            if request.on_visible_output is not None:
                await request.on_visible_output("reasoning")
        if request.partial_output is not None:
            request.partial_output["reasoning_buf"] = state.reasoning_buf
    if content_delta:
        if request.on_output_candidate is not None:
            request.on_output_candidate("content", content_candidate_time)
        state.content_buf += content_delta
        if not request.defer_output:
            await append_stream_delta(
                request=request,
                chunk_type="answering",
                content=content_delta,
                block_id=request.text_block_id,
            )
            if request.on_visible_output is not None:
                await request.on_visible_output("content")
        if request.partial_output is not None:
            request.partial_output["content_buf"] = state.content_buf


async def maybe_check_lock_owner(*, request: LLMStreamRequest, state: LLMStreamState) -> bool:
    state.chunk_count += 1
    if state.chunk_count % LOCK_CHECK_INTERVAL != 0:
        return True
    if await check_lock_owner(request.conversation_id, request.task_id):
        return True
    logger.info(f"流式调用被踢掉: conv_id={request.conversation_id}")
    state.finish_reason = "cancelled"
    return False


def build_tool_calls_list(tool_calls_acc: dict[int, dict]) -> list[dict]:
    return [tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys()) if tool_calls_acc[idx]["name"]]


async def process_stream_choice(*, request: LLMStreamRequest, state: LLMStreamState, choice, chunk) -> bool:
    delta = choice.delta
    finish_reason = choice.finish_reason

    candidate_time = (
        request.capture_output_candidate_time()
        if request.capture_output_candidate_time is not None
        else None
    )
    accumulate_tool_calls(state.tool_calls_acc, delta)
    if getattr(delta, "tool_calls", None) and request.on_output_candidate is not None:
        request.on_output_candidate("tool_call", candidate_time)
    raw_reasoning_delta = extract_reasoning_delta(delta, request.should_use_reasoning)
    content_delta = extract_content_delta(delta, raw_reasoning_delta)
    tag_reasoning_delta, content_delta = filter_reasoning_tag_content_delta(state, content_delta)
    previous_reasoning_candidate = _final_reasoning_candidate(state)
    reasoning_delta = filter_internal_mcp_reasoning_delta(
        state,
        raw_reasoning_delta + (tag_reasoning_delta if request.should_use_reasoning else ""),
    )
    if (
        state.pending_reasoning_candidate_time is None
        and _final_reasoning_candidate(state) != previous_reasoning_candidate
    ):
        state.pending_reasoning_candidate_time = candidate_time
    await append_reasoning_and_content(
        request=request,
        state=state,
        reasoning_delta=reasoning_delta,
        content_delta=content_delta,
        reasoning_candidate_time=state.pending_reasoning_candidate_time,
        content_candidate_time=candidate_time,
    )
    if reasoning_delta:
        state.pending_reasoning_candidate_time = None

    usage_data = extract_usage(chunk)
    if usage_data:
        state.usage_data = usage_data
    if finish_reason == "tool_calls":
        state.finish_reason = "tool_calls"
    elif finish_reason == "stop":
        state.finish_reason = "stop"

    return await maybe_check_lock_owner(request=request, state=state)


async def consume_stream_round(response, request: LLMStreamRequest) -> LLMStreamOutcome:
    state = LLMStreamState(
        reasoning_transport_mode=initial_reasoning_transport_mode(
            model_id=request.model_id,
            explicit_mode=request.reasoning_transport_mode,
        )
    )
    async for chunk in response:
        choice = get_first_choice(chunk)
        if choice is None:
            usage_data = extract_usage(chunk)
            if usage_data:
                state.usage_data = usage_data
            continue
        should_continue = await process_stream_choice(request=request, state=state, choice=choice, chunk=chunk)
        if not should_continue:
            break

    await flush_pending_internal_mcp_aliases(request=request, state=state)

    visible_content = strip_reasoning_tag_blocks(state.raw_content_buf)
    dsml_marker_index = visible_content.find(_DSML_TOOL_CALLS_OPEN)
    dsml_pending_start = _pending_dsml_tool_protocol_start(visible_content) if dsml_marker_index < 0 else None
    generic_candidate = _generic_tool_protocol_candidate(visible_content)
    has_malformed_protocol = dsml_marker_index >= 0 or (
        dsml_pending_start is not None and visible_content[dsml_pending_start:].startswith(_DSML_DISTINCT_PREFIX)
    )
    if generic_candidate is not None:
        _generic_start, generic_status = generic_candidate
        has_malformed_protocol = has_malformed_protocol or _generic_protocol_is_malformed(generic_status)

    tool_calls = build_tool_calls_list(state.tool_calls_acc)
    if not tool_calls:
        raw_id_prefix = request.step_id or request.task_id
        id_prefix = re.sub(r"[^A-Za-z0-9_-]+", "-", raw_id_prefix).strip("-") or "round"
        tool_calls = parse_dsml_tool_calls(state.raw_content_buf, id_prefix=id_prefix)
        if tool_calls:
            logger.warning(
                "检测到正文通道中的 DSML 工具协议，已转换为标准工具调用: "
                f"conv_id={request.conversation_id} step_id={request.step_id} calls={len(tool_calls)}"
            )
            state.finish_reason = "tool_calls"
            has_malformed_protocol = False
        elif has_malformed_protocol:
            logger.warning(
                "检测到无法完整解析的 DSML 工具协议，已拒绝写入与执行: "
                f"conv_id={request.conversation_id} step_id={request.step_id}"
            )
            state.finish_reason = "tool_protocol_error"

    return LLMStreamOutcome(
        reasoning_buf=state.reasoning_buf,
        raw_reasoning_buf=state.raw_reasoning_buf,
        content_buf=state.content_buf,
        raw_content_buf=state.raw_content_buf,
        tool_calls=tool_calls,
        finish_reason=state.finish_reason,
        usage_data=state.usage_data,
    )


async def stream_round(
    response,
    conversation_id: str,
    task_id: str,
    should_use_reasoning: bool,
    thinking_block_id: str,
    text_block_id: str,
    provider: Optional[str] = None,
    model_id: Optional[str] = None,
    reasoning_transport_mode: Optional[ReasoningTransportMode] = None,
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    defer_output: bool = False,
    allow_deferred_reasoning_output: bool = False,
    partial_output: dict[str, str] | None = None,
    on_visible_output: Callable[[str], Awaitable[None]] | None = None,
    on_output_candidate: Callable[[str, float | None], None] | None = None,
    capture_output_candidate_time: Callable[[], float | None] | None = None,
) -> tuple[str, str, list[dict], str, Optional[Usage]]:
    """
    通用 LLM 流式响应处理。
    返回 (reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)。
    tool_calls_list 格式: [{"id": str, "name": str, "arguments": str}, ...]

    run_id / step_id 透传给 reasoning / answering chunk，让 FE 把 token 流挂回
    agent_event 控制面对应的 step（spec §4.6）。两者均可为空（非 agent 路径）。
    """
    outcome = await consume_stream_round(
        response,
        LLMStreamRequest(
            conversation_id=conversation_id,
            task_id=task_id,
            should_use_reasoning=should_use_reasoning,
            thinking_block_id=thinking_block_id,
            text_block_id=text_block_id,
            provider=provider,
            model_id=model_id,
            reasoning_transport_mode=reasoning_transport_mode,
            run_id=run_id,
            step_id=step_id,
            defer_output=defer_output,
            allow_deferred_reasoning_output=allow_deferred_reasoning_output,
            partial_output=partial_output,
            on_visible_output=on_visible_output,
            on_output_candidate=on_output_candidate,
            capture_output_candidate_time=capture_output_candidate_time,
        ),
    )
    return StreamRoundTuple(
        outcome.reasoning_buf,
        outcome.content_buf,
        outcome.tool_calls,
        outcome.finish_reason,
        outcome.usage_data,
        protocol_reasoning_buf=outcome.raw_reasoning_buf,
        protocol_content_buf=strip_pending_dsml_tool_protocol(outcome.raw_content_buf, final=True),
    )


@backoff.on_exception(
    backoff.constant,
    Exception,
    max_tries=AGENT_LLM_MAX_RETRIES + 1,
    interval=2,
    giveup=lambda e: not _is_llm_error_retryable(e),
    on_backoff=_log_llm_retry,
)
async def llm_call_with_retry(
    litellm_model: str,
    litellm_kwargs: dict,
    messages: list[dict],
    **call_kwargs,
):
    """带重试的 LLM 调用，返回 streaming response。

    可重试错误只接受结构化 429 / 502 / 503 / 504、已知连接/超时异常
    和明确错误码，固定 2s 间隔。其它错误立即抛出。
    """
    merged_call_kwargs = merge_litellm_kwargs(
        "chat_stream",
        {
            **litellm_kwargs,
            **call_kwargs,
        },
    )
    return await litellm.acompletion(
        model=litellm_model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **merged_call_kwargs,
    )
