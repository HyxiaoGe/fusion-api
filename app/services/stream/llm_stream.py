"""LLM SSE 消费 + 调用重试。

spec §4.2。stream_round 把 litellm streaming response 消费成
(reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)；
每个 delta 通过 append_chunk 写 Redis Stream，每 LOCK_CHECK_INTERVAL
个 chunk 检查锁所有权（被踢则提前返回 finish_reason="cancelled"）。
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional

import backoff
import litellm

from app.ai.llm_observability import merge_litellm_kwargs
from app.core.logger import app_logger as logger
from app.schemas.chat import Usage
from app.services.stream_state_service import append_chunk, check_lock_owner

# 每 N 个 chunk 检查一次锁状态
LOCK_CHECK_INTERVAL = 20

# LLM 调用重试次数
AGENT_LLM_MAX_RETRIES = 1

# 可重试的错误关键字
_LLM_RETRYABLE_KEYWORDS = ("429", "rate", "503", "502", "timeout")

_OPEN_THINK_TAG_RE = re.compile(r"<think\b[^>]*>", re.IGNORECASE)
_CLOSE_THINK_TAG_RE = re.compile(r"</think\s*>", re.IGNORECASE)

_DSML_TOOL_CALLS_OPEN = "<｜｜DSML｜｜tool_calls>"
_DSML_TOOL_CALLS_CLOSE = "</｜｜DSML｜｜tool_calls>"
_DSML_DISTINCT_PREFIX = "<｜｜DSML"
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
_MCP_ALIAS_PREFIX = "mcp_"
_MCP_ALIAS_TOKEN_LENGTH = 43
_MCP_ALIAS_RE = re.compile(rf"{_MCP_ALIAS_PREFIX}[A-Za-z0-9_-]{{{_MCP_ALIAS_TOKEN_LENGTH}}}")
_MCP_ALIAS_PARTIAL_RE = re.compile(rf"{_MCP_ALIAS_PREFIX}[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class LLMStreamRequest:
    conversation_id: str
    task_id: str
    should_use_reasoning: bool
    thinking_block_id: str
    text_block_id: str
    run_id: Optional[str] = None
    step_id: Optional[str] = None
    defer_output: bool = False


@dataclass
class LLMStreamState:
    reasoning_buf: str = ""
    raw_reasoning_buf: str = ""
    content_buf: str = ""
    raw_content_buf: str = ""
    usage_data: Optional[Usage] = None
    chunk_count: int = 0
    finish_reason: str = "stop"
    tool_calls_acc: dict[int, dict] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStreamOutcome:
    reasoning_buf: str
    content_buf: str
    tool_calls: list[dict]
    finish_reason: str
    usage_data: Optional[Usage]


def _is_llm_error_retryable(exc: Exception) -> bool:
    err = str(exc).lower()
    return any(kw in err for kw in _LLM_RETRYABLE_KEYWORDS)


def extract_usage(chunk) -> Optional[Usage]:
    if not hasattr(chunk, "usage") or not chunk.usage:
        return None
    return Usage(
        input_tokens=chunk.usage.prompt_tokens or 0,
        output_tokens=chunk.usage.completion_tokens or 0,
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


def strip_reasoning_tag_blocks(text: str) -> str:
    """移除被错误写入正文通道的 <think>...</think> 片段。"""
    output: list[str] = []
    cursor = 0
    while cursor < len(text):
        open_match = _OPEN_THINK_TAG_RE.search(text, cursor)
        if not open_match:
            remainder = text[cursor:]
            pending_start = _pending_open_think_tag_start(remainder)
            output.append(remainder if pending_start is None else remainder[:pending_start])
            break

        output.append(text[cursor : open_match.start()])
        close_match = _CLOSE_THINK_TAG_RE.search(text, open_match.end())
        if not close_match:
            break
        cursor = close_match.end()
    return "".join(output)


def strip_pending_dsml_tool_protocol(text: str) -> str:
    """隐藏 DSML 工具协议及末尾尚未收全的跨 chunk 前缀。"""
    marker_index = text.find(_DSML_TOOL_CALLS_OPEN)
    if marker_index >= 0:
        return text[:marker_index]

    pending_start = _pending_dsml_tool_protocol_start(text)
    return text if pending_start is None else text[:pending_start]


def _pending_mcp_alias_start(text: str) -> int | None:
    max_suffix_length = min(len(text), len(_MCP_ALIAS_PREFIX) + _MCP_ALIAS_TOKEN_LENGTH - 1)
    for suffix_length in range(max_suffix_length, 0, -1):
        suffix = text[-suffix_length:]
        if _MCP_ALIAS_PREFIX.startswith(suffix):
            return len(text) - suffix_length
        if (
            suffix.startswith(_MCP_ALIAS_PREFIX)
            and len(suffix) < len(_MCP_ALIAS_PREFIX) + _MCP_ALIAS_TOKEN_LENGTH
            and all(char.isalnum() or char in {"_", "-"} for char in suffix[len(_MCP_ALIAS_PREFIX) :])
        ):
            return len(text) - suffix_length
    return None


def sanitize_internal_mcp_aliases(text: str, *, final: bool = False) -> str:
    """内部运行级 MCP alias 不得进入用户可见 reasoning/answering。"""
    visible_source = text
    if not final:
        pending_start = _pending_mcp_alias_start(text)
        if pending_start is not None:
            visible_source = text[:pending_start]
    sanitized = _MCP_ALIAS_RE.sub("外部工具", visible_source)
    if final:
        sanitized = _MCP_ALIAS_PARTIAL_RE.sub("外部工具", sanitized)
    return sanitized


def _visible_delta(visible_text: str, emitted_text: str, *, channel: str) -> str:
    if visible_text.startswith(emitted_text):
        return visible_text[len(emitted_text) :]

    common_prefix_length = 0
    for left, right in zip(visible_text, emitted_text):
        if left != right:
            break
        common_prefix_length += 1
    logger.warning(f"{channel} 内容过滤出现非单调输出，已保留可追加部分")
    return visible_text[common_prefix_length:]


def _pending_dsml_tool_protocol_start(text: str) -> int | None:
    max_prefix_length = min(len(text), len(_DSML_TOOL_CALLS_OPEN) - 1)
    for prefix_length in range(max_prefix_length, 0, -1):
        if text.endswith(_DSML_TOOL_CALLS_OPEN[:prefix_length]):
            return len(text) - prefix_length
    return None


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


def filter_reasoning_tag_content_delta(state: LLMStreamState, content_delta: str) -> str:
    """基于完整原始正文重算可见正文，避免跨 chunk 的 <think> 前缀先泄漏。"""
    if not content_delta:
        return ""
    state.raw_content_buf += content_delta
    visible_content = strip_reasoning_tag_blocks(state.raw_content_buf)
    visible_content = strip_pending_dsml_tool_protocol(visible_content)
    visible_content = sanitize_internal_mcp_aliases(visible_content)
    return _visible_delta(visible_content, state.content_buf, channel="answering")


def filter_internal_mcp_reasoning_delta(state: LLMStreamState, reasoning_delta: str) -> str:
    if not reasoning_delta:
        return ""
    state.raw_reasoning_buf += reasoning_delta
    visible_reasoning = sanitize_internal_mcp_aliases(state.raw_reasoning_buf)
    return _visible_delta(visible_reasoning, state.reasoning_buf, channel="reasoning")


async def flush_pending_internal_mcp_aliases(*, request: LLMStreamRequest, state: LLMStreamState) -> None:
    final_reasoning = sanitize_internal_mcp_aliases(state.raw_reasoning_buf, final=True)
    final_content = strip_reasoning_tag_blocks(state.raw_content_buf)
    final_content = strip_pending_dsml_tool_protocol(final_content)
    final_content = sanitize_internal_mcp_aliases(final_content, final=True)
    await append_reasoning_and_content(
        request=request,
        state=state,
        reasoning_delta=_visible_delta(final_reasoning, state.reasoning_buf, channel="reasoning"),
        content_delta=_visible_delta(final_content, state.content_buf, channel="answering"),
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
) -> None:
    if reasoning_delta:
        state.reasoning_buf += reasoning_delta
        if not request.defer_output:
            await append_stream_delta(
                request=request,
                chunk_type="reasoning",
                content=reasoning_delta,
                block_id=request.thinking_block_id,
            )
    if content_delta:
        state.content_buf += content_delta
        if not request.defer_output:
            await append_stream_delta(
                request=request,
                chunk_type="answering",
                content=content_delta,
                block_id=request.text_block_id,
            )


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

    accumulate_tool_calls(state.tool_calls_acc, delta)
    if finish_reason == "tool_calls":
        state.finish_reason = "tool_calls"
        return True

    if has_tool_call_delta(delta):
        return True

    raw_reasoning_delta = extract_reasoning_delta(delta, request.should_use_reasoning)
    content_delta = extract_content_delta(delta, raw_reasoning_delta)
    reasoning_delta = filter_internal_mcp_reasoning_delta(state, raw_reasoning_delta)
    content_delta = filter_reasoning_tag_content_delta(state, content_delta)
    await append_reasoning_and_content(
        request=request,
        state=state,
        reasoning_delta=reasoning_delta,
        content_delta=content_delta,
    )

    usage_data = extract_usage(chunk)
    if usage_data:
        state.usage_data = usage_data
    if finish_reason == "stop":
        state.finish_reason = "stop"

    return await maybe_check_lock_owner(request=request, state=state)


async def consume_stream_round(response, request: LLMStreamRequest) -> LLMStreamOutcome:
    state = LLMStreamState()
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
    marker_index = visible_content.find(_DSML_TOOL_CALLS_OPEN)
    pending_start = _pending_dsml_tool_protocol_start(visible_content) if marker_index < 0 else None
    has_malformed_protocol = marker_index >= 0
    if marker_index < 0 and pending_start is not None:
        pending_candidate = visible_content[pending_start:]
        has_malformed_protocol = pending_candidate.startswith(_DSML_DISTINCT_PREFIX)
        if not has_malformed_protocol:
            safe_visible_content = sanitize_internal_mcp_aliases(visible_content, final=True)
            pending_delta = safe_visible_content[len(state.content_buf) :]
            await append_reasoning_and_content(
                request=request,
                state=state,
                reasoning_delta="",
                content_delta=pending_delta,
            )

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
        content_buf=state.content_buf,
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
    run_id: Optional[str] = None,
    step_id: Optional[str] = None,
    defer_output: bool = False,
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
            run_id=run_id,
            step_id=step_id,
            defer_output=defer_output,
        ),
    )
    return (
        outcome.reasoning_buf,
        outcome.content_buf,
        outcome.tool_calls,
        outcome.finish_reason,
        outcome.usage_data,
    )


@backoff.on_exception(
    backoff.constant,
    Exception,
    max_tries=AGENT_LLM_MAX_RETRIES + 1,
    interval=2,
    giveup=lambda e: not _is_llm_error_retryable(e),
    on_backoff=lambda d: logger.warning(
        f"LLM 调用失败（第 {d['tries']} 次），{d['wait']:.0f}s 后重试: {d['exception']}"
    ),
)
async def llm_call_with_retry(
    litellm_model: str,
    litellm_kwargs: dict,
    messages: list[dict],
    **call_kwargs,
):
    """带重试的 LLM 调用，返回 streaming response。

    可重试错误：429 / rate / 503 / 502 / timeout，固定 2s 间隔。
    其它错误立即抛出。重试逻辑由 @backoff.on_exception 装饰器实现。
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
