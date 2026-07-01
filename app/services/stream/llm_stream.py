"""LLM SSE 消费 + 调用重试。

spec §4.2。stream_round 把 litellm streaming response 消费成
(reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)；
每个 delta 通过 append_chunk 写 Redis Stream，每 LOCK_CHECK_INTERVAL
个 chunk 检查锁所有权（被踢则提前返回 finish_reason="cancelled"）。
"""

import re
from dataclasses import dataclass, field
from typing import Optional

import backoff
import litellm

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


@dataclass(frozen=True)
class LLMStreamRequest:
    conversation_id: str
    task_id: str
    should_use_reasoning: bool
    thinking_block_id: str
    text_block_id: str
    run_id: Optional[str] = None
    step_id: Optional[str] = None


@dataclass
class LLMStreamState:
    reasoning_buf: str = ""
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


def filter_reasoning_tag_content_delta(state: LLMStreamState, content_delta: str) -> str:
    """基于完整原始正文重算可见正文，避免跨 chunk 的 <think> 前缀先泄漏。"""
    if not content_delta:
        return ""
    state.raw_content_buf += content_delta
    visible_content = strip_reasoning_tag_blocks(state.raw_content_buf)
    if visible_content.startswith(state.content_buf):
        return visible_content[len(state.content_buf) :]

    common_prefix_length = 0
    for left, right in zip(visible_content, state.content_buf):
        if left != right:
            break
        common_prefix_length += 1
    logger.warning("answering 内容过滤出现非单调输出，已保留可追加部分")
    return visible_content[common_prefix_length:]


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
        await append_stream_delta(
            request=request,
            chunk_type="reasoning",
            content=reasoning_delta,
            block_id=request.thinking_block_id,
        )
    if content_delta:
        state.content_buf += content_delta
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

    reasoning_delta = extract_reasoning_delta(delta, request.should_use_reasoning)
    content_delta = extract_content_delta(delta, reasoning_delta)
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

    return LLMStreamOutcome(
        reasoning_buf=state.reasoning_buf,
        content_buf=state.content_buf,
        tool_calls=build_tool_calls_list(state.tool_calls_acc),
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
    return await litellm.acompletion(
        model=litellm_model,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
        **litellm_kwargs,
        **call_kwargs,
    )
