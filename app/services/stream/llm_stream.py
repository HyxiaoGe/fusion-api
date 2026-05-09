"""LLM SSE 消费 + 调用重试。

spec §4.2。stream_round 把 litellm streaming response 消费成
(reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data)；
每个 delta 通过 append_chunk 写 Redis Stream，每 LOCK_CHECK_INTERVAL
个 chunk 检查锁所有权（被踢则提前返回 finish_reason="cancelled"）。
"""

import asyncio
from typing import Optional

import litellm

from app.core.logger import app_logger as logger
from app.schemas.chat import Usage
from app.services.stream_state_service import append_chunk, check_lock_owner

# 每 N 个 chunk 检查一次锁状态
LOCK_CHECK_INTERVAL = 20

# LLM 调用重试次数
AGENT_LLM_MAX_RETRIES = 1


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
    reasoning_buf = ""
    content_buf = ""
    usage_data: Optional[Usage] = None
    chunk_count = 0
    finish_reason = "stop"

    # tool_call 累积缓冲区（支持多个并行 tool_calls）
    tool_calls_acc: dict[int, dict] = {}  # index → {"id", "name", "arguments"}

    async for chunk in response:
        choice = chunk.choices[0] if chunk.choices else None

        if not choice:
            if hasattr(chunk, "usage") and chunk.usage:
                usage_data = Usage(
                    input_tokens=chunk.usage.prompt_tokens or 0,
                    output_tokens=chunk.usage.completion_tokens or 0,
                )
            continue

        delta = choice.delta
        fr = choice.finish_reason

        # ===== tool_call 累积（支持多个）=====
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc in delta.tool_calls:
                idx = tc.index if hasattr(tc, "index") and tc.index is not None else 0
                if idx not in tool_calls_acc:
                    tool_calls_acc[idx] = {"id": None, "name": None, "arguments": ""}
                if tc.id:
                    tool_calls_acc[idx]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls_acc[idx]["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls_acc[idx]["arguments"] += tc.function.arguments

        if fr == "tool_calls":
            finish_reason = "tool_calls"
            continue

        if hasattr(delta, "tool_calls") and delta.tool_calls:
            continue

        # ===== reasoning + content =====
        reasoning_delta = ""
        if should_use_reasoning:
            reasoning_delta = getattr(delta, "reasoning_content", None) or ""
            if not reasoning_delta and hasattr(delta, "model_extra") and delta.model_extra:
                reasoning_delta = delta.model_extra.get("reasoning_content", "") or ""

        content_delta = delta.content or ""

        if reasoning_delta and content_delta == reasoning_delta:
            content_delta = ""

        if reasoning_delta:
            reasoning_buf += reasoning_delta
            await append_chunk(
                conversation_id,
                "reasoning",
                reasoning_delta,
                thinking_block_id,
                run_id=run_id,
                step_id=step_id,
            )

        if content_delta:
            content_buf += content_delta
            await append_chunk(
                conversation_id,
                "answering",
                content_delta,
                text_block_id,
                run_id=run_id,
                step_id=step_id,
            )

        if hasattr(chunk, "usage") and chunk.usage:
            usage_data = Usage(
                input_tokens=chunk.usage.prompt_tokens or 0,
                output_tokens=chunk.usage.completion_tokens or 0,
            )

        if fr == "stop":
            finish_reason = "stop"

        chunk_count += 1
        if chunk_count % LOCK_CHECK_INTERVAL == 0:
            if not await check_lock_owner(conversation_id, task_id):
                logger.info(f"流式调用被踢掉: conv_id={conversation_id}")
                finish_reason = "cancelled"
                break

    tool_calls_list = [tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys()) if tool_calls_acc[idx]["name"]]

    return reasoning_buf, content_buf, tool_calls_list, finish_reason, usage_data


async def llm_call_with_retry(
    litellm_model: str,
    litellm_kwargs: dict,
    messages: list[dict],
    max_retries: int = AGENT_LLM_MAX_RETRIES,
    **call_kwargs,
):
    """带重试的 LLM 调用，返回 streaming response。

    可重试错误：429 / rate / 503 / 502 / timeout，固定 2s 间隔。
    其它错误立即抛出。
    """
    for attempt in range(max_retries + 1):
        try:
            return await litellm.acompletion(
                model=litellm_model,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
                **litellm_kwargs,
                **call_kwargs,
            )
        except Exception as e:
            error_str = str(e).lower()
            is_retryable = any(kw in error_str for kw in ["429", "rate", "503", "502", "timeout"])
            if is_retryable and attempt < max_retries:
                logger.warning(f"LLM 调用失败（{attempt + 1}/{max_retries + 1}），2s 后重试: {e}")
                await asyncio.sleep(2)
                continue
            raise
