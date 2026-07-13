"""上下文状态的流式安全协议适配。"""

from __future__ import annotations

from inspect import iscoroutinefunction
from typing import Any

from app.schemas.chat import ContextUsage, Usage
from app.services.chat.context_manager import ContextPlan


def build_context_usage(
    plan: ContextPlan,
    usage: Usage | None = None,
    *,
    round_index: int | None = None,
) -> ContextUsage:
    """把 ContextPlan 映射到固定白名单，并补充提供商返回的真实输入 Token。"""
    actual_prompt_tokens = usage.input_tokens if usage is not None and usage.input_tokens > 0 else None
    return plan.to_usage_context(actual_prompt_tokens=actual_prompt_tokens, round_index=round_index)


async def emit_context_status(
    emitter: Any,
    *,
    phase: str,
    context: ContextUsage,
) -> None:
    """兼容没有新协议方法的旧 emitter/test double。"""
    method = getattr(emitter, "context_status_updated", None)
    if method is None or not iscoroutinefunction(method):
        return
    await method(phase=phase, **context.model_dump())
