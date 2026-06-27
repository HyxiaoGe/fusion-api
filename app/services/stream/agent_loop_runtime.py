"""Agent loop driver 运行时依赖集合。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.services.stream.agent_loop_policy import AgentLoopLimits


@dataclass(frozen=True)
class AgentLoopRuntime:
    conversation_id: str
    task_id: str
    run_id: str
    user_id: str
    model_id: str
    provider: str
    litellm_model: str
    litellm_kwargs: dict
    should_use_reasoning: bool
    call_kwargs: dict
    assistant_message_id: str
    run_start: float
    limits: AgentLoopLimits
    emitter: Any
    session_cache: Any
    network_budget: Any
    start_step_fn: Callable[..., Awaitable[Any]]
    complete_step_fn: Callable[..., Awaitable[Any]]
    run_round_fn: Callable[..., Awaitable[Any]]
    handle_tool_calls_round_fn: Callable[..., Awaitable[Any]]
    run_limit_summary_step_fn: Callable[..., Awaitable[Any]]
    llm_call_fn: Callable[..., Awaitable[Any]]
    stream_round_fn: Callable[..., Awaitable[Any]]
    execute_tools_fn: Callable[..., Awaitable[Any]]
    persist_message_fn: Callable[..., None]
    log_round_summary_fn: Callable[..., None]
    warning_fn: Callable[[str], None]
    clock: Callable[[], float]
