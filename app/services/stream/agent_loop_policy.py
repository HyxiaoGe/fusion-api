"""Agent loop 纯策略。

本模块只做限制判断与终态映射，不触碰 Redis、数据库、LLM 或事件发送。
"""

from dataclasses import dataclass
from typing import Literal

AgentLoopLimitReason = Literal["timeout", "max_steps", "max_tool_calls"]
RunCompletedFinishReason = Literal["stop", "limit_reached", "incomplete"]
AgentSessionStatus = Literal["completed", "limit_reached", "incomplete"]


@dataclass(frozen=True)
class AgentLoopLimits:
    max_steps: int
    max_tool_calls: int
    total_timeout_s: float


@dataclass(frozen=True)
class AgentRunTerminalState:
    run_finish_reason: RunCompletedFinishReason
    session_status: AgentSessionStatus


def check_agent_loop_limit(
    *,
    elapsed_seconds: float,
    step: int,
    total_tool_calls: int,
    limits: AgentLoopLimits,
) -> AgentLoopLimitReason | None:
    if elapsed_seconds > limits.total_timeout_s:
        return "timeout"
    if step >= limits.max_steps:
        return "max_steps"
    if total_tool_calls >= limits.max_tool_calls:
        return "max_tool_calls"
    return None


def map_run_terminal_state(
    *,
    unknown_terminated: bool,
    limit_reason: AgentLoopLimitReason | None,
) -> AgentRunTerminalState:
    if unknown_terminated:
        return AgentRunTerminalState(run_finish_reason="incomplete", session_status="incomplete")
    if limit_reason is not None:
        return AgentRunTerminalState(run_finish_reason="limit_reached", session_status="limit_reached")
    return AgentRunTerminalState(run_finish_reason="stop", session_status="completed")
