"""Agent loop 内存状态与纯状态转移。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.chat import Usage
from app.services.stream.agent_loop_policy import AgentLoopLimitReason
from app.services.stream.run_finalizer import AgentRunStats


@dataclass
class AgentLoopState:
    content_blocks: list[Any] = field(default_factory=list)
    accumulated_usage: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    step: int = 0
    total_tool_calls: int = 0
    current_step_id: str | None = None
    finish_reason: str = "stop"
    limit_reason: AgentLoopLimitReason | None = None
    unknown_terminated: bool = False
    terminal_emitted: bool = False

    def next_step_number(self) -> int:
        self.step += 1
        return self.step

    def mark_current_step(self, step_id: str) -> None:
        self.current_step_id = step_id

    def clear_current_step(self) -> None:
        self.current_step_id = None

    def record_executed_tool_calls(self, tool_call_count: int) -> None:
        self.total_tool_calls += tool_call_count

    def update_usage(self, usage: Usage) -> None:
        self.accumulated_usage = usage

    def final_usage(self) -> Usage | None:
        if self.accumulated_usage.input_tokens <= 0:
            return None
        return self.accumulated_usage

    def mark_unknown_terminated(self) -> None:
        self.unknown_terminated = True

    def mark_terminal_emitted(self) -> None:
        self.terminal_emitted = True

    def run_stats(self, run_id: str) -> AgentRunStats:
        return AgentRunStats(
            run_id=run_id,
            total_steps=self.step,
            total_tool_calls=self.total_tool_calls,
        )
