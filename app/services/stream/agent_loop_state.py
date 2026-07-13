"""Agent loop 内存状态与纯状态转移。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.schemas.chat import ContextUsage, Usage
from app.services.stream.agent_loop_policy import AgentLoopLimitReason
from app.services.stream.run_finalizer import AgentRunStats


@dataclass
class AgentLoopState:
    content_blocks: list[Any] = field(default_factory=list)
    accumulated_usage: Usage = field(default_factory=lambda: Usage(input_tokens=0, output_tokens=0))
    last_context: ContextUsage | None = None
    step: int = 0
    total_tool_calls: int = 0
    current_step_id: str | None = None
    finish_reason: str = "stop"
    limit_reason: AgentLoopLimitReason | None = None
    unknown_terminated: bool = False
    terminal_emitted: bool = False
    plan_items: dict[str, dict] = field(default_factory=dict)

    def next_step_number(self) -> int:
        self.step += 1
        return self.step

    def mark_current_step(self, step_id: str) -> None:
        self.current_step_id = step_id

    def set_plan_items(self, items: list[dict]) -> None:
        self.plan_items = {str(item.get("id")): dict(item) for item in items if item.get("id")}

    def clear_current_step(self) -> None:
        self.current_step_id = None

    def record_executed_tool_calls(self, tool_call_count: int) -> None:
        self.total_tool_calls += tool_call_count

    def update_usage(self, usage: Usage) -> None:
        self.accumulated_usage = usage

    def update_context(self, context: ContextUsage | None) -> None:
        if context is not None:
            self.last_context = context

    def final_usage(self) -> Usage | None:
        if self.accumulated_usage.input_tokens <= 0 and self.last_context is None:
            return None
        return Usage(
            input_tokens=self.accumulated_usage.input_tokens,
            output_tokens=self.accumulated_usage.output_tokens,
            context=self.last_context,
        )

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
