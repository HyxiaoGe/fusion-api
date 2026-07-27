"""Agent loop 内存状态与纯状态转移。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from app.schemas.chat import ContextUsage, Usage
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.stream.agent_loop_policy import AgentLoopLimitReason
from app.services.stream.itinerary_observability import ItineraryToolObservation
from app.services.stream.run_finalizer import AgentRunStats

NO_PROGRESS_SEARCH_SUMMARY_THRESHOLD = 2


@dataclass(frozen=True)
class ProductToolOutcome:
    """Agent run 内的安全产品工具摘要，不携带内部错误或第三方 payload。"""

    tool_name: str
    mode: Literal["flight", "train", "weather", "route"]
    status: Literal["available", "unavailable"]
    block_id: str | None = None
    origin: str | None = None
    destination: str | None = None
    departure_date: str | None = None
    location: str | None = None

    @property
    def identity_key(self) -> tuple[str, str | None, str | None, str | None, str | None]:
        return (
            self.tool_name,
            self.origin,
            self.destination,
            self.departure_date,
            self.location,
        )


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
    consecutive_no_progress_search_results: int = 0
    context_wait_seconds: float = 0.0
    runtime_contexts: dict[str, Any] = field(default_factory=dict)
    unavailable_contexts: dict[str, str] = field(default_factory=dict)
    product_tool_attempted: bool = False
    product_tool_outcomes: list[ProductToolOutcome] = field(default_factory=list)
    successful_tool_call_signatures: set[str] = field(default_factory=set)
    argument_repair_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_tool_repairs: dict[str, dict[str, Any]] = field(default_factory=dict)
    itinerary_tool_observations: list[ItineraryToolObservation] = field(default_factory=list)
    plan_coordinator: PlanCoordinator = field(default_factory=lambda: PlanCoordinator(run_id=""))

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

    def record_no_progress_search_results(self, results: tuple[bool, ...]) -> None:
        for is_no_progress_search in results:
            if is_no_progress_search:
                self.consecutive_no_progress_search_results += 1
            else:
                self.consecutive_no_progress_search_results = 0

    def record_product_tool_attempt(self, attempted: bool) -> None:
        self.product_tool_attempted = self.product_tool_attempted or attempted

    def record_product_tool_outcomes(self, outcomes: tuple[ProductToolOutcome, ...]) -> None:
        for outcome in outcomes:
            existing_index = next(
                (
                    index
                    for index, existing in enumerate(self.product_tool_outcomes)
                    if existing.identity_key == outcome.identity_key
                ),
                None,
            )
            if existing_index is None:
                self.product_tool_outcomes.append(outcome)
                continue
            existing = self.product_tool_outcomes[existing_index]
            if existing.status == "available" and outcome.status == "unavailable":
                continue
            self.product_tool_outcomes[existing_index] = outcome

    def record_itinerary_tool_observations(self, observations: list[ItineraryToolObservation]) -> None:
        self.itinerary_tool_observations.extend(observations)

    def should_summarize_no_progress_search(self) -> bool:
        return self.consecutive_no_progress_search_results >= NO_PROGRESS_SEARCH_SUMMARY_THRESHOLD

    def update_usage(self, usage: Usage) -> None:
        self.accumulated_usage = usage

    def update_context(self, context: ContextUsage | None) -> None:
        if context is not None:
            self.last_context = context

    def record_context_wait(self, seconds: float) -> None:
        self.context_wait_seconds += max(0.0, seconds)

    def active_elapsed_seconds(self, *, now: float, run_start: float) -> float:
        return max(0.0, now - run_start - self.context_wait_seconds)

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
