"""agent_event 协议 — 10 个事件模型 + 共享 envelope."""
from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentEventBase(BaseModel):
    """所有 agent_event 的共享 envelope 字段."""
    model_config = ConfigDict(extra="forbid")
    type: str
    run_id: str
    parent_run_id: str | None = None
    step_id: str | None = None
    parent_step_id: str | None = None
    tool_call_id: str | None = None
    sequence: int
    trace_id: str
    ts: float


class RunStarted(AgentEventBase):
    type: Literal["run_started"]
    conversation_id: str
    message_id: str
    model: str
    tools: list[str]
    config: dict[str, Any]


class StepStarted(AgentEventBase):
    type: Literal["step_started"]
    step_number: int


class ToolCallStarted(AgentEventBase):
    type: Literal["tool_call_started"]
    tool_name: str
    arguments: dict[str, Any]


class ToolCallDelta(AgentEventBase):
    type: Literal["tool_call_delta"]
    tool_name: str
    delta: dict[str, Any]


class ToolCallCompleted(AgentEventBase):
    type: Literal["tool_call_completed"]
    tool_name: str
    status: Literal["success", "failed", "degraded"]
    duration_ms: int
    result_summary: dict[str, Any]
    error: str | None = None


class StepCompleted(AgentEventBase):
    type: Literal["step_completed"]
    step_number: int
    tool_call_count: int
    duration_ms: int


class RunLimitReached(AgentEventBase):
    type: Literal["run_limit_reached"]
    reason: Literal["max_steps", "max_tool_calls", "timeout"]


class RunInterrupted(AgentEventBase):
    type: Literal["run_interrupted"]
    reason: Literal["user_cancelled", "superseded"]


class RunFailed(AgentEventBase):
    type: Literal["run_failed"]
    error_code: str
    message: str


class RunCompleted(AgentEventBase):
    type: Literal["run_completed"]
    total_steps: int
    total_tool_calls: int
    finish_reason: Literal["stop", "limit_reached"]


AnyAgentEvent = Annotated[
    RunStarted | StepStarted | ToolCallStarted | ToolCallDelta | ToolCallCompleted
    | StepCompleted | RunLimitReached | RunInterrupted | RunFailed | RunCompleted,
    Field(discriminator="type"),
]
