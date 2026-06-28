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
    # incomplete: LLM 返回 unknown finish_reason 退化时（雷点 3 修复路径），
    # 保留已 emit 的 reasoning/content 并报 incomplete，让前端区分于正常 stop。
    finish_reason: Literal["stop", "limit_reached", "incomplete"]


AgentProgressPhase = Literal[
    "planning", "thinking", "researching", "reading", "synthesizing", "answering", "recovering"
]
AgentPlanItemStatus = Literal["pending", "running", "completed", "failed", "skipped", "blocked"]
AgentPlanItemKind = Literal["reasoning", "search", "read", "synthesis", "answer", "other"]


class AgentPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    status: AgentPlanItemStatus
    kind: AgentPlanItemKind
    summary: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    evidence_item_ids: list[str] = Field(default_factory=list)


class AgentEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["web", "file", "tool", "model"]
    status: Literal["candidate", "used", "discarded"]
    title: str
    url: str | None = None
    domain: str | None = None
    claim: str
    snippet: str | None = None
    used_by_final_answer: bool = False


class RunProgressUpdated(AgentEventBase):
    type: Literal["run_progress_updated"]
    protocol_version: Literal[2]
    phase: AgentProgressPhase
    label: str
    completed_steps: int | None = None
    total_steps: int | None = None
    completed_tool_calls: int | None = None
    max_tool_calls: int | None = None


class PlanSnapshot(AgentEventBase):
    type: Literal["plan_snapshot"]
    protocol_version: Literal[2]
    plan_id: str
    revision: int
    items: list[AgentPlanItem]


class PlanStepUpdated(AgentEventBase):
    type: Literal["plan_step_updated"]
    protocol_version: Literal[2]
    plan_id: str
    revision: int
    item: AgentPlanItem


class ToolResultDigest(AgentEventBase):
    type: Literal["tool_result_digest"]
    protocol_version: Literal[2]
    tool_name: str
    status: Literal["success", "failed", "degraded", "interrupted"]
    title: str
    summary: str
    key_findings: list[str] = Field(default_factory=list)
    source_refs: list[str] = Field(default_factory=list)
    truncated: bool = False


class EvidenceItemUpserted(AgentEventBase):
    type: Literal["evidence_item_upserted"]
    protocol_version: Literal[2]
    evidence: AgentEvidenceItem


AnyAgentEvent = Annotated[
    RunStarted
    | StepStarted
    | ToolCallStarted
    | ToolCallDelta
    | ToolCallCompleted
    | StepCompleted
    | RunLimitReached
    | RunInterrupted
    | RunFailed
    | RunCompleted
    | RunProgressUpdated
    | PlanSnapshot
    | PlanStepUpdated
    | ToolResultDigest
    | EvidenceItemUpserted,
    Field(discriminator="type"),
]
