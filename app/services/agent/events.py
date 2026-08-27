"""agent_event 协议模型与共享 envelope。"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.chat import ContextStatus, KnowledgeEvidenceBlock, ProductResultBlock
from app.schemas.trajectory import TrajectoryCapabilityResolution


class AgentEventBase(BaseModel):
    """所有 agent_event 的共享 envelope 字段."""

    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
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
    task_id: str
    model: str
    tools: list[str]
    config: dict[str, Any]
    capability_resolution: TrajectoryCapabilityResolution | None = None

    @model_validator(mode="after")
    def _require_external_tools_match_resolution(self) -> RunStarted:
        if self.capability_resolution is not None and self.tools != self.capability_resolution.external_tool_names:
            raise ValueError("Run 公告工具必须与能力路由外部工具一致")
        return self


class StepStarted(AgentEventBase):
    type: Literal["step_started"]
    step_number: int


class ToolCallStarted(AgentEventBase):
    type: Literal["tool_call_started"]
    tool_name: str
    arguments: dict[str, Any]
    plan_item_id: str | None = None


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
    plan_item_id: str | None = None


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


class LLMRoundStarted(AgentEventBase):
    type: Literal["llm_round_started"]
    llm_round_id: str
    round_index: int = Field(ge=1)
    model: str
    provider: str
    system_prompt_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class LLMRoundFirstOutputDelta(AgentEventBase):
    type: Literal["llm_round_first_output_delta"]
    llm_round_id: str
    delta_kind: Literal["reasoning", "content", "tool_call"]
    ttft_ms: int = Field(ge=0)


class LLMRoundCompleted(AgentEventBase):
    type: Literal["llm_round_completed"]
    llm_round_id: str
    status: Literal["success"]
    finish_reason: str | None
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_read_tokens: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    reasoning_tokens: int | None = Field(default=None, ge=0)
    ttft_ms: int | None = Field(default=None, ge=0)
    duration_ms: int = Field(ge=0)


class LLMRoundFailed(AgentEventBase):
    type: Literal["llm_round_failed"]
    llm_round_id: str
    status: Literal["failed"]
    error_code: str | None
    message: str | None = Field(default=None, max_length=120)


class LLMRoundCancelled(AgentEventBase):
    type: Literal["llm_round_cancelled"]
    llm_round_id: str
    status: Literal["cancelled"]
    reason: Literal["user_cancelled", "superseded", "shutdown"]


class RetrievalStarted(AgentEventBase):
    type: Literal["retrieval_started"]
    retrieval_id: str
    query_summary: str | None = Field(default=None, max_length=120)


class RetrievalCompleted(AgentEventBase):
    type: Literal["retrieval_completed"]
    retrieval_id: str
    status: Literal["success"]
    document_count: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


class RetrievalFailed(AgentEventBase):
    type: Literal["retrieval_failed"]
    retrieval_id: str
    status: Literal["failed"]
    error_code: str | None
    message: str | None = Field(default=None, max_length=120)


class RetrievalCancelled(AgentEventBase):
    type: Literal["retrieval_cancelled"]
    retrieval_id: str
    status: Literal["cancelled"]
    reason: Literal["user_cancelled", "superseded", "shutdown"]


class ToolAttemptStarted(AgentEventBase):
    type: Literal["tool_attempt_started"]
    tool_attempt_id: str
    tool_call_id: str
    tool_name: str
    attempt_index: int = Field(ge=1)


class ToolAttemptCompleted(AgentEventBase):
    type: Literal["tool_attempt_completed"]
    tool_attempt_id: str
    status: Literal["success", "failed", "cancelled", "timeout"]
    error_code: str | None
    duration_ms: int = Field(ge=0)


class SuggestedQuestionsPending(AgentEventBase):
    """推荐问题已领取版本，生成任务将在 SSE 终态后异步执行。"""

    type: Literal["suggested_questions_pending"]
    protocol_version: Literal[2]
    message_id: str
    revision: int = Field(ge=1)
    status: Literal["pending"] = "pending"


AgentProgressPhase = Literal[
    "planning", "thinking", "researching", "reading", "synthesizing", "answering", "recovering"
]
AgentPlanItemStatus = Literal["pending", "running", "completed", "failed", "skipped", "blocked"]
AgentPlanItemKind = Literal["reasoning", "search", "read", "synthesis", "answer", "other"]


class AgentPlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    title: str
    phase_id: str | None = None
    phase_title: str | None = None
    status: AgentPlanItemStatus
    kind: AgentPlanItemKind
    summary: str | None = None
    tool_names: list[str] = Field(default_factory=list)
    evidence_item_ids: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    planned_tools: list[str] = Field(default_factory=list)


class AgentEvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    kind: Literal["web", "file", "tool", "model", "knowledge"]
    status: Literal["candidate", "selected", "read_success", "read_degraded", "read_failed", "used", "discarded"]
    title: str
    url: str | None = None
    domain: str | None = None
    claim: str
    snippet: str | None = None
    used_by_final_answer: bool = False
    citation_index: int | None = Field(default=None, ge=1)


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
    mode: Literal["auto", "on", "off"] = "auto"
    source: Literal["model", "observed"] = "observed"
    revision: int
    reason: str = "legacy_observed"
    items: list[AgentPlanItem]


class PlanStepUpdated(AgentEventBase):
    type: Literal["plan_step_updated"]
    protocol_version: Literal[2]
    plan_id: str
    mode: Literal["auto", "on", "off"] = "auto"
    source: Literal["model", "observed"] = "observed"
    revision: int
    reason: str = "legacy_observed"
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
    repair_state: Literal["retrying", "requires_user_input", "exhausted", "resolved"] | None = None
    repair_id: str | None = None
    plan_item_id: str | None = None


class EvidenceItemUpserted(AgentEventBase):
    type: Literal["evidence_item_upserted"]
    protocol_version: Literal[2]
    evidence: AgentEvidenceItem


class ContentBlockUpserted(AgentEventBase):
    """将完整、严格白名单的产品结果块增量发送给前端。"""

    type: Literal["content_block_upserted"]
    protocol_version: Literal[2]
    content_block: ProductResultBlock | KnowledgeEvidenceBlock


class ContentBlockDiscarded(AgentEventBase):
    """撤回已流式发送、但不应进入最终消息的过程性 block。"""

    type: Literal["content_block_discarded"]
    protocol_version: Literal[2]
    block_id: str = Field(min_length=1, max_length=128)


class SystemPromptPrepared(AgentEventBase):
    """本地系统提示词组装的终态元数据，不携带提示词或用户偏好正文。"""

    type: Literal["system_prompt_prepared"]
    protocol_version: Literal[2]
    status: Literal["ready", "failed"]
    source: Literal["code"]
    template_version: str = Field(min_length=1, max_length=64)
    section_ids: list[str] = Field(max_length=50)
    fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    char_count: int | None = Field(default=None, ge=0)
    detail_status: Literal["available", "degraded"] | None = None
    duration_ms: int = Field(ge=0)
    error_code: str | None = None
    message: str | None = Field(default=None, max_length=120)


class ContextStatusUpdated(AgentEventBase):
    """单轮 LLM 上下文状态；字段严格白名单，不携带 prompt 或内部来源。"""

    type: Literal["context_status_updated"]
    protocol_version: Literal[2]
    message_id: str
    phase: Literal["estimated", "final", "error"]
    status: ContextStatus
    round_index: int = Field(ge=1)
    window_tokens: int | None = Field(default=None, ge=0)
    estimated_tokens_before: int | None = Field(default=None, ge=0)
    estimated_tokens_after: int | None = Field(default=None, ge=0)
    actual_prompt_tokens: int | None = Field(default=None, ge=0)
    removed_turns: int = Field(default=0, ge=0)
    removed_messages: int = Field(default=0, ge=0)
    removed_tool_transactions: int = Field(default=0, ge=0)


class ContextRequired(AgentEventBase):
    """请求客户端补充运行上下文；不得包含精确位置。"""

    type: Literal["context_required"]
    protocol_version: Literal[2]
    context_type: Literal["geolocation"]
    request_id: str
    purpose: Literal["nearby_search", "route_origin", "route_destination", "local_weather"]
    reason: str
    expires_at: float


class ContextResult(AgentEventBase):
    """上下文握手结果状态；不得包含精确位置。"""

    type: Literal["context_result"]
    protocol_version: Literal[2]
    context_type: Literal["geolocation"]
    request_id: str
    status: Literal["provided", "denied", "timeout", "unavailable"]


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
    | LLMRoundStarted
    | LLMRoundFirstOutputDelta
    | LLMRoundCompleted
    | LLMRoundFailed
    | LLMRoundCancelled
    | RetrievalStarted
    | RetrievalCompleted
    | RetrievalFailed
    | RetrievalCancelled
    | ToolAttemptStarted
    | ToolAttemptCompleted
    | SuggestedQuestionsPending
    | RunProgressUpdated
    | PlanSnapshot
    | PlanStepUpdated
    | ToolResultDigest
    | EvidenceItemUpserted
    | ContentBlockUpserted
    | ContentBlockDiscarded
    | SystemPromptPrepared
    | ContextStatusUpdated
    | ContextRequired
    | ContextResult,
    Field(discriminator="type"),
]
