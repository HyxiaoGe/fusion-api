"""轨迹历史读侧的稳定 DTO。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.utils.run_capability_contract import validate_capability_resolution_semantics

CapabilityPackageId = Literal[
    "direct",
    "transform",
    "date",
    "fresh_web",
    "verified_web",
    "url_read",
    "weather",
    "place_discovery",
    "mobility_route",
    "flight",
    "train",
    "travel_air_rail",
    "mobility_intercity",
    "mixed_itinerary",
    "deep_research",
    "knowledge_grounded",
    "tools_unavailable",
    "clarification_only",
    "mcp_explicit",
]
CapabilityReasonCode = Literal[
    "direct_greeting",
    "assistant_identity_question",
    "stable_knowledge_question",
    "simple_calculation",
    "text_transform_request",
    "current_date_question",
    "fresh_external_fact",
    "verified_source_request",
    "explicit_url_read",
    "explicit_weather_request",
    "explicit_place_discovery",
    "explicit_route_task",
    "explicit_flight_request",
    "explicit_train_request",
    "air_rail_comparison",
    "mixed_itinerary_request",
    "origin_destination_relation",
    "intercity_locations",
    "adjacent_route_followup",
    "deep_research_mode",
    "knowledge_grounded_mode",
    "tools_disabled",
    "function_calling_unavailable",
    "search_capability_unavailable",
    "required_tools_unavailable",
    "explicit_authorized_tool_alias",
    "insufficient_capability_signal",
]


class TrajectoryCapabilityResolution(BaseModel):
    """Run 级能力路由在实时与历史协议中的显式安全 DTO。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1]
    router_version: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}\.\d+$", max_length=32)
    package_id: CapabilityPackageId
    confidence: Literal["high", "medium", "low"]
    resolution_mode: Literal["routed", "degraded", "clarification"]
    reason_codes: list[CapabilityReasonCode] = Field(min_length=1, max_length=4)
    external_tool_names: list[str] = Field(max_length=3)
    effective_plan_mode: Literal["auto", "on", "off"]
    include_current_date: bool
    network_boundary_required: bool
    bundle_fingerprint: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @field_validator("reason_codes", "external_tool_names")
    @classmethod
    def _require_unique_items(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("能力路由列表字段不得重复")
        return value

    @field_validator("external_tool_names")
    @classmethod
    def _validate_tool_names(cls, value: list[str]) -> list[str]:
        if any(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]{0,127}", name) is None for name in value):
            raise ValueError("能力路由工具名格式非法")
        return value

    @model_validator(mode="after")
    def _validate_package_semantics(self) -> TrajectoryCapabilityResolution:
        validate_capability_resolution_semantics(
            package_id=self.package_id,
            confidence=self.confidence,
            resolution_mode=self.resolution_mode,
            reason_codes=self.reason_codes,
            external_tool_names=self.external_tool_names,
            effective_plan_mode=self.effective_plan_mode,
            include_current_date=self.include_current_date,
            network_boundary_required=self.network_boundary_required,
        )
        return self


@dataclass(frozen=True)
class UserTrajectoryMetaRow:
    """普通读取所需的窄 meta 数据，不携带 terminal intent 详情。"""

    trajectory_status: str
    event_count: int
    expected_last_sequence: int | None
    degraded_reason: str | None
    has_pending_terminal_intent: bool
    llm_detail_schema_version: int | None
    llm_round_count: int


class TrajectoryEventRecord(BaseModel):
    """从账本读取后交给纯投影器的事件记录。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    sequence: int
    event_type: str
    schema_version: int | None = None
    timestamp: datetime
    step_id: str | None = None
    tool_call_id: str | None = None
    parent_step_id: str | None = None
    trace_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class TrajectoryRecord(BaseModel):
    """账本事件在快照中的只读投影。"""

    model_config = ConfigDict(extra="forbid")

    sequence: int
    event_type: str
    schema_version: int
    timestamp: datetime
    step_id: str | None = None
    tool_call_id: str | None = None
    parent_step_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    payload: dict[str, Any]


class TrajectorySpan(BaseModel):
    """由账本生命周期事件重建出的执行区间。"""

    model_config = ConfigDict(extra="forbid")

    span_id: str
    kind: str
    name: str
    parent_span_id: str | None = None
    start_sequence: int
    end_sequence: int | None = None
    started_at: datetime
    ended_at: datetime | None = None
    duration_ms: int | None = None
    status: str
    terminal_source: str | None = None
    inferred_reason: str | None = None
    ttft_ms: int | None = None
    record_sequences: list[int] = Field(default_factory=list)


class TrajectoryProjection(BaseModel):
    """单个 run 的事件与 span 投影结果。"""

    model_config = ConfigDict(extra="forbid")

    records: list[TrajectoryRecord] = Field(default_factory=list)
    spans: list[TrajectorySpan] = Field(default_factory=list)


class TrajectoryRunSummary(BaseModel):
    """AgentSession 权威摘要在普通读取端点中的稳定形状。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    message_id: str | None = None
    turn_message_id: str | None = None
    attempt_index: int | None = None
    status: str
    trajectory_status: str
    total_steps: int
    total_tool_calls: int
    duration_ms: int | None = None
    started_at: datetime
    ended_at: datetime | None = None
    llm_detail_schema_version: int | None = None
    llm_round_count: int = 0
    capability_resolution: TrajectoryCapabilityResolution | None = None


class TrajectoryRunListResponse(BaseModel):
    """会话内有界的 run 尝试列表。"""

    model_config = ConfigDict(extra="forbid")

    items: list[TrajectoryRunSummary] = Field(default_factory=list)
    truncated: bool = False


class TrajectoryCompleteness(BaseModel):
    """账本读取前缀与持久化完整性状态。"""

    model_config = ConfigDict(extra="forbid")

    status: str
    degraded_reason: str | None = None
    event_count: int | None = None
    expected_last_sequence: int | None = None
    loaded_event_count: int
    first_sequence: int | None = None
    last_sequence: int | None = None


class TrajectoryLlmRoundSummary(BaseModel):
    """快照中用于高密度账本展示的有界 LLM 正文预览。"""

    model_config = ConfigDict(extra="forbid")

    llm_round_id: str
    reasoning_preview: str | None = None
    output_preview: str | None = None


class TrajectorySnapshot(BaseModel):
    """普通用户可读取的脱敏账本快照。"""

    model_config = ConfigDict(extra="forbid")

    run: TrajectoryRunSummary
    records: list[TrajectoryRecord] = Field(default_factory=list)
    spans: list[TrajectorySpan] = Field(default_factory=list)
    completeness: TrajectoryCompleteness
    truncated: bool = False
    llm_round_summaries: list[TrajectoryLlmRoundSummary] = Field(default_factory=list)


class ToolNodeDetail(BaseModel):
    """普通用户可读取的 Tool 节点安全详情。"""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    tool_name: str
    status: str
    duration_ms: int | None = None
    payload: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class LlmNodeDetail(BaseModel):
    """普通用户可读取的单个 LLM Round 正文详情。"""

    model_config = ConfigDict(extra="forbid")

    llm_round_id: str
    reasoning_text: str | None = None
    output_text: str | None = None


class SystemPromptSection(BaseModel):
    """运行时实际组装并持久化的有序系统提示词段落。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    section_id: str = Field(min_length=1)
    content: str


class SystemPromptNodeDetail(BaseModel):
    """仅通过独立详情端点返回的历史系统提示词正文。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    template_version: str = Field(min_length=1)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_count: int = Field(ge=0)
    sections: list[SystemPromptSection] = Field(min_length=1)


class SystemPromptSnapshot(SystemPromptNodeDetail):
    """持久化格式校验；版本号不接受布尔值或字符串隐式转换。"""

    schema_version: int = Field(ge=1, le=1)


class TrajectoryNodeDetailResponse(BaseModel):
    """轨迹节点详情的统一稳定响应信封。"""

    model_config = ConfigDict(extra="forbid")

    status: Literal["available", "pending", "not_recorded", "degraded"]
    node_type: Literal["tool", "llm", "system_prompt"] = "tool"
    available_sections: list[
        Literal["summary", "payload", "result", "timing", "schema", "thinking", "output", "prompt"]
    ] = Field(default_factory=list)
    detail: ToolNodeDetail | LlmNodeDetail | SystemPromptNodeDetail | None = None
    redacted_fields: list[str] = Field(default_factory=list)
    truncated_fields: list[str] = Field(default_factory=list)
    reason: str | None = None
