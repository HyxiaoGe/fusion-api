"""轨迹历史读侧的稳定 DTO。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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
