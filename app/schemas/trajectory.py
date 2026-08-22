"""轨迹历史读侧的稳定 DTO。"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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
