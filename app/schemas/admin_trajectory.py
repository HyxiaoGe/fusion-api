"""管理员轨迹诊断的独立 DTO。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.trajectory import TrajectorySnapshot


class AdminTrajectoryToolCall(BaseModel):
    """按 run 汇总的脱敏工具诊断；不承诺与账本 span 精确关联。"""

    model_config = ConfigDict(extra="forbid")

    association: Literal["run"] = "run"
    id: str
    message_id: str | None = None
    step_number: int | None = None
    tool_name: str
    status: str
    duration_ms: int | None = None
    model_id: str
    provider: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result_preview: dict[str, Any] = Field(default_factory=dict)
    error: dict[str, str] | None = None
    redacted_fields: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class AdminTrajectorySnapshot(BaseModel):
    """管理员诊断响应，仅将普通快照作为嵌套基础数据。"""

    model_config = ConfigDict(extra="forbid")

    snapshot: TrajectorySnapshot
    tool_calls: list[AdminTrajectoryToolCall] = Field(default_factory=list)
    tool_calls_truncated: bool = False
