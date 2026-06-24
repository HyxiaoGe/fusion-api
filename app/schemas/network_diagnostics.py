from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

ToolStatus = Literal["success", "failed", "degraded", "interrupted"]


class NetworkDiagnosticsSummary(BaseModel):
    total_duration_ms: int | None = None
    total_steps: int = 0
    total_tool_calls: int = 0
    search_calls: int = 0
    url_read_calls: int = 0
    success_count: int = 0
    failed_count: int = 0
    degraded_count: int = 0
    interrupted_count: int = 0
    limit_reason: str | None = None
    run_status: str | None = None


class NetworkDiagnosticsToolItem(BaseModel):
    tool_call_log_id: str
    tool_name: str
    status: ToolStatus
    duration_ms: int | None = None
    target: str = ""
    result_count: int | None = None
    reason: str | None = None
    started_at: datetime | None = None
    admin: dict[str, Any] | None = None


class NetworkDiagnosticsResponse(BaseModel):
    conversation_id: str
    message_id: str
    run_id: str | None = None
    visibility: Literal["user", "admin"] = "user"
    summary: NetworkDiagnosticsSummary = Field(default_factory=NetworkDiagnosticsSummary)
    tools: list[NetworkDiagnosticsToolItem] = Field(default_factory=list)
    is_empty: bool = False
