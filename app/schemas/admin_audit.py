"""管理员审计中心 API 协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SafeCode = Annotated[str, Field(pattern=r"^[a-z0-9:_-]{1,80}$")]


class PerformanceStageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["http", "sse"]
    concurrency: int = Field(ge=1, le=10000)
    requests: int | None = Field(default=None, ge=0)
    flows: int | None = Field(default=None, ge=0)
    successful: int | None = Field(default=None, ge=0)
    failed: int | None = Field(default=None, ge=0)
    requests_per_second: float | None = Field(default=None, ge=0)
    rps: float | None = Field(default=None, ge=0)
    p50_ms: float | None = Field(default=None, ge=0)
    p90_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    p99_ms: float | None = Field(default=None, ge=0)
    max_ms: float | None = Field(default=None, ge=0)
    p50_ttft_ms: float | None = Field(default=None, ge=0)
    p95_ttft_ms: float | None = Field(default=None, ge=0)
    p99_ttft_ms: float | None = Field(default=None, ge=0)
    p95_total_ms: float | None = Field(default=None, ge=0)
    error_rate: float | None = Field(default=None, ge=0, le=1)
    timeout_rate: float | None = Field(default=None, ge=0, le=1)
    error_frames: int | None = Field(default=None, ge=0)


class PerformanceCleanupSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversations_deleted: int = Field(default=0, ge=0)
    tokens_revoked: int = Field(default=0, ge=0)
    users_deleted: int | None = Field(default=None, ge=0)
    agent_steps_deleted: int | None = Field(default=None, ge=0)
    errors: list[SafeCode] = Field(default_factory=list, max_length=100)


class PerformanceResourceMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cpu_percent: float | None = Field(default=None, ge=0)
    memory_mib: float | None = Field(default=None, ge=0)
    memory_percent: float | None = Field(default=None, ge=0)
    connections: int | None = Field(default=None, ge=0)
    restarts: int | None = Field(default=None, ge=0)
    rejected_connections: int | None = Field(default=None, ge=0)
    evicted_keys: int | None = Field(default=None, ge=0)
    oom: bool | None = None


class PerformanceResourcesSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api: PerformanceResourceMetrics | None = None
    postgres: PerformanceResourceMetrics | None = None
    redis: PerformanceResourceMetrics | None = None
    host: PerformanceResourceMetrics | None = None
    nginx: PerformanceResourceMetrics | None = None
    litellm: PerformanceResourceMetrics | None = None


class PerformanceSafeSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stages: list[PerformanceStageSummary] = Field(default_factory=list, max_length=100)
    stopped: bool = False
    stop_reasons: list[SafeCode] = Field(default_factory=list, max_length=100)
    cleanup: PerformanceCleanupSummary = Field(default_factory=PerformanceCleanupSummary)
    resources: PerformanceResourcesSummary | None = None
    rps: float | None = Field(default=None, ge=0)
    p50_ms: float | None = Field(default=None, ge=0)
    p90_ms: float | None = Field(default=None, ge=0)
    p95_ms: float | None = Field(default=None, ge=0)
    p99_ms: float | None = Field(default=None, ge=0)
    max_ms: float | None = Field(default=None, ge=0)
    ttft_ms: float | None = Field(default=None, ge=0)
    error_rate: float | None = Field(default=None, ge=0, le=1)


class AdminPerformanceRunImport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^perf-[A-Za-z0-9._:-]{1,150}$")
    environment: str = Field(pattern=r"^[a-z0-9_-]{1,30}$")
    model_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._:/-]{1,100}$")
    status: str = Field(default="completed", pattern=r"^[a-z0-9_-]{1,30}$")
    schema_version: int = Field(default=1, ge=1, le=100)
    safe_summary: PerformanceSafeSummary
    started_at: datetime | None = None
    finished_at: datetime | None = None


class AdminPageParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=25, ge=1, le=100)
