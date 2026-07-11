"""管理员审计中心 API 协议。"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

SafeCode = Annotated[str, Field(pattern=r"^[a-z0-9:_-]{1,80}$")]
SafeCount = Annotated[int, Field(ge=0, le=1_000_000_000)]
SafeMilliseconds = Annotated[float, Field(ge=0, le=86_400_000)]
SafeSeconds = Annotated[float, Field(ge=0, le=2_678_400)]
SafeThroughput = Annotated[float, Field(ge=0, le=1_000_000_000)]
SafeRate = Annotated[float, Field(ge=0, le=1)]


class PerformanceStageSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: SafeCode | None = None
    kind: Literal["http", "sse", "recovery", "stop", "soak"]
    concurrency: int = Field(ge=1, le=10000)
    duration_seconds: SafeSeconds | None = None
    elapsed_seconds: SafeSeconds | None = None
    cadence_seconds: SafeSeconds | None = None
    window_seconds: SafeSeconds | None = None
    total: SafeCount | None = None
    requests: SafeCount | None = None
    flows: SafeCount | None = None
    flows_with_output: SafeCount | None = None
    successful: SafeCount | None = None
    failed: SafeCount | None = None
    success_rate: SafeRate | None = None
    requests_per_second: SafeThroughput | None = None
    rps: SafeThroughput | None = None
    p50_ms: SafeMilliseconds | None = None
    p90_ms: SafeMilliseconds | None = None
    p95_ms: SafeMilliseconds | None = None
    p99_ms: SafeMilliseconds | None = None
    max_ms: SafeMilliseconds | None = None
    p50_ttft_ms: SafeMilliseconds | None = None
    p95_ttft_ms: SafeMilliseconds | None = None
    p99_ttft_ms: SafeMilliseconds | None = None
    p95_total_ms: SafeMilliseconds | None = None
    error_rate: SafeRate | None = None
    timeout_rate: SafeRate | None = None
    error_frames: SafeCount | None = None
    output_chunks: SafeCount | None = None
    reasoning_chunks: SafeCount | None = None
    answering_chunks: SafeCount | None = None
    visible_chars: SafeCount | None = None
    reasoning_visible_chars: SafeCount | None = None
    answering_visible_chars: SafeCount | None = None
    approx_tokens: SafeCount | None = None
    first_output_p50_ms: SafeMilliseconds | None = None
    first_output_p95_ms: SafeMilliseconds | None = None
    first_output_max_ms: SafeMilliseconds | None = None
    chunk_interval_count: SafeCount | None = None
    chunk_interval_p50_ms: SafeMilliseconds | None = None
    chunk_interval_p95_ms: SafeMilliseconds | None = None
    chunk_interval_max_ms: SafeMilliseconds | None = None
    output_window_p50_ms: SafeMilliseconds | None = None
    output_window_p95_ms: SafeMilliseconds | None = None
    output_window_max_ms: SafeMilliseconds | None = None
    tokens_per_second: SafeThroughput | None = None
    tokens_per_second_p50: SafeThroughput | None = None
    tokens_per_second_p95: SafeThroughput | None = None
    tokens_per_second_max: SafeThroughput | None = None
    initial_events: SafeCount | None = None
    recovered_events: SafeCount | None = None
    duplicate_events: SafeCount | None = None
    lost_events: SafeCount | None = None
    ordering_errors: SafeCount | None = None
    recovery_latency_ms: SafeMilliseconds | None = None
    recovery_latency_p50_ms: SafeMilliseconds | None = None
    recovery_latency_p95_ms: SafeMilliseconds | None = None
    recovery_latency_max_ms: SafeMilliseconds | None = None
    stop_attempted: bool | None = None
    cancelled: bool | None = None
    persistence_verified: bool | None = None
    stop_attempts: SafeCount | None = None
    cancelled_count: SafeCount | None = None
    persistence_verified_count: SafeCount | None = None
    stop_latency_ms: SafeMilliseconds | None = None
    stop_latency_p50_ms: SafeMilliseconds | None = None
    stop_latency_p95_ms: SafeMilliseconds | None = None
    stop_latency_max_ms: SafeMilliseconds | None = None
    executed_ticks: SafeCount | None = None
    skipped_ticks: SafeCount | None = None
    window_count: SafeCount | None = None
    consecutive_failures: SafeCount | None = None


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
