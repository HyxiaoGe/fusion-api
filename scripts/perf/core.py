"""压测 runner 的纯函数、SSE 解析器与脱敏结果模型。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequestSample:
    latency_ms: float
    status: int | None
    error: str | None = None
    timed_out: bool = False


@dataclass(frozen=True)
class SSEEvent:
    event_id: str | None
    payload: dict[str, Any] | None
    done: bool = False


class SSEParser:
    """逐行解析 SSE；仅在空行结束一个完整事件时返回结果。"""

    def __init__(self) -> None:
        self._event_id: str | None = None
        self._data_lines: list[str] = []

    def feed_line(self, line: str) -> SSEEvent | None:
        line = line.rstrip("\r\n")
        if not line:
            return self._finish_event()
        if line.startswith(":"):
            return None
        field_name, _, raw_value = line.partition(":")
        value = raw_value[1:] if raw_value.startswith(" ") else raw_value
        if field_name == "id":
            self._event_id = value
        elif field_name == "data":
            self._data_lines.append(value)
        return None

    def _finish_event(self) -> SSEEvent | None:
        if not self._data_lines:
            return None
        raw_data = "\n".join(self._data_lines)
        event_id = self._event_id
        self._event_id = None
        self._data_lines = []
        if raw_data == "[DONE]":
            return SSEEvent(event_id=event_id, payload=None, done=True)
        payload = json.loads(raw_data)
        if not isinstance(payload, dict):
            raise ValueError("SSE data 必须是 JSON object")
        return SSEEvent(event_id=event_id, payload=payload)


def percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * percent) - 1)
    return round(ordered[index], 2)


def summarize_samples(samples: list[RequestSample]) -> dict[str, int | float]:
    latencies = [sample.latency_ms for sample in samples]
    successful = sum(1 for sample in samples if sample.error is None and sample.status and sample.status < 400)
    timed_out = sum(1 for sample in samples if sample.timed_out)
    count = len(samples)
    failed = count - successful
    return {
        "requests": count,
        "successful": successful,
        "failed": failed,
        "p50_ms": percentile(latencies, 0.50),
        "p95_ms": percentile(latencies, 0.95),
        "max_ms": round(max(latencies), 2) if latencies else 0.0,
        "error_rate": round(failed / count, 4) if count else 0.0,
        "timeout_rate": round(timed_out / count, 4) if count else 0.0,
    }


@dataclass(frozen=True)
class StopPolicy:
    min_samples: int = 20
    max_error_rate: float = 0.05
    max_timeout_rate: float = 0.05
    max_consecutive_failures: int = 10
    max_p95_ms: float | None = None

    def evaluate(self, summary: dict[str, Any], consecutive_failures: int = 0) -> list[str]:
        reasons: list[str] = []
        if summary.get("requests", 0) >= self.min_samples:
            if summary.get("error_rate", 0) >= self.max_error_rate:
                reasons.append("error_rate")
            if summary.get("timeout_rate", 0) >= self.max_timeout_rate:
                reasons.append("timeout_rate")
        if consecutive_failures >= self.max_consecutive_failures:
            reasons.append("consecutive_failures")
        if self.max_p95_ms is not None and summary.get("p95_ms", 0) >= self.max_p95_ms:
            reasons.append("p95_ms")
        return reasons


@dataclass
class CleanupManifest:
    run_id: str
    email: str
    conversation_ids: set[str] = field(default_factory=set)
    agent_run_ids: set[str] = field(default_factory=set)
    agent_trace_ids: set[str] = field(default_factory=set)
    _refresh_tokens: dict[str, str] = field(default_factory=dict, repr=False)

    def add_conversation(self, conversation_id: str) -> None:
        self.conversation_ids.add(conversation_id)

    def add_refresh_token(self, label: str, token: str) -> None:
        self._refresh_tokens[label] = token

    def add_agent_trace(self, run_id: str | None, trace_id: str | None) -> None:
        if run_id:
            self.agent_run_ids.add(run_id)
        if trace_id:
            self.agent_trace_ids.add(trace_id)

    def refresh_tokens(self) -> list[tuple[str, str]]:
        return sorted(self._refresh_tokens.items())

    def cleanup_plan(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "account_fingerprint": fingerprint(self.email),
            "conversation_ids": sorted(self.conversation_ids),
            "refresh_token_labels": sorted(self._refresh_tokens),
            "agent_run_ids": sorted(self.agent_run_ids),
            "agent_trace_ids": sorted(self.agent_trace_ids),
        }


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]


def extract_agent_trace_ids(payload: dict[str, Any]) -> tuple[str | None, str | None]:
    """从兼容的新旧 agent_event envelope 中提取 run_started 标识。"""
    if payload.get("chunk_type") != "agent_event":
        return None, None
    data = payload.get("data")
    candidates = [data.get("event") if isinstance(data, dict) else None, data, payload]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate.get("type") == "run_started":
            return _optional_string(candidate.get("run_id")), _optional_string(candidate.get("trace_id"))
    return None, None


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def build_safe_result(
    *,
    manifest: CleanupManifest,
    stages: list[dict[str, Any]],
    cleanup: dict[str, Any],
    stopped: bool,
    stop_reasons: list[str],
) -> dict[str, Any]:
    """只输出重建定位所需的 run_id 与不可逆指纹，不输出凭据、邮箱或会话 ID。"""
    return {
        "schema_version": 1,
        "run_id": manifest.run_id,
        "account_fingerprint": fingerprint(manifest.email),
        "agent_run_ids": sorted(manifest.agent_run_ids),
        "agent_trace_ids": sorted(manifest.agent_trace_ids),
        "stages": stages,
        "stopped": stopped,
        "stop_reasons": stop_reasons,
        "cleanup": cleanup,
    }
