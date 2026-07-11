#!/usr/bin/env python3
"""从 Prometheus 查询生产压测窗口资源指标并生成脱敏汇总。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

MAX_QUERY_WINDOW_SECONDS = 2 * 60 * 60
MAX_QUERY_POINTS = 1000
MAX_PROMETHEUS_RESPONSE_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True)
class MetricSeries:
    samples: int
    first: float | None = None
    last: float | None = None
    minimum: float | None = None
    maximum: float | None = None
    delta: float | None = None
    changes: int | None = None


def build_default_queries() -> dict[str, str]:
    mib = "1024/1024"
    return {
        "api_cpu_percent": 'sum(rate(container_cpu_usage_seconds_total{name="fusion-api"}[2m])) * 100',
        "api_memory_mib": f'max(container_memory_working_set_bytes{{name="fusion-api"}}) / {mib}',
        "api_oom_events": 'sum(container_oom_events_total{name="fusion-api"})',
        "api_start_time_seconds": 'max(container_start_time_seconds{name="fusion-api"})',
        "postgres_cpu_percent": 'sum(rate(container_cpu_usage_seconds_total{name="postgres"}[2m])) * 100',
        "postgres_memory_mib": f'max(container_memory_working_set_bytes{{name="postgres"}}) / {mib}',
        "postgres_connections": 'pg_stat_database_numbackends{datname="fusion"}',
        "postgres_start_time_seconds": 'max(container_start_time_seconds{name="postgres"})',
        "redis_cpu_percent": 'sum(rate(container_cpu_usage_seconds_total{name="middleware-redis"}[2m])) * 100',
        "redis_memory_mib": f'max(container_memory_working_set_bytes{{name="middleware-redis"}}) / {mib}',
        "redis_rejected_connections": "sum(redis_rejected_connections_total)",
        "redis_evicted_keys": "sum(redis_evicted_keys_total)",
        "redis_start_time_seconds": 'max(container_start_time_seconds{name="middleware-redis"})',
        "nginx_cpu_percent": 'sum(rate(container_cpu_usage_seconds_total{name="nginx-proxy"}[2m])) * 100',
        "nginx_connections_active": "sum(nginx_connections_active)",
        "nginx_start_time_seconds": 'max(container_start_time_seconds{name="nginx-proxy"})',
        "litellm_cpu_percent": 'sum(rate(container_cpu_usage_seconds_total{name="litellm-proxy"}[2m])) * 100',
        "litellm_memory_mib": f'max(container_memory_working_set_bytes{{name="litellm-proxy"}}) / {mib}',
        "litellm_start_time_seconds": 'max(container_start_time_seconds{name="litellm-proxy"})',
        "cloudflared_concurrent_requests": "sum(cloudflared_tunnel_concurrent_requests_per_tunnel)",
        "cloudflared_request_errors": "sum(cloudflared_tunnel_request_errors)",
        "host_cpu_percent": '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[2m])))',
        "host_memory_percent": "100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)",
        "host_swap_used_mib": f"(node_memory_SwapTotal_bytes - node_memory_SwapFree_bytes) / {mib}",
        "host_load1": "node_load1",
    }


COUNTER_METRICS = {
    "api_oom_events",
    "redis_rejected_connections",
    "redis_evicted_keys",
    "cloudflared_request_errors",
}
CONTAINER_START_TIME_METRICS = {
    "api_start_time_seconds",
    "postgres_start_time_seconds",
    "redis_start_time_seconds",
    "nginx_start_time_seconds",
    "litellm_start_time_seconds",
}


def summarize_query_result(
    payload: dict[str, Any],
    *,
    counter: bool,
    track_changes: bool = False,
) -> MetricSeries:
    if payload.get("status") != "success":
        return MetricSeries(samples=0)
    data = payload.get("data")
    rows = data.get("result") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        return MetricSeries(samples=0)

    values: list[float] = []
    series_deltas: list[float] = []
    series_changes: list[int] = []
    for row in rows:
        raw_values = row.get("values") if isinstance(row, dict) else None
        if not isinstance(raw_values, list):
            continue
        series_values: list[float] = []
        for item in raw_values:
            if not isinstance(item, list) or len(item) != 2:
                continue
            try:
                value = float(item[1])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
                series_values.append(value)
        if counter and series_values:
            delta = 0.0
            for previous, current in zip(series_values, series_values[1:]):
                delta += current - previous if current >= previous else max(0.0, current)
            series_deltas.append(delta)
        if track_changes and series_values:
            series_changes.append(
                sum(current != previous for previous, current in zip(series_values, series_values[1:]))
            )
    if not values:
        return MetricSeries(samples=0)
    return MetricSeries(
        samples=len(values),
        first=round(values[0], 4),
        last=round(values[-1], 4),
        minimum=round(min(values), 4),
        maximum=round(max(values), 4),
        delta=round(sum(series_deltas), 4) if counter else None,
        changes=sum(series_changes) if track_changes else None,
    )


def _parse_query_time(value: str) -> float:
    try:
        timestamp = float(value)
    except ValueError:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise ValueError("Prometheus 查询时间格式无效") from error
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = parsed.timestamp()
    if not math.isfinite(timestamp):
        raise ValueError("Prometheus 查询时间格式无效")
    return timestamp


def _validate_query_window(start: str, end: str, step_seconds: int) -> None:
    if step_seconds < 1:
        raise ValueError("step 必须为正数")
    duration_seconds = _parse_query_time(end) - _parse_query_time(start)
    if duration_seconds < 0:
        raise ValueError("Prometheus 查询结束时间不能早于开始时间")
    if duration_seconds > MAX_QUERY_WINDOW_SECONDS:
        raise ValueError("Prometheus 查询窗口不能超过 2 小时")
    points = math.floor(duration_seconds / step_seconds) + 1
    if points > MAX_QUERY_POINTS:
        raise ValueError("Prometheus 查询点数不能超过 1000")


def fetch_query_range(
    prometheus_url: str,
    query: str,
    *,
    start: str,
    end: str,
    step_seconds: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    _validate_query_window(start, end, step_seconds)
    params = urllib.parse.urlencode({"query": query, "start": start, "end": end, "step": step_seconds})
    url = f"{prometheus_url.rstrip('/')}/api/v1/query_range?{params}"
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "fusion-perf-runner/1"})
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        raw_payload = response.read(MAX_PROMETHEUS_RESPONSE_BYTES + 1)
    if len(raw_payload) > MAX_PROMETHEUS_RESPONSE_BYTES:
        raise ValueError("Prometheus 响应过大")
    payload = json.loads(raw_payload.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Prometheus 响应不是 JSON object")
    return payload


def collect_report(
    prometheus_url: str,
    *,
    start: str,
    end: str,
    step_seconds: int = 30,
    timeout_seconds: float = 20,
) -> dict[str, Any]:
    _validate_query_window(start, end, step_seconds)
    metrics: dict[str, Any] = {}
    for name, query in build_default_queries().items():
        payload = fetch_query_range(
            prometheus_url,
            query,
            start=start,
            end=end,
            step_seconds=step_seconds,
            timeout_seconds=timeout_seconds,
        )
        metrics[name] = asdict(
            summarize_query_result(
                payload,
                counter=name in COUNTER_METRICS,
                track_changes=name in CONTAINER_START_TIME_METRICS,
            )
        )
    return {"start": start, "end": end, "step_seconds": step_seconds, "metrics": metrics}


def build_resources_summary(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """把原始 Prometheus report 收敛为可持久化的安全资源汇总。"""
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        return {}

    def number(name: str, field: str) -> float | None:
        metric = metrics.get(name)
        value = metric.get(field) if isinstance(metric, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            return value
        return None

    def peak(name: str) -> float | None:
        return number(name, "maximum")

    def count(name: str, field: str) -> int | None:
        value = number(name, field)
        return max(0, int(value)) if value is not None else None

    def occurred(name: str) -> bool | None:
        value = number(name, "delta")
        return value > 0 if value is not None else None

    groups: dict[str, dict[str, Any]] = {
        "api": {
            "cpu_percent": peak("api_cpu_percent"),
            "memory_mib": peak("api_memory_mib"),
            "restarts": count("api_start_time_seconds", "changes"),
            "oom": occurred("api_oom_events"),
        },
        "postgres": {
            "cpu_percent": peak("postgres_cpu_percent"),
            "memory_mib": peak("postgres_memory_mib"),
            "connections": count("postgres_connections", "maximum"),
            "restarts": count("postgres_start_time_seconds", "changes"),
        },
        "redis": {
            "cpu_percent": peak("redis_cpu_percent"),
            "memory_mib": peak("redis_memory_mib"),
            "restarts": count("redis_start_time_seconds", "changes"),
            "rejected_connections": count("redis_rejected_connections", "delta"),
            "evicted_keys": count("redis_evicted_keys", "delta"),
        },
        "host": {
            "cpu_percent": peak("host_cpu_percent"),
            "memory_percent": peak("host_memory_percent"),
        },
        "nginx": {
            "cpu_percent": peak("nginx_cpu_percent"),
            "connections": count("nginx_connections_active", "maximum"),
            "restarts": count("nginx_start_time_seconds", "changes"),
        },
        "litellm": {
            "cpu_percent": peak("litellm_cpu_percent"),
            "memory_mib": peak("litellm_memory_mib"),
            "restarts": count("litellm_start_time_seconds", "changes"),
        },
    }
    return {
        group: {key: value for key, value in values.items() if value is not None}
        for group, values in groups.items()
        if any(value is not None for value in values.values())
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Fusion 压测窗口 Prometheus 资源汇总")
    parser.add_argument("--prometheus-url", required=True)
    parser.add_argument("--start", required=True, help="RFC3339 或 Unix 时间")
    parser.add_argument("--end", required=True, help="RFC3339 或 Unix 时间")
    parser.add_argument("--step-seconds", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=20)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.step_seconds < 1 or args.timeout_seconds <= 0:
        print(json.dumps({"error": "step 与 timeout 必须为正数"}, ensure_ascii=False), file=sys.stderr)
        return 2
    try:
        report = collect_report(
            args.prometheus_url,
            start=args.start,
            end=args.end,
            step_seconds=args.step_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(json.dumps({"error": type(error).__name__}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
