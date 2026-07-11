"""生产压测资源硬门禁；Prometheus 原始查询不进入任何输出。"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from typing import Any, Protocol


class MonitoringUnavailable(RuntimeError):
    """Prometheus 无法返回有效标量。异常正文不得进入压测结果。"""


class PrometheusScalarClient(Protocol):
    def query_scalar(self, prometheus_url: str, query: str, *, timeout_seconds: float) -> float: ...


def build_prometheus_query_url(prometheus_url: str, query: str) -> str:
    """规范化 Prometheus instant-query URL，并安全编码 PromQL。"""

    try:
        parsed = urllib.parse.urlsplit(prometheus_url)
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Prometheus URL 无效") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("Prometheus URL 必须是 HTTP(S)")
    if parsed.username or parsed.password:
        raise ValueError("Prometheus URL 不能包含凭据")
    if parsed.query or parsed.fragment:
        raise ValueError("Prometheus URL 不能包含 query 或 fragment")
    normalized_query = query.strip()
    if not normalized_query or len(normalized_query) > 4096:
        raise ValueError("PromQL 长度无效")

    base_path = parsed.path.rstrip("/")
    if base_path.endswith("/api/v1/query"):
        endpoint_path = base_path
    else:
        endpoint_path = f"{base_path}/api/v1/query"
    encoded = urllib.parse.urlencode({"query": normalized_query})
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, endpoint_path, encoded, ""))


class UrllibPrometheusClient:
    """最小 Prometheus instant-query 客户端。"""

    def query_scalar(self, prometheus_url: str, query: str, *, timeout_seconds: float) -> float:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须为正数")
        url = build_prometheus_query_url(prometheus_url, query)
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "fusion-perf-resource-guard/1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _parse_prometheus_scalar(payload)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise MonitoringUnavailable("Prometheus 查询不可用") from exc


def _parse_prometheus_scalar(payload: Any) -> float:
    if not isinstance(payload, dict) or payload.get("status") != "success":
        raise MonitoringUnavailable("Prometheus 响应失败")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise MonitoringUnavailable("Prometheus data 缺失")
    result = data.get("result")
    values: list[float] = []
    if isinstance(result, list):
        for row in result:
            raw_value = row.get("value") if isinstance(row, dict) else None
            parsed_value = _parse_sample_value(raw_value)
            if parsed_value is not None:
                values.append(parsed_value)
    else:
        parsed_value = _parse_sample_value(result)
        if parsed_value is not None:
            values.append(parsed_value)
    if not values:
        raise MonitoringUnavailable("Prometheus 没有有效样本")
    return max(values)


def _parse_sample_value(raw_value: Any) -> float | None:
    if not isinstance(raw_value, list) or len(raw_value) != 2:
        return None
    try:
        value = float(raw_value[1])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


_CONTAINERS = {
    "api": "fusion-api",
    "postgres": "postgres",
    "redis": "middleware-redis",
    "nginx": "nginx-proxy",
    "litellm": "litellm-proxy",
}


def build_resource_queries() -> dict[str, str]:
    mib = "1024/1024"
    queries: dict[str, str] = {}
    for component, container_name in _CONTAINERS.items():
        label = f'name="{container_name}"'
        queries[f"{component}_start_time"] = f"max(container_start_time_seconds{{{label}}})"
        queries[f"{component}_oom_events"] = f"sum(container_oom_events_total{{{label}}})"
        queries[f"{component}_cpu_percent"] = f"sum(rate(container_cpu_usage_seconds_total{{{label}}}[5m])) * 100"
        queries[f"{component}_memory_mib"] = f"max(container_memory_working_set_bytes{{{label}}}) / {mib}"
    queries.update(
        {
            "postgres_connections": 'sum(pg_stat_database_numbackends{datname="fusion"})',
            "redis_rejected_connections": "sum(redis_rejected_connections_total)",
            "redis_evicted_keys": "sum(redis_evicted_keys_total)",
            "nginx_connections": "sum(nginx_connections_active)",
            "host_cpu_percent": '100 * (1 - avg(rate(node_cpu_seconds_total{mode="idle"}[5m])))',
            "host_available_memory_mib": f"min(node_memory_MemAvailable_bytes) / {mib}",
            "host_total_memory_mib": f"min(node_memory_MemTotal_bytes) / {mib}",
        }
    )
    return queries


_BASELINE_METRICS = tuple(
    [f"{component}_start_time" for component in _CONTAINERS]
    + [f"{component}_oom_events" for component in _CONTAINERS]
    + ["redis_rejected_connections", "redis_evicted_keys"]
)
_QUERY_MAX_ATTEMPTS = 5
_QUERY_RETRY_BACKOFF_SECONDS = 0.1


class ResourceGuard:
    """采启动基线并在每个压测档位后执行生产资源硬门禁。"""

    def __init__(
        self,
        client: PrometheusScalarClient,
        prometheus_url: str,
        *,
        timeout_seconds: float = 10,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须为正数")
        # 仅验证规范化，不保存带 query 的 URL。
        build_prometheus_query_url(prometheus_url, "up")
        self._client = client
        self._prometheus_url = prometheus_url.rstrip("/")
        self._timeout_seconds = timeout_seconds
        self._queries = build_resource_queries()
        self._baseline: dict[str, float] = {}
        self._baseline_available = False
        self._peaks: dict[str, float] = {}
        self._minimums: dict[str, float] = {}
        self._latest: dict[str, float] = {}
        self._restarted: set[str] = set()
        self._oom: set[str] = set()
        self._capture_baseline()

    def _capture_baseline(self) -> None:
        try:
            self._baseline = self._collect(_BASELINE_METRICS)
        except Exception:  # noqa: BLE001 — 门禁只向结果暴露固定安全原因码
            self._baseline = {}
            self._baseline_available = False
            return
        self._baseline_available = True

    def _collect(self, metric_names: tuple[str, ...] | list[str]) -> dict[str, float]:
        values: dict[str, float] = {}
        for name in metric_names:
            values[name] = self._query_metric(name)
        return values

    def _query_metric(self, name: str) -> float:
        for attempt in range(_QUERY_MAX_ATTEMPTS):
            try:
                value = self._client.query_scalar(
                    self._prometheus_url,
                    self._queries[name],
                    timeout_seconds=self._timeout_seconds,
                )
                numeric = float(value)
                if not math.isfinite(numeric) or numeric < 0:
                    raise MonitoringUnavailable("Prometheus 返回非有限值")
                return numeric
            except Exception as exc:  # noqa: BLE001 — 最终只暴露固定安全原因码
                if attempt == _QUERY_MAX_ATTEMPTS - 1:
                    raise MonitoringUnavailable("Prometheus 查询重试耗尽") from exc
                time.sleep(_QUERY_RETRY_BACKOFF_SECONDS * (2**attempt))
        raise MonitoringUnavailable("Prometheus 查询重试耗尽")

    def check(self) -> list[str]:
        """返回安全硬停原因列表；监控不可用时 fail closed。"""

        if not self._baseline_available:
            return ["resource:monitoring_unavailable"]
        try:
            current = self._collect(list(self._queries))
            self._record_observations(current)
        except Exception:  # noqa: BLE001 — 不暴露 Prometheus URL、PromQL 或异常正文
            return ["resource:monitoring_unavailable"]
        reasons: list[str] = []
        for component in _CONTAINERS:
            if current[f"{component}_start_time"] != self._baseline[f"{component}_start_time"]:
                self._restarted.add(component)
                reasons.append(f"resource:{component}_restart")
        for component in _CONTAINERS:
            if current[f"{component}_oom_events"] > self._baseline[f"{component}_oom_events"]:
                self._oom.add(component)
                reasons.append(f"resource:{component}_oom")

        if self._counter_delta(current, "redis_rejected_connections") > 0:
            reasons.append("resource:redis_rejected_connections")
        if self._counter_delta(current, "redis_evicted_keys") > 0:
            reasons.append("resource:redis_evicted_keys")
        if current["api_memory_mib"] > 900:
            reasons.append("resource:api_memory")
        if current["postgres_connections"] > 80:
            reasons.append("resource:postgres_connections")
        if current["host_available_memory_mib"] < 1024:
            reasons.append("resource:host_available_memory")
        return reasons

    def _record_observations(self, current: dict[str, float]) -> None:
        available = current["host_available_memory_mib"]
        total = current["host_total_memory_mib"]
        if total <= 0 or available > total:
            raise MonitoringUnavailable("主机内存指标无效")
        self._latest = current
        peak_names = [
            name
            for name in current
            if name.endswith(("_cpu_percent", "_memory_mib", "_connections")) and name != "host_available_memory_mib"
        ]
        for name in peak_names:
            self._peaks[name] = max(self._peaks.get(name, current[name]), current[name])
        self._minimums["host_available_memory_mib"] = min(
            self._minimums.get("host_available_memory_mib", available),
            available,
        )
        used_percent = 100 * (1 - available / total)
        self._peaks["host_memory_percent"] = max(self._peaks.get("host_memory_percent", used_percent), used_percent)

    def _counter_delta(self, current: dict[str, float], name: str) -> float:
        return max(0.0, current[name] - self._baseline[name])

    @staticmethod
    def _rounded(value: float) -> int | float:
        rounded = round(value, 2)
        return int(rounded) if rounded.is_integer() else rounded

    @staticmethod
    def _compact(payload: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in payload.items() if value is not None}

    def resources_summary(self) -> dict[str, dict[str, Any]]:
        """返回 Admin PerformanceResourcesSummary 兼容映射，不含 URL/PromQL。"""

        summary: dict[str, dict[str, Any]] = {}
        for component in _CONTAINERS:
            summary[component] = self._compact(
                {
                    "cpu_percent": self._metric_peak(f"{component}_cpu_percent"),
                    "memory_mib": self._metric_peak(f"{component}_memory_mib"),
                    "restarts": 1 if component in self._restarted else 0,
                    "oom": component in self._oom,
                }
            )
        summary["postgres"]["connections"] = self._metric_peak("postgres_connections")
        summary["redis"]["rejected_connections"] = self._safe_delta("redis_rejected_connections")
        summary["redis"]["evicted_keys"] = self._safe_delta("redis_evicted_keys")
        summary["nginx"]["connections"] = self._metric_peak("nginx_connections")
        summary["host"] = self._compact(
            {
                "cpu_percent": self._metric_peak("host_cpu_percent"),
                # host.memory_mib 表示压测窗口内最低 available memory。
                "memory_mib": self._minimum_metric("host_available_memory_mib"),
                "memory_percent": self._metric_peak("host_memory_percent"),
            }
        )
        return summary

    def _metric_peak(self, name: str) -> int | float | None:
        value = self._peaks.get(name)
        return None if value is None else self._rounded(value)

    def _minimum_metric(self, name: str) -> int | float | None:
        value = self._minimums.get(name)
        return None if value is None else self._rounded(value)

    def _safe_delta(self, name: str) -> int | float:
        if not self._latest or name not in self._baseline:
            return 0
        return self._rounded(self._counter_delta(self._latest, name))
