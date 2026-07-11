import json
import unittest
from unittest.mock import patch

from pydantic import ValidationError

from app.schemas.admin_audit import PerformanceResourcesSummary
from scripts.perf.resource_guard import (
    MonitoringUnavailable,
    ResourceGuard,
    build_prometheus_query_url,
    build_resource_queries,
)


class FakePrometheusClient:
    def __init__(self, values, *, missing=None, error=None, failures_before_success=None):
        self.values = dict(values)
        self.missing = set(missing or [])
        self.error = error
        self.failures_before_success = dict(failures_before_success or {})
        self.calls = []
        self.names_by_query = {query: name for name, query in build_resource_queries().items()}

    def query_scalar(self, prometheus_url, query, *, timeout_seconds):
        name = self.names_by_query[query]
        self.calls.append((prometheus_url, name, timeout_seconds))
        if self.error is not None:
            raise self.error
        remaining_failures = self.failures_before_success.get(name, 0)
        if remaining_failures > 0:
            self.failures_before_success[name] = remaining_failures - 1
            raise MonitoringUnavailable("private transient missing sample")
        if name in self.missing:
            raise MonitoringUnavailable("private upstream error")
        return float(self.values[name])


def healthy_values():
    values = {}
    for component in ("api", "postgres", "redis", "nginx", "litellm"):
        values[f"{component}_start_time"] = 100
        values[f"{component}_oom_events"] = 2
        values[f"{component}_cpu_percent"] = 10
        values[f"{component}_memory_mib"] = 200
    values.update(
        {
            "postgres_connections": 10,
            "redis_rejected_connections": 5,
            "redis_evicted_keys": 7,
            "nginx_connections": 3,
            "host_cpu_percent": 20,
            "host_available_memory_mib": 4096,
            "host_total_memory_mib": 8192,
        }
    )
    return values


class PrometheusQueryUrlTests(unittest.TestCase):
    def test_builds_canonical_encoded_instant_query_url(self):
        url = build_prometheus_query_url(
            "https://monitor.example/prometheus/",
            'sum(rate(container_cpu_usage_seconds_total{name="fusion-api"}[2m])) * 100',
        )

        self.assertTrue(url.startswith("https://monitor.example/prometheus/api/v1/query?query="))
        self.assertIn("fusion-api", url)
        self.assertIn("%7B", url)
        self.assertNotIn(" ", url)

    def test_rejects_credentials_existing_query_fragment_and_non_http_urls(self):
        invalid = [
            "https://user:password@monitor.example",
            "https://monitor.example?token=secret",
            "https://monitor.example/#fragment",
            "file:///tmp/prometheus",
            "not-a-url",
        ]

        for url in invalid:
            with self.subTest(url=url), self.assertRaises(ValueError):
                build_prometheus_query_url(url, "up")


class ResourceGuardTests(unittest.TestCase):
    def test_healthy_check_records_cpu_peaks_without_hard_stopping(self):
        values = healthy_values()
        client = FakePrometheusClient(values)
        guard = ResourceGuard(client, "http://127.0.0.1:9090")

        values.update(
            {
                "api_cpu_percent": 250,
                "postgres_cpu_percent": 180,
                "redis_cpu_percent": 90,
                "nginx_cpu_percent": 70,
                "litellm_cpu_percent": 220,
                "host_cpu_percent": 99,
            }
        )
        client.values = values
        reasons = guard.check()
        summary = guard.resources_summary()

        self.assertEqual(reasons, [])
        self.assertEqual(summary["api"]["cpu_percent"], 250)
        self.assertEqual(summary["postgres"]["cpu_percent"], 180)
        self.assertEqual(summary["host"]["cpu_percent"], 99)
        self.assertNotIn("url", json.dumps(summary).lower())
        self.assertNotIn("query", json.dumps(summary).lower())
        PerformanceResourcesSummary.model_validate(summary)

    def test_hard_gates_detect_restart_oom_counter_deltas_and_thresholds(self):
        values = healthy_values()
        client = FakePrometheusClient(values)
        guard = ResourceGuard(client, "http://127.0.0.1:9090")

        values.update(
            {
                "api_start_time": 101,
                "postgres_oom_events": 3,
                "redis_rejected_connections": 6,
                "redis_evicted_keys": 9,
                "api_memory_mib": 900.01,
                "postgres_connections": 81,
                "host_available_memory_mib": 1023.99,
            }
        )
        client.values = values

        reasons = guard.check()

        self.assertEqual(
            reasons,
            [
                "resource:api_restart",
                "resource:postgres_oom",
                "resource:redis_rejected_connections",
                "resource:redis_evicted_keys",
                "resource:api_memory",
                "resource:postgres_connections",
                "resource:host_available_memory",
            ],
        )
        summary = guard.resources_summary()
        self.assertEqual(summary["api"]["restarts"], 1)
        self.assertTrue(summary["postgres"]["oom"])
        self.assertEqual(summary["redis"]["rejected_connections"], 1)
        self.assertEqual(summary["redis"]["evicted_keys"], 2)

    def test_tracks_peak_and_minimum_values_across_checks(self):
        values = healthy_values()
        client = FakePrometheusClient(values)
        guard = ResourceGuard(client, "http://127.0.0.1:9090")

        client.values = {**values, "api_cpu_percent": 50, "host_available_memory_mib": 3000}
        self.assertEqual(guard.check(), [])
        client.values = {**values, "api_cpu_percent": 30, "host_available_memory_mib": 3500}
        self.assertEqual(guard.check(), [])

        summary = guard.resources_summary()
        self.assertEqual(summary["api"]["cpu_percent"], 50)
        self.assertEqual(summary["host"]["memory_mib"], 3000)
        self.assertEqual(summary["host"]["memory_percent"], 63.38)

    def test_monitoring_unavailable_at_baseline_or_check_fails_closed_without_leaking_details(self):
        baseline_guard = ResourceGuard(
            FakePrometheusClient(healthy_values(), error=RuntimeError("token=private")),
            "http://127.0.0.1:9090",
        )
        self.assertEqual(baseline_guard.check(), ["resource:monitoring_unavailable"])

        check_client = FakePrometheusClient(healthy_values())
        check_guard = ResourceGuard(check_client, "http://127.0.0.1:9090")
        check_client.missing.add("api_memory_mib")
        reasons = check_guard.check()

        self.assertEqual(reasons, ["resource:monitoring_unavailable"])
        self.assertNotIn("private", json.dumps(reasons))

        invalid_total_client = FakePrometheusClient(healthy_values())
        invalid_total_guard = ResourceGuard(invalid_total_client, "http://127.0.0.1:9090")
        invalid_total_client.values["host_total_memory_mib"] = 0
        self.assertEqual(invalid_total_guard.check(), ["resource:monitoring_unavailable"])

    def test_transient_single_metric_missing_sample_retries_then_succeeds(self):
        client = FakePrometheusClient(healthy_values())
        guard = ResourceGuard(client, "http://127.0.0.1:9090")
        client.failures_before_success["api_memory_mib"] = 1
        calls_before = sum(name == "api_memory_mib" for _, name, _ in client.calls)

        with patch("scripts.perf.resource_guard.time.sleep") as sleep:
            reasons = guard.check()

        calls_after = sum(name == "api_memory_mib" for _, name, _ in client.calls)
        self.assertEqual(reasons, [])
        self.assertEqual(calls_after - calls_before, 2)
        sleep.assert_called_once()

    def test_single_metric_retry_exhaustion_still_fails_closed(self):
        client = FakePrometheusClient(healthy_values())
        guard = ResourceGuard(client, "http://127.0.0.1:9090")
        client.failures_before_success["api_memory_mib"] = 3
        calls_before = sum(name == "api_memory_mib" for _, name, _ in client.calls)

        with patch("scripts.perf.resource_guard.time.sleep") as sleep:
            reasons = guard.check()

        calls_after = sum(name == "api_memory_mib" for _, name, _ in client.calls)
        self.assertEqual(reasons, ["resource:monitoring_unavailable"])
        self.assertEqual(calls_after - calls_before, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_summary_shape_forbids_queries_and_arbitrary_payload(self):
        guard = ResourceGuard(FakePrometheusClient(healthy_values()), "http://127.0.0.1:9090")
        guard.check()
        summary = guard.resources_summary()

        PerformanceResourcesSummary.model_validate(summary)
        with self.assertRaises(ValidationError):
            PerformanceResourcesSummary.model_validate({**summary, "query": "private promql"})


if __name__ == "__main__":
    unittest.main()
