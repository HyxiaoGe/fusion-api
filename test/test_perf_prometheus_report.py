import unittest
from unittest.mock import patch

from scripts.perf.prometheus_report import (
    MAX_PROMETHEUS_RESPONSE_BYTES,
    MetricSeries,
    build_default_queries,
    build_resources_summary,
    collect_report,
    fetch_query_range,
    summarize_query_result,
)


class PrometheusReportTests(unittest.TestCase):
    def test_default_queries_cover_api_database_redis_network_and_host(self):
        queries = build_default_queries()

        self.assertIn("api_cpu_percent", queries)
        self.assertIn("api_memory_mib", queries)
        self.assertIn("api_start_time_seconds", queries)
        self.assertIn("postgres_connections", queries)
        self.assertIn("postgres_start_time_seconds", queries)
        self.assertIn("redis_rejected_connections", queries)
        self.assertIn("redis_evicted_keys", queries)
        self.assertIn("redis_start_time_seconds", queries)
        self.assertIn("nginx_start_time_seconds", queries)
        self.assertIn("litellm_start_time_seconds", queries)
        self.assertIn("cloudflared_request_errors", queries)
        self.assertIn("host_swap_used_mib", queries)
        self.assertNotIn("token", " ".join(queries))

    def test_summary_aggregates_all_series_and_counter_delta(self):
        payload = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [
                    {"metric": {"cpu": "0"}, "values": [[10, "1"], [20, "3"]]},
                    {"metric": {"cpu": "1"}, "values": [[10, "2"], [20, "4"]]},
                ],
            },
        }

        gauge = summarize_query_result(payload, counter=False)
        counter = summarize_query_result(payload, counter=True)

        self.assertEqual(gauge, MetricSeries(samples=4, first=1.0, last=4.0, minimum=1.0, maximum=4.0, delta=None))
        self.assertEqual(counter.delta, 4.0)

    def test_counter_delta_accumulates_increments_across_reset_and_single_sample(self):
        reset_payload = {
            "status": "success",
            "data": {
                "result": [
                    {"values": [[10, "5"], [20, "8"], [30, "2"], [40, "4"]]},
                    {"values": [[10, "10"], [20, "10"], [30, "1"]]},
                ]
            },
        }
        single_payload = {
            "status": "success",
            "data": {"result": [{"values": [[10, "7"]]}]},
        }

        self.assertEqual(summarize_query_result(reset_payload, counter=True).delta, 8.0)
        self.assertEqual(summarize_query_result(single_payload, counter=True).delta, 0.0)

    def test_start_time_summary_counts_container_restarts(self):
        payload = {
            "status": "success",
            "data": {"result": [{"values": [[10, "100"], [20, "100"], [30, "200"], [40, "200"], [50, "300"]]}]},
        }

        summary = summarize_query_result(payload, counter=False, track_changes=True)

        self.assertEqual(summary.changes, 2)

    def test_empty_or_non_finite_samples_are_safe(self):
        payload = {
            "status": "success",
            "data": {
                "resultType": "matrix",
                "result": [{"metric": {}, "values": [[10, "NaN"], [20, "+Inf"]]}],
            },
        }

        summary = summarize_query_result(payload, counter=False)

        self.assertEqual(summary, MetricSeries(samples=0))

    def test_collect_report_rejects_oversized_window_or_too_many_points(self):
        with self.assertRaisesRegex(ValueError, "2 小时"):
            collect_report("http://prometheus", start="0", end="7201", step_seconds=30)
        with self.assertRaisesRegex(ValueError, "1000"):
            collect_report("http://prometheus", start="0", end="7200", step_seconds=7)

    def test_fetch_query_range_limits_response_body(self):
        class OversizedResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, size):
                self.requested_size = size
                return b"x" * size

        response = OversizedResponse()
        with patch("scripts.perf.prometheus_report.urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(ValueError, "响应过大"):
                fetch_query_range(
                    "http://prometheus",
                    "up",
                    start="0",
                    end="60",
                    step_seconds=30,
                    timeout_seconds=1,
                )

        self.assertEqual(response.requested_size, MAX_PROMETHEUS_RESPONSE_BYTES + 1)

    def test_resources_summary_maps_supported_groups_without_raw_query_metadata(self):
        def metric(*, maximum=None, delta=None, changes=None):
            return {
                "samples": 3,
                "first": 1,
                "last": maximum,
                "minimum": 1,
                "maximum": maximum,
                "delta": delta,
                "changes": changes,
                "query": "private-promql",
            }

        report = {
            "start": "private-start",
            "end": "private-end",
            "step_seconds": 30,
            "metrics": {
                "api_cpu_percent": metric(maximum=81.5),
                "api_memory_mib": metric(maximum=512.25),
                "api_oom_events": metric(maximum=4, delta=1),
                "api_start_time_seconds": metric(maximum=200, changes=2),
                "postgres_cpu_percent": metric(maximum=31.5),
                "postgres_memory_mib": metric(maximum=1024),
                "postgres_connections": metric(maximum=42),
                "postgres_start_time_seconds": metric(maximum=200, changes=1),
                "redis_cpu_percent": metric(maximum=15),
                "redis_memory_mib": metric(maximum=256),
                "redis_rejected_connections": metric(maximum=5, delta=3),
                "redis_evicted_keys": metric(maximum=7, delta=2),
                "redis_start_time_seconds": metric(maximum=200, changes=0),
                "host_cpu_percent": metric(maximum=72),
                "host_memory_percent": metric(maximum=66.5),
                "nginx_cpu_percent": metric(maximum=12),
                "nginx_connections_active": metric(maximum=88),
                "nginx_start_time_seconds": metric(maximum=200, changes=1),
                "litellm_cpu_percent": metric(maximum=44),
                "litellm_memory_mib": metric(maximum=768),
                "litellm_start_time_seconds": metric(maximum=200, changes=0),
                "cloudflared_request_errors": metric(maximum=9, delta=4),
            },
        }

        summary = build_resources_summary(report)

        self.assertEqual(
            summary,
            {
                "api": {"cpu_percent": 81.5, "memory_mib": 512.25, "restarts": 2, "oom": True},
                "postgres": {"cpu_percent": 31.5, "memory_mib": 1024, "connections": 42, "restarts": 1},
                "redis": {
                    "cpu_percent": 15,
                    "memory_mib": 256,
                    "restarts": 0,
                    "rejected_connections": 3,
                    "evicted_keys": 2,
                },
                "host": {"cpu_percent": 72, "memory_percent": 66.5},
                "nginx": {"cpu_percent": 12, "connections": 88, "restarts": 1},
                "litellm": {"cpu_percent": 44, "memory_mib": 768, "restarts": 0},
            },
        )
        serialized = str(summary)
        for forbidden in ("private-start", "private-end", "private-promql", "cloudflared"):
            self.assertNotIn(forbidden, serialized)


if __name__ == "__main__":
    unittest.main()
