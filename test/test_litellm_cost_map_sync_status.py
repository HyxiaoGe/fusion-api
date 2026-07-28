import json
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch

from scripts import check_litellm_cost_map_sync_status as checker

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=UTC)


def status_payload(
    *,
    scheduled: bool = True,
    interval_hours: int | float | None = 6,
    last_run: str | None = None,
    next_run: str | None = None,
) -> dict:
    return {
        "scheduled": scheduled,
        "interval_hours": interval_hours,
        "last_run": last_run or (NOW - timedelta(hours=1)).isoformat(),
        "next_run": next_run or (NOW + timedelta(hours=5)).isoformat(),
    }


class LiteLLMCostMapSyncStatusTests(unittest.TestCase):
    def test_healthy_status_has_no_issues(self):
        report = checker.evaluate_status(
            status_payload(),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        self.assertTrue(report.healthy)
        self.assertEqual(report.issues, [])
        self.assertEqual(report.status.interval_hours, 6)

    def test_not_scheduled_is_unhealthy(self):
        report = checker.evaluate_status(
            status_payload(scheduled=False),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        self.assertFalse(report.healthy)
        self.assertIn("not_scheduled", [issue.code for issue in report.issues])

    def test_unexpected_interval_is_unhealthy(self):
        report = checker.evaluate_status(
            status_payload(interval_hours=12),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        self.assertFalse(report.healthy)
        self.assertIn("unexpected_interval", [issue.code for issue in report.issues])

    def test_missing_timestamps_are_unhealthy(self):
        payload = status_payload()
        payload["last_run"] = None
        payload["next_run"] = None

        report = checker.evaluate_status(
            payload,
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        self.assertFalse(report.healthy)
        codes = [issue.code for issue in report.issues]
        self.assertIn("missing_last_run", codes)
        self.assertIn("missing_next_run", codes)

    def test_invalid_timestamps_are_unhealthy(self):
        report = checker.evaluate_status(
            status_payload(last_run="not-a-time", next_run="also-not-a-time"),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        self.assertFalse(report.healthy)
        codes = [issue.code for issue in report.issues]
        self.assertIn("invalid_last_run", codes)
        self.assertIn("invalid_next_run", codes)

    def test_stale_last_run_and_overdue_next_run_are_unhealthy(self):
        report = checker.evaluate_status(
            status_payload(
                last_run=(NOW - timedelta(hours=13)).isoformat(),
                next_run=(NOW - timedelta(hours=7)).isoformat(),
            ),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        self.assertFalse(report.healthy)
        codes = [issue.code for issue in report.issues]
        self.assertIn("stale_last_run", codes)
        self.assertIn("overdue_next_run", codes)

    def test_future_last_run_and_far_next_run_are_unhealthy(self):
        report = checker.evaluate_status(
            status_payload(
                last_run=(NOW + timedelta(hours=1)).isoformat(),
                next_run=(NOW + timedelta(hours=9)).isoformat(),
            ),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        codes = [issue.code for issue in report.issues]
        self.assertIn("future_last_run", codes)
        self.assertIn("next_run_too_far", codes)

    def test_reported_fallback_is_unhealthy(self):
        payload = status_payload()
        payload["source"] = "bundled"
        payload["fallback_reason"] = "github unavailable"

        report = checker.evaluate_status(
            payload,
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        codes = [issue.code for issue in report.issues]
        self.assertIn("cost_map_fallback", codes)

    def test_fetch_uses_only_get_status_endpoint(self):
        client = Mock()
        response = Mock()
        response.json.return_value = status_payload()
        client.get.return_value = response

        payload = checker.fetch_status(
            base_url="http://litellm-proxy:4000/",
            master_key="sk-master-secret",
            timeout_seconds=10,
            client=client,
        )

        self.assertTrue(payload["scheduled"])
        client.get.assert_called_once_with(
            "http://litellm-proxy:4000/schedule/model_cost_map_reload/status",
            headers={"Authorization": "Bearer sk-master-secret"},
            timeout=10,
        )
        response.raise_for_status.assert_called_once_with()
        self.assertFalse(client.post.called)

    def test_serialized_report_does_not_include_master_key(self):
        report = checker.evaluate_status(
            status_payload(),
            expected_interval_hours=6,
            max_staleness_hours=12,
            now=NOW,
        )

        serialized = checker.serialize_report(
            report,
            context={
                "litellm_base_url": "http://litellm-proxy:4000",
                "master_key": "sk-master-secret",
            },
        )
        output = json.dumps(serialized, ensure_ascii=False)

        self.assertIn("litellm_base_url", serialized["context"])
        self.assertNotIn("master_key", serialized["context"])
        self.assertNotIn("sk-master-secret", output)

    def test_main_returns_nonzero_for_stale_status_without_printing_key(self):
        actual_now = datetime.now(UTC)
        stale_payload = status_payload(
            last_run=(actual_now - timedelta(hours=13)).isoformat(),
            next_run=(actual_now - timedelta(hours=7)).isoformat(),
        )

        with (
            patch.dict("os.environ", {"LITELLM_MASTER_KEY": "sk-master-secret"}, clear=True),
            patch.object(checker, "fetch_status", return_value=stale_payload),
            patch("builtins.print") as print_mock,
        ):
            exit_code = checker.main([])

        self.assertEqual(exit_code, 1)
        output = print_mock.call_args.args[0]
        self.assertNotIn("sk-master-secret", output)
        self.assertIn("stale_last_run", output)


if __name__ == "__main__":
    unittest.main()
