import json
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

from scripts import ensure_litellm_cost_map_sync as ensure


def status(*, scheduled=True, interval=6, stale=False):
    now = datetime.now(UTC)
    return {
        "scheduled": scheduled,
        "interval_hours": interval,
        "last_run": (now - timedelta(hours=13 if stale else 1)).isoformat(),
        "next_run": (now + timedelta(hours=5)).isoformat(),
    }


class SequenceClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.get_calls = []
        self.post_calls = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        response = Mock()
        response.json.return_value = self.payloads.pop(0)
        return response

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return Mock()


class LiteLLMCostMapSyncEnsureTests(unittest.TestCase):
    def test_healthy_schedule_is_read_only_even_in_apply_mode(self):
        client = SequenceClient([status()])

        result = ensure.ensure_sync(base_url="http://litellm", master_key="secret", apply=True, client=client)

        self.assertTrue(result["healthy"])
        self.assertEqual(result["action"], "none")
        self.assertEqual(client.post_calls, [])

    def test_dry_run_reports_required_schedule_without_post(self):
        client = SequenceClient([status(scheduled=False, interval=None)])

        result = ensure.ensure_sync(base_url="http://litellm", master_key="secret", client=client)

        self.assertFalse(result["healthy"])
        self.assertEqual(result["action"], "schedule_required")
        self.assertEqual(client.post_calls, [])

    def test_apply_schedules_then_verifies_status(self):
        client = SequenceClient([status(scheduled=False, interval=None), status()])

        result = ensure.ensure_sync(base_url="http://litellm", master_key="secret", apply=True, client=client)

        self.assertTrue(result["healthy"])
        self.assertEqual(result["action"], "scheduled")
        self.assertEqual(len(client.post_calls), 1)
        url, kwargs = client.post_calls[0]
        self.assertEqual(url, "http://litellm/schedule/model_cost_map_reload")
        self.assertEqual(kwargs["params"], {"hours": 6})

    def test_wrong_interval_is_repairable(self):
        client = SequenceClient([status(interval=12), status()])

        result = ensure.ensure_sync(base_url="http://litellm", master_key="secret", apply=True, client=client)

        self.assertTrue(result["healthy"])
        self.assertEqual(result["action"], "scheduled")

    def test_stale_running_schedule_fails_without_rescheduling(self):
        client = SequenceClient([status(stale=True)])

        result = ensure.ensure_sync(base_url="http://litellm", master_key="secret", apply=True, client=client)

        self.assertFalse(result["healthy"])
        self.assertEqual(result["action"], "none")
        self.assertEqual(client.post_calls, [])

    def test_serialized_result_never_contains_master_key(self):
        client = SequenceClient([status()])

        result = ensure.ensure_sync(
            base_url="http://litellm",
            master_key="master-super-secret",
            apply=True,
            client=client,
        )

        self.assertNotIn("master-super-secret", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
