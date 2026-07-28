import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from scripts import fetch_litellm_cost_map as fetcher


class FakeResponse:
    def __init__(self, payload, *, headers=None, status_code=200):
        self.payload = payload
        self.headers = headers or {}
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            request = httpx.Request("GET", fetcher.DEFAULT_COST_MAP_URL)
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)

    def json(self):
        return self.payload


class FakeClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class CostMapFetchTests(unittest.TestCase):
    def test_fetch_is_get_only_and_builds_observable_status(self):
        payload = {
            "moonshot/kimi-test": {
                "input_cost_per_token": 0.000001,
                "output_cost_per_token": 0.000003,
            }
        }
        client = FakeClient(FakeResponse(payload, headers={"etag": '"abc"', "last-modified": "today"}))

        result, response = fetcher.fetch_cost_map(
            url=fetcher.DEFAULT_COST_MAP_URL,
            client=client,
        )
        status = fetcher.build_status(
            payload=result,
            response=response,
            source_url=fetcher.DEFAULT_COST_MAP_URL,
        )

        self.assertEqual(result, payload)
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0][1]["follow_redirects"], True)
        self.assertEqual(status["model_count"], 1)
        self.assertEqual(status["etag"], '"abc"')
        self.assertEqual(len(status["sha256"]), 64)

    def test_empty_or_unpriced_payload_is_rejected(self):
        for payload in ({}, {"model": {"max_input_tokens": 1000}}):
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                fetcher.validate_cost_map(payload)

    def test_atomic_write_preserves_old_target_when_replace_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cost-map.json"
            path.write_text('{"old": true}\n', encoding="utf-8")

            with patch.object(fetcher.os, "replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    fetcher.write_json_atomic(path, {"new": True})

            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"old": True})
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
