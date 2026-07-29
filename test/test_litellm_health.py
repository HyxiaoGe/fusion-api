"""LiteLLM 健康缓存状态合并测试。"""

import unittest
from unittest.mock import patch

from app.ai import litellm_health


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, responses, **_kwargs):
        self._responses = responses

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get(self, url):
        return _FakeResponse(self._responses[url.rsplit("/", 1)[-1]])


class LiteLLMHealthTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        with litellm_health._lock:
            litellm_health._by_alias.clear()

    def tearDown(self):
        with litellm_health._lock:
            litellm_health._by_alias.clear()

    def test_record_success_promotes_unknown_with_alias_timestamp(self):
        self.assertEqual(
            litellm_health.get_health("kimi-k2.7-code"),
            {"status": "unknown", "error": None, "checked_at": None},
        )

        litellm_health.record_success("kimi-k2.7-code", checked_at=123.0)

        self.assertEqual(
            litellm_health.get_health("kimi-k2.7-code"),
            {"status": "healthy", "error": None, "checked_at": 123.0},
        )

    def test_new_alias_does_not_inherit_another_alias_checked_at(self):
        litellm_health.record_success("existing-model", checked_at=123.0)

        self.assertEqual(
            litellm_health.get_health("new-model"),
            {"status": "unknown", "error": None, "checked_at": None},
        )

    async def test_full_probe_overrides_runtime_success_as_authoritative_state(self):
        litellm_health.record_success("kimi-k2.7-code", checked_at=123.0)
        responses = {
            "info": {
                "data": [
                    {
                        "model_name": "kimi-k2.7-code",
                        "model_info": {"id": "uuid-k27"},
                    }
                ]
            },
            "health": {
                "healthy_endpoints": [],
                "unhealthy_endpoints": [
                    {
                        "model_id": "uuid-k27",
                        "error": "AuthenticationError: invalid api key",
                    }
                ],
            },
        }

        with (
            patch.object(
                litellm_health.httpx,
                "AsyncClient",
                side_effect=lambda **kwargs: _FakeAsyncClient(responses, **kwargs),
            ),
            patch.object(litellm_health.time, "time", return_value=456.0),
        ):
            await litellm_health._fetch_once()

        self.assertEqual(
            litellm_health.get_health("kimi-k2.7-code"),
            {
                "status": "unhealthy",
                "error": "服务商认证失败：API key 无效或已过期，请联系管理员补全密钥",
                "checked_at": 456.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
