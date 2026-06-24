import unittest
from unittest.mock import patch

import httpx


class SearchClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_search_web_propagates_provider_metadata_to_sources(self):
        from app.services.external.search_client import search_web

        calls = []

        class FakeAsyncClient:
            def __init__(self, timeout: int):
                self.timeout = timeout

            async def __aenter__(self):
                return self

            async def __aexit__(self, _exc_type, _exc, _tb):
                return False

            async def post(self, url: str, json: dict):
                calls.append({"url": url, "json": json, "timeout": self.timeout})
                return httpx.Response(
                    200,
                    json={
                        "provider": "brave",
                        "requested_provider": "firecrawl",
                        "result_provider": "brave",
                        "fallback_used": True,
                        "provider_chain": ["firecrawl", "brave"],
                        "results": [
                            {
                                "title": "Result",
                                "url": "https://example.com",
                                "description": "desc",
                                "favicon": "https://example.com/favicon.ico",
                            }
                        ],
                    },
                    request=httpx.Request("POST", url),
                )

        with patch("app.services.external.search_client.httpx.AsyncClient", FakeAsyncClient):
            sources = await search_web("test query", count=5)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].requested_provider, "firecrawl")
        self.assertEqual(sources[0].result_provider, "brave")
        self.assertTrue(sources[0].fallback_used)
        self.assertEqual(sources[0].provider_chain, ["firecrawl", "brave"])
        self.assertEqual(calls[0]["json"]["freshness"], "pw")
