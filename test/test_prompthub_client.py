import unittest

import httpx


def _bundle_response() -> dict:
    return {
        "code": 0,
        "message": "success",
        "data": {
            "project_id": "project-1",
            "project_slug": "fusion",
            "revision": "a" * 64,
            "prompts": [
                {
                    "id": "prompt-1",
                    "slug": "app-identity",
                    "name": "Fusion 应用身份",
                    "version": "1.0.0",
                    "status": "published",
                    "content": "【Fusion 身份一致性规则】测试",
                    "variables": [],
                    "format": "text",
                    "template_engine": "none",
                    "published_at": "2026-07-10T00:00:00Z",
                }
            ],
        },
    }


class PromptHubPublishedBundleClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetches_published_bundle_with_bearer_token(self):
        from app.services.external.prompthub_client import PromptHubPublishedBundleClient

        async def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/v1/projects/by-slug/fusion/prompts/published")
            self.assertEqual(request.headers["Authorization"], "Bearer service-secret")
            return httpx.Response(200, json=_bundle_response())

        client = PromptHubPublishedBundleClient(
            base_url="http://prompthub.local/",
            api_key="service-secret",
            project_slug="fusion",
            timeout_seconds=3,
            transport=httpx.MockTransport(handler),
        )

        bundle = await client.fetch_published_bundle()

        self.assertEqual(bundle.project_slug, "fusion")
        self.assertEqual(bundle.revision, "a" * 64)
        self.assertEqual(bundle.prompts[0].slug, "app-identity")

    async def test_translates_timeout_without_leaking_token(self):
        from app.services.external.prompthub_client import (
            PromptHubClientError,
            PromptHubPublishedBundleClient,
        )

        async def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("service-secret", request=request)

        client = PromptHubPublishedBundleClient(
            base_url="http://prompthub.local",
            api_key="service-secret",
            project_slug="fusion",
            transport=httpx.MockTransport(handler),
        )

        with self.assertRaises(PromptHubClientError) as raised:
            await client.fetch_published_bundle()

        self.assertEqual(raised.exception.kind, "timeout")
        self.assertNotIn("service-secret", str(raised.exception))

    async def test_translates_http_errors(self):
        from app.services.external.prompthub_client import (
            PromptHubClientError,
            PromptHubPublishedBundleClient,
        )

        for status in (401, 404, 503):
            with self.subTest(status=status):
                transport = httpx.MockTransport(lambda request: httpx.Response(status, request=request))
                client = PromptHubPublishedBundleClient(
                    base_url="http://prompthub.local",
                    api_key="service-secret",
                    project_slug="fusion",
                    transport=transport,
                )

                with self.assertRaises(PromptHubClientError) as raised:
                    await client.fetch_published_bundle()

                self.assertEqual(raised.exception.kind, "http")
                self.assertEqual(raised.exception.status_code, status)

    async def test_rejects_bad_json_and_invalid_envelope(self):
        from app.services.external.prompthub_client import (
            PromptHubClientError,
            PromptHubPublishedBundleClient,
        )

        responses = [
            httpx.Response(200, content=b"{"),
            httpx.Response(200, json={"code": 0, "data": []}),
            httpx.Response(200, json={"code": 500, "message": "failed", "data": None}),
        ]
        for response in responses:
            with self.subTest(body=response.content):
                transport = httpx.MockTransport(
                    lambda request, response=response: httpx.Response(
                        response.status_code,
                        content=response.content,
                        headers={"content-type": "application/json"},
                        request=request,
                    )
                )
                client = PromptHubPublishedBundleClient(
                    base_url="http://prompthub.local",
                    api_key="service-secret",
                    project_slug="fusion",
                    transport=transport,
                )

                with self.assertRaises(PromptHubClientError) as raised:
                    await client.fetch_published_bundle()

                self.assertEqual(raised.exception.kind, "invalid_response")


if __name__ == "__main__":
    unittest.main()
