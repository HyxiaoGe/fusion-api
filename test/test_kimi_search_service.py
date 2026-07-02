import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.external import kimi_search_service


class KimiSearchServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_fetch_trending_questions_does_not_attach_proxy_tags_to_direct_moonshot_call(self):
        message = SimpleNamespace(
            content='[{"category": "news", "question": "今天有什么热点？"}]',
            tool_calls=None,
        )
        response = SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=message)])
        client = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)
        client.close = AsyncMock()

        with (
            patch.object(kimi_search_service.settings, "MOONSHOT_API_KEY", "test-key"),
            patch("app.services.external.kimi_search_service.AsyncOpenAI", return_value=client),
        ):
            result = await kimi_search_service.fetch_trending_questions()

        self.assertEqual(result, [{"category": "news", "question": "今天有什么热点？"}])
        self.assertEqual(
            client.chat.completions.create.await_args.kwargs["extra_body"],
            {"thinking": {"type": "disabled"}},
        )
        client.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
