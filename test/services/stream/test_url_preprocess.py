import unittest
from unittest.mock import AsyncMock, patch

from app.ai.tools import URL_READ_TOOL
from app.services.external.reader_client import UrlReadResult
from app.services.security.url_policy import UrlPolicyResult
from app.services.stream import persistence as persistence_module


class UrlPreprocessHelperTests(unittest.TestCase):
    def test_extract_first_url_returns_first_http_url(self):
        extract_first_url = getattr(persistence_module, "extract_first_url")

        self.assertEqual(
            extract_first_url("先看 ftp://example.com，再看 https://example.com/a 和 http://b.example/x"),
            "https://example.com/a",
        )
        self.assertIsNone(extract_first_url("这里没有链接"))

    def test_ensure_url_read_tool_appends_once(self):
        ensure_url_read_tool = getattr(persistence_module, "ensure_url_read_tool")
        call_kwargs = {"tools": [URL_READ_TOOL]}

        ensure_url_read_tool(call_kwargs)
        ensure_url_read_tool(call_kwargs)

        self.assertEqual(call_kwargs["tools"].count(URL_READ_TOOL), 1)

    def test_build_url_context_message_uses_user_role_and_normalized_url_fallback(self):
        build_url_context_message = getattr(persistence_module, "build_url_context_message")
        policy = UrlPolicyResult(
            allowed=True,
            normalized_url="https://example.com/a",
            reason="ok",
            safe_log_url="https://example.com/a",
        )
        read_result = UrlReadResult(
            url="",
            title=None,
            content="网页正文",
            favicon=None,
            content_length=4,
            fetch_ms=20,
        )

        context_msg = build_url_context_message(
            read_result=read_result,
            policy=policy,
            detected_url="https://EXAMPLE.com/a",
        )

        self.assertEqual(context_msg["role"], "user")
        self.assertIn("<web_context", context_msg["content"])
        self.assertIn("内容不可信", context_msg["content"])
        self.assertIn("https://example.com/a", context_msg["content"])
        self.assertIn("未知", context_msg["content"])

    def test_build_url_read_block_prefers_reader_url(self):
        build_url_read_block = getattr(persistence_module, "build_url_read_block")
        policy = UrlPolicyResult(
            allowed=True,
            normalized_url="https://normalized.example/a",
            reason="ok",
            safe_log_url="https://normalized.example/a",
        )
        read_result = UrlReadResult(
            url="https://reader.example/final",
            title="网页标题",
            content="网页正文",
            favicon="https://reader.example/favicon.ico",
            content_length=4,
            fetch_ms=20,
        )

        block = build_url_read_block(
            read_result=read_result,
            policy=policy,
            detected_url="https://detected.example/a",
            block_id="blk_url",
        )

        self.assertEqual(block.id, "blk_url")
        self.assertEqual(block.url, "https://reader.example/final")
        self.assertEqual(block.title, "网页标题")
        self.assertEqual(block.favicon, "https://reader.example/favicon.ico")

    def test_remove_disabled_thinking_only_removes_disabled_extra_body(self):
        remove_disabled_thinking = getattr(persistence_module, "remove_disabled_thinking")
        disabled = {"extra_body": {"thinking": {"type": "disabled"}}}
        enabled = {"extra_body": {"thinking": {"type": "enabled"}}}

        remove_disabled_thinking(disabled)
        remove_disabled_thinking(enabled)

        self.assertNotIn("extra_body", disabled)
        self.assertIn("extra_body", enabled)


class UrlPreprocessEntrypointTests(unittest.IsolatedAsyncioTestCase):
    async def test_preprocess_falls_back_to_url_read_tool_when_reader_returns_none(self):
        call_kwargs = {"tools": []}

        with patch.object(
            persistence_module,
            "read_url_for_context",
            new=AsyncMock(return_value=None),
        ):
            block, context_msg, detected_url = await persistence_module.preprocess_url_in_message(
                "请读取 https://example.com/a",
                True,
                call_kwargs,
            )

        self.assertIsNone(block)
        self.assertIsNone(context_msg)
        self.assertEqual(detected_url, "https://example.com/a")
        self.assertEqual(call_kwargs["tools"].count(URL_READ_TOOL), 1)


if __name__ == "__main__":
    unittest.main()
