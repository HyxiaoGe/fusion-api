import unittest

from app.services.chat.stream_processor import StreamProcessor


class StreamProcessorTests(unittest.TestCase):
    def test_create_tool_synthesis_messages_adds_user_message_for_google(self):
        messages = StreamProcessor.create_tool_synthesis_messages(
            "latest fusion news",
            "web_search",
            {"results": []},
            provider="google",
        )

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1], {"role": "user", "content": "latest fusion news"})

    def test_create_tool_synthesis_messages_returns_only_system_message_for_other_providers(self):
        messages = StreamProcessor.create_tool_synthesis_messages(
            "latest fusion news",
            "web_search",
            {"results": []},
            provider="openai",
        )

        self.assertEqual(len(messages), 1)


if __name__ == "__main__":
    unittest.main()
