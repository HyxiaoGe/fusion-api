import unittest
from datetime import datetime, timezone

from app.db.models import Message as MessageModel
from app.db.repositories import ConversationRepository


class MessageRepositoryTests(unittest.TestCase):
    def test_convert_message_preserves_search_provider_metadata(self):
        """从 JSONB 重建 SearchBlock 时保留搜索提供方元信息"""
        db_message = MessageModel(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            created_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
            content=[
                {
                    "type": "search",
                    "id": "search-1",
                    "query": "AI 标准",
                    "tool_call_log_id": "log-1",
                    "sources": [{"title": "来源", "url": "https://example.com"}],
                    "requested_provider": "firecrawl",
                    "result_provider": "brave",
                    "fallback_used": True,
                    "provider_chain": ["firecrawl", "brave"],
                }
            ],
        )

        message = ConversationRepository(None)._convert_message_to_schema(db_message)

        search_block = message.content[0]
        self.assertEqual(search_block.requested_provider, "firecrawl")
        self.assertEqual(search_block.result_provider, "brave")
        self.assertTrue(search_block.fallback_used)
        self.assertEqual(search_block.provider_chain, ["firecrawl", "brave"])
