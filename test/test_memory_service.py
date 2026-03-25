import unittest
from datetime import datetime
from unittest.mock import MagicMock

from app.schemas.chat import Conversation
from app.services.memory_service import MemoryService


class MemoryServiceTests(unittest.TestCase):
    def setUp(self):
        self.service = MemoryService(MagicMock())
        self.service.repo = MagicMock()

    def test_save_conversation_creates_when_missing(self):
        conversation = MagicMock(id="conv-1", user_id="user-1")
        self.service.repo.get_by_id.return_value = None

        result = self.service.save_conversation(conversation)

        self.assertTrue(result)
        self.service.repo.create.assert_called_once_with(conversation)
        self.service.repo.update.assert_not_called()

    def test_save_conversation_updates_when_existing(self):
        conversation = MagicMock(id="conv-1", user_id="user-1")
        self.service.repo.get_by_id.return_value = conversation

        result = self.service.save_conversation(conversation)

        self.assertTrue(result)
        self.service.repo.update.assert_called_once_with(conversation)
        self.service.repo.create.assert_not_called()

    def test_get_conversations_paginated_builds_pagination_flags(self):
        now = datetime.now()
        mock_conversations = [
            Conversation(
                id=f"conv-{i}",
                user_id="user-1",
                model_id="qwen-max",
                title=f"Title {i}",
                messages=[],
                created_at=now,
                updated_at=now,
            )
            for i in range(1, 3)
        ]
        self.service.repo.get_paginated.return_value = (mock_conversations, 5)

        result = self.service.get_conversations_paginated("user-1", page=2, page_size=2)

        self.assertEqual(result["total"], 5)
        self.assertEqual(result["page"], 2)
        self.assertEqual(result["page_size"], 2)
        self.assertEqual(result["total_pages"], 3)
        self.assertTrue(result["has_next"])
        self.assertTrue(result["has_prev"])
        # 返回的是 ConversationSummary，不含 messages
        self.assertEqual(len(result["items"]), 2)
        item = result["items"][0]
        self.assertEqual(item.id, "conv-1")
        self.assertEqual(item.model_id, "qwen-max")
        self.assertFalse(hasattr(item, "messages"))


if __name__ == "__main__":
    unittest.main()
