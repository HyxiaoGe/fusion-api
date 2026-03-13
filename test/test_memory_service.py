import unittest
from unittest.mock import MagicMock

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
        self.service.repo.get_paginated.return_value = (["conv-1", "conv-2"], 5)

        result = self.service.get_conversations_paginated("user-1", page=2, page_size=2)

        self.assertEqual(
            result,
            {
                "items": ["conv-1", "conv-2"],
                "total": 5,
                "page": 2,
                "page_size": 2,
                "total_pages": 3,
                "has_next": True,
                "has_prev": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
