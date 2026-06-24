"""session_cache message_id 关联测试"""

import os
import unittest
from unittest.mock import MagicMock, patch

# 防御层：保持与同目录 session_cache 测试一致
os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")

from app.services.agent.session_cache import write_session_started  # noqa: E402


class SessionCacheMessageIdTests(unittest.IsolatedAsyncioTestCase):
    async def test_write_session_started_inserts_message_id(self):
        """新建 agent session 时写入最终 assistant message_id"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None

            await write_session_started(
                run_id="run-1",
                conversation_id="conv-1",
                user_id="user-1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
            )

            session.add.assert_called_once()
            row = session.add.call_args.args[0]
            self.assertEqual(row.message_id, "assistant-1")
            session.commit.assert_called_once()

    async def test_write_session_started_updates_message_id(self):
        """已有 agent session 重开时同步更新最终 assistant message_id"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            existing = MagicMock()
            existing.message_id = "old-assistant"
            session.get.return_value = existing

            await write_session_started(
                run_id="run-1",
                conversation_id="conv-2",
                user_id="user-2",
                model_id="gpt-5",
                provider="anthropic",
                message_id="assistant-1",
            )

            session.add.assert_not_called()
            self.assertEqual(existing.message_id, "assistant-1")
            session.commit.assert_called_once()
