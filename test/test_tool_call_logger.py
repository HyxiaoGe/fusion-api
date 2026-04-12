"""
tool_call_logger 单元测试
"""

import unittest
from unittest.mock import MagicMock, patch

from app.db.models import ToolCallLog
from app.db.repositories import ToolCallLogRepository
from app.services.tool_call_logger import log_tool_call


class ToolCallLogModelTests(unittest.TestCase):
    def test_model_has_expected_columns(self):
        """ToolCallLog 模型包含所有预期字段"""
        expected_columns = {
            "id", "conversation_id", "message_id", "user_id",
            "tool_name", "status", "error_message", "duration_ms",
            "model_id", "provider",
            "input_params", "output_data", "metadata",
            "created_at",
        }
        actual_columns = {c.name for c in ToolCallLog.__table__.columns}
        self.assertEqual(expected_columns, actual_columns)

    def test_tablename(self):
        self.assertEqual(ToolCallLog.__tablename__, "tool_call_logs")


class ToolCallLogRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.mock_db = MagicMock()
        self.repo = ToolCallLogRepository(self.mock_db)

    def test_create_adds_and_commits(self):
        """create() 写入数据库并 commit"""
        log = self.repo.create(
            conversation_id="conv-1",
            message_id="msg-1",
            user_id="user-1",
            tool_name="web_search",
            status="success",
            duration_ms=350,
            model_id="gpt-4",
            provider="openai",
            input_params={"query": "test"},
            output_data={"result_count": 3, "sources": []},
        )

        self.mock_db.add.assert_called_once()
        self.mock_db.commit.assert_called_once()
        self.assertEqual(log.tool_name, "web_search")
        self.assertEqual(log.status, "success")
        self.assertEqual(log.duration_ms, 350)

    def test_create_handles_error(self):
        """create() 异常时 rollback 并返回 None"""
        self.mock_db.commit.side_effect = Exception("DB error")

        log = self.repo.create(
            conversation_id="conv-1",
            message_id="msg-1",
            user_id="user-1",
            tool_name="web_search",
            status="failed",
            duration_ms=100,
            model_id="gpt-4",
            provider="openai",
        )

        self.assertIsNone(log)
        self.mock_db.rollback.assert_called_once()


class LogToolCallTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.services.tool_call_logger.SessionLocal")
    async def test_log_tool_call_creates_record(self, mock_session_cls):
        """log_tool_call 使用独立 session 写入记录"""
        mock_db = MagicMock()
        mock_session_cls.return_value = mock_db

        await log_tool_call(
            conversation_id="conv-1",
            message_id="msg-1",
            user_id="user-1",
            tool_name="web_search",
            status="success",
            duration_ms=200,
            model_id="gpt-4",
            provider="openai",
            input_params={"query": "test"},
            output_data={"result_count": 1, "sources": []},
            log_id="custom-log-id",
        )

        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()
        mock_db.close.assert_called_once()
        # 验证使用了自定义 log_id
        added_obj = mock_db.add.call_args[0][0]
        self.assertEqual(added_obj.id, "custom-log-id")

    @patch("app.services.tool_call_logger.SessionLocal")
    async def test_log_tool_call_handles_error_gracefully(self, mock_session_cls):
        """log_tool_call 异常时不抛出，静默失败"""
        mock_db = MagicMock()
        mock_db.commit.side_effect = Exception("DB down")
        mock_session_cls.return_value = mock_db

        # 不应抛异常
        await log_tool_call(
            conversation_id="conv-1",
            message_id="msg-1",
            user_id="user-1",
            tool_name="web_search",
            status="failed",
            duration_ms=100,
            model_id="gpt-4",
            provider="openai",
        )

        mock_db.rollback.assert_called_once()
        mock_db.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
