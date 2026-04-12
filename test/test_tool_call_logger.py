"""
tool_call_logger 单元测试
"""

import unittest

from app.db.models import ToolCallLog


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


if __name__ == "__main__":
    unittest.main()
