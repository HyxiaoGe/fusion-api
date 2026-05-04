"""session_cache 单元测试 — mock SessionLocal 验证 ORM 操作"""
import unittest
from unittest.mock import MagicMock, patch

from app.services.agent.session_cache import (
    write_session_started,
    write_session_status,
    write_step_completed,
    write_step_started,
    write_step_terminal,
)


class SessionCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_write_session_started_inserts_row(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            await write_session_started(run_id="r1", conversation_id="c1",
                                        user_id="u1", model_id="gpt-4",
                                        provider="openai")
            session.add.assert_called_once()
            session.commit.assert_called_once()
            row = session.add.call_args.args[0]
            self.assertEqual(row.id, "r1")
            self.assertEqual(row.conversation_id, "c1")
            self.assertEqual(row.user_id, "u1")
            self.assertEqual(row.model_id, "gpt-4")
            self.assertEqual(row.provider, "openai")
            self.assertEqual(row.status, "running")
            self.assertEqual(row.total_steps, 0)
            self.assertEqual(row.total_tool_calls, 0)

    async def test_write_step_started_inserts_running(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            await write_step_started(run_id="r1", step_id="s1", step_number=1)
            session.add.assert_called_once()
            session.commit.assert_called_once()
            added = session.add.call_args.args[0]
            self.assertEqual(added.id, "s1")
            self.assertEqual(added.trace_id, "r1")
            self.assertEqual(added.step_number, 1)
            self.assertEqual(added.status, "running")

    async def test_write_step_completed_updates_to_completed(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_step_completed(step_id="s1", tool_names=["web_search"],
                                       duration_ms=42)
            session.get.assert_called_once()
            self.assertEqual(row.status, "completed")
            self.assertEqual(row.tool_names, ["web_search"])
            self.assertEqual(row.duration_ms, 42)
            session.commit.assert_called_once()

    async def test_write_step_completed_missing_row_silently_returns(self):
        """row 不存在时 silently return（极少发生但不报错）"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None
            await write_step_completed(step_id="missing", tool_names=None,
                                       duration_ms=0)
            session.commit.assert_not_called()

    async def test_write_step_terminal_sets_failed(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_step_terminal(step_id="s1", status="failed")
            self.assertEqual(row.status, "failed")
            session.commit.assert_called_once()

    async def test_write_step_terminal_sets_interrupted(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_step_terminal(step_id="s1", status="interrupted")
            self.assertEqual(row.status, "interrupted")

    async def test_write_step_terminal_rejects_invalid_status(self):
        """status 必须是 failed 或 interrupted"""
        with patch("app.services.agent.session_cache.SessionLocal"):
            with self.assertRaises(AssertionError):
                await write_step_terminal(step_id="s1", status="completed")

    async def test_write_session_status_updates_terminal(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_session_status(run_id="r1", status="interrupted",
                                       total_steps=2, total_tool_calls=3)
            self.assertEqual(row.status, "interrupted")
            self.assertEqual(row.total_steps, 2)
            self.assertEqual(row.total_tool_calls, 3)

    async def test_write_session_status_rejects_invalid_status(self):
        """status 必须是 4 个终态值之一"""
        with patch("app.services.agent.session_cache.SessionLocal"):
            with self.assertRaises(AssertionError):
                await write_session_status(run_id="r1", status="bogus",
                                           total_steps=0, total_tool_calls=0)

    async def test_write_session_status_missing_row_silently_returns(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None
            await write_session_status(run_id="missing", status="completed",
                                       total_steps=0, total_tool_calls=0)
            session.commit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
