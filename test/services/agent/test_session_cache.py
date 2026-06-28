"""session_cache 单元测试 — mock SessionLocal 验证 ORM 操作"""

import os
import unittest
from unittest.mock import MagicMock, patch

# 防御层：万一 unittest discover 没把 test/ 当 package，这里兜底
os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")

from app.services.agent.session_cache import (  # noqa: E402
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
            session.get.return_value = None  # 明确无已有行
            await write_session_started(
                run_id="r1", conversation_id="c1", user_id="u1", model_id="gpt-4", provider="openai"
            )
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

    async def test_write_session_started_persists_run_config(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None

            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="msg-1",
                run_config={"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300},
            )

            row = session.add.call_args.args[0]
            self.assertEqual(row.run_config, {"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300})

    async def test_write_session_started_updates_existing_run_config(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            existing = MagicMock()
            session.get.return_value = existing

            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="msg-1",
                run_config={"max_steps": 4, "max_tool_calls": 7, "timeout_s": 90},
            )

            self.assertEqual(existing.run_config, {"max_steps": 4, "max_tool_calls": 7, "timeout_s": 90})

    async def test_write_session_started_upserts_existing_row(self):
        """同 run_id 二次调用：不 add 新行，而是更新已有 row 的字段并重置 totals"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            # 模拟已有行
            existing = MagicMock()
            existing.total_steps = 5
            existing.total_tool_calls = 3
            existing.total_duration_ms = 1234
            existing.limit_reason = "max_steps"
            existing.error_message = "旧错误"
            existing.status = "completed"
            session.get.return_value = existing

            await write_session_started(
                run_id="r1", conversation_id="c2", user_id="u2", model_id="gpt-5", provider="anthropic"
            )

            # 不应 add 新行
            session.add.assert_not_called()
            # 应该更新已有行的元信息 + 重置 totals + 重置 status
            self.assertEqual(existing.conversation_id, "c2")
            self.assertEqual(existing.user_id, "u2")
            self.assertEqual(existing.model_id, "gpt-5")
            self.assertEqual(existing.provider, "anthropic")
            self.assertEqual(existing.status, "running")
            self.assertEqual(existing.total_steps, 0)
            self.assertEqual(existing.total_tool_calls, 0)
            self.assertIsNone(existing.total_duration_ms)
            self.assertIsNone(existing.limit_reason)
            self.assertIsNone(existing.error_message)
            session.commit.assert_called_once()

    async def test_write_session_started_inserts_when_no_existing_row(self):
        """没有已有行：行为同 v1（add 新行）"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None  # 无已有行

            await write_session_started(
                run_id="r1", conversation_id="c1", user_id="u1", model_id="gpt-4", provider="openai"
            )

            session.add.assert_called_once()
            row = session.add.call_args.args[0]
            self.assertEqual(row.id, "r1")
            self.assertEqual(row.status, "running")
            session.commit.assert_called_once()

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
            await write_step_completed(step_id="s1", tool_names=["web_search"], duration_ms=42)
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
            await write_step_completed(step_id="missing", tool_names=None, duration_ms=0)
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
            with self.assertRaises(ValueError):
                await write_step_terminal(step_id="s1", status="completed")

    async def test_write_session_status_updates_terminal(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_session_status(run_id="r1", status="interrupted", total_steps=2, total_tool_calls=3)
            self.assertEqual(row.status, "interrupted")
            self.assertEqual(row.total_steps, 2)
            self.assertEqual(row.total_tool_calls, 3)

    async def test_write_session_status_accepts_incomplete(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_session_status(run_id="r1", status="incomplete", total_steps=1, total_tool_calls=0)
            self.assertEqual(row.status, "incomplete")
            self.assertEqual(row.total_steps, 1)
            self.assertEqual(row.total_tool_calls, 0)
            session.commit.assert_called_once()

    async def test_write_session_status_rejects_invalid_status(self):
        """status 必须是声明过的终态值之一"""
        with patch("app.services.agent.session_cache.SessionLocal"):
            with self.assertRaises(ValueError):
                await write_session_status(run_id="r1", status="bogus", total_steps=0, total_tool_calls=0)

    async def test_write_session_status_missing_row_silently_returns(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None
            await write_session_status(run_id="missing", status="completed", total_steps=0, total_tool_calls=0)
            session.commit.assert_not_called()

    async def test_write_step_completed_sets_tool_calls_count(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_step_completed(
                step_id="s1", tool_names=["web_search", "url_read"], tool_calls_count=2, duration_ms=42
            )
            self.assertEqual(row.tool_calls_count, 2)

    async def test_write_step_completed_tool_calls_count_none_skipped(self):
        """None 时不动 tool_calls_count（不覆盖 row 既有值）"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            # 模拟 row 已有 tool_calls_count=5
            row.tool_calls_count = 5
            session.get.return_value = row
            await write_step_completed(step_id="s1", duration_ms=10)
            self.assertEqual(row.tool_calls_count, 5)  # 未被覆盖

    async def test_write_session_status_sets_total_duration_ms(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_session_status(
                run_id="r1", status="completed", total_steps=2, total_tool_calls=3, total_duration_ms=12345
            )
            self.assertEqual(row.total_duration_ms, 12345)

    async def test_write_session_status_persists_limit_reason(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            await write_session_status(
                run_id="r1",
                status="limit_reached",
                total_steps=2,
                total_tool_calls=3,
                limit_reason="max_steps",
            )
            self.assertEqual(row.limit_reason, "max_steps")


if __name__ == "__main__":
    unittest.main()
