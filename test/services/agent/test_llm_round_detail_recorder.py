import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentLlmRoundDetail, AgentSession, Message
from app.services.agent import llm_round_detail_recorder as recorder_module
from app.services.agent.llm_round_detail_recorder import (
    LLM_DETAIL_PREVIEW_MAX_CHARS,
    LlmRoundDetailDraft,
    schedule_llm_round_detail,
    stop_llm_round_detail_workers,
)


class LlmRoundDetailRecorderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "llm-round-detail.sqlite3"
        self.engine = create_engine(
            f"sqlite:///{database_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        with self.Session() as db:
            db.add(
                AgentSession(
                    id="run-1",
                    conversation_id="conv-1",
                    message_id="msg-1",
                    user_id="user-1",
                    model_id="deepseek-chat",
                    provider="deepseek",
                    status="running",
                )
            )
            db.commit()

    async def asyncTearDown(self) -> None:
        await stop_llm_round_detail_workers()
        self.engine.dispose()
        self.temp_dir.cleanup()

    @staticmethod
    def _draft(**overrides) -> LlmRoundDetailDraft:
        values = {
            "conversation_id": "conv-1",
            "run_id": "run-1",
            "message_id": "msg-1",
            "llm_round_id": "round-1",
            "reasoning_text": "先调用 web_search，再整理结果。" + "推理" * 150,
            "content_text": "最终答案" * 100,
        }
        values.update(overrides)
        return LlmRoundDetailDraft(**values)

    async def test_persists_only_sanitized_visible_text_and_bounded_previews(self):
        task = schedule_llm_round_detail(self._draft(), session_factory=self.Session)
        await task

        with self.Session() as db:
            row = db.scalar(select(AgentLlmRoundDetail))
            self.assertIsNotNone(row)
            self.assertNotIn("web_search", row.reasoning_text)
            self.assertIn("联网搜索", row.reasoning_text)
            self.assertLessEqual(len(row.reasoning_preview), LLM_DETAIL_PREVIEW_MAX_CHARS)
            self.assertLessEqual(len(row.output_preview), LLM_DETAIL_PREVIEW_MAX_CHARS)
            self.assertEqual(row.redacted_fields, ["reasoning_text"])
            self.assertEqual(row.truncated_fields, [])
            self.assertIsNone(row.message_id)

    async def test_duplicate_round_is_idempotent_and_first_write_wins(self):
        first = schedule_llm_round_detail(self._draft(content_text="第一次"), session_factory=self.Session)
        await first
        duplicate = schedule_llm_round_detail(self._draft(content_text="第二次"), session_factory=self.Session)
        await duplicate

        with self.Session() as db:
            rows = db.scalars(select(AgentLlmRoundDetail)).all()
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].content_text, "第一次")

    async def test_existing_message_id_is_preserved(self):
        with self.Session() as db:
            db.add(Message(id="msg-existing", conversation_id="conv-1", role="assistant", content=[]))
            db.commit()

        task = schedule_llm_round_detail(
            self._draft(message_id="msg-existing"),
            session_factory=self.Session,
        )
        await task

        with self.Session() as db:
            row = db.scalar(select(AgentLlmRoundDetail))
            self.assertEqual(row.message_id, "msg-existing")

    async def test_completed_task_is_released_from_controlled_task_set(self):
        task = schedule_llm_round_detail(self._draft(), session_factory=self.Session)
        self.assertIn(task, recorder_module._worker_tasks)

        await task
        await asyncio.sleep(0)

        self.assertNotIn(task, recorder_module._worker_tasks)

    async def test_background_write_failure_marks_run_detail_degraded_without_raising(self):
        on_degraded = Mock()

        def broken_session_factory():
            raise RuntimeError("database unavailable")

        task = schedule_llm_round_detail(
            self._draft(),
            session_factory=broken_session_factory,
            on_degraded=on_degraded,
        )
        await task

        on_degraded.assert_called_once_with("llm_detail_write_failed")

    async def test_shutdown_cancels_and_observes_pending_tasks(self):
        pending = asyncio.create_task(asyncio.sleep(60))
        recorder_module._worker_tasks.add(pending)
        pending.add_done_callback(recorder_module._worker_tasks.discard)

        await stop_llm_round_detail_workers()

        self.assertTrue(pending.cancelled())
        self.assertNotIn(pending, recorder_module._worker_tasks)


if __name__ == "__main__":
    unittest.main()
