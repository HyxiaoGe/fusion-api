import asyncio
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import Conversation, Message, User
from app.services.suggested_question_service import SuggestedQuestionService
from app.services.suggested_question_worker import (
    recover_pending_suggested_questions,
    run_suggested_question_generation_worker,
    schedule_suggested_question_generation,
    stop_suggested_question_workers,
)
from app.services.task_manager import get_task


class SuggestedQuestionWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        db = self.Session()
        db.add(User(id="user-1", username="user-1"))
        db.add(Conversation(id="conv-1", user_id="user-1", title="恢复测试", model_id="deepseek-chat"))
        db.add(
            Message(
                id="assistant-msg-1",
                conversation_id="conv-1",
                role="assistant",
                content=[{"type": "text", "id": "answer-1", "text": "正式回答"}],
                model_id="deepseek-chat",
            )
        )
        db.commit()
        self.claim = SuggestedQuestionService(db).claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )
        db.close()

    def tearDown(self):
        self.engine.dispose()

    async def test_worker_cancellation_marks_owned_revision_failed_with_new_session(self):
        started = asyncio.Event()

        async def wait_forever(_service, _dialog_content, _model_id):
            started.set()
            await asyncio.Event().wait()

        with patch.object(SuggestedQuestionService, "_generate", wait_forever):
            task = asyncio.create_task(
                run_suggested_question_generation_worker(
                    self.claim,
                    session_factory=self.Session,
                )
            )
            await started.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        db = self.Session()
        try:
            message = db.query(Message).filter(Message.id == "assistant-msg-1").one()
            self.assertEqual(message.suggested_questions_status, "failed")
        finally:
            db.close()

    async def test_startup_recovery_reschedules_orphan_pending_with_same_revision(self):
        recovered = []

        count = await recover_pending_suggested_questions(
            session_factory=self.Session,
            schedule_fn=recovered.append,
        )

        self.assertEqual(count, 1)
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].message_id, "assistant-msg-1")
        self.assertEqual(recovered[0].revision, self.claim.revision)

    async def test_worker_exception_marks_owned_revision_failed(self):
        with patch.object(
            SuggestedQuestionService,
            "generate_claimed_questions",
            side_effect=RuntimeError("unexpected worker failure"),
        ):
            await run_suggested_question_generation_worker(
                self.claim,
                session_factory=self.Session,
            )

        db = self.Session()
        try:
            message = db.query(Message).filter(Message.id == "assistant-msg-1").one()
            self.assertEqual(message.suggested_questions_status, "failed")
        finally:
            db.close()

    async def test_worker_is_independent_from_chat_task_registry_and_shutdown_keeps_pending(self):
        started = asyncio.Event()

        async def wait_forever(_service, _dialog_content, _model_id):
            started.set()
            await asyncio.Event().wait()

        with patch.object(SuggestedQuestionService, "_generate", wait_forever):
            task = schedule_suggested_question_generation(
                self.claim,
                session_factory=self.Session,
            )
            await started.wait()
            self.assertIsNone(get_task("conv-1"))
            self.assertFalse(task.done())
            await stop_suggested_question_workers()

        db = self.Session()
        try:
            message = db.query(Message).filter(Message.id == "assistant-msg-1").one()
            self.assertEqual(message.suggested_questions_status, "pending")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
