import asyncio
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import Conversation, Message, User
from app.schemas.chat import SuggestedQuestionsRequest
from app.schemas.response import ApiException
from app.services.suggested_question_service import (
    SuggestedQuestionService,
    SuggestedQuestionTargetInvalidError,
    SuggestedQuestionTargetNotFoundError,
)


class SuggestedQuestionServiceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self.db.add(User(id="user-1", username="user-1"))
        self.db.add(
            Conversation(
                id="conv-1",
                user_id="user-1",
                title="推荐问题时序",
                model_id="deepseek-chat",
            )
        )
        self.db.add_all(
            [
                Message(
                    id="user-msg-1",
                    conversation_id="conv-1",
                    role="user",
                    sequence=1,
                    content=[{"type": "text", "id": "u-1", "text": "第一个问题"}],
                ),
                Message(
                    id="assistant-msg-1",
                    conversation_id="conv-1",
                    role="assistant",
                    sequence=2,
                    content=[{"type": "text", "id": "a-1", "text": "第一版正式回答"}],
                    model_id="deepseek-chat",
                ),
            ]
        )
        self.db.commit()
        self.service = SuggestedQuestionService(self.db)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _message(self):
        return self.db.query(Message).filter(Message.id == "assistant-msg-1").one()

    def test_auto_claim_is_idempotent_for_same_run_and_continuation_gets_new_revision(self):
        first = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )
        duplicate = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )

        message = self._message()
        message.content = [{"type": "text", "id": "a-1", "text": "续写后的第二版正式回答"}]
        message.suggested_questions = ["旧正文对应的问题"]
        message.suggested_questions_status = "ready"
        self.db.commit()

        continuation = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-2",
        )

        self.assertEqual(first.revision, 1)
        self.assertIsNone(duplicate)
        self.assertEqual(continuation.revision, 2)
        self.assertEqual(self._message().suggested_questions, ["旧正文对应的问题"])
        self.assertEqual(self._message().suggested_questions_status, "pending")
        self.assertEqual(self._message().suggested_questions_auto_run_id, "run-2")

    def test_auto_claim_requires_nonempty_formal_text(self):
        message = self._message()
        message.content = [{"type": "thinking", "id": "t-1", "thinking": "仍在思考"}]
        self.db.commit()

        claim = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )

        self.assertIsNone(claim)
        self.assertEqual(self._message().suggested_questions_revision, 0)
        self.assertEqual(self._message().suggested_questions_status, "idle")

    def test_request_rejects_missing_or_unowned_message_as_not_found(self):
        with self.assertRaises(SuggestedQuestionTargetNotFoundError):
            self.service.claim_request_generation(
                conversation_id="conv-1",
                user_id="user-1",
                assistant_message_id="missing-message",
                force_refresh=True,
            )
        with self.assertRaises(SuggestedQuestionTargetNotFoundError):
            self.service.claim_request_generation(
                conversation_id="conv-1",
                user_id="other-user",
                assistant_message_id="assistant-msg-1",
                force_refresh=True,
            )
        with self.assertRaises(SuggestedQuestionTargetNotFoundError):
            self.service.claim_request_generation(
                conversation_id="conv-1",
                user_id="user-1",
                assistant_message_id="user-msg-1",
                force_refresh=True,
            )

    def test_request_rejects_message_without_formal_text_as_invalid(self):
        message = self._message()
        message.content = [{"type": "thinking", "id": "t-1", "thinking": "思考"}]
        self.db.commit()

        with self.assertRaises(SuggestedQuestionTargetInvalidError):
            self.service.claim_request_generation(
                conversation_id="conv-1",
                user_id="user-1",
                assistant_message_id="assistant-msg-1",
                force_refresh=True,
            )

    def test_manual_claim_supersedes_auto_and_late_auto_result_is_discarded(self):
        auto_claim = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )
        manual_claim = self.service.claim_request_generation(
            conversation_id="conv-1",
            user_id="user-1",
            assistant_message_id="assistant-msg-1",
            force_refresh=True,
        ).claim

        auto_applied = self.service.store_generated_questions(
            claim=auto_claim,
            questions=["迟到的自动问题"],
        )
        manual_applied = self.service.store_generated_questions(
            claim=manual_claim,
            questions=["用户换一批的问题"],
        )

        self.assertFalse(auto_applied)
        self.assertTrue(manual_applied)
        self.assertEqual(self._message().suggested_questions_revision, 2)
        self.assertEqual(self._message().suggested_questions, ["用户换一批的问题"])
        self.assertEqual(self._message().suggested_questions_status, "ready")

    def test_claim_refreshes_stale_identity_map_before_incrementing_revision(self):
        stale_message = self._message()
        self.assertEqual(stale_message.suggested_questions_revision, 0)
        other_db = self.Session()
        try:
            first = SuggestedQuestionService(other_db).claim_auto_generation(
                assistant_message_id="assistant-msg-1",
                run_id="run-other-worker",
            )
            second = self.service.claim_request_generation(
                conversation_id="conv-1",
                user_id="user-1",
                assistant_message_id="assistant-msg-1",
                force_refresh=True,
            ).claim
        finally:
            other_db.close()

        self.assertEqual(first.revision, 1)
        self.assertEqual(second.revision, 2)
        self.assertEqual(self._message().suggested_questions_revision, 2)

    def test_non_force_request_reuses_ready_historical_questions(self):
        message = self._message()
        message.suggested_questions = ["历史问题"]
        message.suggested_questions_status = "ready"
        self.db.commit()

        result = self.service.claim_request_generation(
            conversation_id="conv-1",
            user_id="user-1",
            assistant_message_id=None,
            force_refresh=False,
        )

        self.assertIsNone(result.claim)
        self.assertEqual(result.questions, ["历史问题"])
        self.assertEqual(result.revision, 0)
        self.assertEqual(self._message().suggested_questions, ["历史问题"])

    def test_generation_context_is_locked_to_target_assistant_message(self):
        self.db.add_all(
            [
                Message(
                    id="user-msg-2",
                    conversation_id="conv-1",
                    role="user",
                    sequence=3,
                    content=[{"type": "text", "id": "u-2", "text": "后续问题"}],
                ),
                Message(
                    id="assistant-msg-2",
                    conversation_id="conv-1",
                    role="assistant",
                    sequence=4,
                    content=[{"type": "text", "id": "a-2", "text": "后续回答"}],
                ),
            ]
        )
        self.db.commit()

        content = self.service.build_dialog_content("assistant-msg-1")

        self.assertIn("第一个问题", content)
        self.assertIn("第一版正式回答", content)
        self.assertNotIn("后续问题", content)
        self.assertNotIn("后续回答", content)

    def test_old_non_force_request_claims_missing_questions_even_if_auto_is_pending(self):
        self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )

        result = self.service.claim_request_generation(
            conversation_id="conv-1",
            user_id="user-1",
            assistant_message_id=None,
            force_refresh=False,
        )

        self.assertIsNotNone(result.claim)
        self.assertEqual(result.claim.revision, 2)

    def test_generate_claimed_questions_uses_cas_and_limits_output(self):
        claim = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="1. 问题一\n2. 问题二\n3. 问题三\n4. 问题四")
                )
            ]
        )

        with (
            patch("app.services.suggested_question_service.llm_manager.resolve_model") as resolve_model,
            patch(
                "app.services.suggested_question_service.litellm.acompletion",
                new=AsyncMock(return_value=response),
            ) as completion,
        ):
            resolve_model.return_value = ("openai/deepseek-chat", "deepseek", {})
            result = asyncio.run(self.service.generate_claimed_questions(claim))

        self.assertTrue(result.applied)
        self.assertEqual(result.questions, ["问题一", "问题二", "问题三"])
        self.assertEqual(self._message().suggested_questions, ["问题一", "问题二", "问题三"])
        self.assertEqual(completion.await_args.kwargs["max_tokens"], 512)
        self.assertEqual(
            completion.await_args.kwargs["extra_body"],
            {
                "metadata": {
                    "tags": ["app:fusion", "phase:suggest_questions"],
                    "prompt_slug": "generate-suggested-questions",
                    "prompt_version": "code-default",
                }
            },
        )

    def test_non_llm_generation_failure_marks_owned_revision_failed_and_preserves_old_questions(self):
        message = self._message()
        message.suggested_questions = ["上一批问题"]
        message.suggested_questions_status = "ready"
        self.db.commit()
        claim = self.service.claim_auto_generation(
            assistant_message_id="assistant-msg-1",
            run_id="run-1",
        )

        with patch.object(self.service, "build_dialog_content", side_effect=RuntimeError("database read failed")):
            with self.assertRaises(RuntimeError):
                asyncio.run(self.service.generate_claimed_questions(claim))

        self.assertEqual(self._message().suggested_questions, ["上一批问题"])
        self.assertEqual(self._message().suggested_questions_status, "failed")

    def test_legacy_request_omits_force_refresh_for_api_compatibility(self):
        request = SuggestedQuestionsRequest(conversation_id="conv-1")

        self.assertIsNone(request.force_refresh)


class SuggestedQuestionsMigrationTests(unittest.TestCase):
    def test_upgrade_marks_historical_rows_with_questions_ready(self):
        migration_path = (
            Path(__file__).parents[1]
            / "alembic/versions/e8b4c2d7f901_add_suggested_question_generation_state.py"
        )
        spec = importlib.util.spec_from_file_location("suggested_question_migration", migration_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        migration = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(migration)
        statements = []

        with patch.object(migration, "op") as operation:
            operation.execute.side_effect = statements.append
            migration.upgrade()

        rendered = "\n".join(str(statement) for statement in statements)
        self.assertIn("suggested_questions_status = 'ready'", rendered)
        self.assertIn("suggested_questions IS NOT NULL", rendered)
        added_columns = [call.args[1] for call in operation.add_column.call_args_list]
        defaults = {column.name: str(column.server_default.arg) for column in added_columns if column.server_default}
        self.assertEqual(defaults["suggested_questions_revision"], "0")
        self.assertEqual(defaults["suggested_questions_status"], "'idle'")


class SuggestedQuestionsApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_request_defaults_to_force_refresh(self):
        from app.api.chat import suggest_questions

        chat_service = MagicMock()
        chat_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
        chat_service.generate_suggested_questions_result = AsyncMock(
            return_value=SimpleNamespace(
                questions=["问题"],
                message_id="assistant-msg-1",
                revision=1,
                status="ready",
            )
        )

        await suggest_questions(
            SuggestedQuestionsRequest(conversation_id="conv-1"),
            SimpleNamespace(state=SimpleNamespace(request_id="request-1")),
            chat_service=chat_service,
            current_user=SimpleNamespace(id="user-1"),
        )

        self.assertTrue(chat_service.generate_suggested_questions_result.await_args.kwargs["force_refresh"])

    async def test_target_errors_map_to_explicit_http_status(self):
        from app.api.chat import suggest_questions

        for error, expected_status in (
            (SuggestedQuestionTargetNotFoundError("找不到目标助手消息"), 404),
            (SuggestedQuestionTargetInvalidError("目标助手消息没有正式正文"), 400),
        ):
            chat_service = MagicMock()
            chat_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
            chat_service.generate_suggested_questions_result = AsyncMock(side_effect=error)

            with self.assertRaises(ApiException) as raised:
                await suggest_questions(
                    SuggestedQuestionsRequest(
                        conversation_id="conv-1",
                        assistant_message_id="assistant-msg-1",
                        force_refresh=True,
                    ),
                    SimpleNamespace(state=SimpleNamespace(request_id="request-1")),
                    chat_service=chat_service,
                    current_user=SimpleNamespace(id="user-1"),
                )

            self.assertEqual(raised.exception.status_code, expected_status)


if __name__ == "__main__":
    unittest.main()
