import os
import subprocess
import sys
import time
import unittest
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

from sqlalchemy import create_engine, insert
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import Conversation as ConversationModel
from app.db.models import Message as MessageModel
from app.db.models import User as UserModel
from app.db.repositories import ConversationRepository
from app.schemas.chat import ChatResponse, Conversation, Message, TextBlock


@contextmanager
def application_timezone(name: str):
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    if hasattr(time, "tzset"):
        time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        if hasattr(time, "tzset"):
            time.tzset()


class MessageTimestampContractTests(unittest.TestCase):
    def test_message_default_is_utc_aware_even_when_application_runs_in_shanghai(self):
        with application_timezone("Asia/Shanghai"):
            message = Message(role="user", content=[TextBlock(type="text", text="你好")])

        self.assertIsNotNone(message.created_at.tzinfo)
        self.assertEqual(message.created_at.utcoffset(), timedelta(0))

    def test_message_orm_uses_timezone_aware_column(self):
        self.assertTrue(MessageModel.__table__.c.created_at.type.timezone)
        self.assertEqual(str(MessageModel.__table__.c.created_at.server_default.arg), "now()")

    def test_message_sequence_server_default_covers_old_binary_insert(self):
        sequence_column = MessageModel.__table__.c.sequence
        self.assertEqual(
            str(sequence_column.server_default.arg.compile(dialect=postgresql.dialect())),
            "nextval('message_order_sequence')",
        )

        old_binary_insert = insert(MessageModel).values(
            id="message-1",
            conversation_id="conversation-1",
            role="user",
            content=[],
        )
        compiled = str(old_binary_insert.compile(dialect=postgresql.dialect()))

        self.assertNotIn("sequence", compiled)

    def test_migration_keeps_legacy_sequence_null_then_sets_default(self):
        repo_root = os.path.dirname(os.path.dirname(__file__))
        env = {
            **os.environ,
            "DATABASE_URL": "postgresql://user:pass@localhost/fusion",
        }
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "alembic",
                "upgrade",
                "b8d4f7a1c2e6:c4f8a2d1e6b9",
                "--sql",
            ],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        add_column = "ALTER TABLE messages ADD COLUMN sequence BIGINT;"
        set_default = "ALTER TABLE messages ALTER COLUMN sequence SET DEFAULT nextval('message_order_sequence');"
        self.assertIn(add_column, result.stdout)
        self.assertIn(set_default, result.stdout)
        self.assertLess(result.stdout.index(add_column), result.stdout.index(set_default))
        between = result.stdout[result.stdout.index(add_column) + len(add_column) : result.stdout.index(set_default)]
        self.assertNotIn("UPDATE messages SET sequence", between)

    def test_conversation_and_chat_response_defaults_are_utc_aware(self):
        with application_timezone("Asia/Shanghai"):
            conversation = Conversation(
                id="conv-1",
                user_id="user-1",
                model_id="deepseek-chat",
                title="UTC",
            )
            response = ChatResponse(
                conversation_id="conv-1",
                message=Message(role="assistant", content=[]),
            )

        self.assertEqual(conversation.created_at.utcoffset(), timedelta(0))
        self.assertEqual(conversation.updated_at.utcoffset(), timedelta(0))
        self.assertEqual(response.created_at.utcoffset(), timedelta(0))
        self.assertTrue(ConversationModel.__table__.c.created_at.type.timezone)
        self.assertTrue(ConversationModel.__table__.c.updated_at.type.timezone)
        self.assertEqual(str(ConversationModel.__table__.c.created_at.server_default.arg), "now()")


class MessageSequenceOrderingTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def tearDown(self):
        self.engine.dispose()

    def test_conversation_detail_orders_by_sequence_not_inverted_timestamps(self):
        db = self.Session()
        try:
            db.add(UserModel(id="user-1", username="user-1"))
            db.add(
                ConversationModel(
                    id="conv-1",
                    user_id="user-1",
                    title="顺序回归",
                    model_id="deepseek-chat",
                )
            )
            db.add_all(
                [
                    MessageModel(
                        id="assistant-1",
                        conversation_id="conv-1",
                        role="assistant",
                        content=[{"type": "text", "id": "a1", "text": "回答"}],
                        sequence=2,
                        created_at=datetime(2026, 7, 13, 15, 17, tzinfo=timezone.utc),
                    ),
                    MessageModel(
                        id="user-1-message",
                        conversation_id="conv-1",
                        role="user",
                        content=[{"type": "text", "id": "u1", "text": "问题"}],
                        sequence=1,
                        created_at=datetime(2026, 7, 13, 23, 17, tzinfo=timezone.utc),
                    ),
                ]
            )
            db.commit()

            conversation = ConversationRepository(db).get_by_id("conv-1", "user-1")

            self.assertEqual([message.role for message in conversation.messages], ["user", "assistant"])
            self.assertEqual([message.sequence for message in conversation.messages], [1, 2])
        finally:
            db.close()

    def test_latest_assistant_uses_sequence_before_created_at(self):
        db = self.Session()
        try:
            db.add(UserModel(id="user-1", username="user-1"))
            db.add(
                ConversationModel(
                    id="conv-1",
                    user_id="user-1",
                    title="最后消息",
                    model_id="deepseek-chat",
                )
            )
            db.add_all(
                [
                    MessageModel(
                        id="assistant-old",
                        conversation_id="conv-1",
                        role="assistant",
                        content=[],
                        sequence=2,
                        created_at=datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc),
                    ),
                    MessageModel(
                        id="assistant-new",
                        conversation_id="conv-1",
                        role="assistant",
                        content=[],
                        sequence=4,
                        created_at=datetime(2026, 7, 13, 8, 0, tzinfo=timezone.utc),
                    ),
                ]
            )
            db.commit()

            latest = ConversationRepository(db).get_last_assistant_message("conv-1")

            self.assertEqual(latest.id, "assistant-new")
        finally:
            db.close()

    def test_reserve_pair_uses_one_atomic_database_nextval(self):
        db = MagicMock()
        db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        db.execute.return_value.scalar_one.return_value = 101

        reserved = ConversationRepository(db).reserve_message_sequence_pair()

        self.assertEqual(reserved, (101, 102))
        db.execute.assert_called_once()
        self.assertIn("message_order_sequence", str(db.execute.call_args.args[0]).lower())


if __name__ == "__main__":
    unittest.main()
