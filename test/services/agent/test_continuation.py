import unittest
from datetime import datetime
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentSession
from app.db.models import Message as MessageModel
from app.schemas.chat import TextBlock, ThinkingBlock
from app.schemas.response import ApiException
from app.services.agent.continuation import (
    CONTINUATION_SYSTEM_PROMPT,
    build_continuation_context,
    deserialize_content_blocks,
    find_latest_limit_reached_session,
    inject_continuation_prompt,
    resolve_continuation_limits,
)
from app.services.stream.agent_loop_policy import AgentLoopLimits


class FakeQuery:
    def __init__(self, rows):
        self.rows = list(rows)
        self.filters = []
        self.ordered = False

    def filter(self, *criteria):
        self.filters.extend(criteria)
        return self

    def order_by(self, *_criteria):
        self.ordered = True
        return self

    def first(self):
        return self.rows[0] if self.rows else None


class FakeDb:
    def __init__(self, *, message=None, sessions=None):
        self.message_query = FakeQuery([message] if message else [])
        self.session_query = FakeQuery(sessions or [])

    def query(self, model):
        if model is MessageModel:
            return self.message_query
        if model is AgentSession:
            return self.session_query
        raise AssertionError(f"未预期查询模型: {model!r}")


class AgentContinuationTests(unittest.TestCase):
    def test_deserialize_content_blocks_preserves_existing_block_id(self):
        blocks = deserialize_content_blocks([{"type": "text", "id": "blk_old", "text": "旧回答"}])

        self.assertEqual(blocks, [TextBlock(type="text", id="blk_old", text="旧回答")])

    def test_deserialize_content_blocks_sanitizes_legacy_thinking_tool_names(self):
        blocks = deserialize_content_blocks(
            [
                {
                    "type": "thinking",
                    "id": "thinking-old",
                    "thinking": "调用 web_search，再用 url_read 读取网页。",
                }
            ]
        )

        self.assertEqual(
            blocks,
            [ThinkingBlock(type="thinking", id="thinking-old", thinking="调用联网搜索，再用网页读取读取网页。")],
        )

    def test_inject_continuation_prompt_after_existing_system_messages(self):
        messages = [
            {"role": "system", "content": "用户自定义系统提示"},
            {"role": "user", "content": "问题"},
            {"role": "assistant", "content": "旧回答"},
        ]

        result = inject_continuation_prompt(messages)

        self.assertEqual(result[0]["content"], "用户自定义系统提示")
        self.assertEqual(result[1], {"role": "system", "content": CONTINUATION_SYSTEM_PROMPT})
        self.assertEqual(result[2]["role"], "user")

    def test_resolve_continuation_limits_uses_session_config(self):
        session = SimpleNamespace(run_config={"max_steps": 4, "max_tool_calls": 7, "timeout_s": 90})

        limits = resolve_continuation_limits(
            session,
            default_limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
        )

        self.assertEqual(limits, AgentLoopLimits(max_steps=4, max_tool_calls=7, total_timeout_s=90))

    def test_resolve_continuation_limits_falls_back_to_default_for_missing_config(self):
        session = SimpleNamespace(run_config=None)
        default_limits = AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300)

        self.assertEqual(resolve_continuation_limits(session, default_limits=default_limits), default_limits)

    def test_find_latest_limit_reached_session_rejects_missing_session(self):
        db = FakeDb(sessions=[])

        with self.assertRaises(ApiException) as raised:
            find_latest_limit_reached_session(
                db,
                conversation_id="conv-1",
                message_id="msg-1",
                previous_run_id=None,
            )

        self.assertEqual(raised.exception.status_code, 400)

    def test_find_latest_limit_reached_session_rejects_when_latest_session_completed(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            old_limit = AgentSession(
                id="run-old",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                status="limit_reached",
                created_at=datetime(2026, 6, 28, 10, 0, 0),
            )
            latest_completed = AgentSession(
                id="run-new",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                status="completed",
                created_at=datetime(2026, 6, 28, 10, 1, 0),
            )
            db.add_all([old_limit, latest_completed])
            db.commit()

            with self.assertRaises(ApiException) as raised:
                find_latest_limit_reached_session(
                    db,
                    conversation_id="conv-1",
                    message_id="msg-1",
                    previous_run_id=None,
                )

            self.assertEqual(raised.exception.status_code, 400)
        finally:
            db.close()
            engine.dispose()

    def test_find_latest_limit_reached_session_accepts_latest_limit_reached(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            old_completed = AgentSession(
                id="run-old",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                status="completed",
                created_at=datetime(2026, 6, 28, 10, 0, 0),
            )
            latest_limit = AgentSession(
                id="run-new",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                status="limit_reached",
                created_at=datetime(2026, 6, 28, 10, 1, 0),
            )
            other_message_newer_limit = AgentSession(
                id="run-other",
                conversation_id="conv-1",
                message_id="msg-other",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                status="limit_reached",
                created_at=datetime(2026, 6, 28, 10, 2, 0),
            )
            db.add_all([old_completed, latest_limit, other_message_newer_limit])
            db.commit()

            result = find_latest_limit_reached_session(
                db,
                conversation_id="conv-1",
                message_id="msg-1",
                previous_run_id=None,
            )

            self.assertEqual(result.id, "run-new")
        finally:
            db.close()
            engine.dispose()

    def test_build_continuation_context_reuses_assistant_message_blocks_and_limits(self):
        message = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[{"type": "text", "id": "blk_old", "text": "旧回答"}],
        )
        previous_session = SimpleNamespace(
            id="run-old",
            status="limit_reached",
            run_config={"max_steps": 3, "max_tool_calls": 5, "timeout_s": 60},
        )
        db = FakeDb(message=message, sessions=[previous_session])

        context = build_continuation_context(
            db,
            conversation_id="conv-1",
            message_id="msg-1",
            previous_run_id="run-old",
            default_limits=AgentLoopLimits(max_steps=8, max_tool_calls=20, total_timeout_s=300),
        )

        self.assertIs(context.assistant_message, message)
        self.assertIs(context.previous_session, previous_session)
        self.assertEqual(context.limits, AgentLoopLimits(max_steps=3, max_tool_calls=5, total_timeout_s=60))
        self.assertEqual(context.initial_content_blocks, [TextBlock(type="text", id="blk_old", text="旧回答")])


if __name__ == "__main__":
    unittest.main()
