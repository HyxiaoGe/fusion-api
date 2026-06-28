import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentSession
from app.db.models import Conversation as ConversationModel
from app.db.models import Message as MessageModel
from app.db.repositories import ConversationRepository


class MessageRepositoryTests(unittest.TestCase):
    def test_convert_message_preserves_search_provider_metadata(self):
        """从 JSONB 重建 SearchBlock 时保留搜索提供方元信息"""
        db_message = MessageModel(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            created_at=datetime(2026, 6, 24, tzinfo=timezone.utc),
            content=[
                {
                    "type": "search",
                    "id": "search-1",
                    "query": "AI 标准",
                    "tool_call_log_id": "log-1",
                    "sources": [{"title": "来源", "url": "https://example.com"}],
                    "requested_provider": "firecrawl",
                    "result_provider": "brave",
                    "fallback_used": True,
                    "provider_chain": ["firecrawl", "brave"],
                }
            ],
        )

        message = ConversationRepository(None)._convert_message_to_schema(db_message)

        search_block = message.content[0]
        self.assertEqual(search_block.requested_provider, "firecrawl")
        self.assertEqual(search_block.result_provider, "brave")
        self.assertTrue(search_block.fallback_used)
        self.assertEqual(search_block.provider_chain, ["firecrawl", "brave"])

    def test_get_by_id_attaches_latest_agent_run_summary_to_assistant_message(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            conversation = ConversationModel(
                id="conv-1",
                user_id="user-1",
                title="会话",
                model_id="deepseek-chat",
                created_at=datetime(2026, 6, 28, 10, 0, 0),
                updated_at=datetime(2026, 6, 28, 10, 0, 0),
            )
            message = MessageModel(
                id="msg-1",
                conversation_id="conv-1",
                role="assistant",
                content=[{"type": "text", "id": "blk_1", "text": "旧回答"}],
                model_id="deepseek-chat",
                created_at=datetime(2026, 6, 28, 10, 0, 1),
            )
            run = AgentSession(
                id="run-1",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                run_config={"max_steps": 3, "max_tool_calls": 5, "timeout_s": 60},
                total_steps=3,
                total_tool_calls=5,
                status="limit_reached",
                limit_reason="max_steps",
                created_at=datetime(2026, 6, 28, 10, 1, 0),
            )
            db.add_all([conversation, message, run])
            db.commit()

            result = ConversationRepository(db).get_by_id("conv-1", "user-1")

            self.assertIsNotNone(result)
            self.assertEqual(result.messages[0].agent_run.run_id, "run-1")
            self.assertEqual(result.messages[0].agent_run.status, "limit_reached")
            self.assertEqual(result.messages[0].agent_run.limit_reason, "max_steps")
            self.assertEqual(result.messages[0].agent_run.config["max_steps"], 3)
        finally:
            db.close()
            engine.dispose()

    def test_get_by_id_uses_latest_agent_run_not_latest_limit_reached(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            conversation = ConversationModel(
                id="conv-1",
                user_id="user-1",
                title="会话",
                model_id="deepseek-chat",
                created_at=datetime(2026, 6, 28, 10, 0, 0),
                updated_at=datetime(2026, 6, 28, 10, 0, 0),
            )
            message = MessageModel(
                id="msg-1",
                conversation_id="conv-1",
                role="assistant",
                content=[{"type": "text", "id": "blk_1", "text": "旧回答"}],
                model_id="deepseek-chat",
                created_at=datetime(2026, 6, 28, 10, 0, 1),
            )
            old_limit = AgentSession(
                id="run-old",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                total_steps=3,
                total_tool_calls=5,
                status="limit_reached",
                limit_reason="max_tool_calls",
                created_at=datetime(2026, 6, 28, 10, 1, 0),
            )
            latest_completed = AgentSession(
                id="run-new",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                total_steps=1,
                total_tool_calls=0,
                status="completed",
                limit_reason="max_steps",
                created_at=datetime(2026, 6, 28, 10, 2, 0),
            )
            db.add_all([conversation, message, old_limit, latest_completed])
            db.commit()

            result = ConversationRepository(db).get_by_id("conv-1", "user-1")

            self.assertIsNotNone(result)
            self.assertEqual(result.messages[0].agent_run.run_id, "run-new")
            self.assertEqual(result.messages[0].agent_run.status, "completed")
            self.assertIsNone(result.messages[0].agent_run.limit_reason)
        finally:
            db.close()
            engine.dispose()
