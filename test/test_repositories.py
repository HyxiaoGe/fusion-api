import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.database import Base
from app.db.models import AgentProgressSnapshot, AgentSession, ConversationFile, File
from app.db.models import Conversation as ConversationModel
from app.db.models import Message as MessageModel
from app.db.models import User as UserModel
from app.db.repositories import ConversationRepository, FileRepository


class MessageRepositoryTests(unittest.TestCase):
    def test_convert_message_sanitizes_legacy_thinking_tool_names(self):
        db_message = MessageModel(
            id="msg-thinking",
            conversation_id="conv-1",
            role="assistant",
            content=[
                {
                    "type": "thinking",
                    "id": "thinking-1",
                    "thinking": "调用 route_compare，再用 local_place_search 核对。",
                }
            ],
            created_at=datetime(2026, 7, 18, tzinfo=timezone.utc),
        )

        message = ConversationRepository(None)._convert_message_to_schema(db_message)

        self.assertEqual(message.content[0].thinking, "调用路线比较，再用地点搜索核对。")

    def test_convert_message_restores_nested_context_and_accepts_legacy_usage(self):
        current = MessageModel(
            id="msg-current",
            conversation_id="conv-1",
            role="assistant",
            content=[{"type": "text", "id": "answer-1", "text": "回答"}],
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "context": {
                    "status": "trimmed",
                    "window_tokens": 1000,
                    "estimated_tokens_before": 900,
                    "estimated_tokens_after": 700,
                    "actual_prompt_tokens": 690,
                    "removed_turns": 1,
                    "removed_messages": 2,
                    "removed_tool_transactions": 0,
                },
            },
            created_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        )
        legacy = MessageModel(
            id="msg-legacy",
            conversation_id="conv-1",
            role="assistant",
            content=[{"type": "text", "id": "answer-2", "text": "旧回答"}],
            usage={"input_tokens": 12, "output_tokens": 8},
            created_at=datetime(2026, 7, 12, tzinfo=timezone.utc),
        )
        repo = ConversationRepository(None)

        current_message = repo._convert_message_to_schema(current)
        legacy_message = repo._convert_message_to_schema(legacy)

        self.assertEqual(current_message.usage.context.actual_prompt_tokens, 690)
        self.assertEqual(current_message.usage.input_tokens, 100)
        self.assertIsNone(legacy_message.usage.context)

    def test_convert_message_discards_corrupt_context_but_keeps_tokens(self):
        db_message = MessageModel(
            id="msg-corrupt",
            conversation_id="conv-1",
            role="assistant",
            content=[{"type": "text", "id": "answer-1", "text": "回答"}],
            usage={
                "input_tokens": 100,
                "output_tokens": 20,
                "context": {"status": "unknown-future", "window_tokens": -1},
            },
            created_at=datetime(2026, 7, 13, tzinfo=timezone.utc),
        )

        message = ConversationRepository(None)._convert_message_to_schema(db_message)

        self.assertEqual(message.usage.input_tokens, 100)
        self.assertEqual(message.usage.output_tokens, 20)
        self.assertIsNone(message.usage.context)

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

    def test_delete_file_requires_owner_and_removes_conversation_link(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()
        try:
            user = UserModel(id="user-1", username="user-1")
            other_user = UserModel(id="user-2", username="user-2")
            conversation = ConversationModel(
                id="conv-1",
                user_id="user-1",
                title="会话",
                model_id="qwen",
                created_at=datetime(2026, 7, 4, 10, 0, 0),
                updated_at=datetime(2026, 7, 4, 10, 0, 0),
            )
            file = File(
                id="file-1",
                user_id="user-1",
                filename="file-1_photo.png",
                original_filename="photo.png",
                mimetype="image/png",
                size=12,
                path="conv-1/file-1/processed.jpg",
                status="processed",
            )
            link = ConversationFile(conversation_id="conv-1", file_id="file-1")
            db.add_all([user, other_user, conversation, file, link])
            db.commit()

            repo = FileRepository(db)

            self.assertFalse(repo.delete_file("file-1", "user-2"))
            self.assertIsNotNone(db.query(File).filter(File.id == "file-1").first())
            self.assertIsNotNone(db.query(ConversationFile).filter(ConversationFile.file_id == "file-1").first())

            self.assertTrue(repo.delete_file("file-1", "user-1"))
            self.assertIsNone(db.query(File).filter(File.id == "file-1").first())
            self.assertIsNone(db.query(ConversationFile).filter(ConversationFile.file_id == "file-1").first())
        finally:
            db.close()
            engine.dispose()

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

    def test_get_by_id_attaches_latest_agent_progress_snapshot(self):
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
                content=[{"type": "text", "id": "blk_1", "text": "回答"}],
                model_id="deepseek-chat",
                created_at=datetime(2026, 6, 28, 10, 0, 1),
            )
            run = AgentSession(
                id="run-new",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                total_steps=1,
                total_tool_calls=1,
                status="completed",
                created_at=datetime(2026, 6, 28, 10, 2, 0),
            )
            snapshot = AgentProgressSnapshot(
                run_id="run-new",
                conversation_id="conv-1",
                message_id="msg-1",
                user_id="user-1",
                protocol_version=2,
                state={
                    "run_id": "run-new",
                    "message_id": "msg-1",
                    "status": "completed",
                    "progress": {"phase": "answering", "label": "正在整理回答"},
                    "plan": {"plan_id": "plan-run-new", "revision": 1, "items": []},
                    "tool_digests": [],
                    "evidence": [],
                },
            )
            db.add_all([conversation, message, run, snapshot])
            db.commit()

            result = ConversationRepository(db).get_by_id("conv-1", "user-1")

            self.assertIsNotNone(result)
            self.assertEqual(result.messages[0].agent_run.run_id, "run-new")
            self.assertEqual(result.messages[0].agent_run.progress["plan"]["plan_id"], "plan-run-new")
        finally:
            db.close()
            engine.dispose()
