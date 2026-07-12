import unittest
from datetime import datetime

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.admin_audit_repository import AdminAuditRepository
from app.db.database import Base
from app.db.models import (
    AgentProgressSnapshot,
    AgentSession,
    AgentStep,
    Conversation,
    ConversationFile,
    File,
    Message,
    PerformanceRun,
    ToolCallLog,
    User,
)


class AdminAuditRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        event.listen(self.engine, "connect", lambda connection, _: connection.execute("PRAGMA foreign_keys=ON"))
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.db = self.Session()
        self._seed()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _seed(self):
        created = datetime(2026, 7, 11, 12, 0, 0)
        user = User(
            id="user-1",
            username="alice",
            nickname="Alice",
            email="alice@example.com",
            created_at=created,
            updated_at=created,
        )
        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="审计测试",
            model_id="deepseek-chat",
            created_at=created,
            updated_at=created,
        )
        user_message = Message(
            id="msg-user",
            conversation_id="conv-1",
            role="user",
            content=[{"type": "future_block", "secret": "Bearer hidden", "value": "保留"}],
            created_at=created,
        )
        assistant_message = Message(
            id="msg-assistant",
            conversation_id="conv-1",
            role="assistant",
            content=[{"type": "text", "id": "blk-1", "text": "回答"}],
            model_id="deepseek-chat",
            usage={"input_tokens": 12, "output_tokens": 8},
            created_at=created,
        )
        run = AgentSession(
            id="run-1",
            conversation_id="conv-1",
            message_id="msg-assistant",
            user_id="user-1",
            model_id="deepseek-chat",
            provider="deepseek",
            run_config={"authorization": "Bearer hidden", "max_steps": 3},
            total_steps=1,
            total_tool_calls=1,
            total_duration_ms=200,
            status="completed",
            created_at=created,
        )
        step = AgentStep(
            id="step-1",
            trace_id="run-1",
            step_number=1,
            status="completed",
            tool_calls_count=1,
            tool_names=["web_search"],
            duration_ms=100,
            created_at=created,
        )
        snapshot = AgentProgressSnapshot(
            id="snapshot-1",
            run_id="run-1",
            conversation_id="conv-1",
            message_id="msg-assistant",
            user_id="user-1",
            protocol_version=2,
            state={"status": "completed"},
            created_at=created,
            updated_at=created,
        )
        tool = ToolCallLog(
            id="tool-1",
            conversation_id="conv-1",
            message_id="msg-assistant",
            user_id="user-1",
            tool_name="web_search",
            status="success",
            duration_ms=90,
            model_id="deepseek-chat",
            provider="deepseek",
            input_params={"query": "Fusion", "api_key": "secret"},
            output_data={"result_count": 1, "token": "secret"},
            extra_metadata={"authorization": "Bearer secret"},
            trace_id="run-1",
            step_number=1,
            created_at=created,
        )
        file = File(
            id="file-1",
            user_id="user-1",
            filename="stored.png",
            original_filename="visible.png",
            mimetype="image/png",
            size=20,
            path="private/path.png",
            storage_key="secret-storage-key",
            thumbnail_key="secret-thumbnail-key",
            parsed_content="private parsed content",
            status="processed",
            width=10,
            height=20,
            created_at=created,
            updated_at=created,
        )
        link = ConversationFile(conversation_id="conv-1", file_id="file-1", created_at=created)
        self.db.add_all([user, conversation, user_message, assistant_message, file, link])
        self.db.commit()
        self.db.add(run)
        self.db.commit()
        self.db.add_all([step, snapshot, tool])
        self.db.commit()

    def test_user_and_conversation_lists_include_bounded_aggregates(self):
        repo = AdminAuditRepository(self.db)

        users, user_total = repo.list_users(page=1, page_size=25, query="alice")
        conversations, conversation_total = repo.list_conversations(
            page=1,
            page_size=25,
            user_id="user-1",
            query="审计",
            model_id="deepseek-chat",
            has_tools=True,
            has_files=True,
        )

        self.assertEqual(user_total, 1)
        self.assertEqual(users[0]["conversation_count"], 1)
        self.assertEqual(users[0]["message_count"], 2)
        self.assertEqual(users[0]["input_tokens"], 12)
        self.assertEqual(conversation_total, 1)
        self.assertEqual(conversations[0]["tool_call_count"], 1)
        self.assertEqual(conversations[0]["file_count"], 1)
        self.assertEqual(conversations[0]["latest_agent_status"], "completed")

    def test_partition_queries_keep_unknown_blocks_and_group_agent_steps(self):
        repo = AdminAuditRepository(self.db)

        messages, message_total = repo.list_messages("conv-1", page=1, page_size=25)
        tools, tool_total = repo.list_tool_calls("conv-1", page=1, page_size=25)
        runs, run_total = repo.list_agent_runs("conv-1", page=1, page_size=25)
        files, file_total = repo.list_files("conv-1", page=1, page_size=25)

        self.assertEqual(message_total, 2)
        unknown_message = next(message for message in messages if message.id == "msg-user")
        self.assertEqual(unknown_message.content[0]["type"], "future_block")
        self.assertEqual(tool_total, 1)
        self.assertEqual(tools[0].id, "tool-1")
        self.assertEqual(run_total, 1)
        self.assertEqual(runs[0]["session"].id, "run-1")
        self.assertEqual(runs[0]["steps"][0].id, "step-1")
        self.assertEqual(runs[0]["tool_calls"][0].id, "tool-1")
        self.assertEqual(files[0].original_filename, "visible.png")
        self.assertEqual(file_total, 1)

    def test_get_users_by_ids_returns_existing_users_in_one_batch_and_omits_deleted_ids(self):
        repo = AdminAuditRepository(self.db)

        users = repo.get_users_by_ids(["deleted-user", "user-1"])

        self.assertEqual(set(users), {"user-1"})
        self.assertEqual(users["user-1"].username, "alice")
        self.assertFalse(hasattr(users["user-1"], "email"))
        self.assertFalse(hasattr(users["user-1"], "system_prompt"))

    def test_model_operations_stats_batch_real_usage_agent_errors_and_latest_performance(self):
        newer = datetime(2026, 7, 12, 12, 0, 0)
        self.db.add(
            AgentSession(
                id="run-error",
                conversation_id="conv-1",
                message_id="msg-assistant",
                user_id="user-1",
                model_id="deepseek-chat",
                provider="deepseek",
                status="error",
                error_message="private provider error",
                total_duration_ms=999,
                created_at=newer,
            )
        )
        self.db.add_all(
            [
                PerformanceRun(
                    run_id="perf-old",
                    environment="production",
                    model_id="deepseek-chat",
                    status="completed",
                    schema_version=2,
                    safe_summary={"stages": []},
                    imported_by_user_id="admin-1",
                    created_at=datetime(2026, 7, 11, 13, 0, 0),
                ),
                PerformanceRun(
                    run_id="perf-new",
                    environment="production",
                    model_id="deepseek-chat",
                    status="completed",
                    schema_version=2,
                    safe_summary={"stages": []},
                    imported_by_user_id="admin-1",
                    created_at=newer,
                ),
                PerformanceRun(
                    run_id="perf-only",
                    environment="production",
                    model_id="retired/perf-only",
                    status="completed",
                    schema_version=2,
                    safe_summary={"stages": []},
                    imported_by_user_id="admin-1",
                    created_at=newer,
                ),
            ]
        )
        self.db.commit()

        statements = []

        def count_selects(_connection, _cursor, statement, _parameters, _context, _executemany):
            if statement.lstrip().upper().startswith("SELECT"):
                statements.append(statement)

        event.listen(self.engine, "before_cursor_execute", count_selects)
        try:
            stats = AdminAuditRepository(self.db).list_model_operation_stats()
        finally:
            event.remove(self.engine, "before_cursor_execute", count_selects)

        self.assertEqual(set(stats), {"deepseek-chat", "retired/perf-only"})
        deepseek = stats["deepseek-chat"]
        self.assertEqual(deepseek["conversation_count"], 1)
        self.assertEqual(deepseek["user_count"], 1)
        self.assertEqual(deepseek["assistant_message_count"], 1)
        self.assertEqual(deepseek["input_tokens"], 12)
        self.assertEqual(deepseek["output_tokens"], 8)
        self.assertEqual(deepseek["agent_run_count"], 2)
        self.assertEqual(deepseek["agent_error_count"], 1)
        self.assertEqual(deepseek["last_used_at"], newer)
        self.assertEqual(deepseek["latest_performance_run"].run_id, "perf-new")
        self.assertNotIn("latency", deepseek)
        self.assertEqual(stats["retired/perf-only"]["latest_performance_run"].run_id, "perf-only")
        self.assertEqual(len(statements), 4)
        self.assertIn("row_number", statements[-1].lower())


if __name__ == "__main__":
    unittest.main()
