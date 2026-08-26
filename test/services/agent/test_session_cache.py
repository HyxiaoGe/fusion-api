"""session_cache 单元测试 — mock SessionLocal 验证 ORM 操作"""

import os
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from sqlalchemy import create_engine, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# 防御层：万一 unittest discover 没把 test/ 当 package，这里兜底
os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")

from app.db.database import Base  # noqa: E402
from app.db.models import AgentSession, AgentSystemPromptSnapshot, Conversation, User  # noqa: E402
from app.services.agent.session_cache import (  # noqa: E402
    write_session_started,
    write_session_status,
    write_step_completed,
    write_step_started,
    write_step_terminal,
)


class SessionCacheTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_snapshot_is_deleted_with_its_run_or_conversation(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(dbapi_connection, _connection_record):
            dbapi_connection.execute("PRAGMA foreign_keys=ON")

        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        try:
            with factory() as db:
                db.add_all(
                    [
                        User(id="u1", username="run-owner", email="run-owner@example.com"),
                        User(id="u2", username="conversation-owner", email="conversation-owner@example.com"),
                    ]
                )
                db.commit()
                db.add_all(
                    [
                        Conversation(id="c1", user_id="u1", title="Run 级联", model_id="m1"),
                        Conversation(id="c2", user_id="u2", title="会话级联", model_id="m1"),
                    ]
                )
                db.commit()
                db.add_all(
                    [
                        AgentSession(
                            id="r1",
                            conversation_id="c1",
                            user_id="u1",
                            model_id="m1",
                            provider="p1",
                            status="running",
                        ),
                        AgentSession(
                            id="r2",
                            conversation_id="c2",
                            user_id="u2",
                            model_id="m1",
                            provider="p1",
                            status="running",
                        ),
                    ]
                )
                db.commit()
                db.add_all(
                    [
                        AgentSystemPromptSnapshot(
                            run_id="r1", conversation_id="c1", user_id="u1", snapshot={"sections": []}
                        ),
                        AgentSystemPromptSnapshot(
                            run_id="r2", conversation_id="c2", user_id="u2", snapshot={"sections": []}
                        ),
                    ]
                )
                db.commit()

                db.delete(db.get(AgentSession, "r1"))
                db.commit()
                self.assertIsNone(db.get(AgentSystemPromptSnapshot, "r1"))

                db.delete(db.get(Conversation, "c2"))
                db.commit()
                self.assertIsNone(db.get(AgentSession, "r2"))
                self.assertIsNone(db.get(AgentSystemPromptSnapshot, "r2"))
        finally:
            engine.dispose()

    async def test_prompt_snapshot_is_durable_scoped_and_not_overwritten_on_run_reentry(self):
        from app.db.trajectory_repository import TrajectoryRepository
        from app.services.agent import session_cache
        from app.services.trajectory_query_service import TrajectoryQueryService

        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        factory = sessionmaker(bind=engine, expire_on_commit=False)
        snapshot = {
            "schema_version": 1,
            "template_version": "test.1",
            "fingerprint": "a45b3127dea09221b17f29ba070f505393118249e0bfdb59899439b3f78677cd",
            "char_count": 7,
            "sections": [{"section_id": "identity", "content": "原文\n保持换行"}],
        }
        try:
            with factory() as db:
                db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="m1"))
                db.add(
                    AgentSession(
                        id="r1",
                        conversation_id="c1",
                        user_id="u1",
                        message_id="m1",
                        turn_message_id="t1",
                        model_id="m1",
                        provider="p1",
                        status="running",
                        run_config={"max_steps": 3},
                    )
                )
                db.commit()
            with patch("app.services.agent.session_cache.SessionLocal", factory):
                for conversation_id, user_id in [("wrong", "u1"), ("c1", "wrong")]:
                    with self.assertRaises(ValueError):
                        await session_cache.write_system_prompt_snapshot(
                            run_id="r1",
                            conversation_id=conversation_id,
                            user_id=user_id,
                            snapshot=snapshot,
                        )
                await session_cache.write_system_prompt_snapshot(
                    run_id="r1",
                    conversation_id="c1",
                    user_id="u1",
                    snapshot=snapshot,
                )
                await session_cache.write_system_prompt_snapshot(
                    run_id="r1",
                    conversation_id="c1",
                    user_id="u1",
                    snapshot=snapshot,
                )
                with self.assertRaises(ValueError):
                    await session_cache.write_system_prompt_snapshot(
                        run_id="r1",
                        conversation_id="c1",
                        user_id="u1",
                        snapshot={"sections": []},
                    )
                await write_session_status(run_id="r1", status="completed", total_steps=1, total_tool_calls=0)
                await write_session_started(
                    run_id="r1",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="m1",
                    provider="p1",
                    message_id="m1",
                    turn_message_id="t1",
                    run_config={"max_steps": 8},
                )
            with factory() as db:
                saved_run = db.get(AgentSession, "r1")
                saved_snapshot = db.get(AgentSystemPromptSnapshot, "r1")
                self.assertEqual(saved_run.run_config, {"max_steps": 8})
                self.assertNotIn("system_prompt_snapshot", saved_run.run_config)
                self.assertIsNotNone(saved_snapshot)
                self.assertEqual(saved_snapshot.snapshot, snapshot)
                self.assertEqual(saved_snapshot.conversation_id, "c1")
                self.assertEqual(saved_snapshot.user_id, "u1")
                service = TrajectoryQueryService(
                    TrajectoryRepository(db), max_events_per_run=10, max_runs_per_conversation=10
                )
                detail = service.get_user_system_prompt_node_detail("c1", "r1", "u1")
                self.assertEqual(detail.status, "available")
                self.assertEqual(detail.detail.sections[0].content, "原文\n保持换行")
                self.assertEqual(detail.detail.fingerprint, snapshot["fingerprint"])
            with factory() as db:
                db.get(AgentSession, "r1").run_config = {"legacy_writer": True}
                db.commit()
            with factory() as db:
                saved_snapshot = db.get(AgentSystemPromptSnapshot, "r1")
                self.assertEqual(saved_snapshot.snapshot, snapshot)
                service = TrajectoryQueryService(
                    TrajectoryRepository(db), max_events_per_run=10, max_runs_per_conversation=10
                )
                detail = service.get_user_system_prompt_node_detail("c1", "r1", "u1")
                self.assertEqual(detail.status, "available")
                self.assertEqual(detail.detail.sections[0].content, "原文\n保持换行")
        finally:
            engine.dispose()

    @staticmethod
    def _configure_new_run(session, *, conversation_id="c1", max_attempt=None):
        lock_result = MagicMock()
        lock_result.scalar_one_or_none.return_value = conversation_id
        max_result = MagicMock()
        max_result.scalar_one.return_value = max_attempt
        session.execute.side_effect = [lock_result, max_result]

    async def test_write_session_started_inserts_row(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None  # 明确无已有行
            self._configure_new_run(session)
            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                turn_message_id="turn-1",
                run_attempt_kind="initial",
            )
            session.add.assert_called_once()
            session.commit.assert_called_once()
            row = session.add.call_args.args[0]
            self.assertEqual(row.id, "r1")
            self.assertEqual(row.conversation_id, "c1")
            self.assertEqual(row.user_id, "u1")
            self.assertEqual(row.model_id, "gpt-4")
            self.assertEqual(row.provider, "openai")
            self.assertEqual(row.turn_message_id, "turn-1")
            self.assertEqual(row.attempt_index, 1)
            self.assertIsNone(row.previous_run_id)
            self.assertEqual(row.status, "running")
            self.assertIsNone(row.terminal_at)
            self.assertEqual(row.total_steps, 0)
            self.assertEqual(row.total_tool_calls, 0)

    async def test_write_session_started_persists_run_config(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None
            self._configure_new_run(session)

            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="msg-1",
                turn_message_id="turn-1",
                run_attempt_kind="initial",
                run_config={"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300},
            )

            row = session.add.call_args.args[0]
            self.assertEqual(row.run_config, {"max_steps": 8, "max_tool_calls": 20, "timeout_s": 300})

    async def test_write_session_started_updates_existing_run_config(self):
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            existing = MagicMock()
            existing.conversation_id = "c1"
            existing.user_id = "u1"
            existing.message_id = "msg-1"
            existing.turn_message_id = "turn-1"
            existing.previous_run_id = None
            session.get.return_value = existing

            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="msg-1",
                turn_message_id="turn-1",
                run_attempt_kind="initial",
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
            existing.terminal_at = datetime(2026, 8, 22, 1, 0, tzinfo=UTC)
            existing.conversation_id = "c1"
            existing.user_id = "u1"
            existing.message_id = "assistant-stable"
            existing.turn_message_id = "turn-stable"
            existing.previous_run_id = "run-parent"
            existing.attempt_index = 3
            session.get.return_value = existing

            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-5",
                provider="anthropic",
                message_id="assistant-stable",
                turn_message_id="turn-stable",
                previous_run_id="run-parent",
                run_attempt_kind="regenerate",
            )

            # 不应 add 新行
            session.add.assert_not_called()
            # 应该更新已有行的元信息 + 重置 totals + 重置 status
            self.assertEqual(existing.conversation_id, "c1")
            self.assertEqual(existing.user_id, "u1")
            self.assertEqual(existing.model_id, "gpt-5")
            self.assertEqual(existing.provider, "anthropic")
            self.assertEqual(existing.status, "running")
            self.assertIsNone(existing.terminal_at)
            self.assertEqual(existing.total_steps, 0)
            self.assertEqual(existing.total_tool_calls, 0)
            self.assertIsNone(existing.total_duration_ms)
            self.assertIsNone(existing.limit_reason)
            self.assertIsNone(existing.error_message)
            self.assertEqual(existing.turn_message_id, "turn-stable")
            self.assertEqual(existing.previous_run_id, "run-parent")
            self.assertEqual(existing.attempt_index, 3)
            session.commit.assert_called_once()

    async def test_write_session_started_rejects_cross_scope_same_run_reentry(self):
        existing = SimpleNamespace(
            id="run-shared",
            conversation_id="other-conversation",
            user_id="other-user",
            message_id="assistant-other",
            turn_message_id="turn-other",
            previous_run_id=None,
            attempt_index=1,
            status="completed",
        )
        session = MagicMock()
        session.get.return_value = existing
        lock_result = MagicMock()
        lock_result.scalar_one_or_none.return_value = "c1"
        session.execute.return_value = lock_result

        with patch("app.services.agent.session_cache.SessionLocal") as session_local:
            session_local.return_value.__enter__.return_value = session
            with self.assertRaises(ValueError):
                await write_session_started(
                    run_id="run-shared",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-1",
                    turn_message_id="turn-1",
                    run_attempt_kind="initial",
                )

        session.commit.assert_not_called()

    async def test_write_session_started_inserts_when_no_existing_row(self):
        """没有已有行：行为同 v1（add 新行）"""
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None  # 无已有行
            self._configure_new_run(session)

            await write_session_started(
                run_id="r1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                turn_message_id="turn-1",
                run_attempt_kind="initial",
            )

            session.add.assert_called_once()
            row = session.add.call_args.args[0]
            self.assertEqual(row.id, "r1")
            self.assertEqual(row.status, "running")
            session.commit.assert_called_once()

    async def test_write_session_started_locks_conversation_before_allocating_attempt(self):
        class Result:
            def __init__(self, value):
                self.value = value

            def scalar_one(self):
                return self.value

            def scalar_one_or_none(self):
                return self.value

        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            session.get.return_value = None
            previous = SimpleNamespace(
                id="run-old",
                conversation_id="c1",
                user_id="u1",
                turn_message_id="user-stable",
                message_id="assistant-old",
                status="error",
            )
            session.execute.side_effect = [Result("c1"), Result(1), Result(previous)]

            await write_session_started(
                run_id="run-new",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-new",
                turn_message_id="user-stable",
                run_attempt_kind="retry",
            )

        statements = [call.args[0] for call in session.execute.call_args_list]
        lock_sql = str(statements[0].compile(dialect=postgresql.dialect()))
        self.assertIn("FOR UPDATE", lock_sql)
        row = session.add.call_args.args[0]
        self.assertEqual(row.attempt_index, 2)
        self.assertEqual(row.previous_run_id, "run-old")
        self.assertEqual(row.message_id, "assistant-new")

    async def test_write_session_started_retries_only_the_attempt_unique_conflict(self):
        class Result:
            def __init__(self, value):
                self.value = value

            def scalar_one(self):
                return self.value

            def scalar_one_or_none(self):
                return self.value

        conflict_orig = SimpleNamespace(diag=SimpleNamespace(constraint_name="uq_agent_sessions_turn_attempt"))
        sessions = []
        for max_attempt in (1, 2):
            session = MagicMock()
            previous = SimpleNamespace(
                id="run-old",
                conversation_id="c1",
                user_id="u1",
                turn_message_id="user-stable",
                message_id="assistant-old",
                status="error",
            )
            session.get.side_effect = [None, previous]
            session.execute.side_effect = [Result("c1"), Result(max_attempt), Result(previous)]
            sessions.append(session)
        sessions[0].commit.side_effect = IntegrityError("insert", {}, conflict_orig)
        contexts = []
        for session in sessions:
            context = MagicMock()
            context.__enter__.return_value = session
            contexts.append(context)

        with patch("app.services.agent.session_cache.SessionLocal", side_effect=contexts) as session_local:
            await write_session_started(
                run_id="run-new",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                turn_message_id="user-stable",
                previous_run_id="run-old",
                run_attempt_kind="retry",
            )

        self.assertEqual(session_local.call_count, 2)
        sessions[0].rollback.assert_called_once()
        self.assertEqual(sessions[1].add.call_args.args[0].attempt_index, 3)

    async def test_independent_sessions_allocate_distinct_attempts_for_the_same_turn(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine)
        with test_session_local() as db:
            db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="gpt-4"))
            db.commit()

        with patch("app.services.agent.session_cache.SessionLocal", test_session_local):
            await write_session_started(
                run_id="run-1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                turn_message_id="turn-1",
                run_attempt_kind="initial",
            )
            with test_session_local() as db:
                db.get(AgentSession, "run-1").status = "error"
                db.commit()
            await write_session_started(
                run_id="run-2",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                turn_message_id="turn-1",
                previous_run_id="run-1",
                run_attempt_kind="retry",
            )

        with test_session_local() as db:
            attempts = [
                row.attempt_index
                for row in db.query(AgentSession).filter(AgentSession.turn_message_id == "turn-1").all()
            ]
        self.assertEqual(sorted(attempts), [1, 2])
        engine.dispose()

    async def test_explicit_retry_cannot_branch_from_stale_attempt_inside_allocation_lock(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine)
        with test_session_local() as db:
            db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="gpt-4"))
            db.add_all(
                [
                    AgentSession(
                        id="run-old",
                        conversation_id="c1",
                        user_id="u1",
                        model_id="gpt-4",
                        provider="openai",
                        message_id="assistant-1",
                        turn_message_id="turn-1",
                        attempt_index=1,
                        status="error",
                    ),
                    AgentSession(
                        id="run-latest",
                        conversation_id="c1",
                        user_id="u1",
                        model_id="gpt-4",
                        provider="openai",
                        message_id="assistant-2",
                        turn_message_id="turn-1",
                        attempt_index=2,
                        status="error",
                    ),
                ]
            )
            db.commit()

        with patch("app.services.agent.session_cache.SessionLocal", test_session_local):
            with self.assertRaisesRegex(ValueError, "最新"):
                await write_session_started(
                    run_id="run-branch",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-3",
                    turn_message_id="turn-1",
                    previous_run_id="run-old",
                    run_attempt_kind="retry",
                )

        with test_session_local() as db:
            self.assertIsNone(db.get(AgentSession, "run-branch"))
        engine.dispose()

    async def test_write_session_started_rechecks_same_run_after_conversation_lock(self):
        lock_acquired = False
        existing = MagicMock()
        existing.conversation_id = "c1"
        existing.user_id = "u1"
        existing.message_id = "assistant-1"
        existing.turn_message_id = "turn-stable"
        existing.previous_run_id = None
        existing.attempt_index = 1
        session = MagicMock()

        def execute_lock(_statement):
            nonlocal lock_acquired
            lock_acquired = True
            result = MagicMock()
            result.scalar_one_or_none.return_value = "c1"
            return result

        def get_after_lock(_model, _run_id):
            self.assertTrue(lock_acquired, "必须先拿 conversation lock，再读取同 run_id")
            return existing

        session.execute.side_effect = execute_lock
        session.get.side_effect = get_after_lock
        context = MagicMock()
        context.__enter__.return_value = session

        with patch("app.services.agent.session_cache.SessionLocal", return_value=context):
            await write_session_started(
                run_id="run-same",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                turn_message_id="turn-stable",
                run_attempt_kind="initial",
            )

        session.add.assert_not_called()
        self.assertEqual(existing.attempt_index, 1)
        session.commit.assert_called_once()

    async def test_write_session_started_rejects_invalid_previous_run_inside_lock(self):
        previous = SimpleNamespace(
            id="run-other",
            conversation_id="other-conversation",
            user_id="u1",
            turn_message_id="turn-1",
            message_id="assistant-old",
            status="error",
        )
        session = MagicMock()
        session.get.side_effect = [None, previous]
        lock_result = MagicMock()
        lock_result.scalar_one_or_none.return_value = "c1"
        session.execute.return_value = lock_result

        with patch("app.services.agent.session_cache.SessionLocal") as session_local:
            session_local.return_value.__enter__.return_value = session
            with self.assertRaises(ValueError):
                await write_session_started(
                    run_id="run-new",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-new",
                    turn_message_id="turn-1",
                    previous_run_id="run-other",
                    run_attempt_kind="retry",
                )

        session.add.assert_not_called()
        session.commit.assert_not_called()

    async def test_write_session_started_accepts_legal_retry_regenerate_and_continue_lineage(self):
        cases = (
            ("retry", "error", "assistant-old", "assistant-new"),
            ("regenerate", "completed", "assistant-same", "assistant-same"),
            ("continue", "limit_reached", "assistant-same", "assistant-same"),
        )
        for attempt_kind, status, previous_message_id, message_id in cases:
            with self.subTest(attempt_kind=attempt_kind):
                previous = SimpleNamespace(
                    id="run-old",
                    conversation_id="c1",
                    user_id="u1",
                    turn_message_id="turn-1",
                    message_id=previous_message_id,
                    status=status,
                )
                session = MagicMock()
                session.get.side_effect = [None, previous]
                lock_result = MagicMock()
                lock_result.scalar_one_or_none.return_value = "c1"
                max_result = MagicMock()
                max_result.scalar_one.return_value = 1
                latest_result = MagicMock()
                latest_result.scalar_one_or_none.return_value = previous
                session.execute.side_effect = [lock_result, max_result, latest_result]

                with patch("app.services.agent.session_cache.SessionLocal") as session_local:
                    session_local.return_value.__enter__.return_value = session
                    await write_session_started(
                        run_id=f"run-{attempt_kind}",
                        conversation_id="c1",
                        user_id="u1",
                        model_id="gpt-4",
                        provider="openai",
                        message_id=message_id,
                        turn_message_id="turn-1",
                        previous_run_id="run-old",
                        run_attempt_kind=attempt_kind,
                    )

                row = session.add.call_args.args[0]
                self.assertEqual(row.previous_run_id, "run-old")
                self.assertEqual(row.attempt_index, 2)

    async def test_write_session_started_accepts_legacy_regenerate_fallback_and_explicit_continue(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine)
        with test_session_local() as db:
            db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="gpt-4"))
            db.add_all(
                [
                    AgentSession(
                        id="run-legacy-regenerate",
                        conversation_id="c1",
                        user_id="u1",
                        model_id="gpt-4",
                        provider="openai",
                        message_id="assistant-regenerate",
                        turn_message_id="assistant-regenerate",
                        attempt_index=2,
                        status="completed",
                    ),
                    AgentSession(
                        id="run-legacy-continue",
                        conversation_id="c1",
                        user_id="u1",
                        model_id="gpt-4",
                        provider="openai",
                        message_id="assistant-continue",
                        turn_message_id="assistant-continue",
                        attempt_index=3,
                        status="limit_reached",
                    ),
                ]
            )
            db.commit()

        with patch("app.services.agent.session_cache.SessionLocal", test_session_local):
            await write_session_started(
                run_id="run-new-regenerate",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-regenerate",
                turn_message_id="user-regenerate",
                run_attempt_kind="regenerate",
            )
            await write_session_started(
                run_id="run-new-continue",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-continue",
                turn_message_id="user-continue",
                previous_run_id="run-legacy-continue",
                run_attempt_kind="continue",
            )

        with test_session_local() as db:
            regenerated = db.get(AgentSession, "run-new-regenerate")
            continued = db.get(AgentSession, "run-new-continue")
            self.assertEqual(regenerated.previous_run_id, "run-legacy-regenerate")
            self.assertEqual(regenerated.attempt_index, 3)
            self.assertEqual(continued.previous_run_id, "run-legacy-continue")
            self.assertEqual(continued.attempt_index, 4)
        engine.dispose()

    async def test_write_session_started_rejects_forged_legacy_alias(self):
        previous = SimpleNamespace(
            id="run-forged",
            conversation_id="c1",
            user_id="u1",
            turn_message_id="assistant-target",
            message_id="assistant-other",
            status="completed",
        )
        session = MagicMock()
        session.get.side_effect = [None, previous]
        lock_result = MagicMock()
        lock_result.scalar_one_or_none.return_value = "c1"
        max_result = MagicMock()
        max_result.scalar_one.return_value = 1
        session.execute.side_effect = [lock_result, max_result]

        with patch("app.services.agent.session_cache.SessionLocal") as session_local:
            session_local.return_value.__enter__.return_value = session
            with self.assertRaises(ValueError):
                await write_session_started(
                    run_id="run-new",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-target",
                    turn_message_id="user-target",
                    previous_run_id="run-forged",
                    run_attempt_kind="regenerate",
                )

        session.add.assert_not_called()

    async def test_legacy_retry_without_assistant_anchor_does_not_guess_lineage(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine)
        with test_session_local() as db:
            db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="gpt-4"))
            db.add(
                AgentSession(
                    id="run-legacy",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-old",
                    turn_message_id="assistant-old",
                    attempt_index=1,
                    status="error",
                )
            )
            db.commit()

        with patch("app.services.agent.session_cache.SessionLocal", test_session_local):
            with self.assertRaises(ValueError):
                await write_session_started(
                    run_id="run-new",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-new",
                    turn_message_id="user-1",
                    run_attempt_kind="retry",
                )

        with test_session_local() as db:
            self.assertIsNone(db.get(AgentSession, "run-new"))
        engine.dispose()

    async def test_legacy_and_new_turn_attempts_share_one_monotonic_index(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine)
        with test_session_local() as db:
            db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="gpt-4"))
            db.add(
                AgentSession(
                    id="run-legacy",
                    conversation_id="c1",
                    user_id="u1",
                    model_id="gpt-4",
                    provider="openai",
                    message_id="assistant-1",
                    turn_message_id="assistant-1",
                    attempt_index=3,
                    status="completed",
                )
            )
            db.commit()

        with patch("app.services.agent.session_cache.SessionLocal", test_session_local):
            await write_session_started(
                run_id="run-new-1",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                turn_message_id="user-1",
                previous_run_id="run-legacy",
                run_attempt_kind="regenerate",
            )
            with test_session_local() as db:
                db.get(AgentSession, "run-new-1").status = "completed"
                db.commit()
            await write_session_started(
                run_id="run-new-2",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                turn_message_id="user-1",
                previous_run_id="run-new-1",
                run_attempt_kind="regenerate",
            )

        with test_session_local() as db:
            self.assertEqual(db.get(AgentSession, "run-new-1").attempt_index, 4)
            self.assertEqual(db.get(AgentSession, "run-new-2").attempt_index, 5)
        engine.dispose()

    async def test_legacy_fallback_ignores_cross_user_candidate(self):
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        test_session_local = sessionmaker(bind=engine)
        with test_session_local() as db:
            db.add(Conversation(id="c1", user_id="u1", title="测试", model_id="gpt-4"))
            db.add_all(
                [
                    AgentSession(
                        id="run-valid",
                        conversation_id="c1",
                        user_id="u1",
                        model_id="gpt-4",
                        provider="openai",
                        message_id="assistant-1",
                        turn_message_id="assistant-1",
                        attempt_index=1,
                        status="completed",
                    ),
                    AgentSession(
                        id="run-cross-user",
                        conversation_id="c1",
                        user_id="u2",
                        model_id="gpt-4",
                        provider="openai",
                        message_id="assistant-1",
                        turn_message_id="assistant-1",
                        attempt_index=2,
                        status="completed",
                    ),
                ]
            )
            db.commit()

        with (
            patch("app.services.agent.session_cache.SessionLocal", test_session_local),
            self.assertLogs("app", level="INFO"),
        ):
            await write_session_started(
                run_id="run-new",
                conversation_id="c1",
                user_id="u1",
                model_id="gpt-4",
                provider="openai",
                message_id="assistant-1",
                turn_message_id="user-1",
                run_attempt_kind="regenerate",
            )

        with test_session_local() as db:
            created = db.get(AgentSession, "run-new")
            self.assertEqual(created.previous_run_id, "run-valid")
            self.assertEqual(created.attempt_index, 2)
        engine.dispose()

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
        terminal_at = datetime(2026, 8, 22, 4, 0, tzinfo=UTC)
        with patch("app.services.agent.session_cache.SessionLocal") as mock_sl:
            session = MagicMock()
            mock_sl.return_value.__enter__.return_value = session
            row = MagicMock()
            session.get.return_value = row
            with patch("app.services.agent.session_cache.utc_now", return_value=terminal_at):
                await write_session_status(run_id="r1", status="interrupted", total_steps=2, total_tool_calls=3)
            self.assertEqual(row.status, "interrupted")
            self.assertEqual(row.terminal_at, terminal_at)
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
