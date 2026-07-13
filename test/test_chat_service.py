import asyncio
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import Conversation, Message, TextBlock
from app.schemas.response import ApiException
from app.services.chat_service import ChatService, _require_stream_initialized
from app.services.stream_state_service import StreamInitResult


def _populated_query(db):
    query = db.query.return_value
    query.populate_existing.return_value = query
    return query


class ChatServiceTests(unittest.TestCase):
    def test_stop_guard_init_failure_returns_explicit_retryable_message(self):
        with self.assertRaises(ApiException) as raised:
            _require_stream_initialized(
                StreamInitResult(
                    ok=False,
                    error_code="stream_stop_in_progress",
                    message="当前生成正在停止，请稍后重试",
                )
            )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_UNAVAILABLE")
        self.assertEqual(raised.exception.message, "当前生成正在停止，请稍后重试")

    def test_persist_stream_partial_before_stop_rejects_other_user_or_message(self):
        service = ChatService(MagicMock())
        service.conversation_service = MagicMock()

        with self.assertRaises(ApiException) as other_user:
            service.persist_stream_partial_before_stop(
                conversation_id="conv-1",
                user_id="user-1",
                message_id="msg-1",
                partial_content=[TextBlock(type="text", id="answer-1", text="半截")],
                stream_meta={"status": "streaming", "user_id": "user-2", "message_id": "msg-1", "model": "gpt-4"},
            )
        self.assertEqual(other_user.exception.status_code, 404)

        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
        with self.assertRaises(ApiException) as wrong_message:
            service.persist_stream_partial_before_stop(
                conversation_id="conv-1",
                user_id="user-1",
                message_id="msg-old",
                partial_content=[TextBlock(type="text", id="answer-1", text="半截")],
                stream_meta={"status": "streaming", "user_id": "user-1", "message_id": "msg-new", "model": "gpt-4"},
            )
        self.assertEqual(wrong_message.exception.status_code, 409)

    def test_persist_stream_partial_before_stop_rejects_unowned_conversation(self):
        service = ChatService(MagicMock())
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = None

        with self.assertRaises(ApiException) as raised:
            service.persist_stream_partial_before_stop(
                conversation_id="conv-hidden",
                user_id="user-1",
                message_id="msg-1",
                partial_content=[TextBlock(type="text", id="answer-1", text="半截")],
                stream_meta={"status": "streaming", "user_id": "user-1", "message_id": "msg-1", "model": "gpt-4"},
            )

        self.assertEqual(raised.exception.status_code, 404)

    def test_persist_stream_partial_before_stop_updates_existing_assistant(self):
        db = MagicMock()
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[],
            model_id="gpt-old",
        )
        _populated_query(db).filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
        partial = [TextBlock(type="text", id="answer-1", text="半截回答")]

        persisted = service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=partial,
            stream_meta={"status": "streaming", "user_id": "user-1", "message_id": "msg-1", "model": "gpt-4"},
        )

        self.assertTrue(persisted)
        self.assertEqual(existing.content, [partial[0].model_dump()])
        self.assertEqual(existing.model_id, "gpt-old")
        db.add.assert_not_called()
        db.commit.assert_called_once()

    def test_persist_stream_partial_before_stop_acquires_advisory_lock_before_query(self):
        db = MagicMock()
        db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[],
            model_id="gpt-old",
        )
        _populated_query(db).filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")

        service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")],
            stream_meta={
                "status": "streaming",
                "user_id": "user-1",
                "message_id": "msg-1",
                "model": "gpt-4",
            },
        )

        method_names = [call[0] for call in db.mock_calls]
        self.assertLess(method_names.index("execute"), method_names.index("query"))
        db.query.return_value.populate_existing.assert_called_once_with()

    def test_persist_stream_partial_before_stop_locks_before_conversation_ownership_load(self):
        calls = []
        db = MagicMock()
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[],
            model_id="gpt-4",
        )
        _populated_query(db).filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()

        def load_conversation(*_args):
            calls.append("conversation")
            return SimpleNamespace(id="conv-1")

        service.conversation_service.get_conversation.side_effect = load_conversation

        def acquire_lock(*_args):
            calls.append("lock")

        with patch("app.services.chat_service.acquire_message_persistence_lock", side_effect=acquire_lock):
            service.persist_stream_partial_before_stop(
                conversation_id="conv-1",
                user_id="user-1",
                message_id="msg-1",
                partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")],
                stream_meta={
                    "status": "streaming",
                    "user_id": "user-1",
                    "message_id": "msg-1",
                    "model": "gpt-4",
                },
            )

        self.assertEqual(calls[:2], ["lock", "conversation"])

    def test_persist_stream_partial_refreshes_stale_identity_before_merge(self):
        refreshed_full = [{"type": "text", "id": "answer-1", "text": "完整回答"}]
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[{"type": "text", "id": "answer-1", "text": "锁前旧快照"}],
            model_id="gpt-4",
        )
        db = MagicMock()
        query = db.query.return_value

        def populate_existing():
            existing.content = refreshed_full.copy()
            return query

        query.populate_existing.side_effect = populate_existing
        query.filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")

        service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[TextBlock(type="text", id="answer-1", text="完整")],
            stream_meta={
                "status": "streaming",
                "user_id": "user-1",
                "message_id": "msg-1",
                "model": "gpt-4",
            },
        )

        self.assertEqual(existing.content, refreshed_full)

    def test_persist_stream_partial_before_stop_role_conflict_rolls_back_transaction_lock(self):
        db = MagicMock()
        db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        _populated_query(db).filter.return_value.first.return_value = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="user",
            content=[],
        )
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")

        with self.assertRaises(ApiException) as raised:
            service.persist_stream_partial_before_stop(
                conversation_id="conv-1",
                user_id="user-1",
                message_id="msg-1",
                partial_content=[TextBlock(type="text", id="answer-1", text="半截回答")],
                stream_meta={
                    "status": "streaming",
                    "user_id": "user-1",
                    "message_id": "msg-1",
                    "model": "gpt-4",
                },
            )

        self.assertEqual(raised.exception.status_code, 409)
        db.execute.assert_called_once()
        db.rollback.assert_called_once()
        db.commit.assert_not_called()

    def test_persist_stream_partial_before_stop_keeps_existing_text_when_incoming_is_its_prefix(self):
        complete_content = [
            {"type": "text", "id": "answer-1", "text": "后台已经落库的更完整回答，不应被 stop partial 截断"}
        ]
        db = MagicMock()
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=complete_content.copy(),
            model_id="gpt-4",
        )
        _populated_query(db).filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")

        persisted = service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[TextBlock(type="text", id="answer-1", text="后台已经落库")],
            stream_meta={
                "status": "streaming",
                "user_id": "user-1",
                "message_id": "msg-1",
                "model": "gpt-4",
            },
        )

        self.assertTrue(persisted)
        self.assertEqual(existing.content, complete_content)
        db.commit.assert_called_once()

    def test_persist_stream_partial_before_stop_merges_existing_search_and_incoming_text(self):
        search_block = {
            "type": "search",
            "id": "search-1",
            "query": "Fusion",
            "sources": [{"title": "来源", "url": "https://example.com"}],
        }
        db = MagicMock()
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[search_block],
            model_id="gpt-4",
        )
        _populated_query(db).filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
        partial = TextBlock(type="text", id="answer-1", text="搜索后的回答")

        persisted = service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[partial],
            stream_meta={
                "status": "streaming",
                "user_id": "user-1",
                "message_id": "msg-1",
                "model": "gpt-4",
            },
        )

        self.assertTrue(persisted)
        self.assertEqual(existing.content, [search_block, partial.model_dump()])
        db.commit.assert_called_once()

    def test_serialized_finalizer_and_stop_writes_preserve_the_winning_complete_content(self):
        from app.services.stream.persistence import persist_message

        db = MagicMock()
        existing = SimpleNamespace(
            id="msg-1",
            conversation_id="conv-1",
            role="assistant",
            content=[],
            usage=None,
            model_id="gpt-4",
        )
        query = db.query.return_value
        query.populate_existing.return_value = query
        query.filter_by.return_value.first.return_value = existing
        query.filter.return_value.first.return_value = existing
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
        stream_meta = {
            "status": "streaming",
            "user_id": "user-1",
            "message_id": "msg-1",
            "model": "gpt-4",
        }

        # finalizer 先提交时，后获得锁的 stop 会重新查询并按前缀保留完整内容。
        full = TextBlock(type="text", id="answer-1", text="完整回答")
        persist_message(db, "msg-1", "conv-1", "gpt-4", [full], partial=False)
        service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[TextBlock(type="text", id="answer-1", text="完整")],
            stream_meta=stream_meta,
        )
        self.assertEqual(existing.content, [full.model_dump()])

        # stop 先提交时，后获得锁的正常 complete 仍完整覆盖 partial。
        existing.content = []
        service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[TextBlock(type="text", id="answer-1", text="半截")],
            stream_meta=stream_meta,
        )
        final = TextBlock(type="text", id="answer-1", text="最终完整回答")
        persist_message(db, "msg-1", "conv-1", "gpt-4", [final], partial=False)
        self.assertEqual(existing.content, [final.model_dump()])

    def test_persist_stream_partial_before_stop_skips_non_streaming_meta(self):
        db = MagicMock()
        service = ChatService(db)
        service.conversation_service = MagicMock()

        persisted = service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=[TextBlock(type="text", id="answer-1", text="较短 partial")],
            stream_meta={
                "status": "done",
                "user_id": "user-1",
                "message_id": "msg-1",
                "model": "gpt-4",
            },
        )

        self.assertFalse(persisted)
        db.query.assert_not_called()
        db.commit.assert_not_called()

    def test_persist_stream_partial_before_stop_creates_assistant_with_stream_model(self):
        db = MagicMock()
        _populated_query(db).filter.return_value.first.return_value = None
        service = ChatService(db)
        service.conversation_service = MagicMock()
        service.conversation_service.get_conversation.return_value = SimpleNamespace(id="conv-1")
        partial = [TextBlock(type="text", id="answer-1", text="半截回答")]

        persisted = service.persist_stream_partial_before_stop(
            conversation_id="conv-1",
            user_id="user-1",
            message_id="msg-1",
            partial_content=partial,
            stream_meta={"status": "streaming", "user_id": "user-1", "message_id": "msg-1", "model": "gpt-4"},
        )

        self.assertTrue(persisted)
        created = db.add.call_args.args[0]
        self.assertEqual(created.id, "msg-1")
        self.assertEqual(created.conversation_id, "conv-1")
        self.assertEqual(created.role, "assistant")
        self.assertEqual(created.model_id, "gpt-4")
        self.assertEqual(created.content, [partial[0].model_dump()])
        db.commit.assert_called_once()

    def test_build_recent_dialog_content_prefers_latest_user_assistant_pair(self):
        service = object.__new__(ChatService)
        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="old question")],
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    content=[TextBlock(type="text", text="old answer")],
                    model_id="qwen-max-latest",
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    content=[TextBlock(type="text", text="latest question")],
                ),
                Message(
                    id="assistant-msg-2",
                    role="assistant",
                    content=[TextBlock(type="text", text="latest answer")],
                    model_id="qwen-max-latest",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        content = service._build_recent_dialog_content(conversation)

        self.assertEqual(content, "用户: latest question\n助手: latest answer")

    def test_build_recent_dialog_content_falls_back_to_user_messages(self):
        service = object.__new__(ChatService)
        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="question one")],
                ),
                Message(
                    id="user-msg-2",
                    role="user",
                    content=[TextBlock(type="text", text="question two")],
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )

        content = service._build_recent_dialog_content(conversation)

        self.assertEqual(content, "用户: question two")

    # _build_llm_messages 已重构为独立 async 函数 app/services/chat/message_builder.py::build_llm_messages
    # 该函数还注入了 system date prompt + user_system_prompt + 图片 base64 等副作用，
    # 不再适合作为 ChatService 的私有方法测。新测试应该写在
    # test/services/chat/test_message_builder.py（暂未补，等需要时再加）。

    def test_update_message_commits_when_update_succeeds(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()

        updated_message = Message(
            id="assistant-msg-1",
            role="assistant",
            content=[TextBlock(type="text", text="updated")],
            model_id="qwen-max-latest",
        )
        service.conversation_service.update_message.return_value = updated_message

        result = service.update_message(
            "assistant-msg-1",
            {"content": [TextBlock(type="text", text="updated")]},
        )

        self.assertIs(result, updated_message)
        service.db.commit.assert_called_once()

    def test_process_message_non_stream_injects_no_tool_network_boundary(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="OpenAI 最近发布了什么模型？",
                    model_id="qwen-vl-max",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                True,
            )
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="基于已有知识回答。"))],
            usage=None,
        )
        build_messages_mock = AsyncMock(
            return_value=[
                {"role": "system", "content": "日期 system"},
                {"role": "user", "content": "OpenAI 最近发布了什么模型？"},
            ]
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-vl-max", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": True, "agentTools": False},
            ),
            patch(
                "app.services.chat_service.build_llm_messages",
                new=build_messages_mock,
            ),
            patch("app.services.chat_service.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            asyncio.run(
                service.process_message(
                    model_id="qwen-vl-max",
                    message="OpenAI 最近发布了什么模型？",
                    user_id="user-1",
                    stream=False,
                )
            )

        build_messages_mock.assert_awaited_once_with(
            service._get_or_create_conversation.return_value[0].messages,
            has_vision=False,
            file_repo=service.file_repo,
            user_system_prompt=None,
            user_id="user-1",
            conversation_id="conv-1",
        )
        sent_messages = mock_litellm.acompletion.await_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in sent_messages[:3]], ["system", "system", "user"])
        self.assertIn("日期 system", sent_messages[0]["content"])
        self.assertIn("【无联网工具边界规则】", sent_messages[1]["content"])
        self.assertIn("无法实时核验", sent_messages[1]["content"])
        self.assertIn("不要把已有知识包装成最新事实", sent_messages[1]["content"])
        self.assertIn("不要把缺少工具描述成系统故障", sent_messages[1]["content"])
        self.assertEqual(sent_messages[2]["content"], "OpenAI 最近发布了什么模型？")
        self.assertEqual(
            mock_litellm.acompletion.await_args.kwargs["extra_body"],
            {"metadata": {"tags": ["app:fusion", "phase:chat_non_stream"]}},
        )

    def test_handle_non_stream_applies_controlled_max_tokens(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="回答"))],
            usage=None,
        )

        context_plan = MagicMock(messages=[{"role": "user", "content": "有效问题"}])
        with (
            patch("app.services.chat_service.litellm.acompletion", new=AsyncMock(return_value=mock_response)) as call,
            patch(
                "app.services.chat_service.prepare_context",
                new=AsyncMock(return_value=context_plan),
            ) as prepare,
        ):
            asyncio.run(
                service._handle_non_stream(
                    "litellm_proxy/model-1",
                    "model-1",
                    {},
                    [{"role": "user", "content": "问题"}],
                    "conv-1",
                    {"max_tokens": 9999},
                )
            )

        self.assertEqual(call.await_args.kwargs["max_tokens"], 4096)
        self.assertEqual(call.await_args.kwargs["messages"], context_plan.messages)
        self.assertEqual(prepare.await_args.kwargs["model_id"], "model-1")
        self.assertEqual(prepare.await_args.kwargs["litellm_model"], "litellm_proxy/model-1")

    def test_handle_non_stream_ignores_invalid_max_tokens(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="回答"))],
            usage=None,
        )

        with patch("app.services.chat_service.litellm.acompletion", new=AsyncMock(return_value=mock_response)) as call:
            asyncio.run(
                service._handle_non_stream(
                    "litellm_proxy/model-1",
                    "model-1",
                    {},
                    [{"role": "user", "content": "问题"}],
                    "conv-1",
                    {"max_tokens": True},
                )
            )

        self.assertNotIn("max_tokens", call.await_args.kwargs)

    def test_process_message_non_stream_injects_no_vision_boundary_for_image_on_text_model(self):
        db = MagicMock()
        db.query.return_value.filter.return_value.first.return_value = None
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="image-1",
            user_id="user-1",
            original_filename="chart.png",
            mimetype="image/png",
            status="processed",
            storage_key="conv-1/image-1/chart.png",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = True
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="这张图里有什么？",
                    model_id="qwen-vl-max",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                True,
            )
        )
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="当前模型无法查看图片。"))],
            usage=None,
        )
        build_messages_mock = AsyncMock(
            return_value=[
                {"role": "system", "content": "日期 system"},
                {"role": "user", "content": "这张图里有什么？"},
            ]
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-vl-max", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": True, "agentTools": False, "vision": False},
            ),
            patch(
                "app.services.chat_service.build_llm_messages",
                new=build_messages_mock,
            ),
            patch("app.services.chat_service.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)
            asyncio.run(
                service.process_message(
                    model_id="qwen-vl-max",
                    message="这张图里有什么？",
                    user_id="user-1",
                    stream=False,
                    file_ids=["image-1"],
                )
            )

        build_messages_mock.assert_awaited_once_with(
            service._get_or_create_conversation.return_value[0].messages,
            has_vision=False,
            file_repo=service.file_repo,
            user_system_prompt=None,
            user_id="user-1",
            conversation_id="conv-1",
        )
        sent_messages = mock_litellm.acompletion.await_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in sent_messages[:4]], ["system", "system", "system", "user"])
        self.assertIn("日期 system", sent_messages[0]["content"])
        self.assertIn("【无图片理解能力边界规则】", sent_messages[1]["content"])
        self.assertIn("当前模型不能读取或理解图片附件", sent_messages[1]["content"])
        self.assertIn("【无联网工具边界规则】", sent_messages[2]["content"])
        self.assertEqual(sent_messages[3]["content"], "这张图里有什么？")

    def test_generate_suggested_questions_limits_output_to_three(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()
        service.file_repo = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="hello",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="Explain this answer")],
                ),
                Message(
                    id="assistant-msg-1",
                    role="assistant",
                    content=[TextBlock(type="text", text="Here is the answer")],
                    model_id="qwen-max-latest",
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.conversation_service.get_conversation.return_value = conversation

        mock_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="1. Follow-up A\n2. Follow-up B\n3. Follow-up C\n4. Follow-up D")
                )
            ]
        )

        with (
            patch("app.services.chat_service.litellm") as mock_litellm,
            patch("app.services.chat_service.llm_manager") as mock_manager,
        ):
            mock_manager.resolve_model.return_value = ("openai/qwen-max-latest", "qwen", {})
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            questions = asyncio.run(
                service.generate_suggested_questions(
                    user_id="user-1",
                    conversation_id="conv-1",
                )
            )

        self.assertEqual(len(questions), 3)
        self.assertEqual(questions, ["Follow-up A", "Follow-up B", "Follow-up C"])
        self.assertEqual(mock_litellm.acompletion.await_args.kwargs["max_tokens"], 512)
        self.assertEqual(
            mock_litellm.acompletion.await_args.kwargs["extra_body"],
            {
                "metadata": {
                    "tags": ["app:fusion", "phase:suggest_questions"],
                    "prompt_slug": "generate-suggested-questions",
                    "prompt_version": "code-default",
                }
            },
        )

    def test_generate_title_persists_title_to_database(self):
        service = object.__new__(ChatService)
        service.db = MagicMock()
        service.conversation_service = MagicMock()
        service.file_repo = MagicMock()

        conversation = Conversation(
            id="conv-1",
            user_id="user-1",
            title="old",
            model_id="qwen-max-latest",
            messages=[
                Message(
                    id="user-msg-1",
                    role="user",
                    content=[TextBlock(type="text", text="Explain fusion")],
                ),
            ],
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        service.conversation_service.get_conversation.return_value = conversation

        mock_response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Fusion Chat"))])

        with (
            patch("app.services.chat_service.litellm") as mock_litellm,
            patch("app.services.chat_service.llm_manager") as mock_manager,
        ):
            mock_manager.resolve_model.return_value = ("openai/qwen-max-latest", "qwen", {})
            mock_litellm.acompletion = AsyncMock(return_value=mock_response)

            title = asyncio.run(
                service.generate_title(
                    user_id="user-1",
                    conversation_id="conv-1",
                )
            )

        self.assertEqual(title, "Fusion Chat")
        self.assertEqual(mock_litellm.acompletion.await_args.kwargs["max_tokens"], 512)
        self.assertEqual(
            mock_litellm.acompletion.await_args.kwargs["extra_body"],
            {
                "metadata": {
                    "tags": ["app:fusion", "phase:generate_title"],
                    "prompt_slug": "generate-title",
                    "prompt_version": "code-default",
                }
            },
        )
        service.conversation_service.repo.update_title.assert_called_once_with("conv-1", "Fusion Chat")
        service.db.commit.assert_called_once()

    def test_validate_message_files_accepts_processed_same_conversation_file(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        file_record = SimpleNamespace(
            id="file-1",
            user_id="user-1",
            original_filename="note.txt",
            mimetype="text/plain",
            status="processed",
            storage_key="conv-1/file-1/note.txt",
            thumbnail_key=None,
        )
        service.file_repo.get_file_by_id.return_value = file_record
        service.file_repo.is_file_linked_to_conversation.return_value = True

        result = service._validate_message_files(["file-1"], "user-1", "conv-1")

        self.assertEqual(result, [file_record])
        service.file_repo.get_file_by_id.assert_called_once_with("file-1", user_id="user-1")
        service.file_repo.is_file_linked_to_conversation.assert_called_once_with("conv-1", "file-1")

    def test_build_file_block_omits_thumbnail_url_when_storage_object_missing(self):
        service = object.__new__(ChatService)
        file_record = SimpleNamespace(
            id="file-1",
            original_filename="diagram.png",
            mimetype="image/png",
            storage_backend="local",
            thumbnail_key="conv-1/file-1/thumbnail.jpg",
            width=640,
            height=480,
        )
        storage = SimpleNamespace(
            exists=AsyncMock(return_value=False),
            get_url=AsyncMock(return_value="/files/file-1/thumb.png"),
        )

        with patch("app.services.chat_service.get_storage_for_backend", return_value=storage) as get_backend:
            block = asyncio.run(service._build_file_block_from_record(file_record))

        get_backend.assert_called_once_with("local")
        storage.exists.assert_awaited_once_with("conv-1/file-1/thumbnail.jpg")
        storage.get_url.assert_not_awaited()
        self.assertIsNone(block.thumbnail_url)
        self.assertEqual(block.file_id, "file-1")

    def test_validate_message_files_rejects_other_user_file(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = None

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-1"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "文件不存在或无权访问")
        service.file_repo.is_file_linked_to_conversation.assert_not_called()

    def test_validate_message_files_rejects_same_user_file_from_other_conversation(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-2",
            user_id="user-1",
            original_filename="other.txt",
            mimetype="text/plain",
            status="processed",
            storage_key="conv-2/file-2/other.txt",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = False

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-2"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "文件不属于当前会话")

    def test_validate_message_files_rejects_unprocessed_non_image_file(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-3",
            user_id="user-1",
            original_filename="draft.pdf",
            mimetype="application/pdf",
            status="parsing",
            storage_key="conv-1/file-3/draft.pdf",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = True

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-3"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "文件仍在处理，请稍后再发送")

    def test_validate_message_files_rejects_image_missing_storage_key(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-4",
            user_id="user-1",
            original_filename="chart.png",
            mimetype="image/png",
            status="processed",
            storage_key=None,
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = True

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-4"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "图片文件不可用，请重新上传")

    def test_validate_message_files_rejects_unprocessed_image_even_with_storage_key(self):
        service = object.__new__(ChatService)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-5",
            user_id="user-1",
            original_filename="uploading.png",
            mimetype="image/png",
            status="uploading",
            storage_key="conv-1/file-5/original/uploading.png",
            thumbnail_key=None,
        )
        service.file_repo.is_file_linked_to_conversation.return_value = True

        with self.assertRaises(ApiException) as context:
            service._validate_message_files(["file-5"], "user-1", "conv-1")

        self.assertEqual(context.exception.code, "INVALID_PARAM")
        self.assertEqual(context.exception.message, "图片文件不可用，请重新上传")

    def test_process_message_rejects_invalid_file_before_creating_message(self):
        db = MagicMock()
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.file_repo.get_file_by_id.return_value = None
        service.stream_handler = MagicMock()
        service.stream_handler.generate_to_redis = AsyncMock()
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-1",
                    user_id="user-1",
                    title="继续分析",
                    model_id="qwen-max-latest",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                False,
            )
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-max-latest", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": False, "agentTools": False, "vision": False},
            ),
            patch("app.services.chat_service.init_stream", new=AsyncMock()) as init_stream_mock,
            patch("app.services.chat_service.register_task") as register_task_mock,
            patch("app.services.chat_service.asyncio.create_task") as create_task_mock,
            patch("app.services.chat_service.litellm") as mock_litellm,
        ):
            mock_litellm.acompletion = AsyncMock()
            with self.assertRaises(ApiException):
                asyncio.run(
                    service.process_message(
                        model_id="qwen-max-latest",
                        message="继续分析",
                        user_id="user-1",
                        conversation_id="conv-1",
                        stream=True,
                        file_ids=["missing-file"],
                    )
                )

        service.conversation_service.create_message.assert_not_called()
        db.commit.assert_not_called()
        init_stream_mock.assert_not_awaited()
        register_task_mock.assert_not_called()
        create_task_mock.assert_not_called()
        service.stream_handler.generate_to_redis.assert_not_called()
        mock_litellm.acompletion.assert_not_called()

    def test_process_message_fails_before_starting_generation_when_stream_init_fails(self):
        from app.services.stream_state_service import StreamInitResult

        db = MagicMock()
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.stream_handler = MagicMock()
        service.stream_handler.generate_to_redis = AsyncMock()
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-redis-down",
                    user_id="user-1",
                    title="Redis 故障",
                    model_id="qwen-max-latest",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                False,
            )
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-max-latest", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": False, "agentTools": False, "vision": False},
            ),
            patch(
                "app.services.chat_service.init_stream",
                new=AsyncMock(
                    return_value=StreamInitResult(
                        ok=False,
                        error_code="redis_unavailable",
                        message="Redis 不可用",
                    )
                ),
            ),
            patch("app.services.chat_service.register_task") as register_task_mock,
            patch("app.services.chat_service.asyncio.create_task") as create_task_mock,
        ):
            with self.assertRaises(ApiException) as raised:
                asyncio.run(
                    service.process_message(
                        model_id="qwen-max-latest",
                        message="不要调用模型",
                        user_id="user-1",
                        conversation_id="conv-redis-down",
                        stream=True,
                    )
                )

        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.code, "STREAM_UNAVAILABLE")
        db.commit.assert_not_called()
        db.rollback.assert_called_once()
        create_task_mock.assert_not_called()
        register_task_mock.assert_not_called()
        service.stream_handler.generate_to_redis.assert_not_called()

    def test_process_message_records_cas_finalize_failure_when_db_commit_fails(self):
        from app.services.stream_state_service import StreamInitResult

        db = MagicMock()
        db.commit.side_effect = RuntimeError("database unavailable")
        service = ChatService(db)
        service.file_repo = MagicMock()
        service.stream_handler = MagicMock()
        service.stream_handler.generate_to_redis = AsyncMock()
        service.conversation_service = MagicMock()
        service._get_or_create_conversation = MagicMock(
            return_value=(
                Conversation(
                    id="conv-commit-fail",
                    user_id="user-1",
                    title="提交失败",
                    model_id="qwen-max-latest",
                    messages=[],
                    created_at=datetime.now(),
                    updated_at=datetime.now(),
                ),
                False,
            )
        )

        with (
            patch(
                "app.services.chat_service.llm_manager.resolve_model",
                return_value=("openai/qwen-max-latest", "qwen", {}),
            ),
            patch(
                "app.services.chat_service.litellm_catalog.get_capabilities",
                return_value={"functionCalling": False, "agentTools": False, "vision": False},
            ),
            patch(
                "app.services.chat_service.init_stream",
                new=AsyncMock(return_value=StreamInitResult(ok=True)),
            ),
            patch(
                "app.services.chat_service.finalize_stream",
                new=AsyncMock(return_value=False),
            ) as finalize_mock,
            patch("app.services.chat_service.logger.error") as error_log,
            patch("app.services.chat_service.register_task") as register_task_mock,
            patch("app.services.chat_service.asyncio.create_task") as create_task_mock,
        ):
            with self.assertRaisesRegex(RuntimeError, "database unavailable"):
                asyncio.run(
                    service.process_message(
                        model_id="qwen-max-latest",
                        message="不要启动模型",
                        user_id="user-1",
                        conversation_id="conv-commit-fail",
                        stream=True,
                    )
                )

        db.rollback.assert_called_once()
        finalize_mock.assert_awaited_once()
        error_log.assert_called_once()
        self.assertIn("CAS 收尾失败", error_log.call_args.args[0])
        create_task_mock.assert_not_called()
        register_task_mock.assert_not_called()
        service.stream_handler.generate_to_redis.assert_not_called()


if __name__ == "__main__":
    unittest.main()
