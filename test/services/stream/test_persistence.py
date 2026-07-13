import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from app.schemas.chat import ContextUsage, TextBlock, Usage
from app.services.stream.persistence import persist_message


def _populated_query(db):
    query = db.query.return_value
    query.populate_existing.return_value = query
    return query


class MessagePersistenceAdvisoryLockTests(unittest.TestCase):
    def test_postgresql_executes_transaction_advisory_lock(self):
        from app.services.stream.persistence import acquire_message_persistence_lock

        db = MagicMock()
        db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))

        acquire_message_persistence_lock(db, "msg-1")

        db.execute.assert_called_once()
        statement, params = db.execute.call_args.args
        self.assertIn("pg_advisory_xact_lock", str(statement))
        self.assertIn("hashtext", str(statement))
        self.assertEqual(params, {"key": "assistant_message:msg-1"})

    def test_non_postgresql_dialect_is_safe_noop(self):
        from app.services.stream.persistence import acquire_message_persistence_lock

        db = MagicMock()
        db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))

        acquire_message_persistence_lock(db, "msg-1")

        db.execute.assert_not_called()


class PersistMessageMonotonicTests(unittest.TestCase):
    def test_persist_message_acquires_advisory_lock_before_query(self):
        db = MagicMock()
        db.get_bind.return_value = SimpleNamespace(dialect=SimpleNamespace(name="postgresql"))
        _populated_query(db).filter_by.return_value.first.return_value = SimpleNamespace(content=[], usage=None)

        persist_message(
            db,
            "msg-1",
            "conv-1",
            "gpt-4",
            [TextBlock(type="text", id="answer-1", text="最终回答")],
        )

        method_names = [call[0] for call in db.mock_calls]
        self.assertLess(method_names.index("execute"), method_names.index("query"))
        db.query.return_value.populate_existing.assert_called_once_with()

    def test_persist_message_refreshes_stale_identity_before_partial_merge(self):
        refreshed_full = [{"type": "text", "id": "answer-1", "text": "完整回答"}]
        existing = SimpleNamespace(
            content=[{"type": "text", "id": "answer-1", "text": "锁前旧快照"}],
            usage=None,
        )
        db = MagicMock()
        query = db.query.return_value

        def populate_existing():
            existing.content = refreshed_full.copy()
            return query

        query.populate_existing.side_effect = populate_existing
        query.filter_by.return_value.first.return_value = existing

        persist_message(
            db,
            "msg-1",
            "conv-1",
            "gpt-4",
            [TextBlock(type="text", id="answer-1", text="完整")],
            partial=True,
        )

        self.assertEqual(existing.content, refreshed_full)

    def test_cancel_checkpoint_does_not_overwrite_newer_client_partial_with_old_blocks(self):
        client_partial = [{"type": "text", "id": "answer-1", "text": "旧回答已经扩展为客户端可见的新 text"}]
        existing = SimpleNamespace(content=client_partial.copy(), usage=None)
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing

        persist_message(
            db,
            "msg-1",
            "conv-1",
            "gpt-4",
            [TextBlock(type="text", id="answer-1", text="旧回答")],
            partial=True,
        )

        self.assertEqual(existing.content, client_partial)
        db.commit.assert_called_once()

    def test_longer_checkpoint_can_advance_existing_partial(self):
        existing = SimpleNamespace(
            content=[{"type": "text", "id": "answer-1", "text": "短"}],
            usage=None,
        )
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        longer = [TextBlock(type="text", id="answer-1", text="更长的 checkpoint 回答")]

        persist_message(db, "msg-1", "conv-1", "gpt-4", longer, partial=True)

        self.assertEqual(existing.content, [longer[0].model_dump()])
        db.commit.assert_called_once()

    def test_large_existing_search_block_and_new_text_are_both_preserved(self):
        search_block = {
            "type": "search",
            "id": "search-1",
            "query": "Fusion",
            "sources": [{"title": f"来源 {index}", "url": f"https://example.com/{index}"} for index in range(20)],
        }
        existing = SimpleNamespace(content=[search_block], usage=None)
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        incoming = TextBlock(type="text", id="answer-1", text="搜索后的回答")

        persist_message(db, "msg-1", "conv-1", "gpt-4", [incoming], partial=True)

        self.assertEqual(existing.content, [search_block, incoming.model_dump()])
        db.commit.assert_called_once()

    def test_non_prefix_text_checkpoint_prefers_current_visible_incoming_text(self):
        existing = SimpleNamespace(
            content=[{"type": "text", "id": "answer-1", "text": "后台分支内容"}],
            usage=None,
        )
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        incoming = TextBlock(type="text", id="answer-1", text="客户端当前可见内容")

        persist_message(db, "msg-1", "conv-1", "gpt-4", [incoming], partial=True)

        self.assertEqual(existing.content, [incoming.model_dump()])
        db.commit.assert_called_once()

    def test_thinking_checkpoint_uses_same_prefix_merge_rule(self):
        from app.schemas.chat import ThinkingBlock

        existing = SimpleNamespace(
            content=[{"type": "thinking", "id": "thinking-1", "thinking": "先分析"}],
            usage=None,
        )
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        incoming = ThinkingBlock(type="thinking", id="thinking-1", thinking="先分析，再验证")

        persist_message(db, "msg-1", "conv-1", "gpt-4", [incoming], partial=True)

        self.assertEqual(existing.content, [incoming.model_dump()])
        db.commit.assert_called_once()

    def test_full_completion_still_overwrites_and_writes_usage(self):
        existing = SimpleNamespace(
            content=[{"type": "text", "id": "answer-1", "text": "很长的旧 partial 内容"}],
            usage=None,
        )
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        final = [TextBlock(type="text", id="answer-1", text="最终")]
        usage = Usage(input_tokens=3, output_tokens=5)

        persist_message(db, "msg-1", "conv-1", "gpt-4", final, usage_data=usage, partial=False)

        self.assertEqual(existing.content, [final[0].model_dump()])
        self.assertEqual(existing.usage, usage.model_dump())
        db.commit.assert_called_once()

    def test_full_completion_persists_nested_context_status(self):
        existing = SimpleNamespace(content=[], usage=None)
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        context = ContextUsage(
            status="trimmed",
            window_tokens=258_000,
            estimated_tokens_before=230_000,
            estimated_tokens_after=190_000,
            actual_prompt_tokens=189_000,
            removed_turns=1,
            removed_messages=2,
            removed_tool_transactions=0,
        )

        persist_message(
            db,
            "msg-1",
            "conv-1",
            "kimi-k2.5",
            [TextBlock(type="text", id="answer-1", text="完成")],
            usage_data=Usage(input_tokens=240_000, output_tokens=100, context=context),
            partial=False,
        )

        self.assertEqual(existing.usage["input_tokens"], 240_000)
        self.assertEqual(existing.usage["context"]["actual_prompt_tokens"], 189_000)
        self.assertEqual(existing.usage["context"]["removed_turns"], 1)

    def test_failed_partial_persists_context_usage_without_overwriting_content(self):
        existing = SimpleNamespace(content=[], usage=None)
        db = MagicMock()
        _populated_query(db).filter_by.return_value.first.return_value = existing
        context = ContextUsage(status="estimator_unavailable", round_index=1, window_tokens=128_000)

        persist_message(
            db,
            "msg-1",
            "conv-1",
            "gpt-4",
            [],
            usage_data=Usage(input_tokens=0, output_tokens=0, context=context),
            partial=True,
        )

        self.assertEqual(existing.content, [])
        self.assertEqual(existing.usage["context"]["status"], "estimator_unavailable")


if __name__ == "__main__":
    unittest.main()
