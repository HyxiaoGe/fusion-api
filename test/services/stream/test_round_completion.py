import importlib
import unittest
from unittest.mock import Mock

from app.services.stream.step_lifecycle import AgentStepContext


def _subject(test_case):
    try:
        return importlib.import_module("app.services.stream.round_completion")
    except ModuleNotFoundError as exc:
        if exc.name == "app.services.stream.round_completion":
            test_case.fail("缺少 app.services.stream.round_completion 模块")
        raise


class RoundCompletionTests(unittest.IsolatedAsyncioTestCase):
    def test_append_round_content_blocks_appends_reasoning_and_text(self):
        """普通文本回合应按 reasoning、text 顺序追加内容块。"""
        module = _subject(self)
        content_blocks = []

        module.append_round_content_blocks(
            content_blocks,
            "推理内容",
            "正文内容",
            "blk_thinking",
            "blk_text",
        )

        self.assertEqual(len(content_blocks), 2)
        self.assertEqual(content_blocks[0].type, "thinking")
        self.assertEqual(content_blocks[0].id, "blk_thinking")
        self.assertEqual(content_blocks[0].thinking, "推理内容")
        self.assertEqual(content_blocks[1].type, "text")
        self.assertEqual(content_blocks[1].id, "blk_text")
        self.assertEqual(content_blocks[1].text, "正文内容")

    def test_append_round_content_blocks_skips_empty_buffers(self):
        """空 reasoning/text buffer 不应追加内容块。"""
        module = _subject(self)
        content_blocks = []

        module.append_round_content_blocks(
            content_blocks,
            "",
            "",
            "blk_thinking",
            "blk_text",
        )

        self.assertEqual(content_blocks, [])

    async def test_complete_text_response_step_completes_without_tools_and_returns_duration(self):
        """无工具文本回合应以空 tool_names 和 0 tool_call_count 闭合 step。"""
        module = _subject(self)
        context = AgentStepContext(
            step_id="step-1",
            step_number=3,
            started_at=10.0,
            thinking_block_id="blk_thinking",
            text_block_id="blk_text",
        )
        emitter = object()
        session_cache = object()
        clock = Mock(return_value=10.25)
        calls = []

        async def complete_step_fn(**kwargs):
            calls.append(kwargs)
            return 250

        duration_ms = await module.complete_text_response_step(
            context=context,
            emitter=emitter,
            session_cache=session_cache,
            complete_step_fn=complete_step_fn,
            clock=clock,
        )

        self.assertEqual(duration_ms, 250)
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0]["context"], context)
        self.assertIs(calls[0]["emitter"], emitter)
        self.assertIs(calls[0]["session_cache"], session_cache)
        self.assertEqual(calls[0]["tool_names"], [])
        self.assertEqual(calls[0]["tool_call_count"], 0)
        self.assertIs(calls[0]["clock"], clock)
