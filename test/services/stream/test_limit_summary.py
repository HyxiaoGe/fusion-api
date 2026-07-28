import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas.chat import ContextUsage, SearchBlock, SearchSourceSummary, SourceReference, UrlBlock, Usage
from app.services.chat.context_manager import ContextPlan
from app.services.chat.context_manager import prepare_context as prepare_context_real
from app.services.stream.limit_summary import (
    DEEP_RESEARCH_INCOMPLETE_TEXT,
    LIMIT_SUMMARY_PROMPT,
    LimitSummaryStepRequest,
    accumulate_summary_usage,
    append_limit_summary_prompt,
    append_summary_content_blocks,
    build_limit_summary_call_kwargs,
    compute_summary_timeout,
    remove_conflicting_tool_usage_contract,
    run_limit_summary_step,
)
from app.services.stream.research_evidence import ResearchEvidenceWorkset
from app.services.stream.step_lifecycle import AgentStepContext


def _deep_summary_evidence() -> tuple[ResearchEvidenceWorkset, list]:
    blocks = [
        SearchBlock(
            type="search",
            query="研究问题",
            sources=[
                SearchSourceSummary(title="候选 A", url="https://example.com/a"),
                SearchSourceSummary(title="候选 B", url="https://example.com/b"),
            ],
            source_refs=[
                SourceReference(
                    kind="search",
                    title="候选 A",
                    url="https://example.com/a",
                    evidence_id="ev-a",
                    citation_index=2,
                ),
                SourceReference(
                    kind="search",
                    title="候选 B",
                    url="https://example.com/b",
                    evidence_id="ev-b",
                    citation_index=3,
                ),
            ],
            source_count=2,
        ),
        UrlBlock(
            type="url_read",
            url="https://example.com/a",
            title="来源 A",
            source_refs=[
                SourceReference(
                    kind="url_read",
                    title="来源 A",
                    url="https://example.com/a",
                    evidence_id="ev-a",
                    citation_index=2,
                )
            ],
            source_count=1,
        ),
        UrlBlock(
            type="url_read",
            url="https://example.com/b",
            title="来源 B",
            source_refs=[
                SourceReference(
                    kind="url_read",
                    title="来源 B",
                    url="https://example.com/b",
                    evidence_id="ev-b",
                    citation_index=3,
                )
            ],
            source_count=1,
        ),
    ]
    workset = ResearchEvidenceWorkset()
    workset.record_content_blocks(blocks)
    return workset, blocks


class LimitSummaryHelpersTests(unittest.TestCase):
    def test_no_progress_summary_uses_neutral_prompt_without_limit_language(self):
        messages = []

        append_limit_summary_prompt(messages, summary_finish_reason="no_progress_summary")

        content = messages[-1]["content"]
        self.assertIn("现有搜索已不再产生新的有效信息", content)
        self.assertIn("不要向用户提及", content)
        self.assertNotIn("工具调用上限", content)
        self.assertNotIn("额度", content)
        self.assertNotIn("预算", content)

    def test_plan_repair_summary_does_not_claim_tool_limit(self):
        messages = []

        append_limit_summary_prompt(messages, summary_finish_reason="plan_repair_exhausted")

        content = messages[-1]["content"]
        self.assertIn("只基于已经实际取得的结果", content)
        self.assertIn("建议重试", content)
        self.assertNotIn("你已达到工具调用上限", content)

    def test_deep_research_plan_repair_summary_keeps_research_citation_contract(self):
        messages = []

        append_limit_summary_prompt(
            messages,
            summary_finish_reason="plan_repair_exhausted",
            task_mode="deep_research",
        )

        content = messages[-1]["content"]
        self.assertIn("只基于已经实际取得的结果", content)
        self.assertIn("只基于当前已成功取得并读取的来源", content)
        self.assertIn("只能使用研究证据工作集列出的引用编号", content)
        self.assertNotIn("你已达到工具调用上限", content)

    def test_build_limit_summary_call_kwargs_copies_and_removes_tool_controls(self):
        tools = [{"function": {"name": "web_search"}}]
        call_kwargs = {
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
            "extra_body": {"thinking": {"type": "disabled"}},
        }

        result = build_limit_summary_call_kwargs(call_kwargs)

        self.assertIsNot(result, call_kwargs)
        self.assertEqual(
            result,
            {
                "temperature": 0.2,
                "extra_body": {"thinking": {"type": "disabled"}},
            },
        )
        self.assertEqual(call_kwargs["tools"], tools)
        self.assertEqual(call_kwargs["tool_choice"], "auto")

    def test_compute_summary_timeout_uses_remaining_budget(self):
        timeout = compute_summary_timeout(
            total_timeout_s=300,
            run_start=100.0,
            clock=lambda: 125.5,
        )

        self.assertEqual(timeout, 274.5)

    def test_compute_summary_timeout_has_10s_floor(self):
        timeout = compute_summary_timeout(
            total_timeout_s=300,
            run_start=100.0,
            clock=lambda: 450.0,
        )

        self.assertEqual(timeout, 10)

    def test_accumulate_summary_usage_adds_usage_data(self):
        result = accumulate_summary_usage(
            Usage(input_tokens=2, output_tokens=3),
            Usage(input_tokens=5, output_tokens=7),
        )

        self.assertEqual(result, Usage(input_tokens=7, output_tokens=10))

    def test_accumulate_summary_usage_keeps_existing_usage_without_usage_data(self):
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)

        result = accumulate_summary_usage(accumulated_usage, None)

        self.assertIs(result, accumulated_usage)

    def test_deep_summary_removes_tool_history_and_control_prompts_but_keeps_evidence(self):
        messages = [
            {"role": "system", "content": "【Fusion 身份一致性规则】保留"},
            {"role": "system", "content": "【自主联网判断规则】按需调用工具"},
            {"role": "system", "content": "【执行计划控制规则】必须更新计划"},
            {"role": "system", "content": "【深度研究执行约束】先建立计划"},
            {"role": "system", "content": "【本轮研究证据工作集】\n[2] status=read_success"},
            {"role": "assistant", "content": "", "tool_calls": [{"name": "url_read"}]},
            {"role": "tool", "content": "原始工具结果"},
            {"role": "user", "content": "原始研究问题"},
            {"role": "user", "content": "<web_context>已读来源安全投影</web_context>"},
        ]

        remove_conflicting_tool_usage_contract(messages, task_mode="deep_research")

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "【Fusion 身份一致性规则】保留"},
                {"role": "system", "content": "【本轮研究证据工作集】\n[2] status=read_success"},
                {"role": "user", "content": "原始研究问题"},
                {"role": "user", "content": "<web_context>已读来源安全投影</web_context>"},
            ],
        )

    def test_append_summary_content_blocks_adds_reasoning_and_text(self):
        content_blocks = []

        append_summary_content_blocks(
            content_blocks=content_blocks,
            reasoning_buf="推理",
            content_buf="总结正文",
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )

        self.assertEqual([block.type for block in content_blocks], ["thinking", "text"])
        self.assertEqual(content_blocks[0].id, "blk-thinking")
        self.assertEqual(content_blocks[0].thinking, "推理")
        self.assertEqual(content_blocks[1].id, "blk-text")
        self.assertEqual(content_blocks[1].text, "总结正文")

    def test_append_summary_content_blocks_recovers_reasoning_only_as_text(self):
        content_blocks = []

        append_summary_content_blocks(
            content_blocks=content_blocks,
            reasoning_buf="总结正文",
            content_buf="",
            thinking_block_id="blk-thinking",
            text_block_id="blk-text",
        )

        self.assertEqual(len(content_blocks), 1)
        self.assertEqual(content_blocks[0].type, "text")
        self.assertEqual(content_blocks[0].id, "blk-text")
        self.assertEqual(content_blocks[0].text, "总结正文")


class LimitSummaryStepTests(unittest.IsolatedAsyncioTestCase):
    def _deep_request(self, *, workset, content_blocks, answer):
        stream_kwargs = []
        answers = iter(answer if isinstance(answer, (list, tuple)) else [answer])

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-deep-summary",
                step_number=9,
                started_at=100.0,
                thinking_block_id="blk-deep-thinking",
                text_block_id="blk-deep-text",
            )

        async def prepare_context_fn(**kwargs):
            return ContextPlan(
                messages=list(kwargs["messages"]),
                status="no_op",
                context_window_tokens=1000,
                context_window_source="test",
                context_window_status="known",
                estimated_tokens_before=100,
                estimated_tokens_after=100,
            )

        async def stream_round_fn(*_args, **kwargs):
            stream_kwargs.append(kwargs)
            return "", next(answers), [], "stop", Usage(input_tokens=2, output_tokens=3)

        emitter = SimpleNamespace(evidence_item_upserted=AsyncMock())
        request = LimitSummaryStepRequest(
            conversation_id="conv-deep",
            task_id="task-deep",
            run_id="run-deep",
            step_number=9,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=[{"role": "user", "content": "深度研究"}],
            should_use_reasoning=False,
            content_blocks=content_blocks,
            call_kwargs={},
            accumulated_usage=Usage(input_tokens=1, output_tokens=1),
            emitter=emitter,
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=AsyncMock(return_value="response"),
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
            task_mode="deep_research",
            evidence_policy="deep_research_v1",
            research_workset=workset,
        )
        return request, emitter, stream_kwargs, prepare_context_fn

    async def test_deep_summary_defers_and_emits_valid_answer_once_with_used_evidence(self):
        workset, blocks = _deep_summary_evidence()
        request, emitter, stream_kwargs, prepare_context_fn = self._deep_request(
            workset=workset,
            content_blocks=blocks,
            answer="综合结论来自两个已读来源。[2][3]",
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(stream_kwargs[0]["defer_output"])
        self.assertFalse(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], "综合结论来自两个已读来源。[2][3]")
        self.assertEqual(request.content_blocks[-1].text, "综合结论来自两个已读来源。[2][3]")
        self.assertEqual(emitter.evidence_item_upserted.await_count, 2)

    async def test_deep_summary_replaces_invalid_or_insufficient_answer_with_deterministic_text(self):
        complete_workset, complete_blocks = _deep_summary_evidence()
        incomplete_workset = ResearchEvidenceWorkset()
        cases = (
            (incomplete_workset, [], "看似完成的结论。[2]"),
        )

        for workset, blocks, answer in cases:
            with self.subTest(answer=answer):
                request, emitter, _stream_kwargs, prepare_context_fn = self._deep_request(
                    workset=workset,
                    content_blocks=list(blocks),
                    answer=answer,
                )
                with (
                    patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
                    patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
                ):
                    outcome = await run_limit_summary_step(request=request)

                self.assertTrue(outcome.incomplete)
                append_chunk.assert_awaited_once()
                self.assertEqual(append_chunk.await_args.args[2], DEEP_RESEARCH_INCOMPLETE_TEXT)
                self.assertEqual(request.content_blocks[-1].text, DEEP_RESEARCH_INCOMPLETE_TEXT)
                emitter.evidence_item_upserted.assert_not_awaited()

    async def test_deep_summary_repairs_missing_or_invalid_citation_once_before_emitting(self):
        cases = (
            ("缺少引用的完整结论。", "补齐引用后的完整结论。[2][3]"),
            ("使用了错误编号。[99]", "改用已读来源编号后的完整结论。[2][3]"),
        )

        for invalid_answer, repaired_answer in cases:
            with self.subTest(invalid_answer=invalid_answer):
                workset, blocks = _deep_summary_evidence()
                request, emitter, stream_kwargs, prepare_context_fn = self._deep_request(
                    workset=workset,
                    content_blocks=list(blocks),
                    answer=[invalid_answer, repaired_answer],
                )
                warnings = []
                request = LimitSummaryStepRequest(
                    **{
                        **request.__dict__,
                        "warning_fn": warnings.append,
                    }
                )
                with (
                    patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
                    patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
                ):
                    outcome = await run_limit_summary_step(request=request)

                self.assertFalse(outcome.incomplete)
                self.assertEqual(len(stream_kwargs), 2)
                self.assertEqual(request.llm_call_fn.await_count, 2)
                append_chunk.assert_awaited_once()
                self.assertEqual(append_chunk.await_args.args[2], repaired_answer)
                self.assertEqual(request.content_blocks[-1].text, repaired_answer)
                self.assertEqual(emitter.evidence_item_upserted.await_count, 2)
                self.assertTrue(any("引用校验未通过" in warning for warning in warnings))
                self.assertTrue(
                    any(
                        "深度研究完成校验" in str(message.get("content", ""))
                        for message in request.messages
                    )
                )

    async def test_deep_summary_keeps_deterministic_failure_when_citation_repair_still_invalid(self):
        workset, blocks = _deep_summary_evidence()
        request, emitter, stream_kwargs, prepare_context_fn = self._deep_request(
            workset=workset,
            content_blocks=list(blocks),
            answer=["伪造引用。[99]", "仍然伪造引用。[98]"],
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        self.assertEqual(len(stream_kwargs), 2)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], DEEP_RESEARCH_INCOMPLETE_TEXT)
        self.assertEqual(request.content_blocks[-1].text, DEEP_RESEARCH_INCOMPLETE_TEXT)
        emitter.evidence_item_upserted.assert_not_awaited()

    async def test_no_progress_summary_removes_conflicting_tool_usage_contract_before_call(self):
        sent_messages = []

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-summary",
                step_number=3,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def prepare_context_fn(**kwargs):
            return ContextPlan(
                messages=list(kwargs["messages"]),
                status="no_op",
                context_window_tokens=1000,
                context_window_source="test",
                context_window_status="known",
                estimated_tokens_before=100,
                estimated_tokens_after=100,
            )

        async def llm_call_fn(_model, _kwargs, messages, **_call_kwargs):
            sent_messages.extend(messages)
            return "response"

        request = LimitSummaryStepRequest(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=3,
            model_id="deepseek-chat",
            provider="deepseek",
            litellm_model="deepseek/deepseek-v4-flash",
            litellm_kwargs={},
            messages=[
                {
                    "role": "system",
                    "content": "【工具调用一致性规则】需要联网时必须调用 web_search。",
                },
                {"role": "user", "content": "北京周末天气如何？"},
            ],
            should_use_reasoning=True,
            content_blocks=[],
            call_kwargs={"tools": [{"function": {"name": "web_search"}}], "tool_choice": "auto"},
            accumulated_usage=Usage(input_tokens=0, output_tokens=0),
            emitter=AsyncMock(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=llm_call_fn,
            stream_round_fn=AsyncMock(return_value=("", "最终答复", [], "stop", None)),
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
            summary_finish_reason="no_progress_summary",
        )

        with patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn):
            await run_limit_summary_step(request=request)

        system_text = "\n".join(
            str(message.get("content", "")) for message in sent_messages if message.get("role") == "system"
        )
        self.assertNotIn("【工具调用一致性规则】", system_text)
        self.assertIn("现有搜索已不再产生新的有效信息", system_text)

    async def test_no_progress_summary_retries_once_when_model_returns_tool_protocol(self):
        content_blocks = []
        warnings = []

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-summary",
                step_number=3,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def prepare_context_fn(**kwargs):
            return ContextPlan(
                messages=list(kwargs["messages"]),
                status="no_op",
                context_window_tokens=1000,
                context_window_source="test",
                context_window_status="known",
                estimated_tokens_before=100,
                estimated_tokens_after=100,
            )

        stream_round_fn = AsyncMock(
            side_effect=[
                (
                    "内部工具规划",
                    "",
                    [{"id": "dsml-step-summary-1", "name": "web_search", "arguments": "{}"}],
                    "tool_calls",
                    Usage(input_tokens=5, output_tokens=7),
                ),
                ("", "最终答复", [], "stop", Usage(input_tokens=3, output_tokens=4)),
            ]
        )
        llm_call_fn = AsyncMock(side_effect=["response-1", "response-2"])
        request = LimitSummaryStepRequest(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=3,
            model_id="deepseek-chat",
            provider="deepseek",
            litellm_model="deepseek/deepseek-v4-flash",
            litellm_kwargs={},
            messages=[{"role": "user", "content": "北京周末天气如何？"}],
            should_use_reasoning=True,
            content_blocks=content_blocks,
            call_kwargs={"tools": [{"function": {"name": "web_search"}}], "tool_choice": "auto"},
            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
            emitter=AsyncMock(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=lambda **_kwargs: None,
            warning_fn=warnings.append,
            clock=lambda: 120.0,
            summary_finish_reason="no_progress_summary",
        )

        with patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn):
            outcome = await run_limit_summary_step(request=request)

        self.assertEqual(llm_call_fn.await_count, 2)
        self.assertEqual(outcome.accumulated_usage, Usage(input_tokens=10, output_tokens=14))
        self.assertEqual([block.type for block in content_blocks], ["text"])
        self.assertEqual(content_blocks[0].text, "最终答复")
        self.assertNotIn("内部工具规划", content_blocks[0].text)
        self.assertTrue(any("工具协议" in warning for warning in warnings))
        self.assertTrue(any("不要输出任何工具调用" in str(message.get("content", "")) for message in request.messages))

    async def test_no_progress_summary_uses_safe_fallback_after_repeated_tool_protocol(self):
        content_blocks = []

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-summary",
                step_number=3,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def prepare_context_fn(**kwargs):
            return ContextPlan(
                messages=list(kwargs["messages"]),
                status="no_op",
                context_window_tokens=1000,
                context_window_source="test",
                context_window_status="known",
                estimated_tokens_before=100,
                estimated_tokens_after=100,
            )

        tool_protocol_result = (
            "内部工具规划",
            "",
            [{"id": "dsml-step-summary-1", "name": "web_search", "arguments": "{}"}],
            "tool_calls",
            None,
        )
        request = LimitSummaryStepRequest(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=3,
            model_id="deepseek-chat",
            provider="deepseek",
            litellm_model="deepseek/deepseek-v4-flash",
            litellm_kwargs={},
            messages=[{"role": "user", "content": "北京周末天气如何？"}],
            should_use_reasoning=True,
            content_blocks=content_blocks,
            call_kwargs={"tools": [{"function": {"name": "web_search"}}], "tool_choice": "auto"},
            accumulated_usage=Usage(input_tokens=0, output_tokens=0),
            emitter=AsyncMock(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=AsyncMock(side_effect=["response-1", "response-2"]),
            stream_round_fn=AsyncMock(side_effect=[tool_protocol_result, tool_protocol_result]),
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
            summary_finish_reason="no_progress_summary",
        )

        with patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn):
            await run_limit_summary_step(request=request)

        self.assertEqual([block.type for block in content_blocks], ["text"])
        self.assertIn("未能生成可靠的最终答复", content_blocks[0].text)
        self.assertNotIn("内部工具规划", content_blocks[0].text)
        self.assertNotIn("DSML", content_blocks[0].text)

    async def test_limit_summary_replaces_previous_round_context_with_final_summary_round(self):
        emitter = AsyncMock()
        context_updates = []

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-summary",
                step_number=4,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        request = LimitSummaryStepRequest(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=4,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=[],
            should_use_reasoning=False,
            content_blocks=[],
            call_kwargs={},
            accumulated_usage=Usage(input_tokens=100, output_tokens=20),
            emitter=emitter,
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=AsyncMock(return_value="response"),
            stream_round_fn=AsyncMock(return_value=("", "总结", [], "stop", Usage(input_tokens=70, output_tokens=10))),
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
            on_context_updated=context_updates.append,
        )
        plan = ContextPlan(
            messages=[{"role": "system", "content": "总结"}],
            status="trimmed",
            context_window_tokens=1000,
            context_window_source="registry",
            context_window_status="known",
            estimated_tokens_before=900,
            estimated_tokens_after=700,
            removed_turns=1,
            removed_messages=2,
        )

        with patch("app.services.stream.limit_summary.prepare_context", new=AsyncMock(return_value=plan)):
            outcome = await run_limit_summary_step(request=request)

        self.assertEqual(outcome.accumulated_usage, Usage(input_tokens=170, output_tokens=30))
        self.assertEqual(
            outcome.context,
            ContextUsage(
                status="trimmed",
                round_index=4,
                window_tokens=1000,
                estimated_tokens_before=900,
                estimated_tokens_after=700,
                actual_prompt_tokens=70,
                removed_turns=1,
                removed_messages=2,
            ),
        )
        self.assertEqual(context_updates[-1], outcome.context)
        self.assertEqual(emitter.context_status_updated.await_args_list[-1].kwargs["phase"], "final")

    async def test_limit_summary_uses_real_budgeted_snapshot_and_keeps_tail_system_prompt(self):
        messages = [
            {"role": "user", "content": "a" * 100},
            {"role": "assistant", "content": "b" * 100},
            {"role": "user", "content": "最新问题"},
        ]
        sent_messages = []

        def estimator(_model, candidate, _kwargs):
            return sum(
                min(len(message.get("content") or ""), 10)
                if message.get("role") == "system"
                else len(message.get("content") or "")
                for message in candidate
            )

        async def budgeted_prepare(**kwargs):
            return await prepare_context_real(
                **kwargs,
                window_resolver=lambda _model: (100, "test", "known"),
                token_estimator=estimator,
                run_in_thread=False,
                use_fast_path=False,
            )

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-summary",
                step_number=2,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def llm_call_fn(_model, _kwargs, call_messages, **_call_kwargs):
            sent_messages.extend(call_messages)
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "", "总结", [], "stop", None

        request = LimitSummaryStepRequest(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=2,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=messages,
            should_use_reasoning=False,
            content_blocks=[],
            call_kwargs={},
            accumulated_usage=Usage(input_tokens=0, output_tokens=0),
            emitter=object(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
        )
        observation = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=budgeted_prepare),
            patch(
                "app.services.stream.limit_summary.create_llm_round_observation",
                return_value=observation,
            ),
        ):
            await run_limit_summary_step(request=request)

        self.assertEqual([message["role"] for message in sent_messages], ["user", "system"])
        self.assertEqual(sent_messages[0]["content"], "最新问题")
        self.assertEqual(sent_messages[-1]["content"], LIMIT_SUMMARY_PROMPT)
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1]["content"], LIMIT_SUMMARY_PROMPT)

    async def test_limit_summary_records_independent_round_observation(self):
        messages = []
        observation = MagicMock()
        log_round_summary_fn = MagicMock()
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response

        async def llm_call_fn(*_args, **_kwargs):
            return "response"

        async def stream_round_fn(*_args, **_kwargs):
            return "", "总结", [], "cancelled", Usage(input_tokens=5, output_tokens=7)

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-summary",
                step_number=2,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        request = LimitSummaryStepRequest(
            conversation_id="conv-1",
            task_id="task-1",
            run_id="run-1",
            step_number=2,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=messages,
            should_use_reasoning=False,
            content_blocks=[],
            call_kwargs={},
            accumulated_usage=Usage(input_tokens=2, output_tokens=3),
            emitter=object(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=llm_call_fn,
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=log_round_summary_fn,
            clock=lambda: 120.0,
            summary_finish_reason="no_progress_summary",
        )

        context_plan = MagicMock(
            messages=[{"role": "system", "content": LIMIT_SUMMARY_PROMPT}],
            estimated_tokens_after=12,
        )
        context_plan.telemetry.return_value = {"context_management_status": "trimmed"}
        with (
            patch(
                "app.services.stream.limit_summary.create_llm_round_observation",
                return_value=observation,
            ) as create_observation,
            patch(
                "app.services.stream.limit_summary.prepare_context",
                new=AsyncMock(return_value=context_plan),
            ),
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertEqual(outcome.accumulated_usage, Usage(input_tokens=7, output_tokens=10))
        self.assertEqual(create_observation.call_args.kwargs["round_kind"], "limit_summary")
        self.assertEqual(create_observation.call_args.kwargs["round_index"], 2)
        self.assertEqual(create_observation.call_args.kwargs["messages"], context_plan.messages)
        self.assertEqual(
            create_observation.call_args.kwargs["context_management"],
            {"context_management_status": "trimmed"},
        )
        observation.start.assert_called_once_with()
        observation.finish_success.assert_awaited_once_with(
            usage=Usage(input_tokens=5, output_tokens=7),
            finish_reason="cancelled",
        )
        self.assertEqual(log_round_summary_fn.call_args.kwargs["finish_reason"], "no_progress_summary")

    async def test_run_limit_summary_step_appends_prompt_and_records_success(self):
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        emitter = object()
        session_cache = object()
        events = []

        async def start_step_fn(*, emitter, session_cache, run_id, step_number, clock, on_step_started):
            events.append(("start", emitter, session_cache, run_id, step_number, clock))
            on_step_started("step-summary")
            return AgentStepContext(
                step_id="step-summary",
                step_number=step_number,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def llm_call_fn(litellm_model, litellm_kwargs, call_messages, **call_kwargs):
            events.append(("llm", litellm_model, litellm_kwargs, list(call_messages), call_kwargs))
            return "response"

        async def stream_round_fn(
            response,
            conversation_id,
            task_id,
            should_use_reasoning,
            thinking_block_id,
            text_block_id,
            *,
            run_id,
            step_id,
        ):
            events.append(
                (
                    "stream",
                    response,
                    conversation_id,
                    task_id,
                    should_use_reasoning,
                    thinking_block_id,
                    text_block_id,
                    run_id,
                    step_id,
                )
            )
            return "推理", "总结正文", [], "stop", Usage(input_tokens=5, output_tokens=7)

        def log_round_summary_fn(**kwargs):
            events.append(("log", kwargs))

        async def complete_step_fn(*, context, emitter, session_cache, tool_names, tool_call_count, clock):
            events.append(("complete", context, emitter, session_cache, tool_names, tool_call_count, clock))

        warnings = []
        marked_step_ids = []

        def clock():
            return 120.0

        outcome = await run_limit_summary_step(
            request=LimitSummaryStepRequest(
                conversation_id="conv-1",
                task_id="task-1",
                run_id="run-1",
                step_number=4,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={"metadata": {"trace": "x"}},
                messages=messages,
                should_use_reasoning=True,
                content_blocks=content_blocks,
                call_kwargs={
                    "tools": [{"function": {"name": "web_search"}}],
                    "tool_choice": "auto",
                    "temperature": 0.1,
                },
                accumulated_usage=accumulated_usage,
                emitter=emitter,
                session_cache=session_cache,
                total_timeout_s=300,
                run_start=100.0,
                start_step_fn=start_step_fn,
                complete_step_fn=complete_step_fn,
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=log_round_summary_fn,
                warning_fn=warnings.append,
                clock=clock,
                on_step_started=marked_step_ids.append,
            ),
        )

        self.assertEqual(marked_step_ids, ["step-summary"])
        self.assertEqual(messages[-1], {"role": "system", "content": LIMIT_SUMMARY_PROMPT})
        self.assertEqual(outcome.accumulated_usage, Usage(input_tokens=7, output_tokens=10))
        self.assertEqual(len(content_blocks), 2)
        self.assertEqual(content_blocks[0].type, "thinking")
        self.assertEqual(content_blocks[0].id, "blk-thinking")
        self.assertEqual(content_blocks[0].thinking, "推理")
        self.assertEqual(content_blocks[1].type, "text")
        self.assertEqual(content_blocks[1].id, "blk-text")
        self.assertEqual(content_blocks[1].text, "总结正文")
        self.assertEqual(warnings, [])

        self.assertEqual([event[0] for event in events], ["start", "llm", "stream", "log", "complete"])
        self.assertEqual(events[1][4], {"temperature": 0.1})
        self.assertEqual(
            events[2],
            (
                "stream",
                "response",
                "conv-1",
                "task-1",
                True,
                "blk-thinking",
                "blk-text",
                "run-1",
                "step-summary",
            ),
        )
        self.assertEqual(events[3][1]["finish_reason"], "limit_summary")
        self.assertEqual(events[3][1]["tool_calls_count"], 0)
        self.assertEqual(events[3][1]["reasoning_buf"], "推理")
        self.assertEqual(events[3][1]["content_buf"], "总结正文")
        self.assertEqual(events[4][1].step_id, "step-summary")
        self.assertEqual(events[4][4], [])
        self.assertEqual(events[4][5], 0)

    async def test_run_limit_summary_step_swallows_timeout_and_completes_step(self):
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        emitter = object()
        session_cache = object()
        events = []

        async def start_step_fn(*, emitter, session_cache, run_id, step_number, clock, on_step_started):
            on_step_started("step-timeout")
            events.append(("start", step_number))
            return AgentStepContext(
                step_id="step-timeout",
                step_number=step_number,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def llm_call_fn(*_args, **_kwargs):
            events.append(("llm",))
            raise asyncio.TimeoutError

        async def stream_round_fn(*_args, **_kwargs):
            events.append(("stream",))
            return "", "", [], "stop", None

        def log_round_summary_fn(**kwargs):
            events.append(("log", kwargs))

        async def complete_step_fn(*, context, emitter, session_cache, tool_names, tool_call_count, clock):
            events.append(("complete", context.step_id, tool_names, tool_call_count))

        warnings = []
        marked_step_ids = []

        outcome = await run_limit_summary_step(
            request=LimitSummaryStepRequest(
                conversation_id="conv-1",
                task_id="task-1",
                run_id="run-1",
                step_number=5,
                model_id="gpt-4",
                provider="openai",
                litellm_model="openai/gpt-4",
                litellm_kwargs={},
                messages=messages,
                should_use_reasoning=False,
                content_blocks=content_blocks,
                call_kwargs={"tools": [], "tool_choice": "auto"},
                accumulated_usage=accumulated_usage,
                emitter=emitter,
                session_cache=session_cache,
                total_timeout_s=2,
                run_start=0.0,
                start_step_fn=start_step_fn,
                complete_step_fn=complete_step_fn,
                llm_call_fn=llm_call_fn,
                stream_round_fn=stream_round_fn,
                log_round_summary_fn=log_round_summary_fn,
                warning_fn=warnings.append,
                clock=lambda: 10.0,
                on_step_started=marked_step_ids.append,
            ),
        )

        self.assertEqual(marked_step_ids, ["step-timeout"])
        self.assertEqual(messages[-1], {"role": "system", "content": LIMIT_SUMMARY_PROMPT})
        self.assertEqual(outcome.accumulated_usage, accumulated_usage)
        self.assertEqual(content_blocks, [])
        self.assertEqual([event[0] for event in events], ["start", "llm", "complete"])
        self.assertEqual(events[-1], ("complete", "step-timeout", [], 0))
        self.assertEqual(len(warnings), 1)
        self.assertIn("触顶总结超出剩余预算", warnings[0])
        self.assertIn("conv_id=conv-1", warnings[0])

    async def test_run_limit_summary_step_reraises_non_timeout_error(self):
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        events = []

        async def start_step_fn(*, emitter, session_cache, run_id, step_number, clock, on_step_started):
            on_step_started("step-error")
            events.append(("start", step_number))
            return AgentStepContext(
                step_id="step-error",
                step_number=step_number,
                started_at=100.0,
                thinking_block_id="blk-thinking",
                text_block_id="blk-text",
            )

        async def llm_call_fn(*_args, **_kwargs):
            events.append(("llm",))
            raise RuntimeError("upstream LLM 5xx")

        async def stream_round_fn(*_args, **_kwargs):
            events.append(("stream",))
            return "", "", [], "stop", None

        def log_round_summary_fn(**kwargs):
            events.append(("log", kwargs))

        async def complete_step_fn(**_kwargs):
            events.append(("complete",))

        with self.assertRaises(RuntimeError) as cm:
            await run_limit_summary_step(
                request=LimitSummaryStepRequest(
                    conversation_id="conv-1",
                    task_id="task-1",
                    run_id="run-1",
                    step_number=6,
                    model_id="gpt-4",
                    provider="openai",
                    litellm_model="openai/gpt-4",
                    litellm_kwargs={},
                    messages=messages,
                    should_use_reasoning=False,
                    content_blocks=content_blocks,
                    call_kwargs={"tools": [], "tool_choice": "auto"},
                    accumulated_usage=Usage(input_tokens=2, output_tokens=3),
                    emitter=object(),
                    session_cache=object(),
                    total_timeout_s=300,
                    run_start=0.0,
                    start_step_fn=start_step_fn,
                    complete_step_fn=complete_step_fn,
                    llm_call_fn=llm_call_fn,
                    stream_round_fn=stream_round_fn,
                    log_round_summary_fn=log_round_summary_fn,
                    warning_fn=None,
                    clock=lambda: 10.0,
                    on_step_started=lambda _step_id: None,
                ),
            )

        self.assertIn("upstream LLM 5xx", str(cm.exception))
        self.assertEqual([event[0] for event in events], ["start", "llm"])
        self.assertEqual(content_blocks, [])
