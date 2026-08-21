import asyncio
import unittest
from dataclasses import replace
from functools import partial
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.ai.prompts.agent_loop import VISIBLE_RESPONSE_LANGUAGE_PROMPT
from app.schemas.chat import ContextUsage, SearchBlock, SearchSourceSummary, SourceReference, UrlBlock, Usage
from app.services.chat.context_manager import ContextPlan
from app.services.chat.context_manager import prepare_context as prepare_context_real
from app.services.stream import limit_summary as limit_summary_module
from app.services.stream import llm_stream as llm_stream_module
from app.services.stream.limit_summary import (
    DEEP_RESEARCH_INCOMPLETE_TEXT,
    LIMIT_SUMMARY_PROMPT,
    SUMMARY_PROTOCOL_FALLBACK_TEXT,
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
from app.services.stream_state_service import StreamOwnershipLostError


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

    def test_plan_synthesis_observation_has_distinct_round_kind_and_no_tool_controls(self):
        request = SimpleNamespace(
            conversation_id="conv-plan-synthesis",
            run_id="run-plan-synthesis",
            step_number=3,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            assistant_message_id="msg-plan-synthesis",
            summary_finish_reason="plan_synthesis",
        )
        context_plan = MagicMock(
            messages=[{"role": "system", "content": "只输出最终回答"}],
            estimated_tokens_after=12,
        )
        context_plan.telemetry.return_value = {"context_management_status": "no_op"}
        call_kwargs = build_limit_summary_call_kwargs(
            {
                "temperature": 0.2,
                "tools": [{"type": "function", "function": {"name": "web_search"}}],
                "tool_choice": "auto",
            }
        )

        with patch(
            "app.services.stream.limit_summary.create_llm_round_observation",
            return_value=MagicMock(),
        ) as create_observation:
            limit_summary_module._create_limit_summary_observation(
                request=request,
                context_plan=context_plan,
                step_id="step-plan-synthesis",
                call_kwargs=call_kwargs,
            )

        self.assertEqual(create_observation.call_args.kwargs["round_kind"], "plan_synthesis")
        self.assertNotIn("tools", create_observation.call_args.kwargs["call_kwargs"])
        self.assertNotIn("tool_choice", create_observation.call_args.kwargs["call_kwargs"])

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

    def test_accumulate_summary_usage_preserves_optional_cache_tokens(self):
        result = accumulate_summary_usage(
            Usage(input_tokens=2, output_tokens=3, cache_read_tokens=4),
            Usage(input_tokens=5, output_tokens=7, cache_read_tokens=6, cache_write_tokens=8),
        )

        self.assertEqual(result.cache_read_tokens, 10)
        self.assertEqual(result.cache_write_tokens, 8)

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

    def test_standard_final_summary_keeps_context7_and_non_search_tool_transactions(self):
        messages = [
            {"role": "user", "content": "查询 FastAPI 文档并比较路线"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-context7",
                        "type": "function",
                        "function": {
                            "name": "mcp_context7_query",
                            "arguments": '{"libraryId":"/fastapi/fastapi","query":"routing"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-context7",
                "content": "Context7 文档结果",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-route",
                        "type": "function",
                        "function": {
                            "name": "route_compare",
                            "arguments": '{"origin":"A","destination":"B"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-route",
                "content": "路线比较结果",
            },
        ]
        expected = [dict(message) for message in messages]

        remove_conflicting_tool_usage_contract(
            messages,
            task_mode="standard",
            final_synthesis=True,
        )

        self.assertEqual(messages, expected)

    def test_standard_final_summary_removes_verified_research_plan_contract(self):
        messages = [
            {"role": "system", "content": "【可核验证据计划规则】先搜索再读取"},
            {"role": "user", "content": "请综合结论"},
        ]

        remove_conflicting_tool_usage_contract(
            messages,
            task_mode="standard",
            final_synthesis=True,
        )

        self.assertEqual(messages, [{"role": "user", "content": "请综合结论"}])

    def test_standard_final_summary_keeps_entire_parallel_mixed_tool_transaction(self):
        messages = [
            {"role": "user", "content": "同时搜索公告并查询官方库文档"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query":"官方公告"}'},
                    },
                    {
                        "id": "call-context7",
                        "type": "function",
                        "function": {
                            "name": "mcp_context7_query",
                            "arguments": '{"libraryId":"/fastapi/fastapi","query":"release notes"}',
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-search",
                "content": "网页搜索结果",
            },
            {
                "role": "tool",
                "tool_call_id": "call-context7",
                "content": "Context7 文档结果",
            },
        ]
        expected = [dict(message) for message in messages]

        remove_conflicting_tool_usage_contract(
            messages,
            task_mode="standard",
            final_synthesis=True,
        )

        self.assertEqual(messages, expected)

    def test_append_summary_content_blocks_persists_only_final_text(self):
        content_blocks = []

        append_summary_content_blocks(
            content_blocks=content_blocks,
            content_buf="总结正文",
            text_block_id="blk-text",
        )

        self.assertEqual([block.type for block in content_blocks], ["text"])
        self.assertEqual(content_blocks[0].id, "blk-text")
        self.assertEqual(content_blocks[0].text, "总结正文")

    def test_append_summary_content_blocks_never_promotes_reasoning_to_answer(self):
        content_blocks = []

        append_summary_content_blocks(
            content_blocks=content_blocks,
            content_buf="",
            text_block_id="blk-text",
        )

        self.assertEqual(content_blocks, [])


class LimitSummaryStepTests(unittest.IsolatedAsyncioTestCase):
    def _plan_synthesis_request(
        self,
        *,
        round_results,
        llm_call_fn=None,
        events=None,
    ):
        queued_results = iter(round_results)
        stream_kwargs = []
        events = events if events is not None else []

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-plan-synthesis",
                step_number=3,
                started_at=100.0,
                thinking_block_id="blk-plan-thinking",
                text_block_id="blk-plan-text",
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
            result = next(queued_results)
            events.append(("round_complete", result[3], bool(result[2])))
            return result

        request = LimitSummaryStepRequest(
            conversation_id="conv-plan-synthesis",
            task_id="task-plan-synthesis",
            run_id="run-plan-synthesis",
            step_number=3,
            model_id="gpt-4",
            provider="openai",
            litellm_model="openai/gpt-4",
            litellm_kwargs={},
            messages=[{"role": "user", "content": "请给出最终回答"}],
            should_use_reasoning=False,
            content_blocks=[],
            call_kwargs={
                "tools": [{"function": {"name": "web_search"}}],
                "tool_choice": "auto",
            },
            accumulated_usage=Usage(input_tokens=1, output_tokens=1),
            emitter=AsyncMock(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=llm_call_fn or AsyncMock(return_value="response"),
            stream_round_fn=stream_round_fn,
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
            summary_finish_reason="plan_synthesis",
            defer_output=True,
        )
        return request, stream_kwargs, events, prepare_context_fn

    async def test_standard_plan_synthesis_streams_round_and_never_bulk_appends_answer(self):
        request, stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[("综合推理", "第一段第二段", [], "stop", Usage(input_tokens=2, output_tokens=3))]
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertFalse(stream_kwargs[0]["defer_output"])
        append_chunk.assert_not_awaited()
        self.assertFalse(outcome.incomplete)
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[-2].thinking, "综合推理")
        self.assertEqual(request.content_blocks[-1].text, "第一段第二段")

    async def test_standard_plan_synthesis_timeout_persists_exact_streamed_partial_without_fallback(self):
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

        async def response():
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=None, reasoning_content="部分综合推理"),
                        finish_reason=None,
                    )
                ]
            )
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content="部分正文  ", reasoning_content=None),
                        finish_reason=None,
                    )
                ]
            )
            await asyncio.Event().wait()

        async def start_step_fn(**_kwargs):
            return AgentStepContext(
                step_id="step-timeout",
                step_number=3,
                started_at=100.0,
                thinking_block_id="thinking-timeout",
                text_block_id="text-timeout",
            )

        request = LimitSummaryStepRequest(
            conversation_id="conv-timeout",
            task_id="task-timeout",
            run_id="run-timeout",
            step_number=3,
            model_id="kimi-k3",
            provider="moonshot",
            litellm_model="moonshot/kimi-k3",
            litellm_kwargs={},
            messages=[{"role": "user", "content": "总结"}],
            should_use_reasoning=True,
            content_blocks=[],
            call_kwargs={},
            accumulated_usage=Usage(input_tokens=0, output_tokens=0),
            emitter=AsyncMock(),
            session_cache=object(),
            total_timeout_s=300,
            run_start=100.0,
            start_step_fn=start_step_fn,
            complete_step_fn=AsyncMock(),
            llm_call_fn=AsyncMock(return_value=response()),
            stream_round_fn=partial(
                llm_stream_module.stream_round,
                model_id="kimi-k3",
                reasoning_transport_mode="delta",
            ),
            log_round_summary_fn=lambda **_kwargs: None,
            clock=lambda: 120.0,
            summary_finish_reason="plan_synthesis",
        )
        observation = SimpleNamespace(
            start=lambda: None,
            wrap_response=lambda value: value,
            finish_error=AsyncMock(),
            finish_success=AsyncMock(),
        )
        stream_append = AsyncMock()

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.compute_summary_timeout", return_value=0.02),
            patch("app.services.stream.limit_summary._create_limit_summary_observation", return_value=observation),
            patch("app.services.stream.llm_stream.append_chunk", new=stream_append),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as summary_append,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertEqual(
            [(call.args[1], call.args[2]) for call in stream_append.await_args_list],
            [("reasoning", "部分综合推理"), ("answering", "部分正文")],
        )
        summary_append.assert_not_awaited()
        self.assertTrue(outcome.incomplete)
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[-2].thinking, "部分综合推理")
        self.assertEqual(request.content_blocks[-1].text, "部分正文")

    async def test_standard_plan_synthesis_stream_error_persists_sent_partial(self):
        warnings = []
        request, _stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(round_results=[])

        async def stream_round_fn(*_args, partial_output=None, **_kwargs):
            partial_output["reasoning_buf"] = "异常前已发送推理"
            partial_output["content_buf"] = "异常前已发送正文"
            raise RuntimeError("provider stream disconnected")

        request = replace(
            request,
            stream_round_fn=stream_round_fn,
            warning_fn=warnings.append,
        )
        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        append_chunk.assert_not_awaited()
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[0].thinking, "异常前已发送推理")
        self.assertEqual(request.content_blocks[1].text, "异常前已发送正文")
        self.assertTrue(any("RuntimeError" in warning for warning in warnings))

    async def test_standard_plan_synthesis_ownership_loss_is_never_converted_to_partial(self):
        request, _stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(round_results=[])

        async def stream_round_fn(*_args, partial_output=None, **_kwargs):
            partial_output["content_buf"] = "旧任务片段"
            raise StreamOwnershipLostError("任务已被替代")

        request = replace(request, stream_round_fn=stream_round_fn)
        with patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn):
            with self.assertRaises(StreamOwnershipLostError):
                await run_limit_summary_step(request=request)

    def _deep_request(
        self,
        *,
        workset,
        content_blocks,
        answer,
        model_id="gpt-4",
        reasoning_buf="",
    ):
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
            return reasoning_buf, next(answers), [], "stop", Usage(input_tokens=2, output_tokens=3)

        emitter = AsyncMock()
        request = LimitSummaryStepRequest(
            conversation_id="conv-deep",
            task_id="task-deep",
            run_id="run-deep",
            step_number=9,
            model_id=model_id,
            provider="moonshot" if model_id == "kimi-k3" else "openai",
            litellm_model=f"moonshot/{model_id}" if model_id == "kimi-k3" else f"openai/{model_id}",
            litellm_kwargs={},
            messages=[{"role": "user", "content": "深度研究"}],
            should_use_reasoning=bool(reasoning_buf),
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

    async def test_deferred_real_tool_delta_emits_first_before_discard_completion(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        workset, blocks = _deep_summary_evidence()
        request, emitter, _stream_kwargs, prepare_context_fn = self._deep_request(
            workset=workset,
            content_blocks=list(blocks),
            answer="unused",
        )
        events = []
        emitter.llm_round_started.side_effect = lambda **_kwargs: events.append("started")
        emitter.llm_round_first_output_delta.side_effect = lambda **_kwargs: events.append("first")
        emitter.llm_round_completed.side_effect = lambda **_kwargs: events.append("completed")
        now = [10.0]
        observation = LLMRoundObservation(
            metadata=RoundMetadata("conv", "run", 9, "step", "limit_summary", "model", "provider"),
            litellm_model="test/model",
            messages=[],
            call_kwargs={},
            clock=lambda: now[0],
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "test", "known"),
            run_context_in_thread=False,
        )

        async def response():
            now[0] = 10.2
            yield SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(
                            content=None,
                            reasoning_content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    index=0,
                                    id="tool-1",
                                    function=SimpleNamespace(name="web_search", arguments="{}"),
                                )
                            ],
                        ),
                        finish_reason="tool_calls",
                    )
                ],
                usage=None,
            )

        request = replace(
            request,
            llm_call_fn=AsyncMock(return_value=response()),
            stream_round_fn=partial(llm_stream_module.stream_round, model_id="model"),
        )
        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary._create_limit_summary_observation", return_value=observation),
            patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
        ):
            result = await limit_summary_module.call_limit_summary_round(
                request=request,
                thinking_block_id="thinking",
                text_block_id="text",
                step_id="step",
            )
            await limit_summary_module._finish_summary_round_lifecycle(result, model_output_visible=False)

        emitter.llm_round_first_output_delta.assert_awaited_once()
        self.assertEqual(emitter.llm_round_first_output_delta.await_args.kwargs["delta_kind"], "tool_call")
        self.assertEqual(emitter.llm_round_first_output_delta.await_args.kwargs["ttft_ms"], 200)
        self.assertEqual(events, ["started", "first", "completed"])

    async def test_post_stream_summary_context_error_and_cancel_close_round(self):
        workset, blocks = _deep_summary_evidence()
        for primary in (RuntimeError("final summary context failed"), asyncio.CancelledError()):
            with self.subTest(primary=type(primary).__name__):
                request, emitter, _stream_kwargs, prepare_context_fn = self._deep_request(
                    workset=workset,
                    content_blocks=list(blocks),
                    answer="candidate",
                )

                def build_context(_plan, usage=None, **_kwargs):
                    if usage is None:
                        return ContextUsage(status="no_op")
                    raise primary

                with (
                    patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
                    patch("app.services.stream.limit_summary.build_context_usage", side_effect=build_context),
                ):
                    with self.assertRaises(type(primary)) as raised:
                        await limit_summary_module.call_limit_summary_round(
                            request=request,
                            thinking_block_id="thinking",
                            text_block_id="text",
                            step_id="step",
                        )

                self.assertIs(raised.exception, primary)
                if isinstance(primary, asyncio.CancelledError):
                    emitter.llm_round_cancelled.assert_awaited_once()
                else:
                    emitter.llm_round_failed.assert_awaited_once()

    async def test_streamed_plan_synthesis_with_late_tool_protocol_does_not_retry_or_bulk_append(self):
        events = []
        request, stream_kwargs, events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[
                (
                    "已流式发送的安全推理。",
                    "已流式发送的安全正文。",
                    [{"id": "tc-late", "name": "web_search", "arguments": "{}"}],
                    "tool_calls",
                    Usage(input_tokens=2, output_tokens=3),
                ),
            ],
            llm_call_fn=AsyncMock(return_value="response-1"),
            events=events,
        )

        async def append_answer(_conversation_id, chunk_type, answer, *_args, **_kwargs):
            events.append((chunk_type, answer))

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch(
                "app.services.stream.limit_summary.append_chunk", new=AsyncMock(side_effect=append_answer)
            ) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertEqual(len(stream_kwargs), 1)
        self.assertFalse(stream_kwargs[0]["defer_output"])
        self.assertTrue(outcome.incomplete)
        append_chunk.assert_not_awaited()
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[-2].thinking, "已流式发送的安全推理。")
        self.assertEqual(request.content_blocks[-1].text, "已流式发送的安全正文。")
        self.assertEqual([event[0] for event in events], ["round_complete"])

    async def test_empty_plan_synthesis_emits_safe_fallback_and_marks_incomplete(self):
        request, stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[("", "", [], "stop", None)]
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertFalse(stream_kwargs[0]["defer_output"])
        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual(request.content_blocks[-1].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

    async def test_reasoning_only_plan_synthesis_persists_reasoning_and_appends_safe_fallback(self):
        request, stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[("仅供内部使用的推理", "", [], "stop", None)]
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertFalse(stream_kwargs[0]["defer_output"])
        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[0].thinking, "仅供内部使用的推理")
        self.assertEqual(request.content_blocks[1].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

    async def test_reasoning_only_timed_out_plan_synthesis_keeps_thinking_and_appends_fallback(self):
        request, _stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(round_results=[])

        async def stream_round_fn(*_args, partial_output=None, **_kwargs):
            partial_output["reasoning_buf"] = "超时前已发送的推理"
            await asyncio.Event().wait()

        request = replace(request, stream_round_fn=stream_round_fn)
        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.compute_summary_timeout", return_value=0.01),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[0].thinking, "超时前已发送的推理")
        self.assertEqual(request.content_blocks[1].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

    async def test_reasoning_only_repeated_protocol_keeps_thinking_and_appends_fallback(self):
        protocol_call = [{"id": "tc-protocol", "name": "web_search", "arguments": "{}"}]
        request, stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[
                ("首轮已发送推理", "", protocol_call, "tool_calls", None),
                ("重试已发送推理", "", protocol_call, "tool_calls", None),
            ],
            llm_call_fn=AsyncMock(side_effect=["response-1", "response-2"]),
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(all(not kwargs["defer_output"] for kwargs in stream_kwargs))
        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual([block.type for block in request.content_blocks], ["thinking", "text"])
        self.assertEqual(request.content_blocks[0].thinking, "首轮已发送推理重试已发送推理")
        self.assertEqual(request.content_blocks[1].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

    async def test_empty_repeated_protocol_appends_and_persists_one_fallback(self):
        protocol_call = [{"id": "tc-protocol", "name": "web_search", "arguments": "{}"}]
        request, _stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[
                ("", "", protocol_call, "tool_calls", None),
                ("", "", protocol_call, "tool_calls", None),
            ],
            llm_call_fn=AsyncMock(side_effect=["response-1", "response-2"]),
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual([block.type for block in request.content_blocks], ["text"])
        self.assertEqual(request.content_blocks[0].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

    async def test_whitespace_only_plan_synthesis_appends_and_persists_fallback(self):
        request, _stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[("", "   ", [], "stop", None)]
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual([block.type for block in request.content_blocks], ["text"])
        self.assertEqual(request.content_blocks[0].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

    async def test_timed_out_plan_synthesis_emits_safe_fallback_and_marks_incomplete(self):
        request, _stream_kwargs, _events, prepare_context_fn = self._plan_synthesis_request(
            round_results=[],
            llm_call_fn=AsyncMock(side_effect=asyncio.TimeoutError),
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertEqual(request.content_blocks[-1].text, SUMMARY_PROTOCOL_FALLBACK_TEXT)

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

    async def test_deep_summary_streams_and_persists_reasoning_for_k3_and_non_k3(self):
        for model_id in ("kimi-k3", "gpt-4"):
            with self.subTest(model_id=model_id):
                workset, blocks = _deep_summary_evidence()
                request, _emitter, stream_kwargs, prepare_context_fn = self._deep_request(
                    workset=workset,
                    content_blocks=blocks,
                    answer="综合结论来自两个已读来源。[2][3]",
                    model_id=model_id,
                    reasoning_buf="只供总结调用内部使用的推理",
                )

                with (
                    patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
                    patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()),
                ):
                    outcome = await run_limit_summary_step(request=request)

                self.assertFalse(outcome.incomplete)
                self.assertTrue(stream_kwargs[0]["allow_deferred_reasoning_output"])
                self.assertEqual(request.content_blocks[-1].type, "text")
                thinking_blocks = [block for block in request.content_blocks if block.type == "thinking"]
                self.assertEqual(len(thinking_blocks), 1)
                self.assertEqual(thinking_blocks[0].thinking, "只供总结调用内部使用的推理")

    async def test_deep_summary_replaces_invalid_or_insufficient_answer_with_deterministic_text(self):
        complete_workset, complete_blocks = _deep_summary_evidence()
        incomplete_workset = ResearchEvidenceWorkset()
        cases = ((incomplete_workset, [], "看似完成的结论。[2]"),)

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
                sent_snapshots = [call.args[2] for call in request.llm_call_fn.await_args_list]
                self.assertTrue(
                    all(
                        sum(str(item.get("content") or "").count(VISIBLE_RESPONSE_LANGUAGE_PROMPT) for item in snapshot)
                        == 1
                        for snapshot in sent_snapshots
                    )
                )
                self.assertIn("深度研究完成校验", sent_snapshots[1][-1]["content"])
                self.assertTrue(sent_snapshots[1][-1]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
                self.assertFalse(
                    any(VISIBLE_RESPONSE_LANGUAGE_PROMPT in str(item.get("content") or "") for item in request.messages)
                )
                append_chunk.assert_awaited_once()
                self.assertEqual(append_chunk.await_args.args[2], repaired_answer)
                self.assertEqual(request.content_blocks[-1].text, repaired_answer)
                self.assertEqual(emitter.evidence_item_upserted.await_count, 2)
                self.assertTrue(any("引用校验未通过" in warning for warning in warnings))
                self.assertTrue(
                    any("深度研究完成校验" in str(message.get("content", "")) for message in request.messages)
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

    async def test_deep_summary_closes_malformed_citation_repair_round(self):
        workset, blocks = _deep_summary_evidence()
        request, emitter, _stream_kwargs, prepare_context_fn = self._deep_request(
            workset=workset,
            content_blocks=list(blocks),
            answer="未引用的候选",
        )
        request = replace(
            request,
            stream_round_fn=AsyncMock(
                side_effect=[
                    ("", "未引用的候选", [], "stop", Usage(input_tokens=1, output_tokens=1)),
                    ("隐藏协议", "", [], "tool_protocol_error", Usage(input_tokens=1, output_tokens=1)),
                ]
            ),
        )

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()),
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertTrue(outcome.incomplete)
        self.assertEqual(emitter.llm_round_completed.await_count, 2)
        self.assertTrue(all(call.kwargs["ttft_ms"] is None for call in emitter.llm_round_completed.await_args_list))

    async def test_malformed_summary_retry_exception_closes_both_rounds(self):
        workset, blocks = _deep_summary_evidence()
        request, emitter, _stream_kwargs, prepare_context_fn = self._deep_request(
            workset=workset,
            content_blocks=list(blocks),
            answer="unused",
        )
        primary = RuntimeError("retry provider failed")
        request = replace(
            request,
            task_mode="standard",
            stream_round_fn=AsyncMock(
                side_effect=[
                    ("隐藏协议", "", [], "tool_protocol_error", Usage(input_tokens=1, output_tokens=1)),
                    primary,
                ]
            ),
        )

        with patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn):
            with self.assertRaises(RuntimeError) as raised:
                await run_limit_summary_step(request=request)

        self.assertIs(raised.exception, primary)
        self.assertEqual(emitter.llm_round_completed.await_count, 1)
        self.assertIsNone(emitter.llm_round_completed.await_args.kwargs["ttft_ms"])
        emitter.llm_round_failed.assert_awaited_once()

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

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()),
        ):
            await run_limit_summary_step(request=request)

        system_text = "\n".join(
            str(message.get("content", "")) for message in sent_messages if message.get("role") == "system"
        )
        self.assertNotIn("【工具调用一致性规则】", system_text)
        self.assertIn("现有搜索已不再产生新的有效信息", system_text)

    async def test_all_final_summaries_strip_tool_transactions_and_control_prompts(self):
        source_messages = [
            {"role": "system", "content": "【Fusion 身份一致性规则】保留身份规则"},
            {"role": "system", "content": "【自主联网判断规则】按需调用工具"},
            {"role": "system", "content": "【工具调用一致性规则】必须保持工具协议"},
            {"role": "system", "content": "【执行计划控制规则】必须更新计划"},
            {"role": "system", "content": "【计划控制修正】必须先建立计划"},
            {"role": "system", "content": "【计划执行修正】必须继续执行工具"},
            {"role": "user", "content": "请总结 Redis 更新"},
            {"role": "assistant", "content": "普通历史回答应保留"},
            {
                "role": "assistant",
                "content": "准备调用工具",
                "tool_calls": [
                    {
                        "id": "call-search",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-search", "content": "原始工具结果"},
        ]

        for summary_finish_reason in (
            "limit_summary",
            "no_progress_summary",
            "plan_repair_exhausted",
            "plan_synthesis",
        ):
            with self.subTest(summary_finish_reason=summary_finish_reason):
                captured_messages: list[dict] = []
                request, _stream_kwargs, _events, _prepare_context_fn = self._plan_synthesis_request(
                    round_results=[("", "最终总结", [], "stop", None)]
                )
                request = LimitSummaryStepRequest(
                    **{
                        **request.__dict__,
                        "messages": [dict(message) for message in source_messages],
                        "summary_finish_reason": summary_finish_reason,
                    }
                )

                async def prepare_context_fn(**kwargs):
                    captured_messages.extend(kwargs["messages"])
                    return ContextPlan(
                        messages=list(kwargs["messages"]),
                        status="no_op",
                        context_window_tokens=1000,
                        context_window_source="test",
                        context_window_status="known",
                        estimated_tokens_before=100,
                        estimated_tokens_after=100,
                    )

                with (
                    patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
                    patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()),
                ):
                    await run_limit_summary_step(request=request)

                self.assertIn(
                    {"role": "system", "content": "【Fusion 身份一致性规则】保留身份规则"},
                    captured_messages,
                )
                self.assertIn(
                    {"role": "user", "content": "请总结 Redis 更新"},
                    captured_messages,
                )
                self.assertIn(
                    {"role": "assistant", "content": "普通历史回答应保留"},
                    captured_messages,
                )
                self.assertFalse(
                    any(
                        message.get("role") == "assistant" and message.get("tool_calls")
                        for message in captured_messages
                    )
                )
                self.assertFalse(any(message.get("role") == "tool" for message in captured_messages))
                control_text = "\n".join(
                    str(message.get("content", "")) for message in captured_messages if message.get("role") == "system"
                )
                for marker in (
                    "【自主联网判断规则】",
                    "【工具调用一致性规则】",
                    "【执行计划控制规则】",
                    "【计划控制修正】",
                    "【计划执行修正】",
                ):
                    self.assertNotIn(marker, control_text)

    async def test_no_progress_summary_retries_once_when_model_returns_tool_protocol(self):
        content_blocks = []
        warnings = []
        emitter = AsyncMock()

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
                    "这段工具协议前缀不得展示",
                    [],
                    "tool_protocol_error",
                    Usage(input_tokens=5, output_tokens=7),
                ),
                ("最终总结推理", "最终答复", [], "stop", Usage(input_tokens=3, output_tokens=4)),
            ]
        )
        sent_snapshots = []

        async def llm_call_fn(_model, _kwargs, call_messages, **_call_kwargs):
            sent_snapshots.append([dict(message) for message in call_messages])
            return f"response-{len(sent_snapshots)}"

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
            emitter=emitter,
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

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        self.assertEqual(len(sent_snapshots), 2)
        self.assertTrue(
            all(
                sum(str(item.get("content") or "").count(VISIBLE_RESPONSE_LANGUAGE_PROMPT) for item in snapshot) == 1
                for snapshot in sent_snapshots
            )
        )
        self.assertIn("不要输出任何工具调用", sent_snapshots[1][-1]["content"])
        self.assertTrue(sent_snapshots[1][-1]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
        self.assertFalse(
            any(VISIBLE_RESPONSE_LANGUAGE_PROMPT in str(item.get("content") or "") for item in request.messages)
        )
        self.assertTrue(all(call.kwargs["defer_output"] for call in stream_round_fn.await_args_list))
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], "最终答复")
        self.assertEqual(outcome.accumulated_usage, Usage(input_tokens=10, output_tokens=14))
        self.assertEqual([block.type for block in content_blocks], ["thinking", "text"])
        self.assertEqual(content_blocks[0].thinking, "内部工具规划最终总结推理")
        self.assertEqual(content_blocks[1].text, "最终答复")
        self.assertNotIn("工具协议前缀", content_blocks[1].text)
        self.assertTrue(any("工具协议" in warning for warning in warnings))
        self.assertTrue(any("不要输出任何工具调用" in str(message.get("content", "")) for message in request.messages))
        self.assertEqual(emitter.llm_round_completed.await_count, 2)
        self.assertIsNone(emitter.llm_round_completed.await_args_list[0].kwargs["ttft_ms"])

    async def test_k3_plan_synthesis_streams_normally_while_limit_summary_stays_deferred(self):
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

        async def response(chunks):
            for chunk in chunks:
                yield chunk

        def chunk(*, content=None, reasoning=None, tool_calls=None, finish_reason=None):
            delta = SimpleNamespace(content=content, reasoning_content=reasoning)
            if tool_calls is not None:
                delta.tool_calls = tool_calls
            return SimpleNamespace(
                choices=[SimpleNamespace(delta=delta, finish_reason=finish_reason)],
            )

        protocol_call = SimpleNamespace(
            index=0,
            id="tc-summary-protocol",
            function=SimpleNamespace(name="web_search", arguments='{"query":"候选"}'),
        )

        for summary_finish_reason in ("plan_synthesis", "no_progress_summary"):
            with self.subTest(summary_finish_reason=summary_finish_reason):
                responses = iter(
                    [
                        response(
                            [
                                chunk(reasoning="候选协议推理"),
                                chunk(tool_calls=[protocol_call], finish_reason="tool_calls"),
                            ]
                        ),
                        response(
                            [
                                chunk(reasoning="最终综合推理"),
                                chunk(content="最终答复", finish_reason="stop"),
                            ]
                        ),
                    ]
                )

                async def llm_call_fn(*_args, **_kwargs):
                    return next(responses)

                async def start_step_fn(**_kwargs):
                    return AgentStepContext(
                        step_id=f"step-{summary_finish_reason}",
                        step_number=3,
                        started_at=100.0,
                        thinking_block_id=f"thinking-{summary_finish_reason}",
                        text_block_id=f"text-{summary_finish_reason}",
                    )

                observation = SimpleNamespace(
                    start=lambda: None,
                    wrap_response=lambda value: value,
                    finish_error=AsyncMock(),
                    finish_success=AsyncMock(),
                )
                stream_append = AsyncMock()
                summary_append = AsyncMock()
                content_blocks = []
                request = LimitSummaryStepRequest(
                    conversation_id="conv-k3-summary",
                    task_id="task-k3-summary",
                    run_id="run-k3-summary",
                    step_number=3,
                    model_id="kimi-k3",
                    provider="moonshot",
                    litellm_model="moonshot/kimi-k3",
                    litellm_kwargs={},
                    messages=[{"role": "user", "content": "总结"}],
                    should_use_reasoning=True,
                    content_blocks=content_blocks,
                    call_kwargs={"tools": [{"function": {"name": "web_search"}}]},
                    accumulated_usage=Usage(input_tokens=0, output_tokens=0),
                    emitter=AsyncMock(),
                    session_cache=object(),
                    total_timeout_s=300,
                    run_start=100.0,
                    start_step_fn=start_step_fn,
                    complete_step_fn=AsyncMock(),
                    llm_call_fn=llm_call_fn,
                    stream_round_fn=partial(llm_stream_module.stream_round, model_id="kimi-k3"),
                    log_round_summary_fn=lambda **_kwargs: None,
                    clock=lambda: 120.0,
                    summary_finish_reason=summary_finish_reason,
                )

                with (
                    patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
                    patch(
                        "app.services.stream.limit_summary._create_limit_summary_observation", return_value=observation
                    ),
                    patch("app.services.stream.llm_stream.append_chunk", new=stream_append),
                    patch("app.services.stream.llm_stream.check_lock_owner", new=AsyncMock(return_value=True)),
                    patch("app.services.stream.limit_summary.append_chunk", new=summary_append),
                ):
                    outcome = await run_limit_summary_step(request=request)

                if summary_finish_reason == "plan_synthesis":
                    self.assertEqual(
                        [(call.args[1], call.args[2]) for call in stream_append.await_args_list],
                        [
                            ("reasoning", "候选协议推理"),
                            ("answering", "最终答复"),
                            ("reasoning", "最终综合推理"),
                        ],
                    )
                    summary_append.assert_not_awaited()
                else:
                    self.assertEqual(
                        [(call.args[1], call.args[2]) for call in stream_append.await_args_list],
                        [
                            ("reasoning", "候选协议推理"),
                            ("reasoning", "最终综合推理"),
                        ],
                    )
                    summary_append.assert_awaited_once()
                    self.assertEqual(summary_append.await_args.args[1:3], ("answering", "最终答复"))
                self.assertFalse(outcome.incomplete)
                if summary_finish_reason == "plan_synthesis":
                    self.assertEqual([block.type for block in content_blocks], ["thinking", "text"])
                    self.assertEqual(content_blocks[-2].thinking, "候选协议推理最终综合推理")
                else:
                    self.assertEqual([block.type for block in content_blocks], ["thinking", "text"])
                    self.assertEqual(content_blocks[-2].thinking, "候选协议推理最终综合推理")

    async def test_no_progress_summary_uses_safe_fallback_after_repeated_tool_protocol(self):
        content_blocks = []
        emitter = AsyncMock()

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
            [],
            "tool_protocol_error",
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
            emitter=emitter,
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

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=prepare_context_fn),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk,
        ):
            outcome = await run_limit_summary_step(request=request)

        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertTrue(outcome.incomplete)
        self.assertEqual([block.type for block in content_blocks], ["thinking", "text"])
        self.assertEqual(content_blocks[0].thinking, "内部工具规划内部工具规划")
        self.assertIn("未能生成可靠的最终答复", content_blocks[1].text)
        self.assertNotIn("DSML", content_blocks[1].text)
        self.assertEqual(emitter.llm_round_completed.await_count, 2)
        self.assertTrue(all(call.kwargs["ttft_ms"] is None for call in emitter.llm_round_completed.await_args_list))

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

        with (
            patch("app.services.stream.limit_summary.prepare_context", new=AsyncMock(return_value=plan)),
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()),
        ):
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
            patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()),
        ):
            await run_limit_summary_step(request=request)

        self.assertEqual([message["role"] for message in sent_messages], ["user", "system"])
        self.assertEqual(sent_messages[0]["content"], "最新问题")
        self.assertTrue(sent_messages[-1]["content"].startswith(LIMIT_SUMMARY_PROMPT))
        self.assertTrue(sent_messages[-1]["content"].endswith(VISIBLE_RESPONSE_LANGUAGE_PROMPT))
        self.assertEqual(
            sum(str(item.get("content") or "").count(VISIBLE_RESPONSE_LANGUAGE_PROMPT) for item in sent_messages),
            1,
        )
        self.assertEqual(len(messages), 4)
        self.assertEqual(messages[-1]["content"], LIMIT_SUMMARY_PROMPT)

    async def test_limit_summary_records_independent_round_observation(self):
        messages = []
        observation = MagicMock()
        log_round_summary_fn = MagicMock()
        emitter = AsyncMock()
        call_order = []
        emitter.llm_round_started.side_effect = lambda **_kwargs: call_order.append("started")
        emitter.llm_round_cancelled.side_effect = lambda **_kwargs: call_order.append("cancelled")
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response

        async def llm_call_fn(*_args, **_kwargs):
            call_order.append("network")
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
            emitter=emitter,
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
        self.assertEqual(call_order, ["started", "network", "cancelled"])
        emitter.llm_round_completed.assert_not_awaited()

    async def test_run_limit_summary_step_appends_prompt_and_records_success(self):
        messages = [{"role": "user", "content": "hi"}]
        content_blocks = []
        accumulated_usage = Usage(input_tokens=2, output_tokens=3)
        emitter = AsyncMock()
        session_cache = object()
        events = []
        emitter.llm_round_started.side_effect = lambda **_kwargs: events.append(("llm_started",))
        emitter.llm_round_first_output_delta.side_effect = lambda **_kwargs: events.append(("first",))
        emitter.llm_round_completed.side_effect = lambda **_kwargs: events.append(("llm_completed",))

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
            defer_output,
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
                    defer_output,
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

        observation = MagicMock(first_output_delta_kind="content", first_output_delta_ms=50, duration_ms=80)
        observation.finish_success = AsyncMock()
        observation.finish_error = AsyncMock()
        observation.wrap_response.side_effect = lambda response: response
        append_chunk = AsyncMock(side_effect=lambda *_args, **_kwargs: events.append(("answer_chunk",)))
        with (
            patch("app.services.stream.limit_summary.append_chunk", new=append_chunk),
            patch(
                "app.services.stream.limit_summary.create_llm_round_observation",
                return_value=observation,
            ),
        ):
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
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], "总结正文")
        self.assertFalse(outcome.incomplete)
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

        self.assertEqual(
            [event[0] for event in events],
            [
                "start",
                "llm_started",
                "llm",
                "stream",
                "log",
                "answer_chunk",
                "first",
                "llm_completed",
                "complete",
            ],
        )
        self.assertEqual(events[2][4], {"temperature": 0.1})
        self.assertEqual(
            events[3],
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
                True,
            ),
        )
        self.assertEqual(events[4][1]["finish_reason"], "limit_summary")
        self.assertEqual(events[4][1]["tool_calls_count"], 0)
        self.assertEqual(events[4][1]["reasoning_buf"], "推理")
        self.assertEqual(events[4][1]["content_buf"], "总结正文")
        self.assertEqual(events[8][1].step_id, "step-summary")
        self.assertEqual(events[8][4], [])
        self.assertEqual(events[8][5], 0)

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

        with patch("app.services.stream.limit_summary.append_chunk", new=AsyncMock()) as append_chunk:
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
        append_chunk.assert_awaited_once()
        self.assertEqual(append_chunk.await_args.args[2], SUMMARY_PROTOCOL_FALLBACK_TEXT)
        self.assertTrue(outcome.incomplete)
        self.assertEqual(messages[-1], {"role": "system", "content": LIMIT_SUMMARY_PROMPT})
        self.assertEqual(outcome.accumulated_usage, accumulated_usage)
        self.assertEqual([block.text for block in content_blocks], [SUMMARY_PROTOCOL_FALLBACK_TEXT])
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
