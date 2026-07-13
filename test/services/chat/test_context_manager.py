import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from unittest.mock import patch

from app.schemas.chat import Message, TextBlock
from app.services.chat.context_manager import (
    ContextBudgetExceededError,
    ContextEstimationUnavailableError,
    prepare_context,
)
from app.services.chat.message_builder import build_llm_messages
from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig, _prepare_url_context


def _length_estimator(_model, messages, call_kwargs):
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text") or part.get("image_url") or ""))
        total += len(str(message.get("tool_calls") or ""))
    total += len(str(call_kwargs.get("tools") or ""))
    total += len(str(call_kwargs.get("tool_choice") or ""))
    return total


def _known_window(_model_id):
    return 100, "test", "known"


class ContextManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_4k_fixture_trims_old_history_and_keeps_system_and_latest_turn(self):
        history = []
        for index in range(6):
            history.extend(
                [
                    Message(
                        role="user",
                        content=[TextBlock(type="text", text=f"user-marker-{index} " + "背景资料 " * 65)],
                    ),
                    Message(
                        role="assistant",
                        content=[TextBlock(type="text", text=f"assistant-marker-{index} " + "历史回答 " * 65)],
                    ),
                ]
            )
        with (
            patch("app.services.chat.message_builder.build_current_date_system_prompt", return_value="固定日期"),
            patch("app.services.chat.message_builder.get_app_identity_prompt", return_value="固定身份"),
        ):
            canonical = await build_llm_messages(history)

        plan = await prepare_context(
            messages=canonical,
            model_id="small-model",
            litellm_model="gpt-4",
            call_kwargs={},
            window_resolver=lambda _model_id: (4096, "test", "known"),
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "trimmed")
        self.assertLessEqual(plan.estimated_tokens_after, int(4096 * 0.75))
        self.assertEqual([message["role"] for message in plan.messages[:2]], ["system", "system"])
        self.assertIn("user-marker-5", plan.messages[-2]["content"])
        self.assertIn("assistant-marker-5", plan.messages[-1]["content"])
        self.assertNotIn("user-marker-0", str(plan.messages))
        self.assertEqual(len(canonical), 14)

    async def test_under_trigger_returns_equal_snapshot_without_mutating_canonical(self):
        canonical = [
            {"role": "system", "content": "s" * 5},
            {"role": "user", "content": "u" * 10},
        ]

        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=_known_window,
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "no_op")
        self.assertEqual(plan.messages, canonical)
        self.assertIsNot(plan.messages, canonical)
        self.assertEqual(canonical[1]["content"], "u" * 10)
        self.assertEqual(plan.estimated_tokens_before, 15)
        self.assertEqual(plan.estimated_tokens_after, 15)

    async def test_over_trigger_drops_oldest_complete_turn_to_target(self):
        canonical = [
            {"role": "system", "content": "s" * 10},
            {"role": "user", "content": "a" * 30},
            {"role": "assistant", "content": "b" * 20},
            {"role": "user", "content": "c" * 30},
        ]

        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=_known_window,
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "trimmed")
        self.assertEqual(
            plan.messages,
            [canonical[0], canonical[3]],
        )
        self.assertEqual(plan.removed_turns, 1)
        self.assertEqual(plan.removed_messages, 2)
        self.assertLessEqual(plan.estimated_tokens_after, plan.target_tokens)
        self.assertEqual(len(canonical), 4)

    async def test_system_messages_keep_original_positions_when_turns_are_removed(self):
        canonical = [
            {"role": "system", "content": "a" * 10},
            {"role": "user", "content": "b" * 30},
            {"role": "assistant", "content": "c" * 20},
            {"role": "system", "content": "d" * 10},
            {"role": "user", "content": "e" * 30},
            {"role": "system", "content": "f" * 10},
        ]

        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=lambda _model: (120, "test", "known"),
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "trimmed")
        self.assertEqual(
            plan.messages,
            [canonical[0], canonical[3], canonical[4], canonical[5]],
        )

    async def test_old_tool_transaction_is_removed_as_one_turn_without_orphans(self):
        canonical = [
            {"role": "system", "content": "s" * 5},
            {"role": "user", "content": "q" * 20},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call-1", "function": {"name": "a", "arguments": "{}"}},
                    {"id": "call-2", "function": {"name": "b", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "r" * 20},
            {"role": "tool", "tool_call_id": "call-2", "content": "t" * 20},
            {"role": "user", "content": "n" * 30},
        ]

        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=lambda _model: (160, "test", "known"),
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "trimmed")
        self.assertEqual(plan.messages, [canonical[0], canonical[5]])
        self.assertFalse(any(message.get("role") == "tool" for message in plan.messages))
        self.assertFalse(any(message.get("tool_calls") for message in plan.messages))

    async def test_latest_tool_transaction_is_mandatory_and_keeps_all_results(self):
        canonical = [
            {"role": "user", "content": "o" * 50},
            {"role": "assistant", "content": "p" * 20},
            {"role": "user", "content": "n" * 10},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "call-1", "function": {"name": "a", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "r" * 10},
        ]

        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=lambda _model: (180, "test", "known"),
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "trimmed")
        self.assertEqual(plan.messages, canonical[2:])
        self.assertEqual(plan.messages[-1]["tool_call_id"], "call-1")

    async def test_same_user_old_tool_transaction_is_removed_but_latest_stays_atomic(self):
        canonical = [
            {"role": "system", "content": "s" * 5},
            {"role": "user", "content": "q" * 10},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"id": "old-1", "function": {"name": "old", "arguments": "{}"}}],
            },
            {"role": "tool", "tool_call_id": "old-1", "content": "o" * 70},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "new-1", "function": {"name": "new", "arguments": "{}"}},
                    {"id": "new-2", "function": {"name": "new", "arguments": "{}"}},
                ],
            },
            {"role": "tool", "tool_call_id": "new-1", "content": "n" * 10},
            {"role": "tool", "tool_call_id": "new-2", "content": "m" * 10},
        ]

        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=lambda _model: (200, "test", "known"),
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertIn(plan.status, {"trimmed", "trimmed_required_above_target"})
        self.assertEqual(plan.removed_turns, 0)
        self.assertEqual(plan.removed_tool_transactions, 1)
        self.assertNotIn("old-1", str(plan.messages))
        self.assertIn("new-1", str(plan.messages))
        self.assertIn("new-2", str(plan.messages))
        self.assertEqual([message["role"] for message in plan.messages[-3:]], ["assistant", "tool", "tool"])

    async def test_current_url_context_and_latest_user_are_one_mandatory_turn(self):
        messages = [
            {"role": "system", "content": "s" * 5},
            {"role": "user", "content": "o" * 80},
            {"role": "assistant", "content": "a" * 80},
            {"role": "user", "content": "summarize it"},
        ]

        async def preprocess_url(_message, _supports_function_calling, _call_kwargs):
            return (
                None,
                {"role": "user", "content": "URL_EVIDENCE " + "e" * 60},
                "https://example.com",
            )

        prepared, _blocks = await _prepare_url_context(
            messages=messages,
            original_message="summarize it",
            call_config=AgentLoopCallConfig(
                should_use_reasoning=False,
                supports_function_calling=True,
                call_kwargs={},
                announced_tools=[],
            ),
            preprocess_url_in_message_fn=preprocess_url,
        )
        plan = await prepare_context(
            messages=prepared,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=lambda _model: (200, "test", "known"),
            token_estimator=_length_estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "trimmed")
        self.assertIn("URL_EVIDENCE", str(plan.messages))
        self.assertIn("summarize it", str(plan.messages))
        self.assertNotIn("o" * 80, str(plan.messages))
        self.assertEqual([message["role"] for message in plan.messages], ["system", "user", "user"])

    async def test_required_context_over_trigger_raises_without_truncating_user_input(self):
        canonical = [
            {"role": "system", "content": "s" * 20},
            {"role": "user", "content": "u" * 70},
        ]

        with self.assertRaises(ContextBudgetExceededError) as raised:
            await prepare_context(
                messages=canonical,
                model_id="model-a",
                litellm_model="litellm_proxy/model-a",
                call_kwargs={},
                window_resolver=_known_window,
                token_estimator=_length_estimator,
                run_in_thread=False,
                use_fast_path=False,
            )

        self.assertEqual(raised.exception.plan.status, "required_context_over_budget")
        self.assertEqual(canonical[1]["content"], "u" * 70)

    async def test_unknown_window_bypasses_without_calling_estimator(self):
        calls = []

        def estimator(*args):
            calls.append(args)
            return 999

        canonical = [{"role": "user", "content": "hello"}]
        plan = await prepare_context(
            messages=canonical,
            model_id="model-a",
            litellm_model="litellm_proxy/model-a",
            call_kwargs={},
            window_resolver=lambda _model: (None, "catalog", "missing"),
            token_estimator=estimator,
            run_in_thread=False,
            use_fast_path=False,
        )

        self.assertEqual(plan.status, "bypass_unknown_window")
        self.assertEqual(plan.messages, canonical)
        self.assertEqual(calls, [])

    async def test_estimator_error_fails_closed_without_exposing_details(self):
        canonical = [{"role": "user", "content": "hello"}]

        with self.assertRaises(ContextEstimationUnavailableError) as raised:
            await prepare_context(
                messages=canonical,
                model_id="model-a",
                litellm_model="litellm_proxy/model-a",
                call_kwargs={},
                window_resolver=_known_window,
                token_estimator=lambda *_args: (_ for _ in ()).throw(RuntimeError("private prompt")),
                run_in_thread=False,
                use_fast_path=False,
            )

        self.assertEqual(raised.exception.plan.status, "estimator_unavailable")
        self.assertEqual(raised.exception.plan.messages, canonical)
        self.assertNotIn("private", str(raised.exception.plan.telemetry()))

    async def test_estimator_timeout_fails_closed_and_does_not_block_event_loop(self):
        release = threading.Event()

        def slow_estimator(*_args):
            release.wait(timeout=1)
            return 99

        try:
            with self.assertRaises(ContextEstimationUnavailableError) as raised:
                await prepare_context(
                    messages=[{"role": "user", "content": "x" * 100}],
                    model_id="model-a",
                    litellm_model="litellm_proxy/model-a",
                    call_kwargs={},
                    window_resolver=_known_window,
                    token_estimator=slow_estimator,
                    estimator_timeout_seconds=0.01,
                    use_fast_path=False,
                )
        finally:
            release.set()

        self.assertEqual(raised.exception.plan.status, "estimator_unavailable")

    async def test_cancellation_propagates_while_estimator_is_running(self):
        started = threading.Event()
        release = threading.Event()

        def slow_estimator(*_args):
            started.set()
            release.wait(timeout=1)
            return 99

        task = asyncio.create_task(
            prepare_context(
                messages=[{"role": "user", "content": "x" * 100}],
                model_id="model-a",
                litellm_model="litellm_proxy/model-a",
                call_kwargs={},
                window_resolver=_known_window,
                token_estimator=slow_estimator,
                estimator_timeout_seconds=1,
                use_fast_path=False,
            )
        )
        self.assertTrue(await asyncio.to_thread(started.wait, 1))
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        release.set()

    async def test_estimator_admission_fails_closed_instead_of_sending_unchecked_context(self):
        from app.services.chat import context_manager as module

        started = 0
        started_lock = threading.Lock()
        two_started = threading.Event()
        release = threading.Event()

        def slow_estimator(*_args):
            nonlocal started
            with started_lock:
                started += 1
                if started == 2:
                    two_started.set()
            release.wait(timeout=1)
            return 10

        executor = ThreadPoolExecutor(max_workers=2)
        admission = BoundedSemaphore(value=2)
        try:
            with (
                patch.object(module, "_TOKEN_EXECUTOR", executor),
                patch.object(module, "_TOKEN_ADMISSION", admission),
            ):
                tasks = [
                    asyncio.create_task(
                        prepare_context(
                            messages=[{"role": "user", "content": "x" * 100}],
                            model_id="model-a",
                            litellm_model="litellm_proxy/model-a",
                            call_kwargs={},
                            window_resolver=_known_window,
                            token_estimator=slow_estimator,
                            estimator_timeout_seconds=1,
                            use_fast_path=False,
                        )
                    )
                    for _index in range(4)
                ]
                self.assertTrue(await asyncio.to_thread(two_started.wait, 1))
                await asyncio.sleep(0)
                release.set()
                results = await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            release.set()
            executor.shutdown(wait=True)

        self.assertEqual(sum(isinstance(result, ContextEstimationUnavailableError) for result in results), 2)
        self.assertEqual(sum(getattr(result, "status", None) == "no_op" for result in results), 2)


if __name__ == "__main__":
    unittest.main()
