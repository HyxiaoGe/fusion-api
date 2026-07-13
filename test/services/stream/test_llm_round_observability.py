import asyncio
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore
from types import SimpleNamespace
from unittest.mock import patch

from app.schemas.chat import Message, TextBlock, Usage
from app.services.chat.message_builder import build_llm_messages


class LLMRoundObservabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_context_management_metadata_is_allowlisted_and_reuses_estimate(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        estimator_calls = []
        observation = LLMRoundObservation(
            metadata=RoundMetadata(
                conversation_id="conv-1",
                run_id="run-1",
                round_index=1,
                step_id="step-1",
                round_kind="agent",
                model_id="small-model",
                provider="test",
            ),
            litellm_model="test/small-model",
            messages=[{"role": "user", "content": "正文不可出现在日志"}],
            call_kwargs={},
            token_estimator=lambda *_args: estimator_calls.append(1) or 999,
            context_window_resolver=lambda _model_id: (100, "catalog", "known"),
            run_context_in_thread=False,
            estimated_prompt_tokens=70,
            context_management={
                "context_management_status": "trimmed",
                "context_management_removed_turns": 2,
                "context_management_private_prompt": "不可泄露",
            },
        )

        observation.start()
        await observation.finish_success(usage=None, finish_reason="stop")
        await observation.wait_for_log()

        self.assertEqual(estimator_calls, [])
        self.assertEqual(observation.last_payload["estimated_prompt_tokens"], 70)
        self.assertEqual(observation.last_payload["estimator_status"], "reused_context_manager")
        self.assertEqual(observation.last_payload["context_management_status"], "trimmed")
        self.assertEqual(observation.last_payload["context_management_removed_turns"], 2)
        self.assertNotIn("context_management_private_prompt", observation.last_payload)

    async def test_4k_over_budget_metrics_are_observed_without_trimming_claim(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

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
            messages = await build_llm_messages(history)
        self.assertEqual(len(messages), 14)
        self.assertIn("user-marker-0", messages[2]["content"])
        self.assertIn("assistant-marker-5", messages[-1]["content"])

        observation = LLMRoundObservation(
            metadata=RoundMetadata(
                conversation_id="conv-1",
                run_id="run-1",
                round_index=1,
                step_id="step-1",
                round_kind="agent",
                model_id="small-model",
                provider="test",
            ),
            litellm_model="test/small-model",
            messages=messages,
            call_kwargs={},
            context_window_resolver=lambda _model_id: (4096, "litellm_catalog", "known"),
            run_context_in_thread=False,
        )

        observation.start()
        await observation.finish_success(
            usage=Usage(input_tokens=5600, output_tokens=20),
            finish_reason="stop",
        )
        await observation.wait_for_log()

        payload = observation.last_payload
        self.assertGreater(payload["estimated_prompt_tokens"], 4096)
        self.assertEqual(payload["context_window_tokens"], 4096)
        self.assertGreater(payload["estimated_utilization_ratio"], 1)
        self.assertIs(payload["estimated_over_budget"], True)
        self.assertGreater(payload["actual_utilization_ratio"], 1)
        self.assertIs(payload["actual_over_budget"], True)

    async def test_unknown_window_keeps_budget_fields_unknown(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        observation = LLMRoundObservation(
            metadata=RoundMetadata(
                conversation_id="conv-1",
                run_id="run-1",
                round_index=1,
                step_id="step-1",
                round_kind="agent",
                model_id="unknown-model",
                provider="test",
            ),
            litellm_model="test/unknown-model",
            messages=[],
            call_kwargs={},
            token_estimator=lambda *_args, **_kwargs: 10,
            context_window_resolver=lambda _model_id: (None, "unknown", "missing"),
            run_context_in_thread=False,
        )

        observation.start()
        await observation.finish_success(usage=None, finish_reason="stop")
        await observation.wait_for_log()

        payload = observation.last_payload
        self.assertIsNone(payload["context_window_tokens"])
        self.assertIsNone(payload["estimated_utilization_ratio"])
        self.assertIsNone(payload["estimated_over_budget"])
        self.assertIsNone(payload["actual_utilization_ratio"])
        self.assertIsNone(payload["actual_over_budget"])
        self.assertIsNone(payload["round_prompt_tokens"])
        self.assertIsNone(payload["round_completion_tokens"])

    async def test_log_never_contains_prompt_tool_file_or_error_details(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        secrets = [
            "prompt-canary-91f2",
            "https://private.example/path",
            "sk-secret-api-key",
            "data:image/png;base64,PRIVATE_IMAGE",
            "private-report.pdf",
            "tool-result-secret",
            "provider echoed secret prompt",
        ]
        messages = [
            {"role": "user", "content": f"{secrets[0]} {secrets[1]} {secrets[2]}"},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": secrets[3]}},
                    {"type": "text", "text": secrets[4]},
                ],
            },
            {"role": "tool", "content": secrets[5], "tool_call_id": "tool-1"},
        ]
        observation = LLMRoundObservation(
            metadata=RoundMetadata(
                conversation_id="conv-1",
                run_id="run-1",
                round_index=1,
                step_id="step-1",
                round_kind="agent",
                model_id="safe-model",
                provider="test",
            ),
            litellm_model="test/safe-model",
            messages=messages,
            call_kwargs={
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "secret_tool", "description": secrets[1]},
                    }
                ]
            },
            token_estimator=lambda *_args, **_kwargs: 100,
            context_window_resolver=lambda _model_id: (4096, "litellm_catalog", "known"),
            run_context_in_thread=False,
        )

        with patch("app.ai.llm_round_observability.logger.info") as info:
            observation.start()
            await observation.finish_error(RuntimeError(secrets[6]))
            await observation.wait_for_log()

        serialized = json.dumps(observation.last_payload, ensure_ascii=False)
        logged = " ".join(str(arg) for arg in info.call_args.args)
        for secret in secrets:
            self.assertNotIn(secret, serialized)
            self.assertNotIn(secret, logged)
        self.assertEqual(observation.last_payload["error_type"], "RuntimeError")
        self.assertEqual(observation.last_payload["outcome"], "error")

    async def test_first_model_text_delta_skips_empty_usage_and_tool_only_chunks(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        times = iter([10.0, 10.4, 11.0])
        observation = LLMRoundObservation(
            metadata=RoundMetadata(
                conversation_id="conv-1",
                run_id="run-1",
                round_index=1,
                step_id="step-1",
                round_kind="agent",
                model_id="safe-model",
                provider="test",
            ),
            litellm_model="test/safe-model",
            messages=[],
            call_kwargs={},
            clock=lambda: next(times),
            token_estimator=lambda *_args, **_kwargs: 1,
            context_window_resolver=lambda _model_id: (4096, "litellm_catalog", "known"),
            run_context_in_thread=False,
        )
        chunks = [
            SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1)),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="", tool_calls=None))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=[object()]))],
                usage=None,
            ),
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content="答案", tool_calls=None))],
                usage=None,
            ),
        ]

        async def response():
            for chunk in chunks:
                yield chunk

        observation.start()
        observed = observation.wrap_response(response())
        async for _chunk in observed:
            pass
        await observation.finish_success(usage=None, finish_reason="stop")
        await observation.wait_for_log()

        self.assertEqual(observation.last_payload["first_model_text_delta_ms"], 400.0)
        self.assertEqual(observation.last_payload["total_duration_ms"], 1000.0)

    async def test_fast_error_still_emits_completed_background_estimate(self):
        from app.ai.llm_round_observability import LLMRoundObservation, RoundMetadata

        estimator_started = threading.Event()
        estimator_release = threading.Event()

        def slow_estimator(*_args, **_kwargs):
            estimator_started.set()
            estimator_release.wait(timeout=1)
            return 5641

        observation = LLMRoundObservation(
            metadata=RoundMetadata(
                conversation_id="conv-fast-error",
                run_id="run-fast-error",
                round_index=1,
                step_id="step-fast-error",
                round_kind="agent",
                model_id="small-model",
                provider="test",
            ),
            litellm_model="test/small-model",
            messages=[{"role": "user", "content": "private prompt"}],
            call_kwargs={},
            token_estimator=slow_estimator,
            context_window_resolver=lambda _model_id: (4096, "litellm_catalog", "known"),
        )

        observation.start()
        self.assertTrue(await asyncio.to_thread(estimator_started.wait, 1))
        await observation.finish_error(RuntimeError("private provider error"))
        self.assertIsNone(observation.last_payload)
        estimator_release.set()
        await observation.wait_for_log()

        self.assertEqual(observation.last_payload["estimated_prompt_tokens"], 5641)
        self.assertEqual(observation.last_payload["estimator_status"], "success")
        self.assertEqual(observation.last_payload["outcome"], "error")

    def test_context_window_resolution_never_refreshes_catalog(self):
        from app.ai.llm_round_observability import resolve_context_window

        with (
            patch(
                "app.ai.llm_round_observability.litellm_catalog.get_cached_model_entry",
                return_value=({"max_input_tokens": 8192}, "stale"),
            ),
            patch("app.ai.litellm_catalog._fetch_catalog") as fetch_catalog,
        ):
            result = resolve_context_window("cached-model")

        self.assertEqual(result, (8192, "litellm_catalog_cache", "stale"))
        fetch_catalog.assert_not_called()

    async def test_estimator_admission_bounds_concurrent_full_tokenization(self):
        from app.ai import llm_round_observability as module

        estimator_release = threading.Event()
        estimator_calls = 0
        estimator_lock = threading.Lock()

        def slow_estimator(*_args, **_kwargs):
            nonlocal estimator_calls
            with estimator_lock:
                estimator_calls += 1
            estimator_release.wait(timeout=1)
            return 100

        observations = [
            module.LLMRoundObservation(
                metadata=module.RoundMetadata(
                    conversation_id=f"conv-{index}",
                    run_id=f"run-{index}",
                    round_index=1,
                    step_id=f"step-{index}",
                    round_kind="agent",
                    model_id="bounded-model",
                    provider="test",
                ),
                litellm_model="test/bounded-model",
                messages=[{"role": "user", "content": "x" * 100_000}],
                call_kwargs={},
                token_estimator=slow_estimator,
                context_window_resolver=lambda _model_id: (200_000, "litellm_catalog", "known"),
            )
            for index in range(6)
        ]
        executor = ThreadPoolExecutor(max_workers=1)
        admission = BoundedSemaphore(value=2)

        try:
            with (
                patch.object(module, "_ESTIMATE_EXECUTOR", executor),
                patch.object(module, "_ESTIMATE_ADMISSION", admission),
                patch.object(module.logger, "info"),
            ):
                for observation in observations:
                    observation.start()
                    await observation.finish_success(usage=None, finish_reason="stop")
                estimator_release.set()
                await asyncio.gather(*(observation.wait_for_log() for observation in observations))
        finally:
            executor.shutdown(wait=True)

        statuses = [observation.last_payload["estimator_status"] for observation in observations]
        self.assertEqual(estimator_calls, 2)
        self.assertEqual(statuses.count("success"), 2)
        self.assertEqual(statuses.count("skipped_overload"), 4)


if __name__ == "__main__":
    unittest.main()
