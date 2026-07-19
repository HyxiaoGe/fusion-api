import argparse
import json
import unittest
import urllib.parse
from decimal import Decimal
from unittest.mock import patch

from scripts.context_ladder_eval import (
    LadderPlan,
    RunnerError,
    StagePlan,
    StageResult,
    _execution_stages,
    _lookup_round_context,
    _parse_loki_round_payloads,
    _validate_round_context,
    build_chat_payload,
    build_ladder_plan,
    build_loki_query_range_url,
    build_parser,
    consume_sse_lines,
    execute,
    validate_args,
)

VALID_INTERNAL_TOKEN = "test-internal-auth-token-0123456789abcdef"


def _required_args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--model-id",
            "model-a",
            "--context-window",
            "100000",
            "--input-usd-per-million",
            "2",
            "--output-usd-per-million",
            "6",
            "--max-cost-usd",
            "1",
            "--max-request-bytes",
            "1000000",
            "--prometheus-url",
            "https://prometheus.example",
            "--loki-url",
            "https://loki.example",
            *extra,
        ]
    )


def _line(payload: dict) -> list[bytes]:
    return [f"data: {json.dumps(payload)}\n".encode(), b"\n"]


class ArgumentsTest(unittest.TestCase):
    def test_required_contract_arguments_are_enforced(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([])

    def test_default_is_network_free_dry_run(self):
        args = _required_args()

        validate_args(args)

        self.assertFalse(args.apply)

        class ForbiddenClient:
            def __init__(self, *_args, **_kwargs):
                raise AssertionError("dry-run 不得创建网络客户端")

        with (
            patch("scripts.context_ladder_eval._build_litellm_estimator", return_value=len),
            patch("scripts.context_ladder_eval.HttpClient", ForbiddenClient),
        ):
            result = execute(args)

        self.assertEqual(result["status"], "planned_not_executed")
        self.assertFalse(result["executed"])
        self.assertFalse(result["cleanup"]["attempted"])

    def test_production_apply_requires_both_explicit_gates(self):
        args = _required_args("--apply")
        with self.assertRaisesRegex(RunnerError, "confirm-production"):
            validate_args(args)

        args = _required_args("--apply", "--confirm-production")
        with self.assertRaisesRegex(RunnerError, "generation-controls-verified"):
            validate_args(args)

        args = _required_args(
            "--apply",
            "--confirm-production",
            "--generation-controls-verified",
        )
        with self.assertRaisesRegex(RunnerError, "allow-account-residue"):
            validate_args(args)

        args = _required_args(
            "--apply",
            "--confirm-production",
            "--generation-controls-verified",
            "--allow-account-residue",
        )
        validate_args(args)

    def test_production_host_check_is_case_and_trailing_dot_safe(self):
        args = _required_args(
            "--apply",
            "--target-url",
            "https://FUSION.SEANFIELD.ORG./",
            "--auth-url",
            "https://AUTH.SEANFIELD.ORG./",
            "--generation-controls-verified",
        )

        with self.assertRaisesRegex(RunnerError, "confirm-production"):
            validate_args(args)


class PlanTest(unittest.TestCase):
    def test_start_case_allows_independent_window_but_rejects_dependent_track_tail(self):
        plan = build_ladder_plan(
            model_id="model-a",
            context_window=1000,
            input_price=Decimal("2"),
            output_price=Decimal("6"),
            max_cost=Decimal("1"),
            max_request_bytes=1_000_000,
            low_targets=(50, 100, 200, 400),
            token_estimator=len,
        )

        self.assertEqual(_execution_stages(plan, "window-60")[0].case_id, "window-60")
        with self.assertRaisesRegex(RunnerError, "依赖"):
            _execution_stages(plan, "multi-100")
        with self.assertRaisesRegex(RunnerError, "依赖"):
            _execution_stages(plan, "managed-trim-90")

    def test_builds_serial_cold_multi_and_window_ladder(self):
        plan = build_ladder_plan(
            model_id="model-a",
            context_window=1000,
            input_price=Decimal("2"),
            output_price=Decimal("6"),
            max_cost=Decimal("1"),
            max_request_bytes=1_000_000,
            low_targets=(50, 100, 200, 400),
            token_estimator=len,
        )

        self.assertEqual(len(plan.stages), 12)
        self.assertEqual([stage.track for stage in plan.stages[:4]], ["cold"] * 4)
        self.assertEqual([stage.track for stage in plan.stages[4:8]], ["multi_turn"] * 4)
        self.assertEqual([stage.target_context_tokens for stage in plan.stages[8:10]], [600, 800])
        self.assertEqual([stage.track for stage in plan.stages[-2:]], ["managed_multi", "managed_multi"])
        self.assertEqual(plan.stages[-1].expected_management_status, "trimmed")
        self.assertEqual(len({stage.conversation_key for stage in plan.stages[:4]}), 4)
        self.assertEqual(len({stage.conversation_key for stage in plan.stages[4:8]}), 1)
        self.assertEqual(
            plan.projected_cost_usd,
            (Decimal("4200") * Decimal("2") + Decimal("12288") * Decimal("6")) / Decimal("1000000"),
        )

    def test_final_multi_canaries_reference_their_original_turns(self):
        plan = build_ladder_plan(
            model_id="model-a",
            context_window=1000,
            input_price=Decimal("2"),
            output_price=Decimal("6"),
            max_cost=Decimal("1"),
            max_request_bytes=1_000_000,
            low_targets=(50, 100, 200, 400),
            token_estimator=len,
        )

        final_multi = next(stage for stage in plan.stages if stage.case_id == "multi-400")

        self.assertIn("multi-50 的 early", final_multi.prompt)
        self.assertIn("multi-200 的 middle", final_multi.prompt)
        self.assertIn("multi-400 的 recent", final_multi.prompt)
        self.assertNotIn(final_multi.expected_canaries["early"], final_multi.prompt)
        self.assertNotIn(final_multi.expected_canaries["middle"], final_multi.prompt)
        self.assertIn(final_multi.expected_canaries["recent"], final_multi.prompt)

    def test_rejects_cost_or_request_size_above_budget(self):
        common = dict(
            model_id="model-a",
            context_window=1000,
            input_price=Decimal("2"),
            output_price=Decimal("6"),
            low_targets=(50, 100, 200, 400),
            token_estimator=len,
        )
        with self.assertRaisesRegex(RunnerError, "成本"):
            build_ladder_plan(**common, max_cost=Decimal("0.001"), max_request_bytes=1_000_000)
        with self.assertRaisesRegex(RunnerError, "请求体"):
            build_ladder_plan(**common, max_cost=Decimal("1"), max_request_bytes=100)

    def test_safe_plan_never_serializes_prompt_or_canary(self):
        plan = build_ladder_plan(
            model_id="model-a",
            context_window=1000,
            input_price=Decimal("2"),
            output_price=Decimal("6"),
            max_cost=Decimal("1"),
            max_request_bytes=1_000_000,
            low_targets=(50, 100, 200, 400),
            token_estimator=len,
        )
        serialized = json.dumps(plan.safe_dict())

        for stage in plan.stages:
            self.assertNotIn(stage.prompt, serialized)
            for canary in stage.expected_canaries.values():
                self.assertNotIn(canary, serialized)
        self.assertNotIn("conversation_key", serialized)


class PayloadAndSSETest(unittest.TestCase):
    def setUp(self):
        self.stage = StagePlan(
            case_id="cold-50",
            track="cold",
            target_context_tokens=50,
            ratio=None,
            conversation_key="secret-conversation-key",
            prompt="secret filler CANARY-ONE",
            expected_canaries={"early": "CANARY-ONE"},
            local_estimated_prompt_tokens=50,
            request_bytes=300,
            planned_input_tokens=50,
            prompt_hash="abc123",
        )

    def test_chat_payload_has_fixed_generation_controls(self):
        payload = build_chat_payload(self.stage, "model-a", "conv-secret")

        self.assertEqual(
            payload["options"],
            {"use_reasoning": False, "disable_tools": True, "max_tokens": 1024},
        )
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["file_ids"], [])

    def test_success_result_contains_only_hash_length_hits_and_metrics(self):
        lines = []
        lines += _line(
            {
                "chunk_type": "agent_event",
                "data": {"type": "run_started", "run_id": "run-safe", "message_id": "message-safe"},
            }
        )
        lines += _line({"chunk_type": "agent_event", "data": {"type": "step_started"}})
        lines += _line({"chunk_type": "answering", "data": {"delta": "CANARY-"}})
        lines += _line({"chunk_type": "answering", "data": {"delta": "ONE"}})
        lines += [b"data: [DONE]\n", b"\n"]

        result = consume_sse_lines(self.stage, lines, observed_times=iter([10, 20, 30, 40, 50]))
        serialized = json.dumps(result.safe_dict())

        self.assertTrue(result.success)
        self.assertEqual(result.canary_hits, {"early": True})
        self.assertEqual(result.round_count, 1)
        self.assertEqual(result.answer_length, len("CANARY-ONE"))
        self.assertEqual(result.agent_run_id, "run-safe")
        self.assertEqual(result.assistant_message_id, "message-safe")
        self.assertNotIn("CANARY-ONE", serialized)
        self.assertNotIn(self.stage.prompt, serialized)
        self.assertNotIn("secret-conversation", serialized)

    def test_tool_second_round_error_and_canary_miss_are_hard_stops(self):
        cases = {
            "tool_event": _line({"chunk_type": "agent_event", "data": {"event": {"type": "tool_call_started"}}}),
            "second_round": _line({"chunk_type": "agent_event", "data": {"type": "step_started"}})
            + _line({"chunk_type": "agent_event", "data": {"type": "step_started"}}),
            "error_frame": _line({"chunk_type": "error", "data": {"message": "secret error"}}),
            "canary_miss": _line({"chunk_type": "answering", "data": {"delta": "not present"}}),
        }
        run_started = _line(
            {
                "chunk_type": "agent_event",
                "data": {"type": "run_started", "run_id": "run-safe", "message_id": "message-safe"},
            }
        )
        for expected_reason, prefix in cases.items():
            with self.subTest(expected_reason=expected_reason):
                lines = run_started + prefix + [b"data: [DONE]\n", b"\n"]
                result = consume_sse_lines(
                    self.stage,
                    lines,
                    observed_times=iter(range(10, 200, 10)),
                )
                self.assertFalse(result.success)
                self.assertIn(expected_reason, result.stop_reasons)
                self.assertNotIn("secret error", json.dumps(result.safe_dict()))

    def test_tool_event_aborts_stream_consumption_immediately(self):
        consumed = []

        def lines():
            for line in _line(
                {
                    "chunk_type": "agent_event",
                    "data": {"type": "run_started", "run_id": "run-safe", "message_id": "message-safe"},
                }
            ) + _line({"chunk_type": "agent_event", "data": {"type": "tool_call_started"}}):
                consumed.append(line)
                yield line
            raise AssertionError("硬门禁命中后不得继续消费后续流")

        result = consume_sse_lines(self.stage, lines(), observed_times=iter(range(10, 200, 10)))

        self.assertIn("tool_event", result.stop_reasons)
        self.assertIn("incomplete_stream", result.stop_reasons)
        self.assertEqual(len(consumed), 4)

    def test_wall_clock_deadline_is_not_extended_by_heartbeat(self):
        clock = iter([0.0, 100.0, 1_100.0])

        result = consume_sse_lines(
            self.stage,
            [": keepalive\n", ": keepalive\n"],
            max_duration_seconds=1,
            deadline_clock_ms=lambda: next(clock),
        )

        self.assertIn("timeout", result.stop_reasons)
        self.assertIn("incomplete_stream", result.stop_reasons)


class LokiTest(unittest.TestCase):
    def test_query_uses_verified_production_label_and_encoded_run_filter(self):
        url = build_loki_query_range_url(
            "https://loki.example",
            "run-safe-1",
            start_ns=1,
            end_ns=2,
        )

        self.assertIn("/loki/api/v1/query_range?", url)
        decoded = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        self.assertIn('{container="fusion-api"}', decoded["query"][0])
        self.assertIn('\\"run_id\\":\\"run-safe-1\\"', decoded["query"][0])

    def test_parses_only_matching_round_context_without_exposing_log_line(self):
        safe = {
            "event": "llm_round_context",
            "run_id": "run-safe",
            "conversation_id": "conv-secret",
            "estimated_prompt_tokens": 123,
        }
        other = {**safe, "run_id": "run-other"}
        payload = {
            "status": "success",
            "data": {
                "result": [
                    {
                        "values": [
                            ["1", f"INFO LLM_ROUND_CONTEXT {json.dumps(safe)}"],
                            ["2", json.dumps({"log": f"LLM_ROUND_CONTEXT {json.dumps(other)}"})],
                        ]
                    }
                ]
            },
        }

        matches = _parse_loki_round_payloads(payload, "run-safe")

        self.assertEqual(matches, [safe])

    def test_lookup_fails_closed_on_duplicate_log(self):
        class FakeLoki:
            def __init__(self, matches):
                self.matches = matches

            def query_range(self, *_args, **_kwargs):
                return self.matches

        duplicate = _lookup_round_context(
            FakeLoki([{"run_id": "run-safe"}, {"run_id": "run-safe"}]),
            "https://loki.example",
            "run-safe",
            start_ns=1,
            timeout_seconds=1,
        )

        self.assertEqual(duplicate, (None, "round_context_duplicate"))

    def test_round_context_requires_fresh_window_single_round_target_and_output_cap(self):
        stage = StagePlan(
            case_id="cold-5000",
            track="cold",
            target_context_tokens=5000,
            ratio=None,
            conversation_key="secret",
            prompt="secret",
            expected_canaries={},
            local_estimated_prompt_tokens=5000,
            request_bytes=100,
            planned_input_tokens=5000,
            prompt_hash="hash",
        )
        result = StageResult(
            case_id=stage.case_id,
            success=True,
            done=True,
            round_count=1,
            tool_event_count=0,
            error_frame_count=0,
            canary_hits={},
            answer_hash="hash",
            answer_length=1,
            metrics={},
            stop_reasons=(),
            agent_run_id="run-safe",
            assistant_message_id="message-safe",
        )
        payload = {
            "event": "llm_round_context",
            "run_id": "run-safe",
            "assistant_message_id": "message-safe",
            "model_id": "model-a",
            "context_window_tokens": 100000,
            "context_window_status": "known",
            "round_index": 1,
            "round_kind": "agent",
            "estimated_prompt_tokens": 5100,
            "estimator_status": "success",
            "round_prompt_tokens": 5200,
            "round_completion_tokens": 64,
            "request_tool_definition_count": 0,
            "outcome": "success",
        }

        _safe, reasons = _validate_round_context(
            stage,
            result,
            payload,
            model_id="model-a",
            context_window=100000,
        )
        self.assertEqual(reasons, [])

        invalid = {
            **payload,
            "context_window_status": "stale",
            "round_index": 2,
            "round_prompt_tokens": 9000,
            "round_completion_tokens": 1025,
        }
        _safe, reasons = _validate_round_context(
            stage,
            result,
            invalid,
            model_id="model-a",
            context_window=100000,
        )

        self.assertIn("round_context_window_not_fresh", reasons)
        self.assertIn("round_context_not_single_agent_round", reasons)
        self.assertIn("round_context_target_deviation", reasons)
        self.assertIn("round_context_output_cap_failed", reasons)


class ExecuteSafetyTest(unittest.TestCase):
    @staticmethod
    def _stage(case_id: str = "cold-50") -> StagePlan:
        return StagePlan(
            case_id=case_id,
            track="cold",
            target_context_tokens=50,
            ratio=None,
            conversation_key="internal-conversation-track",
            prompt="secret prompt CANARY",
            expected_canaries={},
            local_estimated_prompt_tokens=50,
            request_bytes=100,
            planned_input_tokens=50,
            prompt_hash="hash",
        )

    @classmethod
    def _plan(cls, *stages: StagePlan) -> LadderPlan:
        return LadderPlan(
            model_id="model-a",
            context_window=100000,
            stages=tuple(stages),
            projected_input_tokens=sum(stage.planned_input_tokens for stage in stages),
            projected_output_tokens=1024 * len(stages),
            projected_cost_usd=Decimal("0.01"),
        )

    def test_failed_stage_stop_failure_is_sanitized_before_cleanup_and_loki_missing_charges_fallback(self):
        events = []
        stage = self._stage()

        class ResponseContext:
            def __enter__(self):
                return []

            def __exit__(self, *_args):
                return False

        class FakeClient:
            def open_sse(self, *_args, **_kwargs):
                return ResponseContext()

            def request_json(self, method, url, **_kwargs):
                if "stream-status" in url:
                    events.append("status")
                    return type("Response", (), {"data": {"data": {"status": "streaming", "message_id": "msg-safe"}}})()
                if "/stop/" in url:
                    events.append("stop")
                    return type("Response", (), {"data": {"data": {"cancelled": False}}})()
                raise AssertionError("unexpected request")

        failed = StageResult(
            case_id=stage.case_id,
            success=False,
            done=False,
            round_count=1,
            tool_event_count=0,
            error_frame_count=0,
            canary_hits={},
            answer_hash="hash",
            answer_length=0,
            metrics={},
            stop_reasons=("timeout",),
            agent_run_id="run-safe",
            assistant_message_id="msg-safe",
        )
        args = _required_args(
            "--apply",
            "--confirm-production",
            "--generation-controls-verified",
            "--allow-account-residue",
        )

        with (
            patch("scripts.context_ladder_eval.build_ladder_plan", return_value=self._plan(stage)),
            patch("scripts.context_ladder_eval.HttpClient", return_value=FakeClient()),
            patch("scripts.context_ladder_eval.ResourceGuard") as guard_cls,
            patch("scripts.context_ladder_eval._verify_live_model"),
            patch("scripts.context_ladder_eval.authenticate", return_value="secret-token"),
            patch(
                "scripts.context_ladder_eval.generate_identity", return_value=("perf-safe", "secret@example", "secret")
            ),
            patch("scripts.context_ladder_eval.consume_sse_lines", return_value=failed),
            patch(
                "scripts.context_ladder_eval._lookup_round_context",
                side_effect=lambda *_args, **_kwargs: events.append("loki") or (None, "round_context_missing"),
            ),
            patch(
                "scripts.context_ladder_eval.cleanup_run",
                side_effect=lambda *_args: (
                    events.append("cleanup") or {"conversations_deleted": 1, "tokens_revoked": 1, "errors": []}
                ),
            ),
            patch.dict("os.environ", {"FUSION_PERF_INTERNAL_AUTH_TOKEN": VALID_INTERNAL_TOKEN}),
        ):
            guard_cls.return_value.check.return_value = []
            guard_cls.return_value.resources_summary.return_value = {}
            result = execute(args)

        self.assertLess(events.index("stop"), events.index("loki"))
        self.assertEqual(events[-1], "cleanup")
        self.assertEqual(result["actual_cost_usd"], "0.006244")
        self.assertIn("active_stream_stop_failed", result["results"][0]["stop_reasons"])
        self.assertEqual(result["cleanup"]["account_rows_retained"], 1)
        self.assertFalse(result["cleanup"]["account_cleanup_supported"])
        serialized = json.dumps(result)
        self.assertNotIn("secret prompt", serialized)
        self.assertNotIn("CANARY", serialized)
        self.assertNotIn("secret@example", serialized)
        self.assertNotIn("secret-token", serialized)

    def test_stage_reserve_blocks_request_that_would_cross_remaining_budget(self):
        stage = self._stage()
        args = _required_args(
            "--apply",
            "--confirm-production",
            "--generation-controls-verified",
            "--allow-account-residue",
            "--max-cost-usd",
            "0.001",
        )

        class FakeClient:
            def open_sse(self, *_args, **_kwargs):
                raise AssertionError("预算不足时不得发起 stage")

        with (
            patch("scripts.context_ladder_eval.build_ladder_plan", return_value=self._plan(stage)),
            patch("scripts.context_ladder_eval.HttpClient", return_value=FakeClient()),
            patch("scripts.context_ladder_eval.ResourceGuard") as guard_cls,
            patch("scripts.context_ladder_eval._verify_live_model"),
            patch("scripts.context_ladder_eval.authenticate", return_value="token") as auth_mock,
            patch(
                "scripts.context_ladder_eval.generate_identity", return_value=("perf-safe", "safe@example", "secret")
            ),
            patch(
                "scripts.context_ladder_eval.cleanup_run",
                return_value={"conversations_deleted": 0, "tokens_revoked": 1, "errors": []},
            ),
            patch.dict("os.environ", {"FUSION_PERF_INTERNAL_AUTH_TOKEN": VALID_INTERNAL_TOKEN}),
        ):
            guard_cls.return_value.check.return_value = []
            guard_cls.return_value.resources_summary.return_value = {}
            result = execute(args)

        self.assertTrue(result["stopped"])
        self.assertIn("cold-50:cost:insufficient_remaining_budget", result["stop_reasons"])
        self.assertEqual(result["results"], [])
        self.assertEqual(auth_mock.call_args.kwargs["internal_auth_token"], VALID_INTERNAL_TOKEN)
        self.assertNotIn(VALID_INTERNAL_TOKEN, json.dumps(result))

    def test_setup_failure_returns_auditable_cleanup_result(self):
        stage = self._stage()
        args = _required_args(
            "--apply",
            "--confirm-production",
            "--generation-controls-verified",
            "--allow-account-residue",
        )
        cleanup_calls = []

        with (
            patch("scripts.context_ladder_eval.build_ladder_plan", return_value=self._plan(stage)),
            patch("scripts.context_ladder_eval.HttpClient", return_value=object()),
            patch("scripts.context_ladder_eval.ResourceGuard") as guard_cls,
            patch("scripts.context_ladder_eval._verify_live_model", side_effect=RunnerError("secret setup detail")),
            patch(
                "scripts.context_ladder_eval.generate_identity", return_value=("perf-safe", "secret@example", "secret")
            ),
            patch(
                "scripts.context_ladder_eval.cleanup_run",
                side_effect=lambda *_args: (
                    cleanup_calls.append(True) or {"conversations_deleted": 0, "tokens_revoked": 0, "errors": []}
                ),
            ),
        ):
            guard_cls.return_value.check.return_value = []
            guard_cls.return_value.resources_summary.return_value = {}
            result = execute(args)

        self.assertEqual(cleanup_calls, [True])
        self.assertTrue(result["stopped"])
        self.assertIn("setup:RunnerError", result["stop_reasons"])
        self.assertEqual(result["cleanup"]["account_rows_retained"], 0)
        self.assertFalse(result["cleanup"]["account_cleanup_supported"])
        self.assertNotIn("secret setup detail", json.dumps(result))

    def test_missing_internal_auth_token_reports_zero_account_rows_retained(self):
        stage = self._stage()
        args = _required_args(
            "--apply",
            "--confirm-production",
            "--generation-controls-verified",
            "--allow-account-residue",
        )

        with (
            patch("scripts.context_ladder_eval.build_ladder_plan", return_value=self._plan(stage)),
            patch("scripts.context_ladder_eval.HttpClient", return_value=object()),
            patch("scripts.context_ladder_eval.ResourceGuard") as guard_cls,
            patch("scripts.context_ladder_eval._verify_live_model"),
            patch("scripts.context_ladder_eval.authenticate") as auth_mock,
            patch(
                "scripts.context_ladder_eval.generate_identity", return_value=("perf-safe", "safe@example", "secret")
            ),
            patch(
                "scripts.context_ladder_eval.cleanup_run",
                return_value={"conversations_deleted": 0, "tokens_revoked": 0, "errors": []},
            ),
            patch.dict("os.environ", {}, clear=True),
        ):
            guard_cls.return_value.check.return_value = []
            guard_cls.return_value.resources_summary.return_value = {}
            result = execute(args)

        auth_mock.assert_not_called()
        self.assertTrue(result["stopped"])
        self.assertIn("setup:InternalAuthGateError", result["stop_reasons"])
        self.assertEqual(result["cleanup"]["account_rows_retained"], 0)

    def test_managed_trim_round_requires_real_trim_telemetry(self):
        stage = StagePlan(
            case_id="managed-trim-90",
            track="managed_multi",
            target_context_tokens=9000,
            ratio=Decimal("0.9"),
            conversation_key="secret",
            prompt="secret",
            expected_canaries={},
            local_estimated_prompt_tokens=7000,
            request_bytes=100,
            planned_input_tokens=9000,
            prompt_hash="hash",
            expected_management_status="trimmed",
        )
        result = StageResult(
            case_id=stage.case_id,
            success=True,
            done=True,
            round_count=1,
            tool_event_count=0,
            error_frame_count=0,
            canary_hits={},
            answer_hash="hash",
            answer_length=1,
            metrics={},
            stop_reasons=(),
            agent_run_id="run-safe",
            assistant_message_id="message-safe",
        )
        payload = {
            "event": "llm_round_context",
            "run_id": "run-safe",
            "assistant_message_id": "message-safe",
            "model_id": "model-a",
            "context_window_tokens": 10000,
            "context_window_status": "known",
            "round_index": 1,
            "round_kind": "agent",
            "estimated_prompt_tokens": 7000,
            "estimator_status": "reused_context_manager",
            "round_prompt_tokens": 6800,
            "round_completion_tokens": 64,
            "request_tool_definition_count": 0,
            "outcome": "success",
            "context_management_status": "trimmed",
            "context_management_estimated_tokens_before": 9000,
            "context_management_estimated_tokens_after": 7000,
            "context_management_target_tokens": 7500,
            "context_management_removed_turns": 1,
            "context_management_removed_messages": 2,
        }

        _safe, reasons = _validate_round_context(
            stage,
            result,
            payload,
            model_id="model-a",
            context_window=10000,
        )

        self.assertEqual(reasons, [])


if __name__ == "__main__":
    unittest.main()
