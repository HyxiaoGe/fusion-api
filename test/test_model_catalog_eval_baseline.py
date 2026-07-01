import json
import unittest

from scripts import model_catalog_eval_baseline as baseline


class ModelCatalogEvalBaselineTests(unittest.TestCase):
    def test_default_scenarios_cover_product_eval_matrix(self):
        scenarios = baseline.select_scenarios()

        self.assertEqual(
            [scenario.scenario_id for scenario in scenarios],
            [
                "basic_chat",
                "cn_factual",
                "coding_reasoning",
                "autonomous_search",
                "no_search_simple",
                "long_answer",
            ],
        )
        expectation_by_id = {scenario.scenario_id: scenario.expected_tool_use for scenario in scenarios}
        self.assertEqual(expectation_by_id["autonomous_search"], "expected")
        self.assertEqual(expectation_by_id["no_search_simple"], "forbidden")

    def test_select_scenarios_filters_in_requested_order(self):
        scenarios = baseline.select_scenarios(["autonomous_search", "basic_chat"])

        self.assertEqual([scenario.scenario_id for scenario in scenarios], ["autonomous_search", "basic_chat"])

    def test_select_scenarios_rejects_unknown_id(self):
        with self.assertRaisesRegex(ValueError, "unknown-scenario"):
            baseline.select_scenarios(["unknown-scenario"])

    def test_select_models_defaults_to_models_not_marked_unhealthy(self):
        models = [
            {"modelId": "deepseek-chat", "provider": "deepseek", "health": {"status": "healthy"}},
            {"modelId": "mimo-v2.5-pro", "provider": "xiaomi", "health": {"status": "unknown"}},
            {"modelId": "mimo-v2-pro", "provider": "xiaomi", "health": {"status": "unhealthy"}},
            {"modelId": "qwen-max-latest", "provider": "qwen", "health": {"status": "healthy"}},
        ]

        selected = baseline.select_models(models)

        self.assertEqual(
            [model["modelId"] for model in selected],
            ["deepseek-chat", "mimo-v2.5-pro", "qwen-max-latest"],
        )

    def test_select_models_can_include_unhealthy_and_filter_ids(self):
        models = [
            {"modelId": "deepseek-chat", "provider": "deepseek", "health": {"status": "healthy"}},
            {"modelId": "mimo-v2-pro", "provider": "xiaomi", "health": {"status": "unhealthy"}},
        ]

        selected = baseline.select_models(
            models,
            include_unhealthy=True,
            model_ids=["mimo-v2-pro"],
        )

        self.assertEqual([model["modelId"] for model in selected], ["mimo-v2-pro"])

    def test_success_result_jsonl_contains_required_fields(self):
        scenario = baseline.select_scenarios(["basic_chat"])[0]
        result = baseline.build_success_result(
            model={"modelId": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek"},
            scenario=scenario,
            transport="nonstream",
            elapsed_ms=1234,
            response_payload={
                "data": {
                    "message": {
                        "content": "你好，我是 Fusion AI。",
                    }
                }
            },
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertEqual(row["model_id"], "deepseek-chat")
        self.assertEqual(row["provider"], "deepseek")
        self.assertEqual(row["scenario_id"], "basic_chat")
        self.assertEqual(row["scenario_category"], "basic")
        self.assertEqual(row["question"], scenario.question)
        self.assertEqual(row["transport"], "nonstream")
        self.assertTrue(row["success"])
        self.assertEqual(row["elapsed_ms"], 1234)
        self.assertIn("Fusion AI", row["answer_preview"])
        self.assertEqual(row["observed_tool_calls"], 0)
        self.assertTrue(row["tool_expectation_met"])
        self.assertEqual(row["quality_flags"], [])
        self.assertIsNone(row["error"])

    def test_failure_result_jsonl_contains_error(self):
        scenario = baseline.select_scenarios(["basic_chat"])[0]
        result = baseline.build_failure_result(
            model={"modelId": "mimo-v2-pro", "provider": "xiaomi", "name": "MiMo"},
            scenario=scenario,
            transport="stream",
            elapsed_ms=321,
            error=RuntimeError("服务商暂时不可用"),
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertEqual(row["model_id"], "mimo-v2-pro")
        self.assertFalse(row["success"])
        self.assertEqual(row["error"]["category"], "unknown_error")
        self.assertEqual(row["error"]["type"], "RuntimeError")
        self.assertIn("服务商暂时不可用", row["error"]["message"])
        self.assertEqual(row["quality_flags"], [])

    def test_parse_sse_events_extracts_json_envelopes(self):
        events = baseline.parse_sse_events(
            [
                "id: 1-0",
                'data: {"chunk_type":"answering","data":{"delta":"你好"}}',
                "",
                'data: {"chunk_type":"agent_event","data":{"type":"tool_call_started","tool_name":"web_search"}}',
                "",
                "data: [DONE]",
                "",
            ]
        )

        self.assertEqual([event["chunk_type"] for event in events], ["answering", "agent_event"])
        self.assertEqual(events[0]["data"]["delta"], "你好")
        self.assertEqual(events[1]["data"]["tool_name"], "web_search")

    def test_stream_result_records_answer_and_tool_observation(self):
        scenario = baseline.select_scenarios(["autonomous_search"])[0]
        result = baseline.build_stream_result(
            model={
                "modelId": "deepseek-chat",
                "provider": "deepseek",
                "name": "DeepSeek",
                "capabilities": {"functionCalling": True, "agentTools": True},
            },
            scenario=scenario,
            elapsed_ms=2500,
            events=[
                {"chunk_type": "agent_event", "data": {"type": "run_started", "message_id": "msg-1"}},
                {"chunk_type": "agent_event", "data": {"type": "tool_call_started", "tool_name": "web_search"}},
                {"chunk_type": "answering", "data": {"delta": "最新消息如下。"}},
                {"chunk_type": "agent_event", "data": {"type": "run_completed", "finish_reason": "stop"}},
            ],
            response_payload={"conversation_id": "conv-1"},
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertTrue(row["success"])
        self.assertEqual(row["scenario_id"], "autonomous_search")
        self.assertEqual(row["conversation_id"], "conv-1")
        self.assertEqual(row["message_id"], "msg-1")
        self.assertEqual(row["observed_tool_calls"], 1)
        self.assertEqual(row["observed_tool_names"], ["web_search"])
        self.assertTrue(row["tool_expectation_met"])
        self.assertIn("最新消息", row["answer_preview"])
        self.assertEqual(row["quality_flags"], [])

    def test_stream_result_flags_reasoning_tag_leak(self):
        scenario = baseline.select_scenarios(["basic_chat"])[0]
        result = baseline.build_stream_result(
            model={"modelId": "MiniMax-M2.7", "provider": "minimax", "name": "MiniMax M2.7"},
            scenario=scenario,
            elapsed_ms=1800,
            events=[
                {"chunk_type": "answering", "data": {"delta": "<think>用户问我能做什么</think>"}},
                {"chunk_type": "answering", "data": {"delta": "我可以帮你整理信息。"}},
            ],
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertTrue(row["success"])
        self.assertIn("reasoning_tag_leak", row["quality_flags"])

    def test_stream_result_counts_repeated_tool_calls_without_duplicate_names(self):
        scenario = baseline.select_scenarios(["autonomous_search"])[0]
        result = baseline.build_stream_result(
            model={
                "modelId": "deepseek-chat",
                "provider": "deepseek",
                "name": "DeepSeek",
                "capabilities": {"functionCalling": True, "agentTools": True},
            },
            scenario=scenario,
            elapsed_ms=2500,
            events=[
                {"chunk_type": "agent_event", "data": {"type": "tool_call_started", "tool_name": "web_search"}},
                {"chunk_type": "agent_event", "data": {"type": "tool_call_started", "tool_name": "web_search"}},
                {"chunk_type": "answering", "data": {"delta": "最新消息如下。"}},
            ],
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertEqual(row["observed_tool_calls"], 2)
        self.assertEqual(row["observed_tool_names"], ["web_search"])

    def test_stream_error_event_builds_failure_result(self):
        scenario = baseline.select_scenarios(["basic_chat"])[0]
        result = baseline.build_stream_result(
            model={"modelId": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek"},
            scenario=scenario,
            elapsed_ms=500,
            events=[
                {
                    "chunk_type": "error",
                    "data": {"code": "provider_error", "message": "服务商异常"},
                }
            ],
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertFalse(row["success"])
        self.assertEqual(row["error"]["category"], "stream_error")
        self.assertIn("服务商异常", row["error"]["message"])

    def test_stream_empty_answer_is_classified(self):
        scenario = baseline.select_scenarios(["basic_chat"])[0]
        result = baseline.build_stream_result(
            model={"modelId": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek"},
            scenario=scenario,
            elapsed_ms=500,
            events=[{"chunk_type": "agent_event", "data": {"type": "run_completed"}}],
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertFalse(row["success"])
        self.assertEqual(row["error"]["category"], "empty_answer")

    def test_tool_expectation_flags_missing_expected_tool(self):
        scenario = baseline.select_scenarios(["autonomous_search"])[0]
        result = baseline.build_stream_result(
            model={
                "modelId": "deepseek-chat",
                "provider": "deepseek",
                "name": "DeepSeek",
                "capabilities": {"functionCalling": True, "agentTools": True},
            },
            scenario=scenario,
            elapsed_ms=1000,
            events=[{"chunk_type": "answering", "data": {"delta": "我直接回答。"}}],
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertFalse(row["tool_expectation_met"])

    def test_tool_expectation_allows_no_tools_for_models_without_agent_tools(self):
        scenario = baseline.select_scenarios(["autonomous_search"])[0]
        result = baseline.build_stream_result(
            model={
                "modelId": "qwen-vl-max",
                "provider": "qwen",
                "name": "Qwen VL Max",
                "capabilities": {"functionCalling": True, "agentTools": False},
            },
            scenario=scenario,
            elapsed_ms=1000,
            events=[{"chunk_type": "answering", "data": {"delta": "需要联网搜索。"}}],
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertFalse(row["agent_tools_supported"])
        self.assertTrue(row["tool_expectation_met"])
        self.assertEqual(row["quality_flags"], ["expected_search_without_agent_tools"])

    def test_build_summary_groups_results_and_failures(self):
        basic = baseline.select_scenarios(["basic_chat"])[0]
        search = baseline.select_scenarios(["autonomous_search"])[0]
        success = baseline.build_stream_result(
            model={"modelId": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek"},
            scenario=basic,
            elapsed_ms=1000,
            events=[{"chunk_type": "answering", "data": {"delta": "你好。"}}],
        )
        failure = baseline.build_failure_result(
            model={"modelId": "mimo-v2.5-pro", "provider": "xiaomi", "name": "MiMo"},
            scenario=search,
            transport="stream",
            elapsed_ms=90000,
            error=TimeoutError("timeout"),
        )

        summary = baseline.build_summary([success, failure])

        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["failure_count"], 1)
        self.assertEqual(summary["failure_types"], {"timeout": 1})
        self.assertEqual(summary["by_model"]["deepseek-chat"]["success_count"], 1)
        self.assertEqual(summary["by_scenario"]["autonomous_search"]["failure_count"], 1)
        self.assertEqual(summary["quality_flags"], {})

    def test_build_summary_counts_quality_flags(self):
        basic = baseline.select_scenarios(["basic_chat"])[0]
        result = baseline.build_stream_result(
            model={"modelId": "MiniMax-M2.7", "provider": "minimax", "name": "MiniMax M2.7"},
            scenario=basic,
            elapsed_ms=1800,
            events=[{"chunk_type": "answering", "data": {"delta": "<think>思考</think>正文"}}],
        )

        summary = baseline.build_summary([result])

        self.assertEqual(summary["quality_flags"], {"reasoning_tag_leak": 1})

    def test_build_summary_counts_agent_tool_gap_flags(self):
        scenario = baseline.select_scenarios(["autonomous_search"])[0]
        result = baseline.build_stream_result(
            model={
                "modelId": "qwen-vl-max",
                "provider": "qwen",
                "name": "Qwen VL Max",
                "capabilities": {"functionCalling": True, "agentTools": False},
            },
            scenario=scenario,
            elapsed_ms=1200,
            events=[{"chunk_type": "answering", "data": {"delta": "我无法联网。"}}],
        )

        summary = baseline.build_summary([result])

        self.assertEqual(summary["tool_expectation_mismatch_count"], 0)
        self.assertEqual(summary["quality_flags"], {"expected_search_without_agent_tools": 1})

    def test_run_eval_calls_result_callback_after_each_item(self):
        scenario = baseline.select_scenarios(["basic_chat"])[0]
        models = [
            {"modelId": "model-a", "provider": "mock", "name": "Model A"},
            {"modelId": "model-b", "provider": "mock", "name": "Model B"},
        ]
        observed: list[baseline.EvalResult] = []
        original_call_chat_send_stream = baseline.call_chat_send_stream

        def fake_call_chat_send_stream(**kwargs):
            model_id = kwargs["model_id"]
            if model_id == "model-b":
                raise TimeoutError("timeout")
            return (
                [{"chunk_type": "answering", "data": {"delta": f"{model_id} 回答"}}],
                {"conversation_id": f"conv-{model_id}"},
            )

        baseline.call_chat_send_stream = fake_call_chat_send_stream
        try:
            results = baseline.run_eval(
                base_url="http://fusion.local",
                auth_token="token",
                models=models,
                scenarios=[scenario],
                on_result=observed.append,
            )
        finally:
            baseline.call_chat_send_stream = original_call_chat_send_stream

        self.assertEqual([result.model_id for result in results], ["model-a", "model-b"])
        self.assertEqual([result.model_id for result in observed], ["model-a", "model-b"])
        self.assertTrue(observed[0].success)
        self.assertFalse(observed[1].success)
        self.assertEqual(observed[1].error["category"], "timeout")


if __name__ == "__main__":
    unittest.main()
