"""Agent 内部计划控制调用测试。"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services.agent.plan_coordinator import PlanCoordinator, PlanUpdateResult
from app.services.stream.plan_control import process_plan_control_calls


def _update_call(*, call_id: str = "plan-1", arguments: object | None = None) -> dict:
    return {
        "id": call_id,
        "name": "update_plan",
        "arguments": arguments
        if arguments is not None
        else {
            "reason": "先查路线再比较",
            "items": [
                {
                    "id": "route",
                    "title": "查询路线",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["route_compare"],
                },
                {
                    "id": "answer",
                    "title": "比较结果",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["route"],
                    "planned_tools": [],
                },
            ],
        },
    }


class PlanControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_rejection_returns_current_canonical_plan_for_repair(self):
        coordinator = PlanCoordinator(run_id="run-canonical-repair", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "查询路线后回答",
                    "items": [
                        {
                            "id": "route",
                            "title": "查询通勤路线",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["route_compare"],
                        },
                        {
                            "id": "answer",
                            "title": "整理通勤建议",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["route"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tools_started(["route"])

        result = await process_plan_control_calls(
            tool_calls=[
                _update_call(
                    arguments={
                        "reason": "错误替换已执行步骤",
                        "items": [
                            {
                                "id": "answer",
                                "title": "整理通勤建议",
                                "status": "running",
                                "kind": "answer",
                                "depends_on": [],
                                "planned_tools": [],
                            },
                            {
                                "id": "followup",
                                "title": "补充说明",
                                "status": "pending",
                                "kind": "other",
                                "depends_on": ["answer"],
                                "planned_tools": [],
                            },
                        ],
                    }
                )
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        response = json.loads(result.tool_responses["plan-1"])
        self.assertEqual(response["reason"], "attempted_item_removed")
        self.assertEqual(
            response["canonical_plan"],
            [
                {
                    "id": "route",
                    "step": "查询通勤路线",
                    "status": "pending",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["route_compare"],
                },
                {
                    "id": "answer",
                    "step": "整理通勤建议",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["route"],
                    "planned_tools": [],
                },
            ],
        )
        self.assertIn("canonical_plan", response["hint"])

    async def test_mixed_round_applies_plan_before_returning_external_calls(self):
        emitter = AsyncMock()
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        external = {
            "id": "tool-1",
            "name": "route_compare",
            "arguments": {
                "origin": "南景新村",
                "destination": "双子塔",
                "_plan_item_id": "route",
            },
        }

        result = await process_plan_control_calls(
            tool_calls=[_update_call(), external],
            coordinator=coordinator,
            emitter=emitter,
        )

        self.assertEqual(
            result.external_tool_calls,
            [
                {
                    **external,
                    "arguments": {"origin": "南景新村", "destination": "双子塔"},
                    "plan_item_id": "route",
                }
            ],
        )
        self.assertEqual(set(result.tool_responses), {"plan-1"})
        self.assertEqual(json.loads(result.tool_responses["plan-1"])["status"], "accepted")
        self.assertEqual(result.plan_item_ids, {"tool-1": "route"})
        emitter.plan_snapshot.assert_awaited_once()

    async def test_explicit_plan_item_binding_disambiguates_reused_tool_names(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {"id": "outbound", "status": "running", "planned_tools": ["search_trains"]},
            {"id": "return", "status": "pending", "planned_tools": ["search_trains"]},
        ]
        calls = [
            {
                "id": "outbound-call",
                "name": "search_trains",
                "arguments": {"origin": "广州", "_plan_item_id": "outbound"},
            },
            {
                "id": "return-call",
                "name": "search_trains",
                "arguments": '{"origin":"杭州","_plan_item_id":"return"}',
            },
        ]

        result = await process_plan_control_calls(
            tool_calls=calls,
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(
            result.plan_item_ids,
            {"outbound-call": "outbound"},
        )
        self.assertEqual(
            result.external_tool_calls,
            [
                {
                    "id": "outbound-call",
                    "name": "search_trains",
                    "arguments": {"origin": "广州"},
                    "plan_item_id": "outbound",
                },
            ],
        )
        blocked_response = json.loads(result.tool_responses["return-call"])
        self.assertEqual(blocked_response["status"], "not_executed")
        self.assertEqual(blocked_response["reason"], "plan_item_required")

    async def test_invalid_explicit_plan_item_binding_is_blocked_in_on_mode(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        self.assertTrue(coordinator.apply_model_update(_update_call()["arguments"]).accepted)

        result = await process_plan_control_calls(
            tool_calls=[
                {
                    "id": "tool-1",
                    "name": "route_compare",
                    "arguments": {"_plan_item_id": "answer"},
                }
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(result.external_tool_calls, [])
        response = json.loads(result.tool_responses["tool-1"])
        self.assertEqual(response["reason"], "plan_item_required")
        self.assertIn("_plan_item_id", response["hint"])

    async def test_on_mode_blocks_external_call_without_valid_plan_recoverably(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        external = {"id": "tool-1", "name": "route_compare", "arguments": {}}

        result = await process_plan_control_calls(
            tool_calls=[external],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(result.external_tool_calls, [])
        response = json.loads(result.tool_responses["tool-1"])
        self.assertEqual(response["status"], "not_executed")
        self.assertEqual(response["reason"], "plan_required")

    async def test_invalid_control_structure_is_rejected_without_leaking_payload(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="auto")

        result = await process_plan_control_calls(
            tool_calls=[_update_call(arguments="<｜DSML｜>broken")],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        response = result.tool_responses["plan-1"]
        self.assertIn("invalid_plan_structure", response)
        self.assertNotIn("DSML", response)
        self.assertEqual(result.external_tool_calls, [])

    async def test_multi_tool_plan_item_is_rejected_with_structural_repair_hint(self):
        coordinator = PlanCoordinator(run_id="run-multi-tool-repair", mode="on")

        result = await process_plan_control_calls(
            tool_calls=[
                _update_call(
                    arguments={
                        "reason": "错误地让一个步骤执行两个工具",
                        "items": [
                            {
                                "id": "research",
                                "title": "搜索并读取",
                                "status": "running",
                                "kind": "search",
                                "depends_on": [],
                                "planned_tools": ["web_search", "url_read"],
                            },
                            {
                                "id": "answer",
                                "title": "整理回答",
                                "status": "pending",
                                "kind": "answer",
                                "depends_on": ["research"],
                                "planned_tools": [],
                            },
                        ],
                    }
                )
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        response = json.loads(result.tool_responses["plan-1"])
        self.assertEqual(response["status"], "rejected")
        self.assertEqual(response["reason"], "multiple_tools_per_item")
        self.assertIn("planned_tools 最多声明一个", response["hint"])
        self.assertIn("拆成", response["hint"])
        self.assertFalse(coordinator.has_valid_model_plan)

    async def test_missing_initial_tool_coverage_returns_executable_repair_hint(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 2,
            }
        )

        result = await process_plan_control_calls(
            tool_calls=[
                _update_call(
                    arguments={
                        "explanation": "先搜索再回答",
                        "plan": [
                            {
                                "id": "search",
                                "step": "搜索资料",
                                "status": "in_progress",
                                "planned_tools": ["web_search"],
                            },
                            {
                                "id": "answer",
                                "step": "整理回答",
                                "status": "pending",
                                "planned_tools": [],
                            },
                        ],
                    }
                )
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        response = json.loads(result.tool_responses["plan-1"])
        self.assertEqual(response["reason"], "missing_required_initial_tool_coverage")
        self.assertIn("web_search", response["hint"])
        self.assertIn("url_read×2", response["hint"])
        self.assertIn("独立步骤", response["hint"])
        self.assertNotIn("同一个 url_read 步骤", response["hint"])
        self.assertIn("每个 url_read 计划项负责一个独立来源任务", response["hint"])
        self.assertIn("仅当服务端将同一任务保持为 retryable/running 时", response["hint"])
        self.assertIn("跨轮重试", response["hint"])
        self.assertNotIn("每个 url_read 步骤只读取一个来源", response["hint"])

    async def test_missing_answer_phase_returns_executable_repair_hint(self):
        coordinator = PlanCoordinator(run_id="run-missing-answer", mode="on")

        result = await process_plan_control_calls(
            tool_calls=[
                _update_call(
                    arguments={
                        "reason": "只声明查询阶段",
                        "items": [
                            {
                                "id": "search",
                                "title": "搜索资料",
                                "status": "running",
                                "kind": "search",
                                "depends_on": [],
                                "planned_tools": ["web_search"],
                            },
                            {
                                "id": "read",
                                "title": "核验来源",
                                "status": "pending",
                                "kind": "read",
                                "depends_on": ["search"],
                                "planned_tools": ["url_read"],
                            },
                        ],
                    }
                )
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        response = json.loads(result.tool_responses["plan-1"])
        self.assertEqual(response["reason"], "missing_answer_phase")
        self.assertIn("answer", response["hint"])
        self.assertIn("synthesis", response["hint"])
        self.assertIn("planned_tools", response["hint"])

    async def test_rejection_log_only_records_server_whitelisted_tool_counts(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 2,
            }
        )
        secret_marker = "用户私密内容\\n伪造日志"
        arguments = {
            "reason": "错误计划",
            "items": [
                {
                    "id": "search",
                    "title": "搜索",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["web_search", secret_marker],
                },
                {
                    "id": "answer",
                    "title": "回答",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["search"],
                    "planned_tools": [],
                },
            ],
        }

        with patch("app.services.stream.plan_control.logger.info") as log_info:
            await process_plan_control_calls(
                tool_calls=[_update_call(arguments=arguments)],
                coordinator=coordinator,
                emitter=AsyncMock(),
            )

        rendered_log_arguments = repr([call.args for call in log_info.call_args_list])
        self.assertNotIn(secret_marker, rendered_log_arguments)
        self.assertIn("{'web_search': 1, 'url_read': 0}", rendered_log_arguments)

    async def test_rejection_log_uses_same_distinct_owner_rule_as_validation(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 1,
            }
        )
        arguments = {
            "reason": "把搜索和读取混在同一步",
            "items": [
                {
                    "id": "mixed",
                    "title": "搜索并读取",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["web_search", "url_read"],
                },
                {
                    "id": "answer",
                    "title": "回答",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["mixed"],
                    "planned_tools": [],
                },
            ],
        }

        with patch("app.services.stream.plan_control.logger.info") as log_info:
            result = await process_plan_control_calls(
                tool_calls=[_update_call(arguments=arguments)],
                coordinator=coordinator,
                emitter=AsyncMock(),
            )

        self.assertEqual(
            json.loads(result.tool_responses["plan-1"])["reason"],
            "multiple_tools_per_item",
        )
        self.assertIn(
            "{'web_search': 0, 'url_read': 0}",
            repr([call.args for call in log_info.call_args_list]),
        )

    async def test_off_mode_does_not_gate_external_tools(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="off")
        external = {"id": "tool-1", "name": "web_search", "arguments": {"query": "深圳天气"}}

        result = await process_plan_control_calls(
            tool_calls=[external],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(result.external_tool_calls, [external])
        self.assertEqual(result.tool_responses, {})

    async def test_invalid_control_and_external_gate_only_consume_one_repair_per_round(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        calls = [
            _update_call(arguments="<｜DSML｜>broken"),
            _update_call(call_id="plan-2", arguments={"broken": True}),
            {"id": "tool-1", "name": "web_search", "arguments": {"query": "深圳天气"}},
        ]

        first = await process_plan_control_calls(tool_calls=calls, coordinator=coordinator, emitter=AsyncMock())
        second = await process_plan_control_calls(tool_calls=calls, coordinator=coordinator, emitter=AsyncMock())
        third = await process_plan_control_calls(tool_calls=calls, coordinator=coordinator, emitter=AsyncMock())

        self.assertFalse(first.repair_exhausted)
        self.assertFalse(second.repair_exhausted)
        self.assertTrue(third.repair_exhausted)
        self.assertEqual(coordinator.repair_attempt_count, 3)

    async def test_deep_research_adopts_safe_fallback_plan_after_repair_exhaustion(self):
        coordinator = PlanCoordinator(run_id="run-research", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 2,
            }
        )
        emitter = AsyncMock()
        invalid = [_update_call(arguments={"broken": True})]

        first = await process_plan_control_calls(
            tool_calls=invalid,
            coordinator=coordinator,
            emitter=emitter,
        )
        second = await process_plan_control_calls(
            tool_calls=invalid,
            coordinator=coordinator,
            emitter=emitter,
        )
        with patch("app.services.stream.plan_control.logger.info") as log_info:
            third = await process_plan_control_calls(
                tool_calls=invalid,
                coordinator=coordinator,
                emitter=emitter,
            )

        self.assertFalse(first.repair_exhausted)
        self.assertFalse(second.repair_exhausted)
        self.assertFalse(third.repair_exhausted)
        self.assertEqual((third.repair_attempt_count, third.repair_attempt_limit), (3, 3))
        self.assertEqual(log_info.call_args.args[4:6], (True, True))
        self.assertTrue(coordinator.has_valid_model_plan)
        self.assertEqual(coordinator.source, "model")
        self.assertEqual(coordinator.reason, "system_fallback")
        self.assertEqual(
            [item["planned_tools"] for item in coordinator.items],
            [["web_search"], ["url_read"], ["url_read"], []],
        )
        search_id = coordinator.items[0]["id"]
        read_ids = [item["id"] for item in coordinator.items[1:3]]
        self.assertEqual(
            [item["depends_on"] for item in coordinator.items],
            [[], [search_id], [search_id], read_ids],
        )
        emitter.plan_snapshot.assert_awaited_once()
        emitted_snapshot = emitter.plan_snapshot.await_args.kwargs
        self.assertEqual(emitted_snapshot["source"], "model")
        self.assertEqual(emitted_snapshot["reason"], "system_fallback")

    async def test_existing_plan_tolerates_five_consecutive_repair_rounds(self):
        coordinator = PlanCoordinator(run_id="run-existing-plan", mode="on")
        self.assertTrue(coordinator.apply_model_update(_update_call()["arguments"]).accepted)
        with patch.object(
            coordinator,
            "apply_model_update",
            return_value=PlanUpdateResult(False, "multiple_running_items"),
        ):
            attempts = [
                await process_plan_control_calls(
                    tool_calls=[_update_call()],
                    coordinator=coordinator,
                    emitter=AsyncMock(),
                )
                for _ in range(5)
            ]

        self.assertTrue(all(not attempt.repair_exhausted for attempt in attempts[:4]))
        self.assertTrue(attempts[4].repair_exhausted)
        self.assertEqual(coordinator.repair_attempt_count, 5)

    async def test_existing_plan_keeps_default_limit_for_unrelated_rejection(self):
        coordinator = PlanCoordinator(run_id="run-strict-plan", mode="on")
        self.assertTrue(coordinator.apply_model_update(_update_call()["arguments"]).accepted)

        attempts = [
            await process_plan_control_calls(
                tool_calls=[_update_call(arguments={"broken": True})],
                coordinator=coordinator,
                emitter=AsyncMock(),
            )
            for _ in range(3)
        ]

        self.assertTrue(attempts[2].repair_exhausted)
        self.assertEqual(
            (attempts[2].repair_attempt_count, attempts[2].repair_attempt_limit),
            (3, 3),
        )

    async def test_terminal_status_regression_uses_tolerated_limit(self):
        coordinator = PlanCoordinator(run_id="run-terminal-drift", mode="on")
        payload = _update_call()["arguments"]
        self.assertTrue(coordinator.apply_model_update(payload).accepted)
        coordinator.mark_tool_results({"route": "completed"})
        with patch.object(
            coordinator,
            "apply_model_update",
            return_value=PlanUpdateResult(False, "terminal_status_regression"),
        ):
            attempts = [
                await process_plan_control_calls(
                    tool_calls=[_update_call(arguments=payload)],
                    coordinator=coordinator,
                    emitter=AsyncMock(),
                )
                for _ in range(5)
            ]

        self.assertTrue(all(not attempt.repair_exhausted for attempt in attempts[:4]))
        self.assertTrue(attempts[4].repair_exhausted)
        self.assertEqual(
            (attempts[4].repair_attempt_count, attempts[4].repair_attempt_limit),
            (5, 5),
        )

    async def test_accepted_control_makes_same_round_non_repairable(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")

        result = await process_plan_control_calls(
            tool_calls=[
                _update_call(),
                _update_call(call_id="plan-2", arguments={"broken": True}),
                {
                    "id": "tool-1",
                    "name": "route_compare",
                    "arguments": {"_plan_item_id": "route"},
                },
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(coordinator.repair_attempt_count, 0)
        self.assertFalse(result.repair_exhausted)
        self.assertEqual(result.external_tool_calls[0]["plan_item_id"], "route")

    async def test_on_mode_requires_explicit_binding_even_with_single_candidate(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        accepted = coordinator.apply_model_update(
            {
                "explanation": "先查询路线，再给出建议",
                "plan": [
                    {
                        "id": "route",
                        "step": "查询路线",
                        "status": "in_progress",
                        "planned_tools": ["route_compare"],
                    },
                    {
                        "id": "answer",
                        "step": "给出建议",
                        "status": "pending",
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertTrue(accepted.accepted)

        result = await process_plan_control_calls(
            tool_calls=[
                {
                    "id": "tool-1",
                    "name": "route_compare",
                    "arguments": {},
                }
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(result.external_tool_calls, [])
        self.assertEqual(result.plan_item_ids, {})
        response = json.loads(result.tool_responses["tool-1"])
        self.assertEqual(response["status"], "not_executed")
        self.assertEqual(response["reason"], "plan_item_required")
        self.assertIn("_plan_item_id", response["hint"])
        self.assertEqual(coordinator.repair_attempt_count, 1)

    async def test_on_mode_blocks_unmapped_tool_until_model_declares_exact_plan_item(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        accepted = coordinator.apply_model_update(
            {
                "explanation": "查询路线并给出建议",
                "plan": [
                    {"step": "查询路线", "status": "in_progress"},
                    {"step": "给出建议", "status": "pending"},
                ],
            }
        )
        self.assertTrue(accepted.accepted)

        result = await process_plan_control_calls(
            tool_calls=[{"id": "tool-1", "name": "route_compare", "arguments": {}}],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(result.external_tool_calls, [])
        response = json.loads(result.tool_responses["tool-1"])
        self.assertEqual(response["status"], "not_executed")
        self.assertEqual(response["reason"], "plan_item_required")
        self.assertIn("planned_tools", response["hint"])
        self.assertEqual(coordinator.repair_attempt_count, 1)
        self.assertFalse(result.repair_exhausted)

    async def test_auto_mode_keeps_unmapped_tools_as_compatibility_path(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="auto")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "explanation": "查询路线并给出建议",
                    "plan": [
                        {"step": "查询路线", "status": "in_progress"},
                        {"step": "给出建议", "status": "pending"},
                    ],
                }
            ).accepted
        )
        external = {"id": "tool-1", "name": "route_compare", "arguments": {}}

        result = await process_plan_control_calls(
            tool_calls=[external],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(result.external_tool_calls, [external])
        self.assertEqual(result.tool_responses, {})

    async def test_update_limit_rejection_keeps_existing_plan_and_does_not_consume_repair(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on", max_valid_updates=1)
        self.assertTrue(coordinator.apply_model_update(_update_call()["arguments"]).accepted)

        result = await process_plan_control_calls(
            tool_calls=[
                _update_call(
                    call_id="plan-2",
                    arguments={
                        **_update_call()["arguments"],
                        "items": [
                            _update_call()["arguments"]["items"][0],
                            {
                                **_update_call()["arguments"]["items"][1],
                                "title": "比较结果新版",
                            },
                        ],
                    },
                ),
                {
                    "id": "tool-1",
                    "name": "route_compare",
                    "arguments": {"_plan_item_id": "route"},
                },
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(
            json.loads(result.tool_responses["plan-2"])["reason"],
            "control_update_limit_reached",
        )
        self.assertEqual(coordinator.repair_attempt_count, 0)
        self.assertFalse(result.repair_exhausted)
        self.assertEqual(result.external_tool_calls[0]["plan_item_id"], "route")

    async def test_multiple_candidates_for_same_tool_are_never_guessed(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="auto")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {"id": "search-a", "status": "pending", "planned_tools": ["web_search"]},
            {"id": "search-b", "status": "pending", "planned_tools": ["web_search"]},
        ]
        calls = [
            {"id": "a", "name": "web_search", "arguments": {"query": "A"}},
            {"id": "b", "name": "web_search", "arguments": {"query": "B"}},
        ]

        mapped = await process_plan_control_calls(
            tool_calls=calls,
            coordinator=coordinator,
            emitter=AsyncMock(),
        )
        single = await process_plan_control_calls(
            tool_calls=[calls[0]],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertTrue(all("plan_item_id" not in call for call in mapped.external_tool_calls))
        self.assertNotIn("plan_item_id", single.external_tool_calls[0])

    async def test_single_candidate_maps_first_call_and_rejects_duplicate_binding(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="auto")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {"id": "active", "status": "pending", "planned_tools": ["web_search"]},
            {"id": "blocked", "status": "blocked", "planned_tools": ["web_search"]},
        ]
        calls = [
            {"id": "a", "name": "web_search", "arguments": {"query": "A"}},
            {"id": "b", "name": "web_search", "arguments": {"query": "B"}},
        ]

        result = await process_plan_control_calls(
            tool_calls=calls,
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual([call["plan_item_id"] for call in result.external_tool_calls], ["active"])
        rejected = json.loads(result.tool_responses["b"])
        self.assertEqual(rejected["status"], "not_executed")
        self.assertEqual(rejected["reason"], "plan_item_already_bound")
        self.assertIn("同一批次", rejected["hint"])
        self.assertIn("后续轮次", rejected["hint"])
        self.assertIn("retryable/running", rejected["hint"])
        self.assertNotIn("同一计划项只允许一次真实工具调用", rejected["hint"])

    async def test_same_plan_item_rejects_second_call_in_batch_but_allows_next_round_retry(self):
        coordinator = PlanCoordinator(run_id="run-cross-round-retry", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "执行一个可重试的独立搜索任务",
                    "items": [
                        {
                            "id": "search",
                            "title": "搜索可靠来源",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理答案",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        first_batch = [
            {
                "id": "search-first",
                "name": "web_search",
                "arguments": {"query": "第一种表达", "_plan_item_id": "search"},
            },
            {
                "id": "search-duplicate",
                "name": "web_search",
                "arguments": {"query": "第二种表达", "_plan_item_id": "search"},
            },
        ]

        first_round = await process_plan_control_calls(
            tool_calls=first_batch,
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(
            [call["id"] for call in first_round.external_tool_calls],
            ["search-first"],
        )
        duplicate = json.loads(first_round.tool_responses["search-duplicate"])
        self.assertEqual(duplicate["reason"], "plan_item_already_bound")

        coordinator.mark_tools_started(["search"])
        retryable = coordinator.mark_tool_results({"search": "running"})
        self.assertEqual(retryable["items"][0]["status"], "running")

        second_round = await process_plan_control_calls(
            tool_calls=[
                {
                    "id": "search-retry",
                    "name": "web_search",
                    "arguments": {"query": "修正后的表达", "_plan_item_id": "search"},
                }
            ],
            coordinator=coordinator,
            emitter=AsyncMock(),
        )

        self.assertEqual(
            [call["id"] for call in second_round.external_tool_calls],
            ["search-retry"],
        )
        self.assertNotIn("search-retry", second_round.tool_responses)
        coordinator.mark_tools_started(["search"])
        completed = coordinator.mark_tool_results({"search": "completed"})
        self.assertEqual(
            {item["id"]: item["status"] for item in completed["items"]},
            {"search": "completed", "answer": "pending"},
        )


if __name__ == "__main__":
    unittest.main()
