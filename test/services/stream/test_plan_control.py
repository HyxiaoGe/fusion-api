"""Agent 内部计划控制调用测试。"""

import json
import unittest
from unittest.mock import AsyncMock, patch

from app.services.agent.plan_coordinator import PlanCoordinator
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
                    "status": "in_progress",
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
            {"outbound-call": "outbound", "return-call": "return"},
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
                {
                    "id": "return-call",
                    "name": "search_trains",
                    "arguments": '{"origin":"杭州"}',
                    "plan_item_id": "return",
                },
            ],
        )

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

    async def test_missing_initial_tool_coverage_returns_executable_repair_hint(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 1,
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
        self.assertIn("url_read", response["hint"])
        self.assertIn("独立步骤", response["hint"])

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
                "url_read": 1,
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

        rendered_log_arguments = repr(log_info.call_args.args)
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
            "missing_required_initial_tool_coverage",
        )
        self.assertIn(
            "{'web_search': 0, 'url_read': 0}",
            repr(log_info.call_args.args),
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
                "url_read": 1,
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
        third = await process_plan_control_calls(
            tool_calls=invalid,
            coordinator=coordinator,
            emitter=emitter,
        )

        self.assertFalse(first.repair_exhausted)
        self.assertFalse(second.repair_exhausted)
        self.assertFalse(third.repair_exhausted)
        self.assertTrue(coordinator.has_valid_model_plan)
        self.assertEqual(coordinator.source, "observed")
        self.assertEqual(coordinator.reason, "system_fallback")
        self.assertEqual(
            [item["planned_tools"] for item in coordinator.items],
            [["web_search"], ["url_read"], []],
        )
        emitter.plan_snapshot.assert_awaited_once()

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
                _update_call(call_id="plan-2"),
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

    async def test_single_candidate_maps_all_same_tool_calls_and_blocked_candidate_is_ignored(self):
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

        self.assertEqual([call["plan_item_id"] for call in result.external_tool_calls], ["active", "active"])


if __name__ == "__main__":
    unittest.main()
