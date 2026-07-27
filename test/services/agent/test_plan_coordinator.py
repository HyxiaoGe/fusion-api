"""模型计划控制器测试。"""

import unittest

from app.services.agent.plan_coordinator import PlanCoordinator


class PlanCoordinatorTests(unittest.TestCase):
    def test_valid_model_plan_owns_revision_and_generic_fields(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "准备查询路线并比较结果",
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
                        "title": "比较并给出建议",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["route"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.snapshot["mode"], "on")
        self.assertEqual(result.snapshot["source"], "model")
        self.assertEqual(result.snapshot["revision"], 1)
        self.assertEqual(result.snapshot["reason"], "model_update")
        self.assertEqual(result.snapshot["items"][1]["depends_on"], ["route"])
        self.assertEqual(result.snapshot["items"][0]["planned_tools"], ["route_compare"])
        self.assertTrue(coordinator.has_valid_model_plan)

    def test_accepts_industry_update_plan_shape_and_normalizes_it_to_internal_contract(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")

        result = coordinator.apply_model_update(
            {
                "explanation": "先查询路线，再比较并给出建议",
                "plan": [
                    {
                        "step": "查询通勤路线",
                        "status": "in_progress",
                        "planned_tools": ["route_compare"],
                    },
                    {
                        "step": "比较方案并给出推荐",
                        "status": "pending",
                    },
                ],
            }
        )

        self.assertTrue(result.accepted)
        self.assertEqual([item["id"] for item in result.snapshot["items"]], ["step-1", "step-2"])
        self.assertEqual([item["status"] for item in result.snapshot["items"]], ["running", "pending"])
        self.assertEqual([item["kind"] for item in result.snapshot["items"]], ["other", "answer"])
        self.assertEqual(result.snapshot["items"][1]["depends_on"], ["step-1"])
        self.assertEqual(result.snapshot["items"][0]["planned_tools"], ["route_compare"])

    def test_explicit_plan_item_binding_can_disambiguate_reused_tool_names(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {"id": "outbound", "status": "running", "planned_tools": ["search_trains"]},
            {"id": "return", "status": "pending", "planned_tools": ["search_trains"]},
        ]

        self.assertEqual(
            coordinator.plan_item_id_for_tool("search_trains", requested_item_id="outbound"),
            "outbound",
        )
        self.assertEqual(
            coordinator.plan_item_id_for_tool("search_trains", requested_item_id="return"),
            "return",
        )
        self.assertIsNone(
            coordinator.plan_item_id_for_tool("search_trains", requested_item_id="missing"),
        )

    def test_numeric_string_plan_ids_remain_stable_for_tool_binding(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")

        result = coordinator.apply_model_update(
            {
                "explanation": "依次查询并整理",
                "plan": [
                    {
                        "id": "1",
                        "step": "查询高铁",
                        "status": "in_progress",
                        "planned_tools": ["search_trains"],
                    },
                    {
                        "id": "2",
                        "step": "整理建议",
                        "status": "pending",
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(result.accepted)
        self.assertEqual([item["id"] for item in result.snapshot["items"]], ["1", "2"])
        self.assertEqual(
            coordinator.plan_item_id_for_tool("search_trains", requested_item_id="1"),
            "1",
        )

    def test_compatibility_shape_does_not_allow_model_to_assert_initial_terminal_status(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")

        result = coordinator.apply_model_update(
            {
                "plan": [
                    {"step": "理解任务", "status": "completed"},
                    {"step": "整理回答", "status": "in_progress"},
                ],
            }
        )

        self.assertTrue(result.accepted)
        self.assertEqual([item["status"] for item in result.snapshot["items"]], ["pending", "running"])

    def test_invalid_dependency_is_recoverable_and_does_not_mutate_plan(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "无效依赖",
                "items": [
                    {
                        "id": "answer",
                        "title": "回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["missing"],
                        "planned_tools": [],
                    },
                    {
                        "id": "finish",
                        "title": "完成",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["answer"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "unknown_dependency")
        self.assertEqual(coordinator.revision, 0)
        self.assertFalse(coordinator.has_valid_model_plan)

    def test_control_updates_have_independent_rate_limit(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="auto", max_valid_updates=1)
        payload = {
            "reason": "初始计划",
            "items": [
                {
                    "id": "answer",
                    "title": "回答",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": [],
                    "planned_tools": [],
                },
                {
                    "id": "finish",
                    "title": "完成",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["answer"],
                    "planned_tools": [],
                },
            ],
        }

        self.assertTrue(coordinator.apply_model_update(payload).accepted)
        rejected = coordinator.apply_model_update(payload)

        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "control_update_limit_reached")
        self.assertEqual(coordinator.revision, 1)

    def test_observed_plan_is_compatible_but_cannot_overwrite_model_plan(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="auto")
        coordinator.adopt_observed_items(
            [{"id": "tool", "title": "调用工具", "status": "running", "kind": "other"}],
            reason="legacy_observed",
        )
        self.assertEqual(coordinator.source, "observed")

        coordinator.apply_model_update(
            {
                "reason": "模型计划",
                "items": [
                    {
                        "id": "answer",
                        "title": "回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "finish",
                        "title": "完成",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["answer"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        coordinator.adopt_observed_items(
            [{"id": "tool", "title": "错误覆盖", "status": "completed", "kind": "other"}],
            reason="legacy_observed",
        )

        self.assertEqual(coordinator.source, "model")
        self.assertEqual([item["id"] for item in coordinator.items], ["answer", "finish"])

    def test_rejects_dependency_cycle_and_multiple_running_items(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        cycle = coordinator.apply_model_update(
            {
                "reason": "循环",
                "items": [
                    {
                        "id": "a",
                        "title": "A",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": ["b"],
                        "planned_tools": [],
                    },
                    {
                        "id": "b",
                        "title": "B",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": ["a"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertEqual(cycle.reason, "dependency_cycle")

        multiple_running = coordinator.apply_model_update(
            {
                "reason": "多个运行项",
                "items": [
                    {
                        "id": "a",
                        "title": "A",
                        "status": "running",
                        "kind": "other",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "b",
                        "title": "B",
                        "status": "running",
                        "kind": "other",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertEqual(multiple_running.reason, "multiple_running_items")

    def test_model_plan_uses_new_plan_id_and_terminal_status_cannot_regress(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.adopt_observed_items([{"id": "legacy", "title": "旧计划", "status": "running", "kind": "other"}])
        payload = {
            "reason": "模型计划",
            "items": [
                {
                    "id": "search",
                    "title": "搜索",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["web_search"],
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
        first = coordinator.apply_model_update(payload)
        self.assertEqual(first.snapshot["plan_id"], "plan-run-1-model")
        coordinator.mark_tool_results({"search": "completed"})
        payload["items"][0]["status"] = "running"
        self.assertEqual(coordinator.apply_model_update(payload).reason, "terminal_status_regression")

        terminal = coordinator.terminalize("limit_reached")
        self.assertNotIn("running", [item["status"] for item in terminal["items"]])
        self.assertNotIn("pending", [item["status"] for item in terminal["items"]])

    def test_model_cannot_assert_or_remove_server_owned_terminal_status(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        payload = {
            "reason": "模型计划",
            "items": [
                {
                    "id": "search",
                    "title": "搜索",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["web_search"],
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
        self.assertTrue(coordinator.apply_model_update(payload).accepted)
        coordinator.mark_tool_results({"search": "completed"})

        asserted = {
            **payload,
            "items": [
                {**payload["items"][0], "status": "completed"},
                {**payload["items"][1], "status": "completed"},
            ],
        }
        self.assertEqual(coordinator.apply_model_update(asserted).reason, "unproven_terminal_status")

        removed = {
            "reason": "删除已完成项",
            "items": [
                {
                    "id": "answer",
                    "title": "回答",
                    "status": "running",
                    "kind": "answer",
                    "depends_on": [],
                    "planned_tools": [],
                },
                {
                    "id": "followup",
                    "title": "补充",
                    "status": "pending",
                    "kind": "other",
                    "depends_on": ["answer"],
                    "planned_tools": [],
                },
            ],
        }
        self.assertEqual(coordinator.apply_model_update(removed).reason, "terminal_item_removed")

    def test_terminal_item_metadata_cannot_be_rewritten_after_real_tool_completion(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        payload = {
            "reason": "查询航班后回答",
            "items": [
                {
                    "id": "flight",
                    "title": "查询航班",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["search_flights"],
                },
                {
                    "id": "answer",
                    "title": "整理建议",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["flight"],
                    "planned_tools": [],
                },
            ],
        }
        self.assertTrue(coordinator.apply_model_update(payload).accepted)
        coordinator.mark_tool_results({"flight": "completed"})
        before = coordinator.snapshot()

        rewritten = coordinator.apply_model_update(
            {
                "explanation": "改写已完成事实",
                "plan": [
                    {
                        "id": "flight",
                        "step": "已核验酒店价格",
                        "status": "completed",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["search_hotels"],
                    },
                    {
                        "id": "answer",
                        "step": "整理建议",
                        "status": "in_progress",
                        "kind": "answer",
                        "depends_on": ["flight"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(rewritten.accepted)
        self.assertEqual(rewritten.reason, "terminal_item_mutated")
        self.assertEqual(coordinator.snapshot()["items"], before["items"])

    def test_terminal_item_omitted_optional_metadata_reuses_server_owned_values(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "explanation": "查询航班后回答",
                    "plan": [
                        {
                            "id": "flight",
                            "step": "查询航班",
                            "status": "in_progress",
                            "kind": "search",
                            "planned_tools": ["search_flights"],
                        },
                        {
                            "id": "answer",
                            "step": "整理建议",
                            "status": "pending",
                            "kind": "answer",
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"flight": "completed"})

        updated = coordinator.apply_model_update(
            {
                "explanation": "继续整理",
                "plan": [
                    {
                        "id": "flight",
                        "step": "查询航班",
                        "status": "completed",
                    },
                    {
                        "id": "answer",
                        "step": "整理建议",
                        "status": "in_progress",
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(updated.accepted)
        self.assertEqual(updated.snapshot["items"][0]["kind"], "search")
        self.assertEqual(updated.snapshot["items"][0]["planned_tools"], ["search_flights"])

    def test_terminalize_uses_outcome_and_kind_aware_status_mapping(self):
        def coordinator_for(items):
            coordinator = PlanCoordinator(run_id="run-1", mode="on")
            coordinator.source = "model"
            coordinator.revision = 1
            coordinator.items = items
            return coordinator

        normal = coordinator_for(
            [
                {"id": "blocked", "title": "阻塞", "status": "blocked", "kind": "search"},
                {"id": "search", "title": "搜索", "status": "pending", "kind": "search"},
                {"id": "answer", "title": "回答", "status": "pending", "kind": "answer"},
                {"id": "reasoning", "title": "分析", "status": "running", "kind": "reasoning"},
                {"id": "retry", "title": "重试", "status": "running", "kind": "search"},
            ]
        ).terminalize("stop", has_final_answer=True)
        self.assertEqual(
            [item["status"] for item in normal["items"]],
            ["blocked", "skipped", "completed", "completed", "blocked"],
        )

        failed = coordinator_for(
            [
                {"id": "running", "title": "运行", "status": "running", "kind": "search"},
                {"id": "pending", "title": "等待", "status": "pending", "kind": "answer"},
            ]
        ).terminalize("failed")
        self.assertEqual([item["status"] for item in failed["items"]], ["failed", "skipped"])

        empty_answer = coordinator_for(
            [
                {"id": "reasoning", "title": "分析", "status": "running", "kind": "reasoning"},
                {"id": "answer", "title": "回答", "status": "pending", "kind": "answer"},
            ]
        ).terminalize("stop", has_final_answer=False)
        self.assertEqual([item["status"] for item in empty_answer["items"]], ["blocked", "blocked"])

    def test_six_valid_updates_and_two_repairs_are_independent(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        payload = {
            "reason": "更新",
            "items": [
                {
                    "id": "search",
                    "title": "搜索",
                    "status": "running",
                    "kind": "search",
                    "depends_on": [],
                    "planned_tools": ["web_search"],
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
        for _ in range(6):
            self.assertTrue(coordinator.apply_model_update(payload).accepted)
        self.assertEqual(coordinator.apply_model_update(payload).reason, "control_update_limit_reached")
        self.assertFalse(coordinator.record_repair_round())
        self.assertFalse(coordinator.record_repair_round())
        self.assertTrue(coordinator.record_repair_round())


if __name__ == "__main__":
    unittest.main()
