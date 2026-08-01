"""模型计划控制器测试。"""

import unittest

from app.services.agent.plan_coordinator import PlanCoordinator


class PlanCoordinatorTests(unittest.TestCase):
    def test_rejects_execution_branch_that_depends_on_answer_phase(self):
        coordinator = PlanCoordinator(run_id="run-answer-before-search", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "错误地先回答再搜索",
                "items": [
                    {
                        "id": "answer",
                        "title": "先给出回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "search",
                        "title": "随后搜索",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": ["answer"],
                        "planned_tools": ["web_search"],
                    },
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "execution_depends_on_answer_phase")
        self.assertEqual(coordinator.revision, 0)

    def test_rejects_answer_phase_with_planned_tool(self):
        coordinator = PlanCoordinator(run_id="run-tool-bearing-answer", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "错误地让回答阶段执行搜索",
                "items": [
                    {
                        "id": "answer",
                        "title": "搜索并回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "note",
                        "title": "补充说明",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": ["answer"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "answer_phase_has_tools")
        self.assertEqual(coordinator.revision, 0)

    def test_rejects_plan_without_terminal_answer_phase(self):
        coordinator = PlanCoordinator(run_id="run-non-terminal-answer", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "回答之后仍保留普通步骤",
                "items": [
                    {
                        "id": "answer",
                        "title": "整理回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "note",
                        "title": "补充说明",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": ["answer"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "missing_terminal_answer_phase")

    def test_rejects_terminal_answer_that_does_not_cover_all_execution_branches(self):
        coordinator = PlanCoordinator(run_id="run-uncovered-branch", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "遗漏一个执行分支",
                "items": [
                    {
                        "id": "search-primary",
                        "title": "搜索主资料",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "search-secondary",
                        "title": "搜索对照资料",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "answer",
                        "title": "整理回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["search-primary"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "uncovered_execution_branch")

    def test_accepts_normal_route_and_deep_research_terminal_dags(self):
        cases = (
            (
                "route",
                [
                    {
                        "id": "route",
                        "title": "比较通勤路线",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": [],
                        "planned_tools": ["route_compare"],
                    },
                    {
                        "id": "answer",
                        "title": "给出路线建议",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["route"],
                        "planned_tools": [],
                    },
                ],
            ),
            (
                "deep-research",
                [
                    {
                        "id": "search",
                        "title": "搜索可靠来源",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "read-primary",
                        "title": "读取首个来源",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "read-secondary",
                        "title": "读取独立来源",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "answer",
                        "title": "综合研究结论",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read-primary", "read-secondary"],
                        "planned_tools": [],
                    },
                ],
            ),
        )

        for name, items in cases:
            with self.subTest(name=name):
                coordinator = PlanCoordinator(run_id=f"run-{name}", mode="on")
                result = coordinator.apply_model_update(
                    {
                        "reason": "建立可执行计划",
                        "items": items,
                    }
                )

                self.assertTrue(result.accepted)
                self.assertTrue(coordinator.has_valid_model_plan)

    def test_rejects_planned_tool_that_was_not_announced(self):
        coordinator = PlanCoordinator(
            run_id="run-announced-tools",
            mode="on",
            allowed_tool_names=frozenset({"route_compare"}),
        )

        result = coordinator.apply_model_update(
            {
                "items": [
                    {
                        "id": "search",
                        "step": "搜索地点",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "answer",
                        "step": "整理答案",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["search"],
                        "planned_tools": [],
                    },
                ]
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "unannounced_planned_tool")
        self.assertEqual(coordinator.revision, 0)
        self.assertEqual(coordinator.valid_update_count, 0)

    def test_identical_canonical_plan_is_idempotent(self):
        coordinator = PlanCoordinator(
            run_id="run-idempotent-plan",
            mode="on",
            max_valid_updates=2,
            allowed_tool_names=frozenset({"route_compare"}),
            required_initial_tool_counts={"route_compare": 1},
        )
        payload = {
            "items": [
                {
                    "id": "route",
                    "step": "比较通勤路线",
                    "status": "pending",
                    "kind": "other",
                    "depends_on": [],
                    "planned_tools": ["route_compare"],
                },
                {
                    "id": "answer",
                    "step": "给出推荐选择",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["route"],
                    "planned_tools": [],
                },
            ]
        }

        first = coordinator.apply_model_update(payload)
        coordinator.repair_attempt_count = 2
        second = coordinator.apply_model_update(payload)
        third = coordinator.apply_model_update(payload)

        self.assertTrue(first.accepted)
        self.assertTrue(second.accepted)
        self.assertEqual(second.reason, "no_change")
        self.assertEqual(second.snapshot["reason"], "no_change")
        self.assertTrue(third.accepted)
        self.assertEqual(third.reason, "no_change")
        self.assertEqual(coordinator.revision, 1)
        self.assertEqual(coordinator.valid_update_count, 1)
        self.assertEqual(coordinator.repair_attempt_count, 2)
        self.assertEqual(coordinator.consecutive_no_progress_updates, 2)
        self.assertTrue(coordinator.plan_control_suppressed)

        changed = coordinator.apply_model_update(
            {
                "items": [
                    {
                        "id": "route",
                        "step": "比较通勤路线",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": [],
                        "planned_tools": ["route_compare"],
                    },
                    {
                        "id": "answer",
                        "step": "给出有依据的推荐选择",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["route"],
                        "planned_tools": [],
                    },
                ]
            }
        )
        self.assertTrue(changed.accepted)
        self.assertEqual(changed.reason, "model_update")
        self.assertEqual(coordinator.repair_attempt_count, 0)
        self.assertEqual(coordinator.consecutive_no_progress_updates, 0)
        self.assertFalse(coordinator.plan_control_suppressed)

    def _coordinator_with_search_read_chain(self, run_id: str) -> PlanCoordinator:
        coordinator = PlanCoordinator(run_id=run_id, mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先搜索、再读取、最后回答",
                    "items": [
                        {
                            "id": "search",
                            "title": "搜索资料",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "read",
                            "title": "读取来源",
                            "status": "pending",
                            "kind": "read",
                            "depends_on": ["search"],
                            "planned_tools": ["url_read"],
                        },
                        {
                            "id": "answer",
                            "title": "整理回答",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["read"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        return coordinator

    def test_plan_rejects_more_than_six_items_to_match_user_visible_contract(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        result = coordinator.apply_model_update(
            {
                "reason": "过长计划",
                "items": [
                    {
                        "id": f"step-{index}",
                        "title": f"步骤 {index}",
                        "status": "running" if index == 1 else "pending",
                        "kind": "other",
                        "depends_on": [] if index == 1 else [f"step-{index - 1}"],
                        "planned_tools": [],
                    }
                    for index in range(1, 8)
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "invalid_plan_structure")

    def test_initial_plan_must_cover_configured_tool_counts(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 3,
            }
        )
        missing_reads = coordinator.apply_model_update(
            {
                "reason": "只规划搜索",
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
                        "id": "read-one",
                        "title": "读取来源一",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "answer",
                        "title": "整理答案",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read-one"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(missing_reads.accepted)
        self.assertEqual(missing_reads.reason, "missing_required_initial_tool_coverage")
        self.assertFalse(coordinator.has_valid_model_plan)

        corrected = coordinator.apply_model_update(
            {
                "reason": "补足独立读取步骤",
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
                        "id": "read-one",
                        "title": "读取来源一",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "read-two",
                        "title": "读取来源二",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["read-one"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "read-three",
                        "title": "读取来源三",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["read-two"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "answer",
                        "title": "整理答案",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read-three"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(corrected.accepted)

    def test_initial_required_tools_must_have_distinct_plan_owners(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 3,
            }
        )

        result = coordinator.apply_model_update(
            {
                "reason": "错误地让搜索和读取共用计划项",
                "items": [
                    {
                        "id": f"research-{index}",
                        "title": f"研究来源 {index}",
                        "status": "running" if index == 1 else "pending",
                        "kind": "search",
                        "depends_on": [] if index == 1 else [f"research-{index - 1}"],
                        "planned_tools": ["web_search", "url_read"],
                    }
                    for index in range(1, 4)
                ]
                + [
                    {
                        "id": "answer",
                        "title": "整理答案",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["research-3"],
                        "planned_tools": [],
                    }
                ],
            }
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "multiple_tools_per_item")

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
        self.assertEqual([item["status"] for item in result.snapshot["items"]], ["pending", "pending"])
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
        self.assertIsNone(
            coordinator.plan_item_id_for_tool("search_trains", requested_item_id="return"),
        )
        self.assertIsNone(
            coordinator.plan_item_id_for_tool("search_trains", requested_item_id="missing"),
        )

    def test_active_plan_item_ids_exclude_terminal_steps(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {"id": "done", "status": "completed", "planned_tools": ["url_read"]},
            {"id": "current", "status": "running", "planned_tools": ["url_read"]},
            {"id": "next", "status": "pending", "planned_tools": ["url_read"]},
            {"id": "failed", "status": "failed", "planned_tools": ["url_read"]},
        ]

        self.assertEqual(
            coordinator.active_plan_item_ids_for_tool("url_read"),
            ["current"],
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
        self.assertEqual([item["status"] for item in result.snapshot["items"]], ["pending", "pending"])

    def test_model_echo_cannot_change_existing_nonterminal_server_status(self):
        coordinator = PlanCoordinator(run_id="run-status-owner", mode="on")
        payload = {
            "reason": "先搜索再回答",
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
                    "id": "answer",
                    "title": "整理回答",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["search"],
                    "planned_tools": [],
                },
            ],
        }
        initial = coordinator.apply_model_update(payload)
        self.assertTrue(initial.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in initial.snapshot["items"]},
            {"search": "pending", "answer": "pending"},
        )

        started = coordinator.mark_tools_started(["search"])
        self.assertEqual(started["items"][0]["status"], "running")
        payload["items"][0]["status"] = "pending"
        payload["items"][1]["status"] = "running"

        echoed = coordinator.apply_model_update(payload)

        self.assertTrue(echoed.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in echoed.snapshot["items"]},
            {"search": "running", "answer": "pending"},
        )

    def test_new_model_item_is_pending_even_when_model_reports_in_progress(self):
        coordinator = PlanCoordinator(run_id="run-new-item", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先搜索再回答",
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
                            "id": "answer",
                            "title": "整理回答",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )

        expanded = coordinator.apply_model_update(
            {
                "reason": "补充读取来源",
                "items": [
                    {
                        "id": "search",
                        "title": "搜索资料",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "read",
                        "title": "读取来源",
                        "status": "running",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "answer",
                        "title": "整理回答",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(expanded.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in expanded.snapshot["items"]},
            {"search": "pending", "read": "pending", "answer": "pending"},
        )

    def test_only_execution_apis_advance_tool_and_answer_statuses(self):
        coordinator = PlanCoordinator(run_id="run-status-transitions", mode="on")
        initial = coordinator.apply_model_update(
            {
                "reason": "先搜索再回答",
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
                        "id": "answer",
                        "title": "整理回答",
                        "status": "running",
                        "kind": "answer",
                        "depends_on": ["search"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertTrue(initial.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in initial.snapshot["items"]},
            {"search": "pending", "answer": "pending"},
        )

        started = coordinator.mark_tools_started(["search"])
        completed = coordinator.mark_tool_results({"search": "completed"})
        synthesis = coordinator.begin_synthesis()

        self.assertEqual(
            {item["id"]: item["status"] for item in started["items"]},
            {"search": "running", "answer": "pending"},
        )
        self.assertEqual(
            {item["id"]: item["status"] for item in completed["items"]},
            {"search": "completed", "answer": "pending"},
        )
        self.assertEqual(
            {item["id"]: item["status"] for item in synthesis["items"]},
            {"search": "completed", "answer": "running"},
        )

    def test_plan_item_rejects_multiple_planned_tools(self):
        coordinator = PlanCoordinator(run_id="run-multi-tool-item", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "错误地让一个步骤拥有多个工具",
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

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "multiple_tools_per_item")
        self.assertFalse(coordinator.has_valid_model_plan)

    def test_pending_dependency_prevents_tool_binding_and_start(self):
        coordinator = self._coordinator_with_search_read_chain("run-pending-dependency")

        self.assertIsNone(coordinator.plan_item_id_for_tool("url_read", requested_item_id="read"))
        started = coordinator.mark_tools_started(["read"])

        self.assertIsNone(started)
        self.assertEqual(
            {item["id"]: item["status"] for item in coordinator.items},
            {"search": "pending", "read": "pending", "answer": "pending"},
        )

    def test_completed_dependency_enables_downstream_tool_binding_and_start(self):
        coordinator = self._coordinator_with_search_read_chain("run-completed-dependency")
        coordinator.mark_tools_started(["search"])
        coordinator.mark_tool_results({"search": "completed"})

        self.assertEqual(
            coordinator.plan_item_id_for_tool("url_read", requested_item_id="read"),
            "read",
        )
        started = coordinator.mark_tools_started(["read"])
        self.assertEqual(
            {item["id"]: item["status"] for item in started["items"]},
            {"search": "completed", "read": "running", "answer": "pending"},
        )

    def test_failed_dependency_transitively_blocks_downstream_items(self):
        coordinator = self._coordinator_with_search_read_chain("run-failed-dependency")
        coordinator.mark_tools_started(["search"])

        failed = coordinator.mark_tool_results({"search": "failed"})

        self.assertEqual(
            {item["id"]: item["status"] for item in failed["items"]},
            {"search": "failed", "read": "blocked", "answer": "pending"},
        )
        synthesis = coordinator.begin_synthesis()
        self.assertEqual(
            {item["id"]: item["status"] for item in synthesis["items"]},
            {"search": "failed", "read": "blocked", "answer": "running"},
        )
        terminal = coordinator.terminalize("stop", has_final_answer=True)
        self.assertEqual(
            {item["id"]: item["status"] for item in terminal["items"]},
            {"search": "failed", "read": "blocked", "answer": "completed"},
        )
        self.assertIsNone(coordinator.plan_item_id_for_tool("url_read", requested_item_id="read"))

    def test_real_tool_start_completes_non_tool_reasoning_prerequisite(self):
        coordinator = PlanCoordinator(run_id="run-reasoning-prerequisite", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先判断路径再搜索",
                    "items": [
                        {
                            "id": "reason",
                            "title": "判断资料需求",
                            "status": "pending",
                            "kind": "reasoning",
                            "depends_on": [],
                            "planned_tools": [],
                        },
                        {
                            "id": "search",
                            "title": "搜索资料",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": ["reason"],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "answer",
                            "title": "整理回答",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )

        started = coordinator.mark_tools_started(["search"])

        self.assertEqual(
            {item["id"]: item["status"] for item in started["items"]},
            {"reason": "completed", "search": "running", "answer": "pending"},
        )

    def test_starting_blocked_successor_does_not_demote_running_predecessor(self):
        coordinator = self._coordinator_with_search_read_chain("run-no-running-backjump")
        first = coordinator.mark_tools_started(["search"])
        self.assertEqual(first["items"][0]["status"], "running")

        second = coordinator.mark_tools_started(["read"])

        self.assertIsNone(second)
        self.assertEqual(
            {item["id"]: item["status"] for item in coordinator.items},
            {"search": "running", "read": "pending", "answer": "pending"},
        )

    def test_multi_item_batch_has_single_running_owner_then_uses_real_terminal_results(self):
        coordinator = PlanCoordinator(run_id="run-multi-item-batch", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "并列查询后回答",
                    "items": [
                        {
                            "id": "search-a",
                            "title": "查询 A",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["search_a"],
                        },
                        {
                            "id": "search-b",
                            "title": "查询 B",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["search_b"],
                        },
                        {
                            "id": "answer",
                            "title": "整理回答",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["search-a", "search-b"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )

        started = coordinator.mark_tools_started(["search-a", "search-b"])
        started_statuses = {item["id"]: item["status"] for item in started["items"]}
        self.assertEqual(
            sum(status == "running" for status in started_statuses.values()),
            1,
        )
        self.assertEqual(
            sum(status == "pending" for item_id, status in started_statuses.items() if item_id != "answer"),
            1,
        )

        terminal = coordinator.mark_tool_results({"search-a": "completed", "search-b": "failed"})
        self.assertEqual(
            {item["id"]: item["status"] for item in terminal["items"]},
            {"search-a": "completed", "search-b": "failed", "answer": "pending"},
        )
        synthesis = coordinator.begin_synthesis()
        self.assertEqual(
            {item["id"]: item["status"] for item in synthesis["items"]},
            {"search-a": "completed", "search-b": "failed", "answer": "running"},
        )
        completed = coordinator.terminalize("stop", has_final_answer=True)
        self.assertEqual(
            {item["id"]: item["status"] for item in completed["items"]},
            {"search-a": "completed", "search-b": "failed", "answer": "completed"},
        )

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
        changed_payload = {
            **payload,
            "items": [
                payload["items"][0],
                {**payload["items"][1], "title": "完成新版"},
            ],
        }
        rejected = coordinator.apply_model_update(changed_payload)

        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "control_update_limit_reached")
        self.assertEqual(coordinator.revision, 1)

    def test_new_run_removes_legacy_observed_plan_entrypoint(self):
        coordinator = PlanCoordinator(run_id="run-1", mode="on")
        self.assertFalse(hasattr(coordinator, "adopt_observed_items"))

        accepted = coordinator.apply_model_update(
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

        self.assertTrue(accepted.accepted)
        self.assertEqual(coordinator.source, "model")
        self.assertEqual([item["id"] for item in coordinator.items], ["answer", "finish"])

    def test_rejects_dependency_cycle_and_normalizes_model_running_items(self):
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
                        "kind": "answer",
                        "depends_on": ["a"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertTrue(multiple_running.accepted)
        self.assertEqual(
            [item["status"] for item in multiple_running.snapshot["items"]],
            ["pending", "pending"],
        )

    def test_model_plan_uses_new_plan_id_and_server_terminal_status_is_preserved(self):
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
        first = coordinator.apply_model_update(payload)
        self.assertEqual(first.snapshot["plan_id"], "plan-run-1-model")
        coordinator.mark_tool_results({"search": "completed"})
        payload["items"][0]["status"] = "running"
        echoed = coordinator.apply_model_update(payload)
        self.assertTrue(echoed.accepted)
        self.assertEqual(echoed.snapshot["items"][0]["status"], "completed")

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
        asserted_result = coordinator.apply_model_update(asserted)
        self.assertTrue(asserted_result.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in asserted_result.snapshot["items"]},
            {"search": "completed", "answer": "pending"},
        )

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

    def test_terminal_item_metadata_drift_is_replaced_with_server_canonical_values(self):
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

        self.assertTrue(rewritten.accepted)
        rewritten_flight = next(item for item in rewritten.snapshot["items"] if item["id"] == "flight")
        before_flight = next(item for item in before["items"] if item["id"] == "flight")
        self.assertEqual(rewritten_flight, before_flight)

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
                {
                    "id": "milestone",
                    "title": "整理无工具里程碑",
                    "status": "pending",
                    "kind": "other",
                    "planned_tools": [],
                },
                {"id": "answer", "title": "回答", "status": "pending", "kind": "answer"},
                {"id": "reasoning", "title": "分析", "status": "running", "kind": "reasoning"},
                {"id": "retry", "title": "重试", "status": "running", "kind": "search"},
            ]
        ).terminalize("stop", has_final_answer=True)
        self.assertEqual(
            [item["status"] for item in normal["items"]],
            ["blocked", "skipped", "completed", "completed", "completed", "blocked"],
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
        for index in range(6):
            payload["items"][1]["title"] = f"回答 {index}"
            self.assertTrue(coordinator.apply_model_update(payload).accepted)
        payload["items"][1]["title"] = "回答 7"
        self.assertEqual(coordinator.apply_model_update(payload).reason, "control_update_limit_reached")
        self.assertFalse(coordinator.record_repair_round())
        self.assertFalse(coordinator.record_repair_round())
        self.assertTrue(coordinator.record_repair_round())

    def test_research_fallback_never_replaces_an_existing_valid_plan(self):
        coordinator = PlanCoordinator(run_id="run-research", mode="on")
        coordinator.configure_initial_tool_requirements(
            {
                "web_search": 1,
                "url_read": 1,
            }
        )
        accepted = coordinator.apply_model_update(
            {
                "reason": "搜索后核验来源",
                "items": [
                    {
                        "id": "search",
                        "title": "搜索来源",
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
                    {
                        "id": "answer",
                        "title": "整理结论",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertTrue(accepted.accepted)
        coordinator.mark_tool_results({"search": "completed"})
        existing_items = [dict(item) for item in coordinator.items]

        self.assertFalse(coordinator.record_repair_round_with_fallback().exhausted)
        self.assertFalse(coordinator.record_repair_round_with_fallback().exhausted)
        third = coordinator.record_repair_round_with_fallback()

        self.assertTrue(third.exhausted)
        self.assertIsNone(third.fallback)
        self.assertEqual(coordinator.source, "model")
        self.assertEqual(coordinator.items, existing_items)

    def test_synthesis_start_is_monotonic_and_late_tool_cannot_reopen_completed_stage(self):
        coordinator = PlanCoordinator(run_id="run-answer-preview", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先搜索再回答",
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
        tool_completed = coordinator.mark_tool_results({"search": "completed"})

        self.assertEqual(
            {item["id"]: item["status"] for item in tool_completed["items"]},
            {"search": "completed", "answer": "pending"},
        )
        answering = coordinator.begin_synthesis()
        self.assertEqual(
            {item["id"]: item["status"] for item in answering["items"]},
            {"search": "completed", "answer": "running"},
        )

        late_tool = coordinator.mark_tools_started(["search"])

        self.assertIsNone(late_tool)
        self.assertEqual(
            {item["id"]: item["status"] for item in coordinator.items},
            {"search": "completed", "answer": "running"},
        )
        self.assertEqual(coordinator.active_plan_item_ids_for_tool("web_search"), [])

    def test_model_plan_without_answer_phase_is_rejected(self):
        coordinator = PlanCoordinator(run_id="run-missing-answer", mode="on")

        result = coordinator.apply_model_update(
            {
                "reason": "只声明两个工具阶段",
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

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "missing_answer_phase")
        self.assertFalse(coordinator.has_valid_model_plan)

    def test_tool_results_immediately_publish_terminal_status_without_false_completion(self):
        def coordinator_for(run_id: str) -> PlanCoordinator:
            coordinator = PlanCoordinator(run_id=run_id, mode="on")
            self.assertTrue(
                coordinator.apply_model_update(
                    {
                        "reason": "先搜索再回答",
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
            return coordinator

        successful = coordinator_for("run-tool-success")
        success_snapshot = successful.mark_tool_results({"search": "completed"})
        self.assertEqual(
            {item["id"]: item["status"] for item in success_snapshot["items"]},
            {"search": "completed", "answer": "pending"},
        )

        failed = coordinator_for("run-tool-failed")
        failed_snapshot = failed.mark_tool_results({"search": "failed"})
        self.assertEqual(
            {item["id"]: item["status"] for item in failed_snapshot["items"]},
            {"search": "failed", "answer": "pending"},
        )
        synthesis_snapshot = failed.begin_synthesis()
        self.assertEqual(
            {item["id"]: item["status"] for item in synthesis_snapshot["items"]},
            {"search": "failed", "answer": "running"},
        )
        terminal_snapshot = failed.terminalize("stop", has_final_answer=True)
        self.assertEqual(
            {item["id"]: item["status"] for item in terminal_snapshot["items"]},
            {"search": "failed", "answer": "completed"},
        )

        missing = coordinator_for("run-tool-missing")
        missing_snapshot = missing.mark_tool_results({})
        self.assertIsNone(missing_snapshot)
        self.assertNotEqual(missing.items[0]["status"], "completed")

    def test_every_run_terminal_snapshot_closes_all_running_model_plan_items(self):
        for outcome in ("stop", "limit_reached", "incomplete", "interrupted", "superseded", "failed"):
            with self.subTest(outcome=outcome):
                coordinator = PlanCoordinator(run_id=f"run-{outcome}", mode="on")
                self.assertTrue(
                    coordinator.apply_model_update(
                        {
                            "reason": "先搜索再回答",
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

                terminal = coordinator.terminalize(outcome, has_final_answer=outcome == "stop")

                self.assertNotIn("running", [item["status"] for item in terminal["items"]])

    def test_model_echo_of_server_terminal_status_is_merged_without_regression(self):
        coordinator = PlanCoordinator(run_id="run-terminal-merge", mode="on")
        payload = {
            "reason": "先搜索再回答",
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
                    "id": "answer",
                    "title": "整理答案",
                    "status": "pending",
                    "kind": "answer",
                    "depends_on": ["search"],
                    "planned_tools": [],
                },
            ],
        }
        self.assertTrue(coordinator.apply_model_update(payload).accepted)
        coordinator.mark_tool_results({"search": "completed"})

        echoed = coordinator.apply_model_update(payload)

        self.assertTrue(echoed.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in echoed.snapshot["items"]},
            {"search": "completed", "answer": "pending"},
        )

    def test_successful_update_and_real_tool_progress_reset_consecutive_repair_count(self):
        coordinator = PlanCoordinator(run_id="run-repair-reset", mode="on")
        self.assertFalse(coordinator.record_repair_round())
        self.assertFalse(coordinator.record_repair_round())

        accepted = coordinator.apply_model_update(
            {
                "reason": "建立有效计划",
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
                        "id": "answer",
                        "title": "整理答案",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["search"],
                        "planned_tools": [],
                    },
                ],
            }
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(coordinator.repair_attempt_count, 0)

        self.assertFalse(coordinator.record_repair_round())
        coordinator.reset_repair_attempts()
        self.assertEqual(coordinator.repair_attempt_count, 0)

    def test_unexecuted_plan_tool_names_excludes_items_with_successful_execution(self):
        coordinator = PlanCoordinator(run_id="run-tool-coverage", mode="on")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {"id": "search-a", "status": "running", "planned_tools": ["web_search"]},
            {"id": "search-b", "status": "pending", "planned_tools": ["web_search"]},
            {"id": "read", "status": "pending", "planned_tools": ["url_read"]},
        ]

        coordinator.mark_tool_results({"search-a": "completed"})

        self.assertEqual(
            coordinator.unexecuted_plan_tool_names(),
            {"web_search", "url_read"},
        )
        self.assertEqual(
            coordinator.unexecuted_plan_item_ids_for_tool("web_search"),
            ["search-b"],
        )

    def test_retryable_tool_result_keeps_item_running_until_same_item_succeeds(self):
        def coordinator_for(run_id: str) -> PlanCoordinator:
            coordinator = PlanCoordinator(run_id=run_id, mode="on")
            coordinator.source = "model"
            coordinator.revision = 1
            coordinator.items = [
                {
                    "id": "search",
                    "status": "running",
                    "kind": "search",
                    "planned_tools": ["web_search"],
                }
            ]
            return coordinator

        coordinator = coordinator_for("run-retryable-repair")
        coordinator.mark_tools_started(["search"])
        retryable = coordinator.mark_tool_results({"search": "running"})
        self.assertEqual(
            coordinator.plan_item_id_for_tool("web_search", requested_item_id="search"),
            "search",
        )
        coordinator.mark_tools_started(["search"])
        repaired = coordinator.mark_tool_results({"search": "completed"})

        self.assertEqual(retryable["items"][0]["status"], "running")
        self.assertEqual(repaired["items"][0]["status"], "completed")
        self.assertEqual(coordinator.items[0]["status"], "completed")
        self.assertEqual(coordinator.unexecuted_plan_tool_names(), set())
        self.assertEqual(coordinator.unexecuted_plan_item_ids_for_tool("web_search"), [])

    def test_non_retryable_tool_failure_never_completes_item(self):
        coordinator = PlanCoordinator(run_id="run-non-retryable-failure", mode="on")
        coordinator.source = "model"
        coordinator.revision = 1
        coordinator.items = [
            {
                "id": "search",
                "status": "running",
                "kind": "search",
                "planned_tools": ["web_search"],
            }
        ]

        failed = coordinator.mark_tool_results({"search": "failed"})

        self.assertNotEqual(failed["items"][0]["status"], "completed")
        self.assertNotEqual(coordinator.items[0]["status"], "completed")

    def test_attempted_tool_item_cannot_be_removed_before_run_terminalization(self):
        coordinator = PlanCoordinator(run_id="run-attempted-remove", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先搜索再回答",
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
        coordinator.mark_tool_results({"search": "completed"})
        before = coordinator.snapshot()

        removed = coordinator.apply_model_update(
            {
                "reason": "删除已经执行的步骤",
                "items": [
                    {
                        "id": "answer",
                        "title": "整理答案",
                        "status": "running",
                        "kind": "answer",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "followup",
                        "title": "补充建议",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": ["answer"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(removed.accepted)
        self.assertEqual(removed.reason, "terminal_item_removed")
        self.assertEqual(coordinator.snapshot()["items"], before["items"])

    def test_attempted_tool_item_metadata_drift_is_replaced_with_server_canonical_values(self):
        coordinator = PlanCoordinator(run_id="run-attempted-mutate", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先搜索再回答",
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
        coordinator.mark_tool_results({"search": "completed"})
        before = coordinator.snapshot()

        rewritten = coordinator.apply_model_update(
            {
                "reason": "改写已经执行的步骤",
                "items": [
                    {
                        "id": "search",
                        "title": "查询航班",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["search_flights"],
                    },
                    {
                        "id": "answer",
                        "title": "整理答案",
                        "status": "running",
                        "kind": "answer",
                        "depends_on": ["search"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(rewritten.accepted)
        rewritten_search = next(item for item in rewritten.snapshot["items"] if item["id"] == "search")
        before_search = next(item for item in before["items"] if item["id"] == "search")
        self.assertEqual(
            {key: rewritten_search[key] for key in ("title", "kind", "depends_on", "planned_tools")},
            {key: before_search[key] for key in ("title", "kind", "depends_on", "planned_tools")},
        )
        self.assertEqual(rewritten_search["status"], "completed")

    def test_kimi_style_full_plan_echo_keeps_attempted_item_canonical_and_advances_status(self):
        coordinator = PlanCoordinator(run_id="run-kimi-drift", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "建立四步研究计划",
                    "items": [
                        {
                            "id": "search-primary",
                            "title": "检索核心资料",
                            "status": "running",
                            "kind": "search",
                            "depends_on": [],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "search-secondary",
                            "title": "补充对照资料",
                            "status": "pending",
                            "kind": "search",
                            "depends_on": ["search-primary"],
                            "planned_tools": ["web_search"],
                        },
                        {
                            "id": "compare",
                            "title": "比较关键差异",
                            "status": "pending",
                            "kind": "synthesis",
                            "depends_on": ["search-secondary"],
                            "planned_tools": [],
                        },
                        {
                            "id": "answer",
                            "title": "整理最终建议",
                            "status": "pending",
                            "kind": "answer",
                            "depends_on": ["compare"],
                            "planned_tools": [],
                        },
                    ],
                }
            ).accepted
        )
        coordinator.mark_tool_results({"search-primary": "completed"})

        update = coordinator.apply_model_update(
            {
                "reason": "根据搜索回声更新计划",
                "items": [
                    {
                        "id": "search-primary",
                        "title": "确认 Next.js 与 React 的最新版本",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": [],
                        "planned_tools": [],
                    },
                    {
                        "id": "search-secondary",
                        "title": "补充对照资料",
                        "status": "running",
                        "kind": "search",
                        "depends_on": ["search-primary"],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "compare",
                        "title": "比较关键差异",
                        "status": "pending",
                        "kind": "synthesis",
                        "depends_on": ["search-secondary"],
                        "planned_tools": [],
                    },
                    {
                        "id": "answer",
                        "title": "整理最终建议",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["compare"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(update.accepted)
        primary = next(item for item in update.snapshot["items"] if item["id"] == "search-primary")
        self.assertEqual(
            primary,
            {
                "id": "search-primary",
                "title": "检索核心资料",
                "status": "completed",
                "kind": "search",
                "depends_on": [],
                "planned_tools": ["web_search"],
            },
        )

    def test_model_echo_cannot_advance_answer_after_tool_execution(self):
        coordinator = PlanCoordinator(run_id="run-attempted-status", mode="on")
        self.assertTrue(
            coordinator.apply_model_update(
                {
                    "reason": "先搜索再回答",
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
        coordinator.mark_tool_results({"search": "completed"})

        advanced = coordinator.apply_model_update(
            {
                "reason": "进入答案整理",
                "items": [
                    {
                        "id": "search",
                        "title": "搜索资料",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "answer",
                        "title": "整理答案",
                        "status": "running",
                        "kind": "answer",
                        "depends_on": ["search"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertTrue(advanced.accepted)
        self.assertEqual(
            {item["id"]: item["status"] for item in advanced.snapshot["items"]},
            {"search": "completed", "answer": "pending"},
        )
        terminal = coordinator.terminalize("stop", has_final_answer=True)
        self.assertEqual(
            {item["id"]: item["status"] for item in terminal["items"]},
            {"search": "completed", "answer": "completed"},
        )


if __name__ == "__main__":
    unittest.main()
