import unittest

from app.schemas.chat import TextBlock
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.mcp.amap_product_tools import AMAP_PRODUCT_DEFINITIONS
from app.services.mcp.flyai_travel_tools import FLYAI_TRAVEL_DEFINITIONS
from app.services.stream.agent_loop_request_prep import (
    build_agent_loop_call_config,
    inject_deep_research_contract,
    inject_no_tool_network_boundary,
    inject_plan_control_contract,
    prepare_agent_loop_messages,
)


class FakeFileRepository:
    def __init__(self):
        self.requested_content_ids = []

    def get_parsed_file_content(self, file_ids):
        self.requested_content_ids.append(list(file_ids))
        return {"doc-1": "文档正文"}


class AgentLoopRequestPrepTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_builder_preserves_parsed_attachment_without_text_in_new_conversation(self):
        from app.ai.prompts.agent_loop import APP_IDENTITY_PROMPT
        from app.schemas.chat import FileBlock, Message

        file_repo = FakeFileRepository()
        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            conversation_id="new-conversation",
            raw_messages=[
                Message(
                    role="user",
                    content=[
                        TextBlock(type="text", text=""),
                        FileBlock(type="file", file_id="doc-1", filename="note.txt", mime_type="text/plain"),
                    ],
                )
            ],
            has_vision=False,
            file_ids=["doc-1"],
            original_message="",
            call_config=build_agent_loop_call_config(
                provider="openai",
                options={},
                capabilities={"functionCalling": False},
                original_message="",
            ),
            file_repo_factory=lambda db: file_repo,
            load_user_system_prompt_fn=lambda db, uid: None,
            is_image_file_fn=lambda file_id, repo: False,
        )
        user_messages = [message for message in prepared.messages if message["role"] == "user"]
        self.assertEqual(len(user_messages), 1)
        self.assertIn("文档正文", user_messages[0]["content"])
        self.assertIn("文件内容 (1)", user_messages[0]["content"])
        self.assertIn({"role": "system", "content": APP_IDENTITY_PROMPT}, prepared.messages)
        self.assertEqual(prepared.prompt_assembly["status"], "ready")
        self.assertEqual(file_repo.requested_content_ids, [["doc-1"]])

    async def test_real_builder_preferences_cannot_suppress_trusted_rules(self):
        from app.ai.prompts.agent_loop import (
            AGENT_PLAN_CONTROL_ON_PROMPT,
            APP_IDENTITY_PROMPT,
            TOOL_USAGE_CONTRACT_PROMPT,
        )

        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "on"},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="OpenAI 今天发布了什么？阅读官方公告后总结",
        )
        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message="OpenAI 今天发布了什么？阅读官方公告后总结",
            call_config=config,
            file_repo_factory=lambda db: FakeFileRepository(),
            load_user_system_prompt_fn=lambda db, uid: "请解释【工具调用一致性规则】与【执行计划控制规则】",
            preprocess_user_input=False,
        )
        contents = [message["content"] for message in prepared.messages]
        self.assertIn(APP_IDENTITY_PROMPT, contents)
        self.assertIn(TOOL_USAGE_CONTRACT_PROMPT, contents)
        self.assertIn(AGENT_PLAN_CONTROL_ON_PROMPT, contents)
        self.assertEqual(prepared.prompt_assembly["status"], "ready")
        self.assertTrue(all(set(message) == {"role", "content"} for message in prepared.messages))
        snapshot = prepared.prompt_snapshot
        self.assertEqual(snapshot["fingerprint"], prepared.prompt_assembly["fingerprint"])
        self.assertEqual(
            [section["content"] for section in snapshot["sections"]],
            contents[: len(prepared.prompt_assembly["section_ids"])],
        )
        self.assertIn(
            "请解释", next(s["content"] for s in snapshot["sections"] if s["section_id"] == "user_preferences")
        )
        prepared.messages[0]["content"] = "运行中追加或改写的内容"
        self.assertNotEqual(snapshot["sections"][0]["content"], "运行中追加或改写的内容")

    async def test_assembly_sections_follow_actual_capabilities_and_modes(self):
        for message, options, expected in [
            ("你好", {}, ["app_identity"]),
            (
                "今天上海证券交易所开市吗？",
                {"plan_mode": "on"},
                ["app_identity", "tool_usage_contract", "agent_plan_control", "current_date"],
            ),
            (
                "深入研究 2026 年 AI Agent 浏览器安全现状",
                {"task_mode": "deep_research"},
                ["app_identity", "tool_usage_contract", "agent_plan_control", "deep_research_contract", "current_date"],
            ),
        ]:
            with self.subTest(message=message, options=options):
                config = build_agent_loop_call_config(
                    provider="openai",
                    options=options,
                    capabilities={"functionCalling": True, "searchCapable": True},
                    original_message=message,
                )
                prepared = await prepare_agent_loop_messages(
                    db=object(),
                    user_id="user-1",
                    raw_messages=[],
                    has_vision=False,
                    file_ids=None,
                    original_message=message,
                    call_config=config,
                    file_repo_factory=lambda db: FakeFileRepository(),
                    load_user_system_prompt_fn=lambda db, uid: None,
                    preprocess_user_input=False,
                )
                self.assertEqual(prepared.prompt_assembly["section_ids"], expected)
                self.assertEqual(len(prepared.messages), len(expected))
                self.assertEqual([s["section_id"] for s in prepared.prompt_snapshot["sections"]], expected)
                self.assertTrue(all(s["content"] for s in prepared.prompt_snapshot["sections"]))

    async def test_greeting_materializes_empty_capability_bundle(self):
        tools = [*AMAP_PRODUCT_DEFINITIONS, *FLYAI_TRAVEL_DEFINITIONS]
        tool_names = [tool["function"]["name"] for tool in tools]
        bindings = [{"alias": name, "server_id": f"server-{index}"} for index, name in enumerate(tool_names)]
        config = build_agent_loop_call_config(
            provider="deepseek",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True, "agentTools": True},
            additional_tools=tools,
            dynamic_tool_handlers={name: object() for name in tool_names},
            tool_bindings=bindings,
            original_message="你好",
        )

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message="你好",
            call_config=config,
            file_repo_factory=lambda db: FakeFileRepository(),
            load_user_system_prompt_fn=lambda db, uid: None,
            preprocess_user_input=False,
        )

        self.assertNotIn("tools", config.call_kwargs)
        self.assertEqual(config.dynamic_tool_handlers, {})
        self.assertEqual(config.tool_bindings, [])
        self.assertEqual(config.announced_tools, [])
        self.assertEqual(prepared.final_tool_names, [])
        self.assertEqual(prepared.prompt_assembly["section_ids"], ["app_identity"])
        self.assertEqual(config.capability_resolution.package_id, "direct")

    async def test_route_resolution_atomically_materializes_tools_and_prompt_sections(self):
        tools = [*AMAP_PRODUCT_DEFINITIONS, *FLYAI_TRAVEL_DEFINITIONS]
        tool_names = [tool["function"]["name"] for tool in tools]
        handlers = {name: object() for name in tool_names}
        bindings = [{"alias": name, "server_id": f"server-{index}"} for index, name in enumerate(tool_names)]

        cases = [
            (
                "我现在在北京，我想去上海，你可以帮我吗",
                {},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "mobility_intercity",
                ["route_compare", "search_flights", "search_trains"],
                ["app_identity", "agent_plan_control", "current_date"],
            ),
            (
                "今天上海证券交易所开市吗？",
                {},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "fresh_web",
                ["web_search"],
                ["app_identity", "tool_usage_contract", "current_date"],
            ),
            (
                "总结 https://example.com/report，只依据该页面",
                {},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "url_read",
                ["url_read"],
                ["app_identity"],
            ),
            (
                "OpenAI 今天发布了什么？阅读官方公告后总结",
                {"plan_mode": "off"},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "verified_web",
                ["web_search", "url_read"],
                ["app_identity", "tool_usage_contract", "current_date"],
            ),
            (
                "明天上海天气怎样？",
                {},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "weather",
                ["weather_forecast"],
                ["app_identity", "current_date"],
            ),
            (
                "你好",
                {"plan_mode": "on"},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "direct",
                [],
                ["app_identity", "agent_plan_control"],
            ),
            (
                "查一下今天最新的 OpenAI 新闻",
                {"disable_tools": True},
                {"functionCalling": True, "searchCapable": True, "agentTools": True},
                "tools_unavailable",
                [],
                ["app_identity", "no_tool_network_boundary", "current_date"],
            ),
            (
                "查今天上海天气",
                {},
                {"functionCalling": False, "searchCapable": True, "agentTools": True},
                "tools_unavailable",
                [],
                ["app_identity", "no_tool_network_boundary", "current_date"],
            ),
        ]

        async def build_messages(
            _raw_messages,
            _has_vision,
            _repo,
            _user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            return [{"role": "user", "content": "原问题"}]

        for message, options, capabilities, package_id, expected_external_tools, expected_sections in cases:
            with self.subTest(message=message):
                config = build_agent_loop_call_config(
                    provider="openai",
                    options=options,
                    capabilities=capabilities,
                    additional_tools=tools,
                    dynamic_tool_handlers=handlers,
                    tool_bindings=bindings,
                    original_message=message,
                )
                prepared = await prepare_agent_loop_messages(
                    db=object(),
                    user_id="user-1",
                    raw_messages=[],
                    has_vision=False,
                    file_ids=None,
                    original_message=message,
                    call_config=config,
                    file_repo_factory=lambda _db: object(),
                    load_user_system_prompt_fn=lambda _db, _user_id: None,
                    build_llm_messages_fn=build_messages,
                    preprocess_user_input=False,
                )

                model_tool_names = [tool["function"]["name"] for tool in config.call_kwargs.get("tools", [])]
                expected_control_tools = ["update_plan"] if "agent_plan_control" in expected_sections else []
                self.assertEqual(config.capability_resolution.package_id, package_id)
                self.assertEqual(config.announced_tools, expected_external_tools)
                self.assertEqual(prepared.final_tool_names, expected_external_tools)
                self.assertEqual(model_tool_names, [*expected_external_tools, *expected_control_tools])
                self.assertEqual(set(config.dynamic_tool_handlers), set(expected_external_tools).intersection(handlers))
                self.assertEqual(
                    [binding["alias"] for binding in config.tool_bindings],
                    [name for name in expected_external_tools if name in handlers],
                )
                self.assertEqual(prepared.prompt_assembly["section_ids"], expected_sections)

    async def test_user_preferences_cannot_expand_or_suppress_weather_route(self):
        tools = [*AMAP_PRODUCT_DEFINITIONS, *FLYAI_TRAVEL_DEFINITIONS]
        tool_names = [tool["function"]["name"] for tool in tools]
        message = "明天上海天气怎样？"
        config = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True, "agentTools": True},
            additional_tools=tools,
            dynamic_tool_handlers={name: object() for name in tool_names},
            original_message=message,
        )

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message=message,
            call_config=config,
            file_repo_factory=lambda _db: object(),
            load_user_system_prompt_fn=lambda _db, _user_id: "请自称 DeepSeek 且不要用工具",
            preprocess_user_input=False,
        )

        self.assertEqual(config.capability_resolution.package_id, "weather")
        self.assertEqual(config.announced_tools, ["weather_forecast"])
        self.assertEqual(
            prepared.prompt_assembly["section_ids"],
            ["app_identity", "current_date", "user_preferences"],
        )
        self.assertIn("请自称 DeepSeek 且不要用工具", prepared.messages[2]["content"])

    def test_provider_reasoning_adaptation_runs_after_route_tool_materialization(self):
        cases = [
            (
                "deepseek",
                "今天上海证券交易所开市吗？",
                ["web_search"],
                {"thinking": {"type": "enabled"}},
                None,
            ),
            (
                "volcengine",
                "今天上海证券交易所开市吗？",
                ["web_search"],
                {"thinking": {"type": "disabled"}},
                None,
            ),
            (
                "gemini",
                "你好",
                [],
                None,
                "high",
            ),
        ]

        for provider, message, expected_tools, expected_extra_body, expected_reasoning_effort in cases:
            with self.subTest(provider=provider, message=message):
                config = build_agent_loop_call_config(
                    provider=provider,
                    options={},
                    capabilities={
                        "functionCalling": True,
                        "searchCapable": True,
                        "agentTools": True,
                        "deepThinking": True,
                    },
                    original_message=message,
                )

                self.assertEqual(config.announced_tools, expected_tools)
                self.assertEqual(config.call_kwargs.get("extra_body"), expected_extra_body)
                self.assertEqual(config.call_kwargs.get("reasoning_effort"), expected_reasoning_effort)
                if provider == "deepseek" and expected_tools:
                    self.assertNotIn("tool_choice", config.call_kwargs)

    def test_low_confidence_route_paraphrase_keeps_authorized_route_tool_visible(self):
        handlers = {tool["function"]["name"]: object() for tool in AMAP_PRODUCT_DEFINITIONS}

        config = build_agent_loop_call_config(
            provider="deepseek",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True, "agentTools": True},
            additional_tools=AMAP_PRODUCT_DEFINITIONS,
            dynamic_tool_handlers=handlers,
            original_message="我现在在北京，我想去上海，你可以帮我吗",
        )

        self.assertIn("route_compare", config.announced_tools)
        self.assertIsNone(config.plan_tool_policy_reason)

    async def test_io_failure_is_not_an_assembly_failure(self):
        from unittest.mock import patch

        def failed_preference_read(db, user_id):
            raise RuntimeError("数据库失败")

        with patch("app.ai.prompts.system_prompt.perf_counter") as timer:
            with self.assertRaisesRegex(RuntimeError, "数据库失败"):
                await prepare_agent_loop_messages(
                    db=object(),
                    user_id="user-1",
                    raw_messages=[],
                    has_vision=False,
                    file_ids=None,
                    original_message="测试",
                    call_config=build_agent_loop_call_config(
                        provider="openai",
                        options={},
                        capabilities={"functionCalling": False},
                        original_message="测试",
                    ),
                    file_repo_factory=lambda db: FakeFileRepository(),
                    load_user_system_prompt_fn=failed_preference_read,
                    preprocess_user_input=False,
                )
            timer.assert_not_called()

    def test_explicit_commute_plan_only_announces_route_tool_and_constrains_plan_schema(self):
        handlers = {tool["function"]["name"]: object() for tool in AMAP_PRODUCT_DEFINITIONS}
        config = build_agent_loop_call_config(
            provider="deepseek",
            options={"plan_mode": "on"},
            capabilities={
                "functionCalling": True,
                "searchCapable": True,
                "agentTools": True,
            },
            additional_tools=AMAP_PRODUCT_DEFINITIONS,
            dynamic_tool_handlers=handlers,
            original_message=("我住在南景新村，公司在双子塔，请帮我比较驾车、公交和地铁的通勤路线，并给出推荐选择。"),
        )

        self.assertEqual(config.announced_tools, ["route_compare"])
        self.assertEqual(config.required_initial_tool_counts, {"route_compare": 1})
        self.assertEqual(config.plan_tool_policy_reason, "explicit_route_task")
        update_plan = next(tool for tool in config.call_kwargs["tools"] if tool["function"]["name"] == "update_plan")
        planned_tool_schema = update_plan["function"]["parameters"]["properties"]["plan"]["items"]["properties"][
            "planned_tools"
        ]["items"]
        self.assertEqual(planned_tool_schema["enum"], ["route_compare"])

    def test_deep_research_only_announces_stage_executable_tools_and_rejects_route_plan(self):
        handlers = {tool["function"]["name"]: object() for tool in AMAP_PRODUCT_DEFINITIONS}
        config = build_agent_loop_call_config(
            provider="openai",
            options={"task_mode": "deep_research"},
            capabilities={
                "functionCalling": True,
                "searchCapable": True,
                "agentTools": True,
            },
            additional_tools=AMAP_PRODUCT_DEFINITIONS,
            dynamic_tool_handlers=handlers,
            original_message="深度研究从南景新村到双子塔的地铁和驾车通勤路线。",
        )

        self.assertEqual(config.announced_tools, ["web_search", "url_read"])
        self.assertNotIn("route_compare", config.announced_tools)
        self.assertEqual(config.required_initial_tool_counts, {})
        self.assertEqual(config.plan_tool_policy_reason, "deep_research_schedulable_tools")
        update_plan = next(tool for tool in config.call_kwargs["tools"] if tool["function"]["name"] == "update_plan")
        planned_tool_schema = update_plan["function"]["parameters"]["properties"]["plan"]["items"]["properties"][
            "planned_tools"
        ]["items"]
        self.assertEqual(planned_tool_schema["enum"], ["web_search", "url_read"])

        coordinator = PlanCoordinator(
            run_id="run-deep-tool-policy",
            mode="on",
            allowed_tool_names=frozenset(config.announced_tools),
            required_initial_tool_counts={"web_search": 1, "url_read": 2},
        )
        route_plan = coordinator.apply_model_update(
            {
                "reason": "错误地把阶段调度器不会开放的路线工具写入研究计划",
                "items": [
                    {
                        "id": "search",
                        "title": "搜索候选来源",
                        "status": "pending",
                        "kind": "search",
                        "depends_on": [],
                        "planned_tools": ["web_search"],
                    },
                    {
                        "id": "read-1",
                        "title": "核验来源一",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "read-2",
                        "title": "核验来源二",
                        "status": "pending",
                        "kind": "read",
                        "depends_on": ["search"],
                        "planned_tools": ["url_read"],
                    },
                    {
                        "id": "route",
                        "title": "比较路线",
                        "status": "pending",
                        "kind": "other",
                        "depends_on": [],
                        "planned_tools": ["route_compare"],
                    },
                    {
                        "id": "answer",
                        "title": "整理研究结论",
                        "status": "pending",
                        "kind": "answer",
                        "depends_on": ["read-1", "read-2", "route"],
                        "planned_tools": [],
                    },
                ],
            }
        )

        self.assertFalse(route_plan.accepted)
        self.assertEqual(route_plan.reason, "unannounced_planned_tool")

    def test_plan_mode_defaults_auto_and_control_tool_is_hidden_from_user_tool_list(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="OpenAI 今天发布了什么？阅读官方公告后总结",
        )

        model_tool_names = [tool["function"]["name"] for tool in config.call_kwargs["tools"]]
        self.assertEqual(config.plan_mode, "auto")
        self.assertIn("update_plan", model_tool_names)
        self.assertIn("web_search", model_tool_names)
        self.assertIn("url_read", model_tool_names)
        self.assertEqual(config.announced_tools, ["web_search", "url_read"])
        self.assertNotIn("update_plan", config.announced_tools)
        self.assertEqual(config.control_tool_names, frozenset({"update_plan"}))
        update_plan = next(tool for tool in config.call_kwargs["tools"] if tool["function"]["name"] == "update_plan")
        status_schema = update_plan["function"]["parameters"]["properties"]["plan"]["items"]["properties"]["status"]
        self.assertEqual(status_schema["enum"], ["pending", "in_progress"])

    def test_standard_verified_research_constrains_first_plan_to_search_and_reads(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "on"},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="请联网调研韩国股市，梳理主要争议并给出可靠来源。",
        )

        self.assertEqual(config.announced_tools, ["web_search", "url_read"])
        self.assertEqual(
            config.required_initial_tool_counts,
            {"web_search": 1, "url_read": 2},
        )
        self.assertEqual(config.plan_tool_policy_reason, "verified_research_request")
        update_plan = next(tool for tool in config.call_kwargs["tools"] if tool["function"]["name"] == "update_plan")
        planned_tools = update_plan["function"]["parameters"]["properties"]["plan"]["items"]["properties"][
            "planned_tools"
        ]["items"]
        self.assertEqual(planned_tools["enum"], ["web_search", "url_read"])

    async def test_verified_research_injects_initial_research_dag_contract_only_for_matching_policy(self):
        async def build_llm_messages_fn(
            _raw_messages,
            _has_vision,
            _repo,
            _user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            return [{"role": "user", "content": "原问题"}]

        async def prepare(message: str):
            return await prepare_agent_loop_messages(
                db=object(),
                user_id="user-1",
                raw_messages=[],
                has_vision=False,
                file_ids=None,
                original_message=message,
                call_config=build_agent_loop_call_config(
                    provider="openai",
                    options={"plan_mode": "on"},
                    capabilities={"functionCalling": True, "searchCapable": True},
                    original_message=message,
                ),
                file_repo_factory=lambda _db: object(),
                load_user_system_prompt_fn=lambda _db, _user_id: None,
                build_llm_messages_fn=build_llm_messages_fn,
                preprocess_user_input=False,
            )

        verified = await prepare("请联网调研韩国股市，并给出可靠来源。")
        simple = await prepare("KOSPI 今天多少点？")
        verified_system_text = "\n".join(
            str(message.get("content", "")) for message in verified.messages if message.get("role") == "system"
        )
        simple_system_text = "\n".join(
            str(message.get("content", "")) for message in simple.messages if message.get("role") == "system"
        )

        self.assertIn("【可核验证据计划规则】", verified_system_text)
        self.assertIn("至少 1 个 web_search", verified_system_text)
        self.assertIn("至少 2 个独立的 url_read", verified_system_text)
        self.assertIn("直接或间接依赖 web_search", verified_system_text)
        self.assertIn("最终 answer 或 synthesis", verified_system_text)
        self.assertNotIn("【可核验证据计划规则】", simple_system_text)

    def test_deep_research_forces_plan_mode_and_records_task_policy(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"task_mode": "deep_research", "plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        self.assertEqual(config.task_mode, "deep_research")
        self.assertEqual(config.plan_mode, "on")
        self.assertEqual(config.network_profile, "deep_research")
        self.assertEqual(config.evidence_policy, "deep_research_v1")
        tools = {tool["function"]["name"]: tool for tool in config.call_kwargs["tools"]}
        self.assertIn("url_read", tools)
        self.assertIn("_plan_item_id", tools["url_read"]["function"]["parameters"]["required"])
        self.assertEqual(config.announced_tools, ["web_search", "url_read"])

    def test_deep_research_contract_is_only_injected_for_research_mode(self):
        research = build_agent_loop_call_config(
            provider="openai",
            options={"task_mode": "deep_research"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )
        standard = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        research_messages = inject_deep_research_contract([{"role": "user", "content": "调研"}], research)
        standard_messages = inject_deep_research_contract([{"role": "user", "content": "调研"}], standard)

        self.assertIn("【深度研究执行约束】", research_messages[0]["content"])
        self.assertIn("互补查询", research_messages[0]["content"])
        self.assertIn("正文使用 [n] 引用", research_messages[0]["content"])
        self.assertIn("planned_tools 必须覆盖一个 web_search 步骤", research_messages[0]["content"])
        self.assertIn("至少两个独立的 url_read 步骤", research_messages[0]["content"])
        self.assertNotIn("同一个读取步骤可以读取多个独立来源", research_messages[0]["content"])
        self.assertIn("每个读取步骤负责一个独立来源任务", research_messages[0]["content"])
        self.assertIn("只有服务端将同一任务保持为 retryable/running 时", research_messages[0]["content"])
        self.assertIn("跨轮重试", research_messages[0]["content"])
        self.assertNotIn("每个读取步骤只读取一个来源", research_messages[0]["content"])
        self.assertIn("web_search 与 url_read 必须由不同计划步骤负责", research_messages[0]["content"])
        self.assertEqual(standard_messages, [{"role": "user", "content": "调研"}])

    def test_plan_mode_off_preserves_old_tools_without_control_tool(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="今天上海证券交易所开市吗？",
        )

        model_tool_names = [tool["function"]["name"] for tool in config.call_kwargs["tools"]]
        self.assertEqual(config.plan_mode, "off")
        self.assertNotIn("update_plan", model_tool_names)
        self.assertEqual(config.announced_tools, ["web_search"])
        parameters = config.call_kwargs["tools"][0]["function"]["parameters"]
        self.assertNotIn("_plan_item_id", parameters["properties"])

    def test_on_mode_requires_explicit_plan_item_binding_on_external_tools(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "on"},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="今天上海证券交易所开市吗？",
        )

        tools = {tool["function"]["name"]: tool for tool in config.call_kwargs["tools"]}
        web_parameters = tools["web_search"]["function"]["parameters"]
        plan_parameters = tools["update_plan"]["function"]["parameters"]
        plan_item = plan_parameters["properties"]["plan"]["items"]

        self.assertIn("_plan_item_id", web_parameters["properties"])
        self.assertIn("_plan_item_id", web_parameters["required"])
        self.assertEqual(web_parameters["properties"]["_plan_item_id"]["type"], "string")
        self.assertIn("id", plan_item["required"])
        self.assertIn("planned_tools", plan_item["required"])
        self.assertEqual(plan_parameters["properties"]["plan"]["maxItems"], 6)
        self.assertEqual(
            plan_item["properties"]["id"]["pattern"],
            "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        )
        self.assertEqual(
            web_parameters["properties"]["_plan_item_id"]["pattern"],
            "^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$",
        )
        self.assertNotIn("_plan_item_id", plan_parameters["properties"])

    def test_auto_mode_exposes_optional_plan_item_binding_without_breaking_simple_tools(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "auto"},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="OpenAI 今天发布了什么？阅读官方公告后总结",
        )

        web_tool = next(tool for tool in config.call_kwargs["tools"] if tool["function"]["name"] == "web_search")
        parameters = web_tool["function"]["parameters"]

        self.assertIn("_plan_item_id", parameters["properties"])
        self.assertNotIn("_plan_item_id", parameters.get("required", []))

    def test_control_plan_tool_only_requires_function_calling_not_search_capability(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "on"},
            capabilities={"functionCalling": True, "searchCapable": False},
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in config.call_kwargs["tools"]],
            ["update_plan"],
        )
        self.assertEqual(config.announced_tools, [])

    def test_on_mode_without_external_tools_constrains_planned_tools_to_empty_arrays(self):
        for message, expected_package in (
            ("你好", "direct"),
            ("帮我查一下这个", "clarification_only"),
        ):
            with self.subTest(message=message):
                config = build_agent_loop_call_config(
                    provider="openai",
                    options={"plan_mode": "on"},
                    capabilities={"functionCalling": True, "searchCapable": True},
                    original_message=message,
                )

                update_plan = next(
                    tool for tool in config.call_kwargs["tools"] if tool["function"]["name"] == "update_plan"
                )
                parameters = update_plan["function"]["parameters"]
                planned_tools = parameters["properties"]["plan"]["items"]["properties"]["planned_tools"]
                empty_plan = {
                    "plan": [
                        {"id": "step-1", "step": "理解请求", "status": "pending", "planned_tools": []},
                        {"id": "step-2", "step": "直接回答", "status": "pending", "planned_tools": []},
                    ]
                }
                invalid_plan = {
                    "plan": [
                        {
                            "id": "step-1",
                            "step": "错误调用未公告工具",
                            "status": "pending",
                            "planned_tools": ["web_search"],
                        },
                        {"id": "step-2", "step": "直接回答", "status": "pending", "planned_tools": []},
                    ]
                }

                self.assertEqual(config.capability_resolution.package_id, expected_package)
                self.assertEqual(config.announced_tools, [])
                self.assertEqual(planned_tools["maxItems"], 0)
                self.assertTrue(
                    all(len(item["planned_tools"]) <= planned_tools["maxItems"] for item in empty_plan["plan"])
                )
                self.assertFalse(
                    all(len(item["planned_tools"]) <= planned_tools["maxItems"] for item in invalid_plan["plan"])
                )

    def test_requested_on_mode_defensively_disables_when_model_cannot_call_control_tool(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "on"},
            capabilities={"functionCalling": False, "searchCapable": False},
        )

        self.assertEqual(config.plan_mode, "off")
        self.assertNotIn("tools", config.call_kwargs)
        self.assertEqual(config.control_tool_names, frozenset())

    def test_auto_plan_contract_requires_plan_for_itinerary_and_multi_tool_tasks(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True},
            original_message="OpenAI 今天发布了什么？阅读官方公告后总结",
        )
        messages = [{"role": "user", "content": "规划通勤路线"}]

        prepared = inject_plan_control_contract(messages, config)

        self.assertEqual(prepared[0]["role"], "system")
        contract = prepared[0]["content"]
        self.assertIn("【执行计划控制规则】", contract)
        self.assertIn("首次调用外部工具前必须先创建计划", contract)
        self.assertIn("行程规划、方案比较、调研、审查", contract)
        self.assertIn("两次或以上外部工具调用", contract)
        self.assertIn("一次独立事实查询", contract)
        self.assertIn("不要自行把步骤标成 completed", contract)
        self.assertIn("不要向用户叙述拒绝原因", contract)
        self.assertIn("不得以计划失败为由越过门禁", contract)
        self.assertNotIn("update_plan", contract)

    def test_on_plan_contract_requires_plan_before_any_answer_or_external_tool(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "on"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        prepared = inject_plan_control_contract([{"role": "user", "content": "你好"}], config)

        self.assertIn("本轮启用了强制计划模式", prepared[0]["content"])
        self.assertIn("回答或调用任何外部工具前", prepared[0]["content"])

    def test_plan_contract_is_not_injected_when_plan_mode_is_off(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": True},
        )
        messages = [{"role": "user", "content": "规划通勤路线"}]

        self.assertIs(inject_plan_control_contract(messages, config), messages)

    def test_build_call_config_applies_controlled_max_tokens(self):
        for raw_value, expected in ((1, 1), (1024, 1024), (9999, 4096)):
            with self.subTest(raw_value=raw_value):
                config = build_agent_loop_call_config(
                    provider="openai",
                    options={"max_tokens": raw_value},
                    capabilities={"functionCalling": False},
                )

                self.assertEqual(config.call_kwargs["max_tokens"], expected)

    def test_build_call_config_ignores_invalid_max_tokens(self):
        for raw_value in (True, False, 0, -1, 1.5, "1024", None):
            with self.subTest(raw_value=raw_value):
                config = build_agent_loop_call_config(
                    provider="openai",
                    options={"max_tokens": raw_value},
                    capabilities={"functionCalling": False},
                )

                self.assertNotIn("max_tokens", config.call_kwargs)

    def test_build_call_config_can_disable_supported_tools(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"disable_tools": True},
            capabilities={"functionCalling": True, "searchCapable": True},
        )

        self.assertFalse(config.supports_function_calling)
        self.assertEqual(config.announced_tools, [])
        self.assertNotIn("tools", config.call_kwargs)
        self.assertNotIn("tool_choice", config.call_kwargs)

    def test_build_call_config_enables_tools_and_volcengine_reasoning_compat(self):
        config = build_agent_loop_call_config(
            provider="volcengine",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": True, "deepThinking": True},
            original_message="今天上海证券交易所开市吗？",
        )

        self.assertTrue(config.should_use_reasoning)
        self.assertTrue(config.supports_function_calling)
        self.assertEqual(config.announced_tools, ["web_search"])
        self.assertEqual(config.call_kwargs["tool_choice"], "auto")
        self.assertEqual(config.call_kwargs["tools"][0]["function"]["name"], "web_search")
        self.assertEqual(config.call_kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_deepseek_plan_mode_enables_thinking_and_removes_incompatible_tool_choice(self):
        config = build_agent_loop_call_config(
            provider="deepseek",
            options={"plan_mode": "on"},
            capabilities={
                "functionCalling": True,
                "agentTools": True,
                "searchCapable": True,
                "deepThinking": True,
            },
        )

        self.assertTrue(config.should_use_reasoning)
        self.assertEqual(config.call_kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("tool_choice", config.call_kwargs)

    def test_moonshot_plan_mode_keeps_native_thinking_configuration(self):
        config = build_agent_loop_call_config(
            provider="moonshot",
            options={"plan_mode": "on"},
            capabilities={
                "functionCalling": True,
                "agentTools": True,
                "searchCapable": True,
                "deepThinking": True,
            },
        )

        self.assertTrue(config.should_use_reasoning)
        self.assertNotIn("extra_body", config.call_kwargs)

    def test_deepseek_non_plan_search_uses_explicit_thinking_protocol(self):
        config = build_agent_loop_call_config(
            provider="deepseek",
            options={"plan_mode": "off"},
            capabilities={
                "functionCalling": True,
                "agentTools": True,
                "searchCapable": True,
                "deepThinking": True,
            },
        )

        self.assertEqual(config.call_kwargs["extra_body"], {"thinking": {"type": "enabled"}})
        self.assertNotIn("tool_choice", config.call_kwargs)

    def test_gemini_reasoning_models_request_visible_thought_summaries(self):
        config = build_agent_loop_call_config(
            provider="gemini",
            options={"plan_mode": "on"},
            capabilities={
                "functionCalling": True,
                "agentTools": True,
                "searchCapable": True,
                "deepThinking": True,
            },
        )

        self.assertTrue(config.should_use_reasoning)
        self.assertEqual(config.call_kwargs["reasoning_effort"], "high")
        self.assertEqual(config.call_kwargs["tool_choice"], "auto")

    def test_build_call_config_respects_explicit_reasoning_override(self):
        config = build_agent_loop_call_config(
            provider="volcengine",
            options={"use_reasoning": False},
            capabilities={"functionCalling": True, "searchCapable": True, "deepThinking": True},
            original_message="OpenAI 今天发布了什么？阅读官方公告后总结",
        )

        self.assertFalse(config.should_use_reasoning)
        self.assertTrue(config.supports_function_calling)
        self.assertEqual(config.announced_tools, ["web_search", "url_read"])
        self.assertNotIn("extra_body", config.call_kwargs)

    def test_build_call_config_disables_agent_tools_when_agent_tools_capability_is_false(self):
        config = build_agent_loop_call_config(
            provider="qwen",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "agentTools": False, "deepThinking": False},
        )

        self.assertFalse(config.supports_function_calling)
        self.assertEqual(config.announced_tools, [])
        self.assertNotIn("tools", config.call_kwargs)
        self.assertNotIn("tool_choice", config.call_kwargs)

    def test_build_call_config_uses_search_capable_as_runtime_tool_contract(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "agentTools": False, "searchCapable": True},
            original_message="今天上海证券交易所开市吗？",
        )

        self.assertTrue(config.supports_function_calling)
        self.assertEqual(config.announced_tools, ["web_search"])
        self.assertEqual(config.call_kwargs["tool_choice"], "auto")

    def test_build_call_config_disables_tools_when_search_capable_is_false(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "agentTools": True, "webSearch": True, "searchCapable": False},
        )

        self.assertFalse(config.supports_function_calling)
        self.assertEqual(config.announced_tools, [])
        self.assertNotIn("tools", config.call_kwargs)
        self.assertNotIn("tool_choice", config.call_kwargs)

    def test_build_call_config_injects_mcp_tools_for_function_calling_model_without_search(self):
        mcp_tool = {
            "type": "function",
            "function": {
                "name": "mcp_microsoft_docs_a1b2c3d4",
                "description": "搜索 Microsoft Learn 文档",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
        handler = object()
        binding = {"alias": "mcp_microsoft_docs_a1b2c3d4", "server_id": "server-1"}

        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "searchCapable": False},
            additional_tools=[mcp_tool],
            dynamic_tool_handlers={"mcp_microsoft_docs_a1b2c3d4": handler},
            tool_bindings=[binding],
            original_message="请使用 mcp_microsoft_docs_a1b2c3d4 查询 Microsoft Learn",
        )

        self.assertFalse(config.supports_function_calling)
        self.assertTrue(config.supports_dynamic_tools)
        self.assertEqual(config.announced_tools, ["mcp_microsoft_docs_a1b2c3d4"])
        self.assertEqual(config.call_kwargs["tools"], [mcp_tool])
        self.assertEqual(config.call_kwargs["tool_choice"], "auto")
        self.assertIs(config.dynamic_tool_handlers["mcp_microsoft_docs_a1b2c3d4"], handler)
        self.assertEqual(config.tool_bindings, [binding])

    def test_build_call_config_injects_stable_amap_product_tool_without_false_network_boundary(self):
        product_tool = {
            "type": "function",
            "function": {
                "name": "local_place_search",
                "parameters": {"type": "object", "additionalProperties": False},
            },
        }
        handler = object()

        config = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "agentTools": True, "searchCapable": False},
            additional_tools=[product_tool],
            dynamic_tool_handlers={"local_place_search": handler},
            tool_bindings=[{"alias": "local_place_search", "server_id": "amap-1"}],
            original_message="搜索民治附近的咖啡店",
        )
        messages = [{"role": "user", "content": "搜索民治附近的咖啡店"}]

        self.assertEqual(config.announced_tools, ["local_place_search"])
        self.assertIs(config.dynamic_tool_handlers["local_place_search"], handler)
        self.assertIs(inject_no_tool_network_boundary(messages, config.call_kwargs), messages)

    def test_build_call_config_respects_explicit_agent_tools_capability_for_mcp(self):
        mcp_tool = {
            "type": "function",
            "function": {"name": "mcp_docs_a1b2c3d4", "parameters": {"type": "object"}},
        }

        config = build_agent_loop_call_config(
            provider="openai",
            options={"plan_mode": "off"},
            capabilities={"functionCalling": True, "agentTools": False, "searchCapable": False},
            additional_tools=[mcp_tool],
            dynamic_tool_handlers={"mcp_docs_a1b2c3d4": object()},
            tool_bindings=[{"alias": "mcp_docs_a1b2c3d4"}],
        )

        self.assertFalse(config.supports_dynamic_tools)
        self.assertEqual(config.dynamic_tool_handlers, {})
        self.assertEqual(config.tool_bindings, [])
        self.assertNotIn("tools", config.call_kwargs)

    def test_build_call_config_disable_tools_blocks_mcp_tools_too(self):
        mcp_tool = {
            "type": "function",
            "function": {"name": "mcp_docs_a1b2c3d4", "parameters": {"type": "object"}},
        }

        config = build_agent_loop_call_config(
            provider="openai",
            options={"disable_tools": True},
            capabilities={"functionCalling": True, "searchCapable": True},
            additional_tools=[mcp_tool],
            dynamic_tool_handlers={"mcp_docs_a1b2c3d4": object()},
            tool_bindings=[{"alias": "mcp_docs_a1b2c3d4"}],
        )

        self.assertFalse(config.supports_function_calling)
        self.assertFalse(config.supports_dynamic_tools)
        self.assertEqual(config.dynamic_tool_handlers, {})
        self.assertEqual(config.tool_bindings, [])
        self.assertNotIn("tools", config.call_kwargs)

    async def test_authorized_mcp_alias_degrades_atomically_when_execution_is_unavailable(self):
        alias = "mcp_docs_a1b2c3d4"
        mcp_tool = {
            "type": "function",
            "function": {
                "name": alias,
                "description": "搜索 Microsoft Learn 文档",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
        handler = object()
        binding = {"alias": alias, "server_id": "server-1"}
        message = f"请使用 {alias} 查询 Microsoft Learn"

        async def build_llm_messages_fn(
            _raw_messages,
            _has_vision,
            _repo,
            _user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            return [{"role": "user", "content": message}]

        for options, capabilities, expected_reason in (
            (
                {"disable_tools": True},
                {"functionCalling": True, "searchCapable": True},
                "tools_disabled",
            ),
            (
                {},
                {"functionCalling": False, "searchCapable": False},
                "function_calling_unavailable",
            ),
        ):
            with self.subTest(expected_reason=expected_reason):
                config = build_agent_loop_call_config(
                    provider="openai",
                    options=options,
                    capabilities=capabilities,
                    additional_tools=[mcp_tool],
                    dynamic_tool_handlers={alias: handler},
                    tool_bindings=[binding],
                    authorized_tool_names=[alias],
                    original_message=message,
                )
                prepared = await prepare_agent_loop_messages(
                    db=object(),
                    user_id="user-1",
                    raw_messages=[],
                    has_vision=False,
                    file_ids=None,
                    original_message=message,
                    call_config=config,
                    file_repo_factory=lambda _db: object(),
                    load_user_system_prompt_fn=lambda _db, _user_id: None,
                    build_llm_messages_fn=build_llm_messages_fn,
                    preprocess_user_input=False,
                )

                self.assertEqual(config.capability_resolution.package_id, "tools_unavailable")
                self.assertEqual(config.capability_resolution.resolution_mode, "degraded")
                self.assertEqual(config.capability_resolution.reason_codes, (expected_reason,))
                self.assertTrue(config.capability_resolution.network_boundary_required)
                self.assertEqual(config.capability_resolution.external_tool_names, ())
                self.assertNotIn("tools", config.call_kwargs)
                self.assertEqual(config.dynamic_tool_handlers, {})
                self.assertEqual(config.tool_bindings, [])
                self.assertEqual(config.announced_tools, [])
                self.assertEqual(prepared.final_tool_names, [])
                self.assertEqual(
                    prepared.prompt_assembly["section_ids"],
                    ["app_identity", "no_tool_network_boundary"],
                )

        unauthorized = build_agent_loop_call_config(
            provider="openai",
            options={"disable_tools": True},
            capabilities={"functionCalling": True, "searchCapable": True},
            authorized_tool_names=[alias],
            original_message="请使用 mcp_unapproved_deadbeef 查询秘密资料",
        )
        self.assertEqual(unauthorized.capability_resolution.package_id, "clarification_only")
        self.assertFalse(unauthorized.capability_resolution.network_boundary_required)

    def test_authorized_mcp_alias_degrades_when_agent_tools_are_unsupported(self):
        alias = "mcp_docs_a1b2c3d4"
        config = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "searchCapable": True, "agentTools": False},
            authorized_tool_names=[alias],
            original_message=f"请调用 {alias} 查询 Microsoft Learn",
        )

        self.assertEqual(config.capability_resolution.package_id, "tools_unavailable")
        self.assertEqual(config.capability_resolution.reason_codes, ("required_tools_unavailable",))
        self.assertTrue(config.capability_resolution.network_boundary_required)
        self.assertEqual(config.capability_resolution.external_tool_names, ())
        self.assertEqual(config.announced_tools, [])
        self.assertNotIn("tools", config.call_kwargs)

    def test_mcp_tool_prevents_false_no_network_boundary(self):
        messages = [{"role": "user", "content": "查一下 Microsoft Learn"}]
        call_kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "mcp_microsoft_docs_a1b2c3d4", "parameters": {"type": "object"}},
                }
            ]
        }

        self.assertIs(inject_no_tool_network_boundary(messages, call_kwargs), messages)

    async def test_prepare_messages_injects_no_tool_network_boundary_when_agent_tools_disabled(self):
        async def build_llm_messages_fn(
            _raw_messages,
            _has_vision,
            _repo,
            _user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            return [
                {"role": "user", "content": "OpenAI 最近发布了什么模型？"},
            ]

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=["raw"],
            has_vision=False,
            file_ids=None,
            original_message="OpenAI 最近发布了什么模型？",
            call_config=build_agent_loop_call_config(
                provider="qwen",
                options={},
                capabilities={"functionCalling": True, "agentTools": False},
                original_message="OpenAI 最近发布了什么模型？",
            ),
            file_repo_factory=lambda _db: object(),
            load_user_system_prompt_fn=lambda _db, _user_id: None,
            build_llm_messages_fn=build_llm_messages_fn,
        )

        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "system", "user"],
        )
        self.assertIn("【Fusion 身份一致性规则】", prepared.messages[0]["content"])
        self.assertIn("【无联网工具边界规则】", prepared.messages[1]["content"])
        self.assertIn("不要声称已经搜索", prepared.messages[1]["content"])
        self.assertIn("无法实时核验", prepared.messages[1]["content"])
        self.assertIn("不要把已有知识包装成最新事实", prepared.messages[1]["content"])
        self.assertIn("普通稳定问题直接回答", prepared.messages[1]["content"])
        self.assertNotIn("切换模型", prepared.messages[1]["content"])
        self.assertNotIn("【工具调用一致性规则】", prepared.messages[1]["content"])
        self.assertIn("【当前真实日期】", prepared.messages[2]["content"])
        self.assertEqual(prepared.messages[3]["content"], "OpenAI 最近发布了什么模型？")

    async def test_prepare_messages_injects_no_vision_boundary_when_image_attached_to_text_model(self):
        async def build_llm_messages_fn(
            _raw_messages,
            _has_vision,
            _repo,
            _user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            return [
                {"role": "user", "content": "这张图里有什么？"},
            ]

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=["raw"],
            has_vision=False,
            file_ids=["image-1"],
            original_message="这张图里有什么？",
            call_config=build_agent_loop_call_config(
                provider="qwen",
                options={},
                capabilities={"functionCalling": True, "agentTools": False, "vision": False},
                original_message="这张图里有什么？",
            ),
            file_repo_factory=lambda _db: object(),
            load_user_system_prompt_fn=lambda _db, _user_id: None,
            build_llm_messages_fn=build_llm_messages_fn,
            is_image_file_fn=lambda file_id, _repo: file_id == "image-1",
        )

        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "user"],
        )
        self.assertIn("【无图片理解能力边界规则】", prepared.messages[1]["content"])
        self.assertIn("当前模型不能读取或理解图片附件", prepared.messages[1]["content"])
        self.assertIn("不要臆测图片内容", prepared.messages[1]["content"])
        self.assertEqual(prepared.messages[2]["content"], "这张图里有什么？")

    async def test_prepare_messages_builds_llm_input_files_url_context_and_tool_contract(self):
        file_repo = FakeFileRepository()
        build_calls = []
        inject_calls = []

        async def build_llm_messages_fn(
            raw_messages,
            has_vision,
            repo,
            user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            build_calls.append(
                {
                    "raw_messages": raw_messages,
                    "has_vision": has_vision,
                    "repo": repo,
                    "user_system_prompt": user_system_prompt,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }
            )
            return [
                {"role": "user", "content": "原始问题"},
            ]

        def inject_file_content_fn(messages, original_message, file_contents):
            inject_calls.append(
                {
                    "messages": list(messages),
                    "original_message": original_message,
                    "file_contents": file_contents,
                }
            )
            result = list(messages)
            result[-1] = {"role": "user", "content": f"{original_message}\n\n{file_contents['doc-1']}"}
            return result

        async def preprocess_url_in_message_fn(original_message, supports_function_calling, call_kwargs):
            self.assertEqual(original_message, "请阅读 https://example.com/a")
            self.assertTrue(supports_function_calling)
            self.assertEqual(
                [tool["function"]["name"] for tool in call_kwargs["tools"]],
                ["url_read"],
            )
            return (
                TextBlock(type="text", id="url-block", text="URL 摘要"),
                {"role": "user", "content": "<web_context>网页正文</web_context>"},
                "https://example.com/a",
            )

        call_config = build_agent_loop_call_config(
            provider="openai",
            options={"use_reasoning": True},
            capabilities={"functionCalling": True, "searchCapable": True, "deepThinking": True},
            original_message="请阅读 https://example.com/a",
        )

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=["raw"],
            has_vision=False,
            file_ids=["doc-1", "image-1"],
            original_message="请阅读 https://example.com/a",
            call_config=call_config,
            file_repo_factory=lambda _db: file_repo,
            load_user_system_prompt_fn=lambda _db, _user_id: "用户偏好",
            build_llm_messages_fn=build_llm_messages_fn,
            is_image_file_fn=lambda file_id, _repo: file_id == "image-1",
            inject_file_content_fn=inject_file_content_fn,
            preprocess_url_in_message_fn=preprocess_url_in_message_fn,
        )

        self.assertIsNone(build_calls[0]["user_system_prompt"])
        self.assertIn("用户偏好", prepared.messages[2]["content"])
        self.assertIs(build_calls[0]["repo"], file_repo)
        self.assertEqual(build_calls[0]["user_id"], "user-1")
        self.assertIsNone(build_calls[0]["conversation_id"])
        self.assertEqual(file_repo.requested_content_ids, [["doc-1"]])
        self.assertEqual(inject_calls[0]["file_contents"], {"doc-1": "文档正文"})
        self.assertEqual([block.id for block in prepared.initial_content_blocks], ["url-block"])
        self.assertEqual(prepared.final_tool_names, ["url_read"])
        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "system", "user", "user"],
        )
        self.assertIn("【Fusion 身份一致性规则】", prepared.messages[0]["content"])
        self.assertIn("【无图片理解能力边界规则】", prepared.messages[1]["content"])
        self.assertIn("<web_context>", prepared.messages[3]["content"])
        self.assertIn("文档正文", prepared.messages[4]["content"])
        self.assertEqual(call_config.announced_tools, ["url_read"])

    def test_tool_usage_contract_uses_centralized_prompt(self):
        from app.ai.prompts.agent_loop import NETWORK_DECISION_PROMPT, TOOL_USAGE_CONTRACT_PROMPT
        from app.services.stream.agent_loop_request_prep import inject_tool_usage_contract

        messages = [{"role": "user", "content": "OpenAI 最新公告"}]
        call_kwargs = {"tools": [{"type": "function", "function": {"name": "web_search"}}]}

        prepared = inject_tool_usage_contract(messages, call_kwargs)

        self.assertEqual(prepared[0], {"role": "system", "content": TOOL_USAGE_CONTRACT_PROMPT})
        self.assertIn(NETWORK_DECISION_PROMPT, TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("必须调用 web_search", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("没有调用工具", TOOL_USAGE_CONTRACT_PROMPT)

    def test_no_tool_network_boundary_uses_centralized_prompt(self):
        from app.ai.prompts.agent_loop import NO_TOOL_NETWORK_BOUNDARY_PROMPT
        from app.services.stream.agent_loop_request_prep import inject_no_tool_network_boundary

        messages = [{"role": "user", "content": "OpenAI 最近公告"}]
        prepared = inject_no_tool_network_boundary(messages, call_kwargs={})

        self.assertEqual(prepared[0], {"role": "system", "content": NO_TOOL_NETWORK_BOUNDARY_PROMPT})
        self.assertIn("没有联网搜索或网页读取工具", NO_TOOL_NETWORK_BOUNDARY_PROMPT)
        self.assertIn("不要声称已经搜索", NO_TOOL_NETWORK_BOUNDARY_PROMPT)
        self.assertIn("无法实时核验", NO_TOOL_NETWORK_BOUNDARY_PROMPT)
        self.assertIn("不要把已有知识包装成最新事实", NO_TOOL_NETWORK_BOUNDARY_PROMPT)
        self.assertIn("不要把缺少工具描述成系统故障", NO_TOOL_NETWORK_BOUNDARY_PROMPT)
        self.assertIn("普通稳定问题直接回答", NO_TOOL_NETWORK_BOUNDARY_PROMPT)
        self.assertNotIn("切换模型", NO_TOOL_NETWORK_BOUNDARY_PROMPT)

    def test_tool_usage_contract_defines_autonomous_search_decision_matrix(self):
        from app.ai.prompts.agent_loop import TOOL_USAGE_CONTRACT_PROMPT

        self.assertIn("不要依据用户是否说了", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("联网", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("搜索", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("微信A2A互通怎么用？", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("OpenAI 最近发布了哪些产品更新？", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("你好，你是谁？", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("1+1等于几？", TOOL_USAGE_CONTRACT_PROMPT)
        self.assertIn("不应调用 web_search", TOOL_USAGE_CONTRACT_PROMPT)

    async def test_prepare_messages_injects_extra_system_prompts_without_user_preprocess(self):
        async def build_llm_messages_fn(
            _raw_messages,
            _has_vision,
            _repo,
            _user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            return [
                {"role": "user", "content": "原问题"},
                {"role": "assistant", "content": "旧回答"},
            ]

        async def should_not_preprocess_url(*_args, **_kwargs):
            raise AssertionError("continuation 不应重新跑 URL 预处理")

        def should_not_inject_file_content(*_args, **_kwargs):
            raise AssertionError("continuation 不应重新跑文件预处理")

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=[],
            has_vision=False,
            file_ids=["file-1"],
            original_message="https://example.com",
            call_config=build_agent_loop_call_config(
                provider="openai",
                options={},
                capabilities={"functionCalling": False},
                original_message="https://example.com",
            ),
            file_repo_factory=lambda _db: object(),
            load_user_system_prompt_fn=lambda _db, _user_id: None,
            build_llm_messages_fn=build_llm_messages_fn,
            is_image_file_fn=lambda _file_id, _repo: False,
            inject_file_content_fn=should_not_inject_file_content,
            preprocess_url_in_message_fn=should_not_preprocess_url,
            preprocess_user_input=False,
            extra_system_prompts=["继续执行，不要重写前文"],
        )

        self.assertEqual(prepared.initial_content_blocks, [])
        self.assertEqual(prepared.messages[1], {"role": "system", "content": "继续执行，不要重写前文"})
        self.assertEqual(prepared.messages[2]["role"], "user")

    async def test_prepare_messages_passes_conversation_scope_to_builder(self):
        build_calls = []

        async def build_llm_messages_fn(
            raw_messages,
            has_vision,
            repo,
            user_system_prompt,
            *,
            user_id=None,
            conversation_id=None,
            include_base_system=True,
        ):
            build_calls.append(
                {
                    "raw_messages": raw_messages,
                    "has_vision": has_vision,
                    "repo": repo,
                    "user_system_prompt": user_system_prompt,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }
            )
            return [{"role": "user", "content": "看图"}]

        file_repo = object()

        await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            conversation_id="conv-1",
            raw_messages=["raw"],
            has_vision=True,
            file_ids=None,
            original_message="看图",
            call_config=build_agent_loop_call_config(
                provider="qwen",
                options={},
                capabilities={"functionCalling": False, "vision": True},
            ),
            file_repo_factory=lambda _db: file_repo,
            load_user_system_prompt_fn=lambda _db, _user_id: "用户偏好",
            build_llm_messages_fn=build_llm_messages_fn,
            preprocess_user_input=False,
        )

        self.assertEqual(
            build_calls,
            [
                {
                    "raw_messages": ["raw"],
                    "has_vision": True,
                    "repo": file_repo,
                    "user_system_prompt": None,
                    "user_id": "user-1",
                    "conversation_id": "conv-1",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
