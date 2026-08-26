import unittest

from app.schemas.chat import TextBlock
from app.services.agent.plan_coordinator import PlanCoordinator
from app.services.mcp.amap_product_tools import AMAP_PRODUCT_DEFINITIONS
from app.services.stream.agent_loop_request_prep import (
    build_agent_loop_call_config,
    inject_amap_fact_boundary,
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
        from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig

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
            call_config=AgentLoopCallConfig(False, False, {}, []),
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
        )
        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=[],
            has_vision=False,
            file_ids=None,
            original_message="测试",
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

    async def test_assembly_sections_follow_actual_capabilities_and_modes(self):
        from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig

        for plan_mode, task_mode, tools, expected in [
            ("off", "standard", [], ["current_date", "app_identity", "no_tool_network_boundary"]),
            (
                "on",
                "standard",
                ["web_search", "update_plan"],
                ["current_date", "app_identity", "tool_usage_contract", "agent_plan_control"],
            ),
            (
                "on",
                "deep_research",
                ["web_search", "url_read", "update_plan"],
                ["current_date", "app_identity", "tool_usage_contract", "agent_plan_control", "deep_research_contract"],
            ),
        ]:
            with self.subTest(plan_mode=plan_mode, task_mode=task_mode):
                config = AgentLoopCallConfig(
                    should_use_reasoning=False,
                    supports_function_calling=bool(tools),
                    call_kwargs={"tools": [{"type": "function", "function": {"name": name}} for name in tools]},
                    announced_tools=tools,
                    plan_mode=plan_mode,
                    task_mode=task_mode,
                    control_tool_names=frozenset({"update_plan"}) if "update_plan" in tools else frozenset(),
                )
                prepared = await prepare_agent_loop_messages(
                    db=object(),
                    user_id="user-1",
                    raw_messages=[],
                    has_vision=False,
                    file_ids=None,
                    original_message="测试",
                    call_config=config,
                    file_repo_factory=lambda db: FakeFileRepository(),
                    load_user_system_prompt_fn=lambda db, uid: None,
                    preprocess_user_input=False,
                )
                self.assertEqual(prepared.prompt_assembly["section_ids"], expected)
                self.assertEqual(len(prepared.messages), len(expected))

    async def test_io_failure_is_not_an_assembly_failure(self):
        from unittest.mock import patch

        from app.services.stream.agent_loop_request_prep import AgentLoopCallConfig

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
                    call_config=AgentLoopCallConfig(False, False, {}, []),
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

    def test_amap_fact_boundary_is_generic_system_prompt_and_preserves_multi_tool_message_order(self):
        messages = [
            {"role": "system", "content": "基础系统提示"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "tc-place", "type": "function", "function": {"name": "local_place_search"}},
                    {"id": "tc-route", "type": "function", "function": {"name": "route_compare"}},
                    {"id": "tc-search", "type": "function", "function": {"name": "web_search"}},
                ],
            },
            {"role": "tool", "tool_call_id": "tc-place", "content": "民治星巴克地点原始数据"},
            {"role": "tool", "tool_call_id": "tc-route", "content": "深圳北站路线原始数据"},
            {"role": "tool", "tool_call_id": "tc-search", "content": "普通搜索上下文"},
        ]
        call_kwargs = {
            "tools": [
                {"type": "function", "function": {"name": "local_place_search"}},
                {"type": "function", "function": {"name": "route_compare"}},
                {"type": "function", "function": {"name": "weather_forecast"}},
                {"type": "function", "function": {"name": "web_search"}},
            ]
        }

        prepared = inject_amap_fact_boundary(messages, call_kwargs)

        self.assertEqual(
            [message["role"] for message in prepared],
            ["system", "system", "assistant", "tool", "tool", "tool"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in prepared[3:]],
            ["tc-place", "tc-route", "tc-search"],
        )
        boundary = prepared[1]["content"]
        self.assertIn("【地点与路线工具选择规则】", boundary)
        self.assertIn("【组合行程必填信息规则】", boundary)
        self.assertIn("存在多个合理的具体出发日期", boundary)
        self.assertIn("必须先向用户确认具体日期", boundary)
        self.assertIn(
            "不得选择任一候选日期调用 search_flights、search_trains、weather_forecast 或 route_compare", boundary
        )
        self.assertIn("两个自然语言起终点", boundary)
        self.assertIn("直接调用 route_compare", boundary)
        self.assertIn("不要先调用 web_search 或 local_place_search", boundary)
        self.assertIn("当前位置", boundary)
        self.assertIn("source=current_location", boundary)
        self.assertIn("【地点与路线事实边界规则】", boundary)
        self.assertIn("工具失败、不可用或未取得可用结果", boundary)
        self.assertIn("不得用训练知识补充具体地点、线路、时间、距离、费用或路况", boundary)
        self.assertIn("不得仅根据地址片区或同村", boundary)
        self.assertIn("步行可达", boundary)
        self.assertIn("隔壁片区", boundary)
        self.assertIn("本次返回候选中", boundary)
        self.assertIn("距离或就近作为选择条件", boundary)
        self.assertIn("只能来自对应 result.places 或 result.routes 中实际返回的字段", boundary)
        self.assertIn("禁止使用常识、品牌印象、店名词义或训练知识", boundary)
        self.assertIn("环境、安静度、座位、出品、通常营业时间、公园步道", boundary)
        self.assertIn("rating 只能称为评分或综合评分", boundary)
        self.assertIn("不得解释为环境、安静度或服务评分", boundary)
        self.assertIn("不得根据品牌、店名或综合评分", boundary)
        self.assertIn("适合聊天、适合三人、品牌稳定或出品稳定", boundary)
        self.assertIn("不得在正文或括号中补充估计", boundary)
        self.assertIn("结果为 0 条时，不得根据常识推荐任何有名称的地点", boundary)
        self.assertIn("reference_cost_yuan", boundary)
        self.assertIn("只能原样称为参考消费", boundary)
        self.assertIn("不得评价为便宜、实惠或性价比高", boundary)
        self.assertIn("允许依据实际返回的 rating 或 open_hours 做有限排序或说明", boundary)
        self.assertIn("必须明确所依据的字段", boundary)
        self.assertIn("不得把排序或说明改写成未返回属性", boundary)
        self.assertIn("不得推断实时排队、预约、空位", boundary)
        self.assertIn("地点之间的时间或距离", boundary)
        self.assertIn("路线选择或比较只能基于实际返回的 duration_s、distance_m、transfers 等字段", boundary)
        self.assertIn("允许说明最快、最慢、换乘次数或距离远近", boundary)
        self.assertIn("必须明确依据的返回字段", boundary)
        self.assertIn("停车位、停车难度、停车费、公交票价或成本", boundary)
        self.assertIn(
            "当前路况、周六路况、进出站或换乘等待时间、出行灵活性、舒适度、环保或免费",
            boundary,
        )
        self.assertIn("不得声称路线耗时包含或不包含停车及其他未返回构成", boundary)
        self.assertIn("未返回的路线属性只能说明无法从本次查询结果确认", boundary)
        self.assertIn("【天气事实边界规则】", boundary)
        self.assertIn("组合行程", boundary)
        self.assertIn("必须调用 weather_forecast", boundary)
        self.assertIn("不得用 web_search 或 url_read 替代", boundary)
        self.assertIn("目的地市内接驳", boundary)
        self.assertIn("先完成航班或高铁查询", boundary)
        self.assertIn("只调用一次 route_compare", boundary)
        self.assertIn("不得猜测机场或车站", boundary)
        self.assertIn("必须把班次结果中的 city", boundary)
        self.assertIn("origin_city 和 destination_city", boundary)
        self.assertIn("实时温度、湿度、空气质量、降雨概率", boundary)
        self.assertIn("不得声称代表具体建筑物、街道或园区的精确天气", boundary)
        self.assertNotIn("民治星巴克", boundary)
        self.assertNotIn("深圳北站", boundary)

    def test_amap_fact_boundary_is_deduplicated(self):
        call_kwargs = {"tools": [{"type": "function", "function": {"name": "local_place_search"}}]}

        prepared = inject_amap_fact_boundary([{"role": "user", "content": "找咖啡店"}], call_kwargs)
        prepared = inject_amap_fact_boundary(prepared, call_kwargs)

        boundaries = [
            message
            for message in prepared
            if message.get("role") == "system" and "【地点与路线事实边界规则】" in str(message.get("content", ""))
        ]
        self.assertEqual(len(boundaries), 1)

    def test_non_amap_tools_do_not_inject_amap_fact_boundary(self):
        messages = [{"role": "user", "content": "深圳天气"}]
        call_kwargs = {"tools": [{"type": "function", "function": {"name": "web_search"}}]}

        prepared = inject_amap_fact_boundary(messages, call_kwargs)

        self.assertIs(prepared, messages)
        self.assertFalse(any("【地点与路线事实边界规则】" in str(message.get("content", "")) for message in prepared))

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
            capabilities={"functionCalling": True, "deepThinking": True},
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
            capabilities={"functionCalling": True, "deepThinking": True},
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
            ),
            file_repo_factory=lambda _db: object(),
            load_user_system_prompt_fn=lambda _db, _user_id: None,
            build_llm_messages_fn=build_llm_messages_fn,
        )

        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "system", "system", "user"],
        )
        self.assertIn("【当前真实日期】", prepared.messages[0]["content"])
        self.assertIn("【执行计划控制规则】", prepared.messages[2]["content"])
        self.assertIn("【无联网工具边界规则】", prepared.messages[3]["content"])
        self.assertIn("不要声称已经搜索", prepared.messages[3]["content"])
        self.assertIn("无法实时核验", prepared.messages[3]["content"])
        self.assertIn("不要把已有知识包装成最新事实", prepared.messages[3]["content"])
        self.assertIn("普通稳定问题直接回答", prepared.messages[3]["content"])
        self.assertNotIn("切换模型", prepared.messages[3]["content"])
        self.assertNotIn("【工具调用一致性规则】", prepared.messages[3]["content"])
        self.assertEqual(prepared.messages[4]["content"], "OpenAI 最近发布了什么模型？")

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
            ),
            file_repo_factory=lambda _db: object(),
            load_user_system_prompt_fn=lambda _db, _user_id: None,
            build_llm_messages_fn=build_llm_messages_fn,
            is_image_file_fn=lambda file_id, _repo: file_id == "image-1",
        )

        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "system", "system", "system", "user"],
        )
        self.assertIn("【无图片理解能力边界规则】", prepared.messages[2]["content"])
        self.assertIn("当前模型不能读取或理解图片附件", prepared.messages[2]["content"])
        self.assertIn("不要臆测图片内容", prepared.messages[2]["content"])
        self.assertIn("【执行计划控制规则】", prepared.messages[3]["content"])
        self.assertIn("【无联网工具边界规则】", prepared.messages[4]["content"])
        self.assertEqual(prepared.messages[5]["content"], "这张图里有什么？")

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
            self.assertEqual(original_message, "请看 https://example.com/a")
            self.assertTrue(supports_function_calling)
            self.assertEqual(call_kwargs["tools"][0]["function"]["name"], "web_search")
            if not any(tool["function"]["name"] == "url_read" for tool in call_kwargs["tools"]):
                call_kwargs["tools"].append({"type": "function", "function": {"name": "url_read"}})
            return (
                TextBlock(type="text", id="url-block", text="URL 摘要"),
                {"role": "user", "content": "<web_context>网页正文</web_context>"},
                "https://example.com/a",
            )

        call_config = build_agent_loop_call_config(
            provider="openai",
            options={"use_reasoning": True},
            capabilities={"functionCalling": True, "deepThinking": True},
        )

        prepared = await prepare_agent_loop_messages(
            db=object(),
            user_id="user-1",
            raw_messages=["raw"],
            has_vision=False,
            file_ids=["doc-1", "image-1"],
            original_message="请看 https://example.com/a",
            call_config=call_config,
            file_repo_factory=lambda _db: file_repo,
            load_user_system_prompt_fn=lambda _db, _user_id: "用户偏好",
            build_llm_messages_fn=build_llm_messages_fn,
            is_image_file_fn=lambda file_id, _repo: file_id == "image-1",
            inject_file_content_fn=inject_file_content_fn,
            preprocess_url_in_message_fn=preprocess_url_in_message_fn,
        )

        self.assertIsNone(build_calls[0]["user_system_prompt"])
        self.assertIn("用户偏好", prepared.messages[1]["content"])
        self.assertIs(build_calls[0]["repo"], file_repo)
        self.assertEqual(build_calls[0]["user_id"], "user-1")
        self.assertIsNone(build_calls[0]["conversation_id"])
        self.assertEqual(file_repo.requested_content_ids, [["doc-1"]])
        self.assertEqual(inject_calls[0]["file_contents"], {"doc-1": "文档正文"})
        self.assertEqual([block.id for block in prepared.initial_content_blocks], ["url-block"])
        self.assertEqual(prepared.final_tool_names, ["web_search", "url_read"])
        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "system", "system", "system", "system", "user", "user"],
        )
        self.assertIn("【当前真实日期】", prepared.messages[0]["content"])
        self.assertIn("【无图片理解能力边界规则】", prepared.messages[3]["content"])
        self.assertIn("【工具调用一致性规则】", prepared.messages[4]["content"])
        self.assertIn("【执行计划控制规则】", prepared.messages[5]["content"])
        self.assertNotIn("【无联网工具边界规则】", prepared.messages[5]["content"])
        self.assertIn("<web_context>", prepared.messages[6]["content"])
        self.assertIn("文档正文", prepared.messages[7]["content"])
        self.assertEqual(call_config.announced_tools, ["web_search", "url_read"])

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
        self.assertEqual(prepared.messages[2], {"role": "system", "content": "继续执行，不要重写前文"})
        self.assertEqual(prepared.messages[3]["role"], "system")
        self.assertIn("【无联网工具边界规则】", prepared.messages[3]["content"])
        self.assertEqual(prepared.messages[4]["role"], "user")

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
