import unittest

from app.schemas.chat import TextBlock
from app.services.stream.agent_loop_request_prep import (
    build_agent_loop_call_config,
    inject_amap_fact_boundary,
    inject_no_tool_network_boundary,
    prepare_agent_loop_messages,
)


class FakeFileRepository:
    def __init__(self):
        self.requested_content_ids = []

    def get_parsed_file_content(self, file_ids):
        self.requested_content_ids.append(list(file_ids))
        return {"doc-1": "文档正文"}


class AgentLoopRequestPrepTests(unittest.IsolatedAsyncioTestCase):
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
            options={},
            capabilities={"functionCalling": True, "deepThinking": True},
        )

        self.assertTrue(config.should_use_reasoning)
        self.assertTrue(config.supports_function_calling)
        self.assertEqual(config.announced_tools, ["web_search"])
        self.assertEqual(config.call_kwargs["tool_choice"], "auto")
        self.assertEqual(config.call_kwargs["tools"][0]["function"]["name"], "web_search")
        self.assertEqual(config.call_kwargs["extra_body"], {"thinking": {"type": "disabled"}})

    def test_build_call_config_respects_explicit_reasoning_override(self):
        config = build_agent_loop_call_config(
            provider="volcengine",
            options={"use_reasoning": False},
            capabilities={"functionCalling": True, "deepThinking": True},
        )

        self.assertFalse(config.should_use_reasoning)
        self.assertTrue(config.supports_function_calling)
        self.assertEqual(config.announced_tools, ["web_search"])
        self.assertNotIn("extra_body", config.call_kwargs)

    def test_build_call_config_disables_agent_tools_when_agent_tools_capability_is_false(self):
        config = build_agent_loop_call_config(
            provider="qwen",
            options={},
            capabilities={"functionCalling": True, "agentTools": False, "deepThinking": False},
        )

        self.assertFalse(config.supports_function_calling)
        self.assertEqual(config.announced_tools, [])
        self.assertNotIn("tools", config.call_kwargs)
        self.assertNotIn("tool_choice", config.call_kwargs)

    def test_build_call_config_uses_search_capable_as_runtime_tool_contract(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={},
            capabilities={"functionCalling": True, "agentTools": False, "searchCapable": True},
        )

        self.assertTrue(config.supports_function_calling)
        self.assertEqual(config.announced_tools, ["web_search"])
        self.assertEqual(config.call_kwargs["tool_choice"], "auto")

    def test_build_call_config_disables_tools_when_search_capable_is_false(self):
        config = build_agent_loop_call_config(
            provider="openai",
            options={},
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
            options={},
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
            options={},
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
            _raw_messages, _has_vision, _repo, _user_system_prompt, *, user_id=None, conversation_id=None
        ):
            return [
                {"role": "system", "content": "日期 system"},
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

        self.assertEqual([message["role"] for message in prepared.messages], ["system", "system", "user"])
        self.assertIn("日期 system", prepared.messages[0]["content"])
        self.assertIn("【无联网工具边界规则】", prepared.messages[1]["content"])
        self.assertIn("不要声称已经搜索", prepared.messages[1]["content"])
        self.assertIn("无法实时核验", prepared.messages[1]["content"])
        self.assertIn("不要把已有知识包装成最新事实", prepared.messages[1]["content"])
        self.assertIn("普通稳定问题直接回答", prepared.messages[1]["content"])
        self.assertNotIn("切换模型", prepared.messages[1]["content"])
        self.assertNotIn("【工具调用一致性规则】", prepared.messages[1]["content"])
        self.assertEqual(prepared.messages[2]["content"], "OpenAI 最近发布了什么模型？")

    async def test_prepare_messages_injects_no_vision_boundary_when_image_attached_to_text_model(self):
        async def build_llm_messages_fn(
            _raw_messages, _has_vision, _repo, _user_system_prompt, *, user_id=None, conversation_id=None
        ):
            return [
                {"role": "system", "content": "日期 system"},
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

        self.assertEqual([message["role"] for message in prepared.messages], ["system", "system", "system", "user"])
        self.assertIn("【无图片理解能力边界规则】", prepared.messages[1]["content"])
        self.assertIn("当前模型不能读取或理解图片附件", prepared.messages[1]["content"])
        self.assertIn("不要臆测图片内容", prepared.messages[1]["content"])
        self.assertIn("【无联网工具边界规则】", prepared.messages[2]["content"])
        self.assertEqual(prepared.messages[3]["content"], "这张图里有什么？")

    async def test_prepare_messages_builds_llm_input_files_url_context_and_tool_contract(self):
        file_repo = FakeFileRepository()
        build_calls = []
        inject_calls = []

        async def build_llm_messages_fn(
            raw_messages, has_vision, repo, user_system_prompt, *, user_id=None, conversation_id=None
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
                {"role": "system", "content": "日期 system"},
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

        self.assertEqual(build_calls[0]["user_system_prompt"], "用户偏好")
        self.assertIs(build_calls[0]["repo"], file_repo)
        self.assertEqual(build_calls[0]["user_id"], "user-1")
        self.assertIsNone(build_calls[0]["conversation_id"])
        self.assertEqual(file_repo.requested_content_ids, [["doc-1"]])
        self.assertEqual(inject_calls[0]["file_contents"], {"doc-1": "文档正文"})
        self.assertEqual([block.id for block in prepared.initial_content_blocks], ["url-block"])
        self.assertEqual(prepared.final_tool_names, ["web_search", "url_read"])
        self.assertEqual(
            [message["role"] for message in prepared.messages],
            ["system", "system", "system", "user", "user"],
        )
        self.assertIn("日期 system", prepared.messages[0]["content"])
        self.assertIn("【无图片理解能力边界规则】", prepared.messages[1]["content"])
        self.assertIn("【工具调用一致性规则】", prepared.messages[2]["content"])
        self.assertNotIn("【无联网工具边界规则】", prepared.messages[2]["content"])
        self.assertIn("<web_context>", prepared.messages[3]["content"])
        self.assertIn("文档正文", prepared.messages[4]["content"])
        self.assertEqual(call_config.announced_tools, ["web_search"])

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
            _raw_messages, _has_vision, _repo, _user_system_prompt, *, user_id=None, conversation_id=None
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
        self.assertEqual(prepared.messages[0], {"role": "system", "content": "继续执行，不要重写前文"})
        self.assertEqual(prepared.messages[1]["role"], "system")
        self.assertIn("【无联网工具边界规则】", prepared.messages[1]["content"])
        self.assertEqual(prepared.messages[2]["role"], "user")

    async def test_prepare_messages_passes_conversation_scope_to_builder(self):
        build_calls = []

        async def build_llm_messages_fn(
            raw_messages, has_vision, repo, user_system_prompt, *, user_id=None, conversation_id=None
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
                    "user_system_prompt": "用户偏好",
                    "user_id": "user-1",
                    "conversation_id": "conv-1",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
