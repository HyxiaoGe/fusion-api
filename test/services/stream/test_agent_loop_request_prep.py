import unittest

from app.schemas.chat import TextBlock
from app.services.stream.agent_loop_request_prep import (
    build_agent_loop_call_config,
    prepare_agent_loop_messages,
)


class FakeFileRepository:
    def __init__(self):
        self.requested_content_ids = []

    def get_parsed_file_content(self, file_ids):
        self.requested_content_ids.append(list(file_ids))
        return {"doc-1": "文档正文"}


class AgentLoopRequestPrepTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_prepare_messages_injects_no_tool_network_boundary_when_agent_tools_disabled(self):
        async def build_llm_messages_fn(_raw_messages, _has_vision, _repo, _user_system_prompt):
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
        self.assertIn("普通稳定问题直接回答", prepared.messages[1]["content"])
        self.assertNotIn("切换模型", prepared.messages[1]["content"])
        self.assertNotIn("【工具调用一致性规则】", prepared.messages[1]["content"])
        self.assertEqual(prepared.messages[2]["content"], "OpenAI 最近发布了什么模型？")

    async def test_prepare_messages_builds_llm_input_files_url_context_and_tool_contract(self):
        file_repo = FakeFileRepository()
        build_calls = []
        inject_calls = []

        async def build_llm_messages_fn(raw_messages, has_vision, repo, user_system_prompt):
            build_calls.append(
                {
                    "raw_messages": raw_messages,
                    "has_vision": has_vision,
                    "repo": repo,
                    "user_system_prompt": user_system_prompt,
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
        self.assertEqual(file_repo.requested_content_ids, [["doc-1"]])
        self.assertEqual(inject_calls[0]["file_contents"], {"doc-1": "文档正文"})
        self.assertEqual([block.id for block in prepared.initial_content_blocks], ["url-block"])
        self.assertEqual(prepared.final_tool_names, ["web_search", "url_read"])
        self.assertEqual([message["role"] for message in prepared.messages], ["system", "system", "user", "user"])
        self.assertIn("日期 system", prepared.messages[0]["content"])
        self.assertIn("【工具调用一致性规则】", prepared.messages[1]["content"])
        self.assertNotIn("【无联网工具边界规则】", prepared.messages[1]["content"])
        self.assertIn("<web_context>", prepared.messages[2]["content"])
        self.assertIn("文档正文", prepared.messages[3]["content"])
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
        async def build_llm_messages_fn(_raw_messages, _has_vision, _repo, _user_system_prompt):
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


if __name__ == "__main__":
    unittest.main()
