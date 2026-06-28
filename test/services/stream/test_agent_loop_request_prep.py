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
        self.assertIn("<web_context>", prepared.messages[2]["content"])
        self.assertIn("文档正文", prepared.messages[3]["content"])
        self.assertEqual(call_config.announced_tools, ["web_search"])

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
        self.assertEqual(prepared.messages[1]["role"], "user")


if __name__ == "__main__":
    unittest.main()
