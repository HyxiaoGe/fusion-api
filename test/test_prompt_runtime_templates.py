import unittest
from unittest.mock import patch


class PromptRuntimeTemplatesTests(unittest.TestCase):
    def test_prompt_manager_uses_runtime_template_override(self):
        from app.ai.prompts.prompt_manager import prompt_manager

        with patch(
            "app.ai.prompts.prompt_manager.get_runtime_config_payload",
            return_value=({"template": "标题：{content}"}, {"source": "test"}),
            create=True,
        ):
            prompt = prompt_manager.format_prompt("generate_title", content="Redis")

        self.assertEqual(prompt, "标题：Redis")

    def test_agent_loop_prompt_getter_uses_runtime_template_override(self):
        from app.ai.prompts import agent_loop

        with patch.object(
            agent_loop,
            "get_runtime_prompt_template",
            return_value="覆盖后的工具规则",
            create=True,
        ):
            prompt = agent_loop.get_tool_usage_contract_prompt()

        self.assertEqual(prompt, "覆盖后的工具规则")

    def test_build_url_read_tool_uses_runtime_description(self):
        from app.ai import tools

        with patch(
            "app.ai.tools.get_url_read_tool_description",
            return_value="动态读取网页说明",
            create=True,
        ):
            tool = tools.build_url_read_tool()

        self.assertEqual(tool["function"]["description"], "动态读取网页说明")

    def test_message_builder_injects_runtime_app_identity_prompt(self):
        from app.services.chat import message_builder

        with patch.object(
            message_builder,
            "get_app_identity_prompt",
            return_value="运行时 Fusion 身份规则",
            create=True,
        ):
            messages = self._run_async(message_builder.build_llm_messages([], has_vision=False, file_repo=None))

        self.assertEqual(messages[1], {"role": "system", "content": "运行时 Fusion 身份规则"})

    def test_agent_loop_request_prep_injects_runtime_tool_contract_prompt(self):
        from app.services.stream import agent_loop_request_prep

        call_kwargs = {"tools": [{"type": "function", "function": {"name": "web_search"}}]}
        with patch.object(
            agent_loop_request_prep,
            "get_tool_usage_contract_prompt",
            return_value="运行时工具一致性规则",
            create=True,
        ):
            messages = agent_loop_request_prep.inject_tool_usage_contract(
                [{"role": "user", "content": "OpenAI 最新公告"}],
                call_kwargs,
            )

        self.assertEqual(messages[0], {"role": "system", "content": "运行时工具一致性规则"})

    def test_limit_summary_uses_runtime_prompt(self):
        from app.services.stream import limit_summary

        messages = []
        with patch.object(
            limit_summary,
            "get_limit_summary_prompt",
            return_value="运行时触顶总结规则",
            create=True,
        ):
            limit_summary.append_limit_summary_prompt(messages)

        self.assertEqual(messages, [{"role": "system", "content": "运行时触顶总结规则"}])

    def test_continuation_injects_runtime_prompt(self):
        from app.services.agent import continuation

        with patch.object(
            continuation,
            "get_continuation_system_prompt",
            return_value="运行时继续回答规则",
            create=True,
        ):
            messages = continuation.inject_continuation_prompt([{"role": "user", "content": "继续"}])

        self.assertEqual(messages[0], {"role": "system", "content": "运行时继续回答规则"})

    def test_url_preprocess_uses_runtime_url_read_tool_builder(self):
        from app.services.stream import persistence

        dynamic_tool = {"type": "function", "function": {"name": "url_read", "description": "运行时读取工具"}}
        call_kwargs = {"tools": []}
        with patch.object(
            persistence,
            "build_url_read_tool",
            return_value=dynamic_tool,
            create=True,
        ):
            persistence.ensure_url_read_tool(call_kwargs)
            persistence.ensure_url_read_tool(call_kwargs)

        self.assertEqual(call_kwargs["tools"], [dynamic_tool])

    @staticmethod
    def _run_async(coro):
        import asyncio

        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()
