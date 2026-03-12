from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.ai.function_call_adapter import FunctionCallAdapter


class FunctionCallAdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.registry = MagicMock()
        self.registry.call_function = AsyncMock(return_value={"status": "ok"})
        self.registry.get_functions_for_provider.return_value = [{"name": "web_search"}]
        self.adapter = FunctionCallAdapter(self.registry)

    def test_prepare_functions_for_tool_provider_uses_tools_key(self):
        result = self.adapter.prepare_functions_for_model("openai", "gpt-4o")
        self.assertEqual(result, {"tools": [{"name": "web_search"}]})

    def test_prepare_functions_for_non_tool_qwen_model_falls_back_to_functions_key(self):
        result = self.adapter.prepare_functions_for_model("qwen", "qwen-max-latest")
        self.assertEqual(result, {"functions": [{"name": "web_search"}]})

    async def test_process_function_call_parses_json_string_arguments(self):
        context = {"db": object()}
        result = await self.adapter.process_function_call(
            "openai",
            {"name": "web_search", "arguments": "{\"query\":\"fusion\"}"},
            context,
        )

        self.assertEqual(result, {"status": "ok"})
        self.registry.call_function.assert_awaited_once_with(
            "web_search",
            {"query": "fusion"},
            context,
        )

    def test_detect_function_call_in_stream_reads_tool_calls_dict_shape(self):
        detected, payload = self.adapter.detect_function_call_in_stream(
            SimpleNamespace(
                additional_kwargs={},
                tool_calls=[
                    {
                        "name": "web_search",
                        "arguments": "{\"query\":\"fusion\"}",
                        "id": "tool-1",
                    }
                ],
            )
        )

        self.assertTrue(detected)
        self.assertEqual(payload["function"]["name"], "web_search")
        self.assertEqual(payload["tool_call_id"], "tool-1")


if __name__ == "__main__":
    unittest.main()
