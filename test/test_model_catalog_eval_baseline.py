import json
import unittest

from scripts import model_catalog_eval_baseline as baseline


class ModelCatalogEvalBaselineTests(unittest.TestCase):
    def test_select_models_defaults_to_models_not_marked_unhealthy(self):
        models = [
            {"modelId": "deepseek-chat", "provider": "deepseek", "health": {"status": "healthy"}},
            {"modelId": "mimo-v2.5-pro", "provider": "xiaomi", "health": {"status": "unknown"}},
            {"modelId": "mimo-v2-pro", "provider": "xiaomi", "health": {"status": "unhealthy"}},
            {"modelId": "qwen-max-latest", "provider": "qwen", "health": {"status": "healthy"}},
        ]

        selected = baseline.select_models(models)

        self.assertEqual(
            [model["modelId"] for model in selected],
            ["deepseek-chat", "mimo-v2.5-pro", "qwen-max-latest"],
        )

    def test_select_models_can_include_unhealthy_and_filter_ids(self):
        models = [
            {"modelId": "deepseek-chat", "provider": "deepseek", "health": {"status": "healthy"}},
            {"modelId": "mimo-v2-pro", "provider": "xiaomi", "health": {"status": "unhealthy"}},
        ]

        selected = baseline.select_models(
            models,
            include_unhealthy=True,
            model_ids=["mimo-v2-pro"],
        )

        self.assertEqual([model["modelId"] for model in selected], ["mimo-v2-pro"])

    def test_success_result_jsonl_contains_required_fields(self):
        result = baseline.build_success_result(
            model={"modelId": "deepseek-chat", "provider": "deepseek", "name": "DeepSeek"},
            question="你好",
            elapsed_ms=1234,
            response_payload={
                "data": {
                    "message": {
                        "content": "你好，我是 Fusion AI。",
                    }
                }
            },
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertEqual(row["model_id"], "deepseek-chat")
        self.assertEqual(row["provider"], "deepseek")
        self.assertEqual(row["question"], "你好")
        self.assertTrue(row["success"])
        self.assertEqual(row["elapsed_ms"], 1234)
        self.assertIn("Fusion AI", row["answer_preview"])
        self.assertIsNone(row["error"])

    def test_failure_result_jsonl_contains_error(self):
        result = baseline.build_failure_result(
            model={"modelId": "mimo-v2-pro", "provider": "xiaomi", "name": "MiMo"},
            question="你好",
            elapsed_ms=321,
            error=RuntimeError("服务商暂时不可用"),
        )

        row = json.loads(baseline.to_jsonl(result).strip())

        self.assertEqual(row["model_id"], "mimo-v2-pro")
        self.assertFalse(row["success"])
        self.assertEqual(row["error"]["type"], "RuntimeError")
        self.assertIn("服务商暂时不可用", row["error"]["message"])


if __name__ == "__main__":
    unittest.main()
