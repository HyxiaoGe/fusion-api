import unittest


class ModelPresentationTests(unittest.TestCase):
    def test_build_model_capability_presentation_uses_configured_weights(self):
        from app.services.model_presentation import build_model_capability_presentation

        presentation = build_model_capability_presentation(
            {
                "name": "Config Model",
                "capabilities": {
                    "searchCapable": True,
                    "agentTools": True,
                    "vision": False,
                    "deepThinking": False,
                },
                "contextWindowTokens": 8192,
                "health": {"status": "healthy"},
            },
            config={
                "long_context_threshold_tokens": 128000,
                "weights": {
                    "base": 10,
                    "network": 80,
                    "vision": 15,
                    "long_context": 15,
                    "deep_thinking": 10,
                },
                "levels": {
                    "recommended": 85,
                    "capable": 70,
                },
            },
        )

        self.assertEqual(presentation["score"], 90)
        self.assertEqual(presentation["level"], "recommended")
        self.assertEqual(presentation["headline"], "推荐：实时资料与复杂查询")

    def test_build_model_capability_presentation_marks_unhealthy_model_unavailable(self):
        from app.services.model_presentation import build_model_capability_presentation

        presentation = build_model_capability_presentation(
            {
                "name": "Offline Model",
                "capabilities": {"searchCapable": True, "vision": True},
                "health": {"status": "unhealthy", "error": "模型已下线"},
            }
        )

        self.assertEqual(presentation["score"], 0)
        self.assertEqual(presentation["level"], "unavailable")
        self.assertEqual(presentation["headline"], "不建议：当前不可用")
        self.assertIn("模型已下线", presentation["warnings"])
        self.assertIn("健康状态异常：模型已下线", presentation["tooltip"])

    def test_build_model_capability_presentation_keeps_non_network_model_usable(self):
        from app.services.model_presentation import build_model_capability_presentation

        presentation = build_model_capability_presentation(
            {
                "name": "Plain Model",
                "capabilities": {
                    "searchCapable": False,
                    "agentTools": False,
                    "functionCalling": True,
                    "vision": False,
                    "deepThinking": False,
                },
                "health": {"status": "healthy"},
            }
        )

        self.assertEqual(presentation["level"], "limited")
        self.assertIn("可处理普通文本任务", presentation["reasons"])
        self.assertIn("不支持实时联网，涉及最新信息时会基于已有知识谨慎回答", presentation["warnings"])
        self.assertTrue(any(label["text"] == "不可联网" for label in presentation["labels"]))


if __name__ == "__main__":
    unittest.main()
