import unittest

from scripts.deployment_smoke import (
    DeploymentSmokeError,
    build_url,
    validate_health_payload,
    validate_models_payload,
)


class DeploymentSmokeTests(unittest.TestCase):
    def test_build_url_trims_base_slash(self):
        self.assertEqual(build_url("http://127.0.0.1:8002/", "/api/models/"), "http://127.0.0.1:8002/api/models/")

    def test_validate_health_requires_healthy_status(self):
        validate_health_payload({"status": "healthy", "database": "connected"})
        with self.assertRaisesRegex(DeploymentSmokeError, "health status"):
            validate_health_payload({"status": "unhealthy"})

    def test_validate_models_requires_non_empty_models_and_providers(self):
        validate_models_payload(
            {
                "code": "SUCCESS",
                "data": {
                    "models": [
                        {
                            "modelId": "deepseek-chat",
                            "name": "DeepSeek",
                            "provider": "deepseek",
                            "enabled": True,
                            "contextWindowTokens": 128000,
                            "maxOutputTokens": 8192,
                            "capabilities": {
                                "functionCalling": True,
                                "agentTools": True,
                                "searchCapable": True,
                                "webSearch": True,
                                "vision": False,
                            },
                        }
                    ],
                    "providers": [{"id": "deepseek", "name": "DeepSeek", "order": 1}],
                },
            }
        )

    def test_validate_models_rejects_missing_search_capable_contract(self):
        with self.assertRaisesRegex(DeploymentSmokeError, "searchCapable"):
            validate_models_payload(
                {
                    "code": "SUCCESS",
                    "data": {
                        "models": [
                            {
                                "modelId": "deepseek-chat",
                                "name": "DeepSeek",
                                "provider": "deepseek",
                                "enabled": True,
                                "contextWindowTokens": 128000,
                                "maxOutputTokens": 8192,
                                "capabilities": {
                                    "functionCalling": True,
                                    "agentTools": True,
                                    "webSearch": True,
                                    "vision": False,
                                },
                            }
                        ],
                        "providers": [{"id": "deepseek", "name": "DeepSeek", "order": 1}],
                    },
                }
            )

    def test_validate_models_rejects_missing_capabilities(self):
        with self.assertRaisesRegex(DeploymentSmokeError, "capabilities"):
            validate_models_payload(
                {
                    "code": "SUCCESS",
                    "data": {
                        "models": [
                            {
                                "modelId": "deepseek-chat",
                                "name": "DeepSeek",
                                "provider": "deepseek",
                                "enabled": True,
                                "contextWindowTokens": 128000,
                                "maxOutputTokens": 8192,
                            }
                        ],
                        "providers": [{"id": "deepseek", "name": "DeepSeek", "order": 1}],
                    },
                }
            )

    def test_validate_models_rejects_invalid_context_token_contract(self):
        with self.assertRaisesRegex(DeploymentSmokeError, "contextWindowTokens"):
            validate_models_payload(
                {
                    "code": "SUCCESS",
                    "data": {
                        "models": [
                            {
                                "modelId": "deepseek-chat",
                                "name": "DeepSeek",
                                "provider": "deepseek",
                                "enabled": True,
                                "contextWindowTokens": "128k",
                                "maxOutputTokens": 8192,
                                "capabilities": {
                                    "functionCalling": True,
                                    "agentTools": True,
                                    "searchCapable": True,
                                    "webSearch": True,
                                    "vision": False,
                                },
                            }
                        ],
                        "providers": [{"id": "deepseek", "name": "DeepSeek", "order": 1}],
                    },
                }
            )


if __name__ == "__main__":
    unittest.main()
