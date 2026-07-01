import unittest

from scripts import govern_litellm_model_catalog as catalog


class ModelCatalogGovernanceTests(unittest.TestCase):
    def test_plan_deletes_deprecated_xiaomi_models(self):
        entries = [
            {
                "model_name": "mimo-v2-pro",
                "model_info": {
                    "id": "uuid-old-pro",
                    "metadata": {"provider_key": "xiaomi", "source": "fusion-migration"},
                },
            }
        ]

        plan = catalog.build_governance_plan(entries, {"XIAOMI_API_KEY": "sk-xiaomi"})

        delete_actions = [action for action in plan.actions if action.action == "delete"]
        self.assertEqual(len(delete_actions), 1)
        self.assertEqual(delete_actions[0].model_name, "mimo-v2-pro")
        self.assertEqual(delete_actions[0].model_uuid, "uuid-old-pro")

    def test_plan_registers_missing_xiaomi_v25_models(self):
        plan = catalog.build_governance_plan([], {"XIAOMI_API_KEY": "sk-xiaomi"})

        create_actions = [action for action in plan.actions if action.action == "create"]
        self.assertEqual(
            [action.model_name for action in create_actions],
            ["mimo-v2.5-pro", "mimo-v2.5-pro-ultraspeed"],
        )
        payload = create_actions[0].payload
        self.assertEqual(payload["model_name"], "mimo-v2.5-pro")
        self.assertEqual(payload["litellm_params"]["model"], "openai/mimo-v2.5-pro")
        self.assertEqual(payload["litellm_params"]["api_base"], "https://api.xiaomimimo.com/v1")
        self.assertEqual(payload["litellm_params"]["api_key"], "sk-xiaomi")
        self.assertEqual(payload["model_info"]["metadata"]["provider_key"], "xiaomi")
        self.assertEqual(payload["model_info"]["metadata"]["provider_display"], "小米 MiMo")
        self.assertEqual(payload["model_info"]["metadata"]["source"], "fusion-governance")
        self.assertTrue(payload["model_info"]["metadata"]["capabilities"]["functionCalling"])

    def test_plan_skips_existing_xiaomi_v25_models(self):
        entries = [
            {
                "model_name": "mimo-v2.5-pro",
                "model_info": {
                    "id": "uuid-new-pro",
                    "metadata": {"provider_key": "xiaomi", "source": "fusion-governance"},
                },
            },
            {
                "model_name": "mimo-v2.5-pro-ultraspeed",
                "model_info": {
                    "id": "uuid-new-fast",
                    "metadata": {"provider_key": "xiaomi", "source": "fusion-governance"},
                },
            },
        ]

        plan = catalog.build_governance_plan(entries, {"XIAOMI_API_KEY": "sk-xiaomi"})

        self.assertFalse([action for action in plan.actions if action.action == "create"])

    def test_missing_xiaomi_key_is_clear_when_registration_needed(self):
        with self.assertRaisesRegex(RuntimeError, "XIAOMI_API_KEY"):
            catalog.build_governance_plan([], {})

    def test_deprecated_model_without_uuid_is_skipped(self):
        entries = [
            {
                "model_name": "mimo-v2-flash",
                "model_info": {"metadata": {"provider_key": "xiaomi", "source": "fusion-migration"}},
            }
        ]

        plan = catalog.build_governance_plan(entries, {"XIAOMI_API_KEY": "sk-xiaomi"})

        skip_actions = [action for action in plan.actions if action.action == "skip"]
        self.assertEqual(len(skip_actions), 1)
        self.assertEqual(skip_actions[0].model_name, "mimo-v2-flash")
        self.assertIn("缺少 LiteLLM model UUID", skip_actions[0].reason)

    def test_serialized_plan_redacts_api_key(self):
        plan = catalog.build_governance_plan([], {"XIAOMI_API_KEY": "sk-secret-xiaomi"})
        action = [action for action in plan.actions if action.action == "create"][0]

        serialized = catalog.serialize_action(action)

        self.assertEqual(serialized["payload"]["litellm_params"]["api_key"], "***")
        self.assertEqual(action.payload["litellm_params"]["api_key"], "sk-secret-xiaomi")

    def test_replace_deprecated_xiaomi_models_in_key_allowlist(self):
        models = [
            "deepseek-chat",
            "mimo-v2-flash",
            "mimo-v2-pro",
            "qwen-max-latest",
        ]

        updated = catalog.replace_deprecated_models_in_allowlist(models)

        self.assertEqual(
            updated,
            [
                "deepseek-chat",
                "qwen-max-latest",
                "mimo-v2.5-pro",
                "mimo-v2.5-pro-ultraspeed",
            ],
        )


if __name__ == "__main__":
    unittest.main()
