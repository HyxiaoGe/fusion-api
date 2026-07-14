import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class PromptCatalogServiceTests(unittest.TestCase):
    def test_default_catalog_covers_home_starters_and_library_templates(self):
        from app.services.runtime_config_defaults import DEFAULT_HOME_PROMPT_CATALOG

        items = DEFAULT_HOME_PROMPT_CATALOG["items"]
        ids = [item["id"] for item in items]

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(item["kind"] == "starter" for item in items), 8)
        self.assertGreaterEqual(sum(item["kind"] == "template" for item in items), 3)

    def test_catalog_filters_disabled_items_and_exposes_runtime_version(self):
        from app.services.prompt_catalog_service import get_home_prompt_catalog

        payload = {
            "items": [
                {
                    "id": "enabled",
                    "kind": "starter",
                    "title": "可用模板",
                    "description": "描述",
                    "content": "提示词",
                    "category": "通用",
                    "icon_key": "search",
                    "tone": "blue",
                    "sort_order": 20,
                    "enabled": True,
                    "required_capabilities": [],
                },
                {
                    "id": "disabled",
                    "kind": "template",
                    "title": "停用模板",
                    "description": "",
                    "content": "提示词",
                    "category": "通用",
                    "icon_key": "file-text",
                    "tone": "violet",
                    "sort_order": 10,
                    "enabled": False,
                    "required_capabilities": [],
                },
            ]
        }

        with patch(
            "app.services.prompt_catalog_service.get_runtime_config_payload",
            return_value=(payload, {"source": "db", "version": "2026-07-14.v2"}),
        ):
            result = get_home_prompt_catalog()

        self.assertEqual([item["id"] for item in result["items"]], ["enabled"])
        self.assertEqual(result["source"], "db")
        self.assertEqual(result["version"], "2026-07-14.v2")

    def test_public_templates_endpoint_uses_unified_response(self):
        from app.api import prompts

        request = SimpleNamespace(state=SimpleNamespace(request_id="request-1"))
        catalog = {"items": [], "source": "default", "version": "code-default"}

        with patch.object(prompts, "get_home_prompt_catalog", return_value=catalog):
            response = asyncio.run(prompts.get_templates(request))

        self.assertEqual(response.code, "SUCCESS")
        self.assertEqual(response.data, catalog)
        self.assertEqual(response.request_id, "request-1")


if __name__ == "__main__":
    unittest.main()
