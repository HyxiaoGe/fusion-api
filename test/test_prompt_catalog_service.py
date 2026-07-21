import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class PromptCatalogServiceTests(unittest.TestCase):
    def test_default_catalog_covers_home_starters_and_library_templates(self):
        from app.services.runtime_config_defaults import DEFAULT_HOME_PROMPT_CATALOG

        items = DEFAULT_HOME_PROMPT_CATALOG["items"]
        ids = [item["id"] for item in items]
        travel_ids = {
            "commute-planning",
            "weekend-itinerary",
            "dining-entertainment",
        }

        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(sum(item["kind"] == "starter" for item in items), 11)
        self.assertGreaterEqual(sum(item["kind"] == "template" for item in items), 3)
        self.assertTrue(travel_ids.issubset(ids))

        items_by_id = {item["id"]: item for item in items}
        self.assertEqual(
            items_by_id["commute-planning"],
            {
                "id": "commute-planning",
                "kind": "starter",
                "title": "规划通勤",
                "description": "对比路线、耗时和换乘成本",
                "content": (
                    "我从【出发地】前往【目的地】，计划【出发时间】出发。请比较驾车、公共交通、骑行和步行等"
                    "可用方式，给出具体路线，并根据用时、距离、换乘和步行距离推荐合适方案。"
                ),
                "category": "出行",
                "icon_key": "map-pinned",
                "tone": "sky",
                "sort_order": 90,
                "enabled": True,
                "required_capabilities": [],
            },
        )
        self.assertEqual(
            items_by_id["weekend-itinerary"],
            {
                "id": "weekend-itinerary",
                "kind": "starter",
                "title": "安排周末行程",
                "description": "串联地点、时间和交通路线",
                "content": (
                    "我计划【日期/时间】从【出发地】出发，想去【地点1、地点2、地点3】，一共【人数】人，"
                    "偏好【兴趣】，预算【预算】。请推荐合理的游玩顺序，并规划每段路线和时间安排。"
                ),
                "category": "出行",
                "icon_key": "calendar-range",
                "tone": "orange",
                "sort_order": 100,
                "enabled": True,
                "required_capabilities": [],
            },
        )
        self.assertEqual(
            items_by_id["dining-entertainment"],
            {
                "id": "dining-entertainment",
                "kind": "starter",
                "title": "聚餐与娱乐",
                "description": "推荐地点并规划饭后转场",
                "content": (
                    "我计划【日期/时间】在【区域】和【人数】人聚餐，偏好【餐饮类型】，总预算【预算】，"
                    "饭后想【娱乐活动】。请推荐合适地点，并规划聚餐与娱乐之间的转场路线。"
                ),
                "category": "出行",
                "icon_key": "utensils-crossed",
                "tone": "teal",
                "sort_order": 110,
                "enabled": True,
                "required_capabilities": [],
            },
        )
        self.assertEqual(
            {
                item_id: items_by_id[item_id]["sort_order"]
                for item_id in (
                    "template-code-explanation",
                    "template-text-summary",
                    "template-question-answering",
                )
            },
            {
                "template-code-explanation": 210,
                "template-text-summary": 220,
                "template-question-answering": 230,
            },
        )
        for item_id in travel_ids:
            item = items_by_id[item_id]
            self.assertEqual(item["kind"], "starter")
            self.assertEqual(item["category"], "出行")
            self.assertEqual(item["required_capabilities"], [])
            self.assertNotIn("高德", item["content"])
            self.assertNotIn("route_compare", item["content"])
            self.assertNotIn("local_place_search", item["content"])

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
