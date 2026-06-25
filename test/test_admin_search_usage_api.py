import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["AUTH_SERVICE_BASE_URL"] = "http://auth.example:8100"
os.environ["AUTH_SERVICE_CLIENT_ID"] = "fusion-client"
os.environ["AUTH_SERVICE_JWKS_URL"] = "http://auth.example:8100/.well-known/jwks.json"


class AdminSearchUsageApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        main = importlib.import_module("main")
        cls.main = main
        cls.client = TestClient(main.app)

    def tearDown(self):
        self.main.app.dependency_overrides.clear()

    def _set_current_user(self, *, is_superuser: bool):
        from app.api.deps import get_current_user

        user = SimpleNamespace(id="user-123", is_superuser=is_superuser)
        self.main.app.dependency_overrides[get_current_user] = lambda: user

    def test_search_usage_rejects_non_admin_user(self):
        self._set_current_user(is_superuser=False)

        response = self.client.get("/api/admin/search-usage")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["message"], "需要管理员权限")

    def test_search_usage_returns_firecrawl_usage_for_admin(self):
        self._set_current_user(is_superuser=True)

        usage = {
            "provider": "firecrawl",
            "available": True,
            "remaining_credits": 84833,
            "plan_credits": 500000,
            "used_credits": 415167,
            "usage_ratio": 0.830334,
            "billing_period_start": "2026-06-01T00:00:00Z",
            "billing_period_end": "2026-06-30T23:59:59Z",
            "recorded_usage": {
                "provider": "firecrawl",
                "available": True,
                "credits_used": 12,
                "request_count": 3,
                "period_start": "2026-06-01T00:00:00Z",
                "period_end": "2026-06-30T23:59:59Z",
                "source": "search_response_credits_used",
            },
        }
        historical = {
            "provider": "firecrawl",
            "available": True,
            "by_api_key": False,
            "periods": [
                {
                    "start_date": "2026-06-01T00:00:00Z",
                    "end_date": "2026-06-30T23:59:59Z",
                    "api_key": None,
                    "total_credits": 128,
                }
            ],
        }

        with (
            patch("app.services.external.search_usage_client.get_firecrawl_usage", AsyncMock(return_value=usage)),
            patch(
                "app.services.external.search_usage_client.get_firecrawl_historical_usage",
                AsyncMock(return_value=historical),
            ),
        ):
            response = self.client.get("/api/admin/search-usage")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["firecrawl"], usage)
        self.assertEqual(data["historical"], historical)
        self.assertEqual(
            data["providers"],
            [
                {"provider": "firecrawl", "official_usage": True},
                {"provider": "brave", "official_usage": False},
            ],
        )

    def test_search_usage_degrades_when_historical_usage_fails(self):
        from app.services.external.search_usage_client import SearchUsageClientError

        self._set_current_user(is_superuser=True)

        usage = {
            "provider": "firecrawl",
            "available": True,
            "remaining_credits": 84801,
            "plan_credits": 1000,
            "used_credits": None,
            "usage_ratio": None,
            "billing_period_start": "2026-06-22T21:35:09.173Z",
            "billing_period_end": "2026-07-22T21:35:09.173Z",
        }

        with (
            patch("app.services.external.search_usage_client.get_firecrawl_usage", AsyncMock(return_value=usage)),
            patch(
                "app.services.external.search_usage_client.get_firecrawl_historical_usage",
                AsyncMock(side_effect=SearchUsageClientError("historical failed with fc-secret-key")),
            ),
        ):
            response = self.client.get("/api/admin/search-usage")

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertEqual(data["firecrawl"], usage)
        self.assertEqual(
            data["historical"],
            {
                "provider": "firecrawl",
                "available": False,
                "by_api_key": False,
                "periods": [],
            },
        )
        self.assertNotIn("fc-secret-key", response.text)

    def test_search_usage_maps_search_service_failure_without_leaking_secret(self):
        from app.services.external.search_usage_client import SearchUsageClientError

        self._set_current_user(is_superuser=True)

        with patch(
            "app.services.external.search_usage_client.get_firecrawl_usage",
            AsyncMock(side_effect=SearchUsageClientError("upstream failed with fc-secret-key")),
        ):
            response = self.client.get("/api/admin/search-usage")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["message"], "联网用量查询失败")
        self.assertNotIn("fc-secret-key", response.text)
