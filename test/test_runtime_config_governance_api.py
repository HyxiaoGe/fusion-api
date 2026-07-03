import importlib
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"
os.environ["SERVER_HOST"] = "http://dev.example:8002"
os.environ["FRONTEND_URL"] = "http://dev.example:3004"
os.environ["AUTH_SERVICE_BASE_URL"] = "http://auth.example:8100"
os.environ["AUTH_SERVICE_CLIENT_ID"] = "fusion-client"
os.environ["AUTH_SERVICE_JWKS_URL"] = "http://auth.example:8100/.well-known/jwks.json"


class RuntimeConfigGovernanceApiTests(unittest.TestCase):
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

    def test_runtime_config_rejects_non_admin_user(self):
        self._set_current_user(is_superuser=False)

        response = self.client.get("/api/admin/runtime-config")

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["message"], "需要管理员权限")

    def test_runtime_config_returns_snapshot_for_admin(self):
        self._set_current_user(is_superuser=True)

        snapshot = {
            "generated_at": "2026-07-03T00:00:00+00:00",
            "effective": [
                {
                    "namespace": "agent_strategy",
                    "key": "default",
                    "source": "db",
                    "version": "2026-07-03.v2",
                    "valid": True,
                    "issues": [],
                    "skipped_versions": ["2026-07-03.bad"],
                }
            ],
            "entries": [
                {
                    "id": "row-1",
                    "namespace": "agent_strategy",
                    "key": "default",
                    "version": "2026-07-03.v2",
                    "is_active": True,
                    "valid": True,
                    "issues": [],
                    "description": "测试配置",
                    "created_at": None,
                    "updated_at": None,
                }
            ],
        }

        with patch("app.api.admin.build_runtime_config_snapshot", return_value=snapshot):
            response = self.client.get("/api/admin/runtime-config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"], snapshot)

    def test_runtime_config_validate_checks_payload_without_writing(self):
        self._set_current_user(is_superuser=True)

        response = self.client.post(
            "/api/admin/runtime-config/validate",
            json={
                "namespace": "prompt_template",
                "key": "generate_title",
                "payload": {"template": ""},
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()["data"]
        self.assertFalse(data["valid"])
        self.assertIn("template 必须是非空字符串", data["issues"])
        self.assertEqual(data["namespace"], "prompt_template")
        self.assertEqual(data["key"], "generate_title")

    def test_runtime_config_status_patch_updates_active_flag(self):
        self._set_current_user(is_superuser=True)

        updated = {
            "id": "row-1",
            "namespace": "prompt_template",
            "key": "generate_title",
            "version": "2026-07-03.bad",
            "is_active": False,
            "valid": False,
            "issues": ["template 必须是非空字符串"],
        }

        with patch("app.api.admin.set_runtime_config_entry_active", return_value=updated) as patched:
            response = self.client.patch("/api/admin/runtime-config/row-1/status", json={"is_active": False})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"], updated)
        patched.assert_called_once_with("row-1", False)


if __name__ == "__main__":
    unittest.main()
