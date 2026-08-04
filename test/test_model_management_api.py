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
os.environ["LITELLM_MODEL_ADMISSION_WORKER_ENABLED"] = "true"
os.environ["LITELLM_MODEL_ADMISSION_WORKER_TOKEN"] = "worker-secret"


class FakeModelManagementService:
    def __init__(self):
        self.calls = []
        self.claimed = True
        self.operation = SimpleNamespace(
            id="operation-1",
            candidate_fingerprint="a" * 64,
            model_id="kimi-k3",
            governance_run_id="20260804T010203456789Z",
            status="pending",
            result={},
            created_at=None,
            updated_at=None,
        )

    def operation_payload(self, operation):
        result = operation.result or {}
        return {
            "operation_id": operation.id,
            "candidate_fingerprint": operation.candidate_fingerprint,
            "model_id": operation.model_id,
            "status": operation.status,
            "phase": result.get("phase"),
            "error_code": result.get("error_code"),
            "created_at": operation.created_at,
            "updated_at": operation.updated_at,
        }

    def get_snapshot(self):
        operation = self.operation_payload(self.operation)
        return {
            "generated_at": "2026-08-04T00:00:00+00:00",
            "governance": {"available": True, "run_id": self.operation.governance_run_id},
            "capabilities": {"admission_enabled": True, "hard_delete_enabled": False},
            "models": [
                {
                    "model_id": "qwen/max",
                    "name": "Qwen Max",
                    "provider": "qwen",
                    "provider_display": "通义千问",
                    "health": "healthy",
                    "selectable": False,
                    "routable": True,
                    "state": "hidden",
                    "revision": 1,
                    "reason": "暂时隐藏",
                    "updated_at": None,
                }
            ],
            "candidates": [
                {
                    "candidate_fingerprint": "a" * 64,
                    "model_id": "kimi-k3",
                    "name": "Kimi K3",
                    "provider_key": "moonshot",
                    "provider_display": "Moonshot",
                    "state": "admission_ready",
                    "reasons": [],
                    "capabilities": {"functionCalling": True},
                }
            ],
            "operations": [operation],
        }

    def set_visibility(self, **kwargs):
        self.calls.append(("visibility", kwargs))
        return SimpleNamespace(
            model_id=kwargs["model_id"],
            selectable=kwargs["selectable"],
            routable=True,
            revision=2,
            reason=kwargs["reason"],
            updated_by=kwargs["admin"].id,
            updated_at=None,
        )

    def request_admission(self, **kwargs):
        self.calls.append(("admit", kwargs))
        return self.operation

    def claim_operation(self):
        self.calls.append(("claim", {}))
        if not self.claimed:
            return None
        self.operation.status = "running"
        return self.operation, "one-time-lease"

    def invalidate_catalog(self, **kwargs):
        self.calls.append(("invalidate", kwargs))
        return "42"

    def complete_operation(self, **kwargs):
        self.calls.append(("complete", kwargs))
        self.operation.status = kwargs["status"]
        self.operation.result = {"phase": kwargs["phase"], "error_code": kwargs["error_code"]}
        return self.operation


class FakeControlRepository:
    def get_by_model_ids(self, model_ids):
        if "qwen-max-latest" not in model_ids:
            return {}
        return {
            "qwen-max-latest": SimpleNamespace(
                model_id="qwen-max-latest",
                selectable=False,
                routable=True,
                revision=3,
                reason="隐藏",
                updated_by="admin-1",
                updated_at=None,
            )
        }

    def get(self, model_id):
        return self.get_by_model_ids([model_id]).get(model_id)


class ModelManagementApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.pop("main", None)
        cls.main = importlib.import_module("main")
        cls.client = TestClient(cls.main.app)

    def setUp(self):
        from app.api.deps import (
            get_current_admin_user,
            get_model_management_service,
        )

        self.service = FakeModelManagementService()
        self.admin = SimpleNamespace(id="admin-1", is_superuser=True)
        self.main.app.dependency_overrides[get_current_admin_user] = lambda: self.admin
        self.main.app.dependency_overrides[get_model_management_service] = lambda: self.service
        self.worker_enabled_patcher = patch(
            "app.api.deps.settings.LITELLM_MODEL_ADMISSION_WORKER_ENABLED",
            True,
        )
        self.worker_token_patcher = patch(
            "app.api.deps.settings.LITELLM_MODEL_ADMISSION_WORKER_TOKEN",
            "worker-secret",
        )
        self.worker_enabled_patcher.start()
        self.worker_token_patcher.start()

    def tearDown(self):
        self.worker_token_patcher.stop()
        self.worker_enabled_patcher.stop()
        self.main.app.dependency_overrides.clear()

    def test_admin_snapshot_has_exact_contract_and_is_never_cacheable(self):
        response = self.client.get("/api/admin/model-management")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "private, no-store")
        data = response.json()["data"]
        self.assertEqual(
            set(data),
            {"generated_at", "governance", "capabilities", "models", "candidates", "operations"},
        )
        self.assertEqual(
            set(data["models"][0]),
            {
                "model_id",
                "name",
                "provider",
                "provider_display",
                "health",
                "selectable",
                "routable",
                "state",
                "revision",
                "reason",
                "updated_at",
            },
        )
        self.assertEqual(data["candidates"][0]["candidate_fingerprint"], "a" * 64)
        self.assertEqual(
            set(data["candidates"][0]),
            {
                "candidate_fingerprint",
                "model_id",
                "name",
                "provider_key",
                "provider_display",
                "state",
                "reasons",
                "capabilities",
            },
        )
        self.assertEqual(
            set(data["operations"][0]),
            {
                "operation_id",
                "candidate_fingerprint",
                "model_id",
                "status",
                "phase",
                "error_code",
                "created_at",
                "updated_at",
            },
        )

    def test_admin_routes_use_body_model_id_and_return_persistent_operation(self):
        hidden = self.client.patch(
            "/api/admin/model-management/models/visibility",
            json={
                "model_id": "qwen/max",
                "selectable": False,
                "reason": "暂时隐藏",
                "expected_revision": 1,
            },
        )
        admitted = self.client.post(
            f"/api/admin/model-management/candidates/{'a' * 64}/admit",
            json={
                "model_id": "kimi-k3",
                "expected_run_id": "20260804T010203456789Z",
                "reason": "完成验收后准入",
            },
        )

        self.assertEqual(hidden.status_code, 200)
        self.assertEqual(hidden.json()["data"]["model_id"], "qwen/max")
        self.assertEqual(admitted.status_code, 202)
        self.assertEqual(admitted.json()["data"]["status"], "pending")
        self.assertEqual(self.service.calls[0][1]["expected_revision"], 1)
        self.assertEqual(self.service.calls[1][1]["expected_run_id"], "20260804T010203456789Z")

    def test_internal_worker_protocol_is_token_and_lease_scoped(self):
        worker_headers = {"X-Fusion-Worker-Token": "worker-secret"}
        claimed = self.client.post(
            "/api/internal/model-management/admissions/claim",
            headers=worker_headers,
        )
        self.assertEqual(claimed.status_code, 200)
        self.assertEqual(
            set(claimed.json()),
            {"operation_id", "lease_token", "run_id", "candidate_fingerprint", "model_id"},
        )

        invalidated = self.client.post(
            "/api/internal/model-management/admissions/operation-1/invalidate-catalog",
            headers={**worker_headers, "X-Operation-Lease": "one-time-lease"},
        )
        self.assertEqual(invalidated.json(), {"generation": "42"})

        completed = self.client.post(
            "/api/internal/model-management/admissions/operation-1/complete",
            headers={**worker_headers, "X-Operation-Lease": "one-time-lease"},
            json={
                "status": "succeeded",
                "phase": "complete",
                "error_code": None,
                "completed_phases": ["register", "allowlist", "readback"],
                "writes_performed": True,
                "compensation": {"attempted": False},
            },
        )
        self.assertEqual(completed.status_code, 200)
        self.assertEqual(completed.json()["status"], "succeeded")
        self.assertEqual(completed.headers["cache-control"], "private, no-store")

        rejected_extra = self.client.post(
            "/api/internal/model-management/admissions/operation-1/complete",
            headers={**worker_headers, "X-Operation-Lease": "one-time-lease"},
            json={
                "status": "failed",
                "phase": "complete",
                "error_code": "failed",
                "completed_phases": [],
                "writes_performed": False,
                "compensation": {},
                "secret": "must-not-be-accepted",
            },
        )
        self.assertEqual(rejected_extra.status_code, 422)

    def test_internal_claim_returns_204_and_invalid_worker_token_returns_401(self):
        self.service.claimed = False
        no_work = self.client.post(
            "/api/internal/model-management/admissions/claim",
            headers={"X-Fusion-Worker-Token": "worker-secret"},
        )
        self.assertEqual(no_work.status_code, 204)
        self.assertEqual(no_work.content, b"")

        unauthorized = self.client.post(
            "/api/internal/model-management/admissions/claim",
            headers={"X-Fusion-Worker-Token": "wrong"},
        )
        self.assertEqual(unauthorized.status_code, 401)

    def test_admin_endpoint_is_superuser_only(self):
        from app.api.deps import get_current_admin_user, get_current_user

        self.main.app.dependency_overrides.pop(get_current_admin_user)
        self.main.app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(id="user-1", is_superuser=False)
        forbidden = self.client.get("/api/admin/model-management")
        self.assertEqual(forbidden.status_code, 403)

    def test_public_models_list_and_detail_keep_hidden_model_routable(self):
        from app.api.deps import get_model_catalog_control_repository

        catalog = {
            "qwen-max-latest": {
                "underlying": "openai/qwen-max",
                "metadata": {"provider_key": "qwen", "capabilities": {"functionCalling": True}},
                "max_input_tokens": 32768,
                "max_output_tokens": 8192,
                "db_model": True,
            },
            "deepseek-chat": {
                "underlying": "deepseek/deepseek-chat",
                "metadata": {"provider_key": "deepseek"},
                "db_model": True,
            },
        }
        self.main.app.dependency_overrides[get_model_catalog_control_repository] = lambda: FakeControlRepository()

        with (
            patch("app.api.models.litellm_catalog.list_aliases", return_value=catalog),
            patch("app.api.models.litellm_catalog.get_model_entry", side_effect=catalog.get),
        ):
            listed = self.client.get("/api/models/")
            detailed = self.client.get("/api/models/qwen-max-latest")

        self.assertEqual(listed.status_code, 200)
        models = {item["modelId"]: item for item in listed.json()["data"]["models"]}
        self.assertFalse(models["qwen-max-latest"]["selectable"])
        self.assertTrue(models["qwen-max-latest"]["routable"])
        self.assertTrue(models["qwen-max-latest"]["enabled"])
        self.assertTrue(models["deepseek-chat"]["selectable"])
        self.assertEqual(detailed.status_code, 200)
        self.assertFalse(detailed.json()["data"]["selectable"])
        self.assertTrue(detailed.json()["data"]["routable"])
        self.assertEqual(detailed.headers["cache-control"], "private, no-store")


if __name__ == "__main__":
    unittest.main()
