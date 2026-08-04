import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "ops/litellm/fusion-litellm-governance.service"
TIMER = ROOT / "ops/litellm/fusion-litellm-governance.timer"
COST_SERVICE = ROOT / "ops/litellm/fusion-litellm-cost-sync.service"
COST_TIMER = ROOT / "ops/litellm/fusion-litellm-cost-sync.timer"
MODEL_MANAGEMENT_SERVICE = ROOT / "ops/litellm/fusion-litellm-model-management.service"
MODEL_MANAGEMENT_TIMER = ROOT / "ops/litellm/fusion-litellm-model-management.timer"
REQUIREMENTS = ROOT / "ops/litellm/requirements-governance.txt"
GOVERNANCE_ENV = ROOT / "ops/litellm/litellm-governance.env.example"


class LiteLLMGovernanceUnitTests(unittest.TestCase):
    def test_documented_module_entrypoints_start_from_repo_root(self):
        modules = (
            "scripts.run_litellm_governance_unit",
            "scripts.run_litellm_governance_cycle",
            "scripts.ensure_litellm_cost_map_sync",
            "scripts.orchestrate_litellm_model_candidates",
            "scripts.enrich_litellm_model_candidates",
            "scripts.execute_litellm_candidate_admission",
            "scripts.check_litellm_model_management_worker_env",
            "scripts.configure_litellm_model_management_worker_env",
            "scripts.run_litellm_model_management_worker",
        )

        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", module, "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_service_is_read_only_and_has_no_apply_or_secret_literal(self):
        content = SERVICE.read_text(encoding="utf-8")

        self.assertIn("run_litellm_governance_cycle.py", content)
        self.assertNotIn("EnvironmentFile=", content)
        self.assertIn("%h/project/litellm-proxy/.env", content)
        self.assertIn("run_litellm_governance_unit.py", content)
        self.assertIn("-m scripts.run_litellm_governance_unit", content)
        self.assertIn("AssertPathExists=", content)
        self.assertIn("ops/litellm/candidate-overrides.json", content)
        self.assertIn("--overrides", content)
        self.assertNotIn("ConditionPathExists=", content)
        self.assertIn("litellm-governance-venv/bin/python", content)
        self.assertNotIn("ExecStartPre=", content)
        self.assertIn("UnsetEnvironment=PYTHONPATH PYTHONHOME LD_PRELOAD", content)
        self.assertIn("NoNewPrivileges=true", content)
        self.assertIn("ProtectSystem=strict", content)
        self.assertIn("UMask=0077", content)
        self.assertNotIn("--apply", content)
        self.assertNotIn("sk-", content)
        self.assertNotIn("LITELLM_MASTER_KEY=", content)
        self.assertNotIn("/usr/bin/python3", content)

    def test_timer_uses_six_hour_shanghai_cadence_and_persistent_catchup(self):
        content = TIMER.read_text(encoding="utf-8")

        self.assertIn("00,06,12,18:20:00 Asia/Shanghai", content)
        self.assertIn("Persistent=true", content)
        self.assertIn("RandomizedDelaySec=10m", content)
        self.assertIn("fusion-litellm-governance.service", content)

    def test_cost_sync_unit_is_separate_idempotent_apply_guard(self):
        service = COST_SERVICE.read_text(encoding="utf-8")
        timer = COST_TIMER.read_text(encoding="utf-8")

        self.assertIn("ensure_litellm_cost_map_sync.py", service)
        self.assertIn("--apply", service)
        self.assertNotIn("EnvironmentFile=", service)
        self.assertIn("%h/project/litellm-proxy/.env", service)
        self.assertIn("run_litellm_governance_unit.py", service)
        self.assertIn("-m scripts.run_litellm_governance_unit", service)
        self.assertIn("UnsetEnvironment=PYTHONPATH PYTHONHOME LD_PRELOAD", service)
        self.assertIn("AssertPathExists=", service)
        self.assertNotIn("ConditionPathExists=", service)
        self.assertIn("litellm-governance-venv/bin/python", service)
        self.assertNotIn("sk-", service)
        self.assertNotIn("/usr/bin/python3", service)
        self.assertIn("*:00/15:00 Asia/Shanghai", timer)
        self.assertNotIn("OnUnitActiveSec=", timer)
        self.assertIn("Persistent=true", timer)

    def test_host_runtime_dependency_is_version_pinned(self):
        content = REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("httpx==0.28.1", content)
        self.assertNotIn(">=", content)

    def test_model_management_worker_is_isolated_and_timer_driven(self):
        service = MODEL_MANAGEMENT_SERVICE.read_text(encoding="utf-8")
        timer = MODEL_MANAGEMENT_TIMER.read_text(encoding="utf-8")

        self.assertIn("run_litellm_model_management_worker", service)
        self.assertIn("run_litellm_governance_unit", service)
        self.assertIn("--require-env LITELLM_MASTER_KEY", service)
        self.assertIn("--require-env LITELLM_VIRTUAL_KEY", service)
        self.assertIn("--require-env LITELLM_MODEL_ADMISSION_WORKER_TOKEN", service)
        self.assertIn("--require-env LITELLM_GOVERNANCE_MAX_AGE_SECONDS", service)
        self.assertNotIn("EnvironmentFile=", service)
        self.assertIn("ProtectSystem=strict", service)
        self.assertIn("ProtectHome=read-only", service)
        self.assertIn("NoNewPrivileges=true", service)
        self.assertIn("%h/.local/share/fusion/litellm-model-management-current", service)
        self.assertNotIn("%h/project/fusion/fusion-api/scripts/run_litellm_model_management_worker.py", service)
        self.assertIn("%h/backups/litellm-governance", service)
        self.assertIn("%h/.local/state/fusion/litellm-model-management", service)
        self.assertIn("ReadWritePaths=", service)
        self.assertIn("OnCalendar=*-*-* *:*:00 Asia/Shanghai", timer)
        self.assertIn("Persistent=true", timer)

    def test_governance_env_does_not_duplicate_provider_or_master_keys(self):
        content = GOVERNANCE_ENV.read_text(encoding="utf-8")

        self.assertIn("LITELLM_BASE_URL=", content)
        self.assertIn("LITELLM_CANDIDATE_KEY=", content)
        self.assertIn("LITELLM_VIRTUAL_KEY=", content)
        self.assertIn("LITELLM_MODEL_ADMISSION_WORKER_TOKEN=", content)
        self.assertIn("LITELLM_GOVERNANCE_MAX_AGE_SECONDS=", content)
        self.assertNotIn("LITELLM_MASTER_KEY=", content)
        self.assertNotIn("MOONSHOT_API_KEY=", content)
        self.assertNotIn("QWEN_API_KEY=", content)


if __name__ == "__main__":
    unittest.main()
