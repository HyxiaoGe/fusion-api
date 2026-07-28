import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "ops/litellm/fusion-litellm-governance.service"
TIMER = ROOT / "ops/litellm/fusion-litellm-governance.timer"
COST_SERVICE = ROOT / "ops/litellm/fusion-litellm-cost-sync.service"
COST_TIMER = ROOT / "ops/litellm/fusion-litellm-cost-sync.timer"
REQUIREMENTS = ROOT / "ops/litellm/requirements-governance.txt"


class LiteLLMGovernanceUnitTests(unittest.TestCase):
    def test_service_is_read_only_and_has_no_apply_or_secret_literal(self):
        content = SERVICE.read_text(encoding="utf-8")

        self.assertIn("run_litellm_governance_cycle.py", content)
        self.assertIn("EnvironmentFile=", content)
        self.assertIn("litellm-governance-venv/bin/python", content)
        self.assertIn("ExecStartPre=", content)
        self.assertIn("check_litellm_governance_runtime.py", content)
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
        self.assertIn("EnvironmentFile=", service)
        self.assertIn("litellm-governance-venv/bin/python", service)
        self.assertIn("check_litellm_governance_runtime.py", service)
        self.assertNotIn("sk-", service)
        self.assertNotIn("/usr/bin/python3", service)
        self.assertIn("OnUnitActiveSec=15m", timer)
        self.assertIn("Persistent=true", timer)

    def test_host_runtime_dependency_is_version_pinned(self):
        content = REQUIREMENTS.read_text(encoding="utf-8")

        self.assertIn("httpx==0.28.1", content)
        self.assertNotIn(">=", content)


if __name__ == "__main__":
    unittest.main()
