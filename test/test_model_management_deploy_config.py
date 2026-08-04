import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ModelManagementDeployConfigTests(unittest.TestCase):
    def test_dev_deploy_mounts_only_governance_root_read_only(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn(
            "${LITELLM_GOVERNANCE_ROOT_HOST:-/home/heyanxiao/backups/litellm-governance}:/var/lib/fusion/litellm-governance:ro",
            workflow,
        )
        self.assertIn("LITELLM_GOVERNANCE_ROOT=/var/lib/fusion/litellm-governance", workflow)
        self.assertIn("LITELLM_MODEL_MANAGEMENT_ENABLED=${LITELLM_MODEL_MANAGEMENT_ENABLED:-false}", workflow)
        self.assertIn("LITELLM_MODEL_ADMISSION_WORKER_ENABLED=${LITELLM_MODEL_ADMISSION_WORKER_ENABLED:-false}", workflow)
        self.assertIn("LITELLM_MODEL_ADMISSION_WORKER_TOKEN=${LITELLM_MODEL_ADMISSION_WORKER_TOKEN:-}", workflow)
        self.assertNotIn("- LITELLM_MASTER_KEY=", workflow)
        self.assertNotIn("litellm-proxy/.env:/", workflow)

    def test_local_compose_uses_same_read_only_boundary(self):
        compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

        self.assertIn(
            "${LITELLM_GOVERNANCE_ROOT_HOST:-./ops/litellm/local-governance}:/var/lib/fusion/litellm-governance:ro",
            compose,
        )
        self.assertIn("LITELLM_GOVERNANCE_ROOT=/var/lib/fusion/litellm-governance", compose)
        self.assertNotIn("LITELLM_MASTER_KEY", compose)


if __name__ == "__main__":
    unittest.main()
