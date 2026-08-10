import unittest
from pathlib import Path

import yaml

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
        self.assertIn(
            "LITELLM_MODEL_ADMISSION_WORKER_ENABLED=${LITELLM_MODEL_ADMISSION_WORKER_ENABLED:-false}", workflow
        )
        self.assertIn("LITELLM_MODEL_ADMISSION_WORKER_TOKEN=${LITELLM_MODEL_ADMISSION_WORKER_TOKEN:-}", workflow)
        self.assertNotIn("- LITELLM_MASTER_KEY=", workflow)
        self.assertNotIn("litellm-proxy/.env:/", workflow)

    def test_dev_deploy_manages_worker_as_versioned_systemd_unit(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        unit = (ROOT / "ops/litellm/fusion-litellm-model-management.service").read_text(encoding="utf-8")

        self.assertIn("Install model management worker", workflow)
        self.assertIn("Pause model management worker before API deploy", workflow)
        self.assertIn("litellm-model-management-src-${GITHUB_SHA}", workflow)
        self.assertIn("litellm-model-management-current", workflow)
        self.assertIn("fusion-litellm-model-management.service", workflow)
        self.assertIn("fusion-litellm-model-management.timer", workflow)
        self.assertIn("systemctl --user enable --now fusion-litellm-model-management.timer", workflow)
        self.assertIn("systemd-analyze --user verify", workflow)
        self.assertIn("systemctl --user disable --now fusion-litellm-model-management.timer", workflow)
        self.assertIn("systemctl --user stop fusion-litellm-model-management.timer", workflow)
        self.assertIn("systemctl --user is-active --quiet fusion-litellm-model-management.service", workflow)
        self.assertLess(
            workflow.index("Pause model management worker before API deploy"),
            workflow.index("Apply alembic migrations"),
        )
        self.assertLess(
            workflow.index("Pause model management worker before API deploy"),
            workflow.index("Pull and restart fusion-api"),
        )
        self.assertIn("scripts.check_litellm_model_management_worker_env", workflow)
        self.assertIn("scripts.configure_litellm_model_management_worker_env", workflow)
        self.assertIn("verify_litellm_governance_snapshot.py", workflow)
        self.assertIn('source "${HOME}/project/fusion/.env"', workflow)
        self.assertIn("DEPLOY_LITELLM_MODEL_ADMISSION_WORKER_TOKEN", workflow)
        self.assertIn('acceptance_dir="${HOME}/.local/share/fusion/litellm-acceptance"', workflow)
        self.assertIn('install -d -m 0700 "${state_dir}" "${acceptance_dir}"', workflow)
        self.assertIn("os.chmod(path, 0o600, follow_symlinks=False)", workflow)
        self.assertIn("--require-env LITELLM_CANDIDATE_KEY", workflow)
        self.assertIn("--require-env LITELLM_GOVERNANCE_MAX_AGE_SECONDS", unit)
        self.assertIn("--require-env LITELLM_CANDIDATE_KEY", unit)
        self.assertIn("--acceptance-dir %h/.local/share/fusion/litellm-acceptance", unit)
        self.assertIn("ReadWritePaths=%h/.local/share/fusion/litellm-acceptance", unit)
        self.assertNotIn("--governance-max-age-seconds 86400", unit)

    def test_deploy_restores_worker_lifecycle_when_deployment_rolls_back(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        document = yaml.safe_load(workflow)
        rollback_step = next(
            step
            for step in document["jobs"]["deploy-dev"]["steps"]
            if step.get("name") == "Roll back failed deployment"
        )

        self.assertIn(
            'model_management_current_target="$(readlink -f -- "${model_management_current_link}")"', workflow
        )
        self.assertIn('"model_management_timer_enabled=${model_management_timer_enabled}"', workflow)
        self.assertIn('"model_management_timer_active=${model_management_timer_active}"', workflow)
        self.assertIn(
            "ROLLBACK_MODEL_MANAGEMENT_CURRENT_TARGET: "
            "${{ steps.capture_rollback_target.outputs.model_management_current_target }}",
            workflow,
        )
        self.assertIn("Restore model management worker after automatic rollback", workflow)
        self.assertIn("Restore model management worker for manual rollback", workflow)
        self.assertIn("id: pause_model_management_worker", workflow)
        self.assertIn("steps.pause_model_management_worker.outcome != 'skipped'", workflow)
        self.assertIn(
            'target_release="${HOME}/.local/share/fusion/litellm-model-management-src-${DEPLOY_TARGET_SHA}"', workflow
        )
        self.assertIn("if: needs.prepare.outputs.rollback_requested != 'true'", workflow)
        self.assertIn("if: needs.prepare.outputs.rollback_requested == 'true'", workflow)
        self.assertLess(
            workflow.index("Capture current deployment for rollback"),
            workflow.index("Pause model management worker before API deploy"),
        )
        self.assertLess(
            workflow.index("Run deployment smoke"),
            workflow.index("Restore model management worker for manual rollback"),
        )
        self.assertEqual(
            rollback_step["env"]["DEPLOY_LITELLM_MODEL_MANAGEMENT_ENABLED"],
            "${{ vars.LITELLM_MODEL_MANAGEMENT_ENABLED || 'false' }}",
        )
        self.assertEqual(
            rollback_step["env"]["DEPLOY_LITELLM_MODEL_ADMISSION_WORKER_ENABLED"],
            "${{ vars.LITELLM_MODEL_ADMISSION_WORKER_ENABLED || 'false' }}",
        )
        self.assertIn(
            "export LITELLM_MODEL_MANAGEMENT_ENABLED=\"$(printf '%s' "
            "\"${DEPLOY_LITELLM_MODEL_MANAGEMENT_ENABLED:-false}\" | tr '[:upper:]' '[:lower:]')\"",
            rollback_step["run"],
        )

    def test_dev_deploy_manages_discovery_registry_and_governance_timer(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        unit = (ROOT / "ops/litellm/fusion-litellm-governance.service").read_text(encoding="utf-8")

        self.assertIn("Install LiteLLM governance discovery", workflow)
        self.assertIn("litellm-governance-src-${GITHUB_SHA}", workflow)
        self.assertIn("litellm-governance-current", workflow)
        self.assertIn("provider-registry.example.json", workflow)
        self.assertIn("litellm-provider-registry.json", workflow)
        self.assertIn("fusion-litellm-governance.service", workflow)
        self.assertIn("fusion-litellm-governance.timer", workflow)
        self.assertIn("systemctl --user enable --now fusion-litellm-governance.timer", workflow)
        self.assertIn("%h/.local/share/fusion/litellm-governance-current", unit)
        self.assertNotIn("%h/project/fusion/fusion-api/scripts", unit)

    def test_dev_deploy_uses_repository_vars_for_feature_flags(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn("vars.LITELLM_MODEL_MANAGEMENT_ENABLED", workflow)
        self.assertIn("vars.LITELLM_MODEL_ADMISSION_WORKER_ENABLED", workflow)
        self.assertIn("vars.LITELLM_GOVERNANCE_MAX_AGE_SECONDS", workflow)
        self.assertIn("secrets.LITELLM_MODEL_ADMISSION_WORKER_TOKEN", workflow)
        self.assertIn("tr '[:upper:]' '[:lower:]'", workflow)

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
