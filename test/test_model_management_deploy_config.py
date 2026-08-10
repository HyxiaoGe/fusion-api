import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


class ModelManagementDeployConfigTests(unittest.TestCase):
    def test_dev_deploy_mounts_only_governance_root_read_only(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn(
            "${LITELLM_GOVERNANCE_ROOT_HOST}:/var/lib/fusion/litellm-governance:ro",
            workflow,
        )
        self.assertIn('expected_governance_root="${HOME}/backups/litellm-governance"', workflow)
        self.assertIn('if [ "${governance_root}" != "${expected_governance_root}" ]', workflow)
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
        self.assertIn("litellm-model-management-src-${DEPLOY_TARGET_SHA}", workflow)
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
        self.assertIn("--require-env FUSION_MODEL_MANAGEMENT_BASE_URL", unit)
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
        self.assertIn("ref: ${{ needs.prepare.outputs.target_sha }}", workflow)
        self.assertIn("id: pause_model_management_worker", workflow)
        self.assertIn("Restore model management worker after pre-deploy failure", workflow)
        self.assertIn("steps.deploy_candidate.outcome != 'skipped'", workflow)
        self.assertIn(
            'target_release="${HOME}/.local/share/fusion/litellm-model-management-src-${DEPLOY_TARGET_SHA}"', workflow
        )
        self.assertIn(
            'target_service_unit="${GITHUB_WORKSPACE}/ops/litellm/fusion-litellm-model-management.service"', workflow
        )
        self.assertIn(
            'target_timer_unit="${GITHUB_WORKSPACE}/ops/litellm/fusion-litellm-model-management.timer"', workflow
        )
        self.assertIn('install -m 0644 "${target_service_unit}" "${target_timer_unit}" "${unit_dir}/"', workflow)
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
            rollback_step["env"]["ROLLBACK_LITELLM_MODEL_MANAGEMENT_ENABLED"],
            "${{ steps.capture_rollback_target.outputs.model_management_enabled }}",
        )
        self.assertEqual(
            rollback_step["env"]["ROLLBACK_LITELLM_MODEL_ADMISSION_WORKER_ENABLED"],
            "${{ steps.capture_rollback_target.outputs.model_admission_worker_enabled }}",
        )
        self.assertIn(
            "export LITELLM_MODEL_MANAGEMENT_ENABLED=\"$(printf '%s' "
            "\"${ROLLBACK_LITELLM_MODEL_MANAGEMENT_ENABLED}\" | tr '[:upper:]' '[:lower:]')\"",
            rollback_step["run"],
        )
        self.assertIn("capture_container_bool_env()", workflow)
        self.assertIn(
            'rollback_model_management_enabled="$(capture_container_bool_env LITELLM_MODEL_MANAGEMENT_ENABLED)"',
            workflow,
        )
        self.assertIn(
            'rollback_model_admission_worker_enabled="$(capture_container_bool_env '
            'LITELLM_MODEL_ADMISSION_WORKER_ENABLED)"',
            workflow,
        )
        self.assertIn('"model_management_enabled=${rollback_model_management_enabled}"', workflow)
        self.assertIn('"model_admission_worker_enabled=${rollback_model_admission_worker_enabled}"', workflow)
        self.assertLess(
            rollback_step["run"].index("systemctl --user stop fusion-litellm-model-management.timer"),
            rollback_step["run"].index("ensure_rollback_image"),
        )
        self.assertLess(
            rollback_step["run"].index("wait_for_user_services"),
            rollback_step["run"].index("ensure_rollback_image"),
        )

    def test_dev_deploy_manages_discovery_registry_and_governance_timer(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        document = yaml.safe_load(workflow)
        unit = (ROOT / "ops/litellm/fusion-litellm-governance.service").read_text(encoding="utf-8")
        install_step = next(
            step
            for step in document["jobs"]["deploy-dev"]["steps"]
            if step.get("name") == "Install LiteLLM governance discovery"
        )

        self.assertIn("Install LiteLLM governance discovery", workflow)
        self.assertIn("litellm-governance-src-${DEPLOY_TARGET_SHA}", workflow)
        self.assertIn("litellm-governance-current", workflow)
        self.assertIn("provider-registry.example.json", workflow)
        self.assertIn("litellm-provider-registry.json", workflow)
        self.assertIn("fusion-litellm-governance.service", workflow)
        self.assertIn("fusion-litellm-governance.timer", workflow)
        self.assertIn("systemctl --user enable --now fusion-litellm-governance.timer", workflow)
        self.assertIn('governance_current_target="$(readlink -f -- "${governance_current_link}")"', workflow)
        self.assertIn('"governance_timer_enabled=${governance_timer_enabled}"', workflow)
        self.assertIn('"governance_timer_active=${governance_timer_active}"', workflow)
        self.assertIn(
            "ROLLBACK_GOVERNANCE_CURRENT_TARGET: "
            "${{ steps.capture_rollback_target.outputs.governance_current_target }}",
            workflow,
        )
        self.assertIn("Restore LiteLLM governance discovery after automatic rollback", workflow)
        self.assertIn("ROLLBACK_GOVERNANCE_TIMER_ENABLED", workflow)
        self.assertIn("ROLLBACK_GOVERNANCE_TIMER_ACTIVE", workflow)
        self.assertIn("governance_service_unit_b64=", workflow)
        self.assertIn("governance_timer_unit_b64=", workflow)
        self.assertIn("ROLLBACK_GOVERNANCE_SERVICE_UNIT_B64", workflow)
        self.assertIn("ROLLBACK_GOVERNANCE_TIMER_UNIT_B64", workflow)
        self.assertIn("ROLLBACK_MODEL_MANAGEMENT_SERVICE_UNIT_B64", workflow)
        self.assertIn("ROLLBACK_MODEL_MANAGEMENT_TIMER_UNIT_B64", workflow)
        self.assertIn("restore_governance_unit", workflow)
        self.assertIn("base64 --decode", workflow)
        self.assertIn('rm -f "${target}"', workflow)
        self.assertEqual(workflow.count("restore_governance_unit \\"), 4)
        self.assertIn("%h/.local/share/fusion/litellm-governance-current", unit)
        self.assertNotIn("%h/project/fusion/fusion-api/scripts", unit)
        install_script = install_step["run"]
        self.assertIn('proxy_env="${HOME}/project/litellm-proxy/.env"', install_script)
        self.assertIn('governance_env="${HOME}/.config/fusion/litellm-governance.env"', install_script)
        self.assertIn("os.chmod(path, 0o600, follow_symlinks=False)", install_script)
        self.assertIn("systemctl --user start fusion-litellm-governance.service", install_script)
        self.assertIn(
            "systemctl --user show fusion-litellm-governance.service --property=Result --value",
            install_script,
        )
        self.assertLess(
            install_script.index("os.chmod(path, 0o600, follow_symlinks=False)"),
            install_script.index("systemctl --user start fusion-litellm-governance.service"),
        )
        self.assertLess(
            install_script.index("systemctl --user start fusion-litellm-governance.service"),
            install_script.index("systemctl --user enable --now fusion-litellm-governance.timer"),
        )

    def test_manual_rollback_health_probe_tolerates_pre_governance_images(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        document = yaml.safe_load(workflow)
        verify_step = next(
            step for step in document["jobs"]["deploy-dev"]["steps"] if step.get("name") == "Verify health"
        )

        self.assertEqual(
            verify_step["env"]["ROLLBACK_REQUESTED"],
            "${{ needs.prepare.outputs.rollback_requested }}",
        )
        self.assertIn('FUSION_ROLLBACK_REQUESTED="${ROLLBACK_REQUESTED}"', verify_step["run"])
        self.assertIn("getattr(settings, name, None)", verify_step["run"])
        self.assertIn("旧版回滚目标不含模型治理配置，跳过该项兼容性探针", verify_step["run"])

    def test_automatic_rollback_wait_budget_fits_deploy_job(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")
        document = yaml.safe_load(workflow)
        deploy_job = document["jobs"]["deploy-dev"]
        rollback_step = next(step for step in deploy_job["steps"] if step.get("name") == "Roll back failed deployment")

        self.assertGreaterEqual(deploy_job["timeout-minutes"], 30)
        self.assertIn("wait_for_user_services", rollback_step["run"])
        self.assertIn("seq 1 24", rollback_step["run"])
        self.assertNotIn("seq 1 120", rollback_step["run"])

    def test_dev_deploy_preserves_existing_provider_registry(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertIn('if [ ! -e "${registry_target}" ]; then', workflow)
        self.assertIn('python3 - "${registry_target}"', workflow)
        self.assertNotIn('python3 - "${registry_temp}"', workflow)
        self.assertIn("LiteLLM provider registry 权限过宽", workflow)

    def test_deploy_refusal_restores_active_governance_and_worker_timers(self):
        workflow = (ROOT / ".github/workflows/deploy.yml").read_text(encoding="utf-8")

        self.assertGreaterEqual(
            workflow.count('if [ "${ROLLBACK_MODEL_MANAGEMENT_TIMER_ACTIVE}" = "true" ]; then'),
            2,
        )
        self.assertIn("systemctl --user start fusion-litellm-model-management.timer", workflow)
        self.assertIn('if [ "${ROLLBACK_GOVERNANCE_TIMER_ACTIVE}" = "true" ]; then', workflow)
        self.assertIn("systemctl --user start fusion-litellm-governance.timer", workflow)
        self.assertIn("定时器已恢复部署前状态", workflow)

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
