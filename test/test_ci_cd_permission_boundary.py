import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PR_WORKFLOW = ROOT / ".github" / "workflows" / "pr-ci.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"
DRIFT_AUDIT_WORKFLOW = ROOT / ".github" / "workflows" / "baseline-drift-audit.yml"
CLEANUP_SCRIPT = ROOT / ".github" / "scripts" / "windows-cleanup.ps1"
BUILD_SCRIPT = ROOT / ".github" / "scripts" / "windows-build-and-test.ps1"
LINUX_BUILD_SCRIPT = ROOT / ".github" / "scripts" / "linux-build-and-test.sh"
CHECKOUT_ACTION = "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803 # v6"
LOGIN_ACTION = "docker/login-action@dbcb813823bdd20940b903addbd779551569679f # v4.6.0"


def github_action_documents() -> list[Path]:
    workflow_directory = ROOT / ".github" / "workflows"
    action_directory = ROOT / ".github" / "actions"
    workflows = sorted((*workflow_directory.glob("*.yml"), *workflow_directory.glob("*.yaml")))
    actions = []
    if action_directory.exists():
        actions = sorted((*action_directory.rglob("action.yml"), *action_directory.rglob("action.yaml")))
    return [*workflows, *actions]


class CICDPermissionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pr_workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        self.release_workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        self.cleanup_script = CLEANUP_SCRIPT.read_text(encoding="utf-8")
        self.build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.linux_build_script = LINUX_BUILD_SCRIPT.read_text(encoding="utf-8")

    def assert_context7_has_no_secret_or_runner_fallback(self, workflow: str) -> None:
        deploy_job = workflow[workflow.index("  deploy-dev:") :]
        restart_step_name = "      - name: Pull and restart fusion-api"
        identity_step_name = "      - name: Verify deployed image identity"
        restart_step = deploy_job[deploy_job.index(restart_step_name) : deploy_job.index(identity_step_name)]
        compose_heredoc = "cat > docker-compose.fusion-api-ghcr.yml <<'EOF'"
        executable_prefix = restart_step[: restart_step.index(compose_heredoc)]
        active_shell = "\n".join(line for line in executable_prefix.splitlines() if not line.lstrip().startswith("#"))

        self.assertNotIn("${{ secrets.CONTEXT7_API_KEY }}", workflow)
        self.assertNotIn("DEPLOY_CONTEXT7_API_KEY", restart_step)
        self.assertRegex(active_shell, r'(?m)^\s*export CONTEXT7_API_KEY=""\s*$')
        self.assertLess(active_shell.index("source .env"), active_shell.index('export CONTEXT7_API_KEY=""'))
        self.assertNotRegex(
            active_shell,
            r"(?m)^\s*export CONTEXT7_API_KEY=.*\$\{(?:DEPLOY_CONTEXT7_API_KEY|CONTEXT7_API_KEY)",
        )
        self.assertIn("- CONTEXT7_API_KEY=${CONTEXT7_API_KEY:-}", restart_step)
        self.assertIn(
            'export MCP_ALLOWED_CREDENTIAL_REFS="$(append_csv_value "${MCP_ALLOWED_CREDENTIAL_REFS}" "CONTEXT7_API_KEY")"',
            restart_step,
        )

    def test_drift_audit_is_owned_by_central_baseline_repository(self) -> None:
        self.assertFalse(
            DRIFT_AUDIT_WORKFLOW.exists(),
            "漂移审计应由 engineering-baseline 中央工作流统一执行",
        )
        active_workflows = "\n".join(
            path.read_text(encoding="utf-8") for path in (ROOT / ".github" / "workflows").glob("*.yml")
        )
        self.assertNotIn("engineering-baseline/.github/actions/audit", active_workflows)

    def test_pr_workflow_only_targets_master_pull_requests(self) -> None:
        self.assertRegex(
            self.pr_workflow,
            r"(?ms)^on:\n\s+pull_request:\n\s+branches:\s*\[master\]",
        )
        self.assertNotRegex(self.pr_workflow, r"(?m)^\s{2}(push|workflow_dispatch):")

    def test_pr_workflow_has_no_release_privileges(self) -> None:
        self.assertEqual(self.pr_workflow.count("name: PR container validation"), 1)
        self.assertNotIn("name: Build on Windows runner", self.pr_workflow)
        self.assertNotIn("过渡检查名", self.pr_workflow)
        self.assertIn("runs-on: ubuntu-latest", self.pr_workflow)
        self.assertNotIn("runs-on: [self-hosted, Windows, X64]", self.pr_workflow)
        self.assertNotIn("environment:", self.pr_workflow)
        self.assertNotIn("secrets.", self.pr_workflow)
        self.assertNotIn("docker/login-action", self.pr_workflow)
        self.assertNotRegex(self.pr_workflow, r"(?m)^\s*docker push\b")
        self.assertNotIn("deploy-dev:", self.pr_workflow)
        self.assertIn(".github/scripts/linux-build-and-test.sh", self.pr_workflow)
        self.assertIn("persist-credentials: false", self.pr_workflow)
        self.assertNotIn("windows-cleanup.ps1", self.pr_workflow)
        self.assertNotIn("builder prune", self.pr_workflow)
        self.assertNotIn("secrets.", self.linux_build_script)
        self.assertNotRegex(self.linux_build_script, r"(?mi)^\s*docker login\b")
        self.assertNotRegex(self.linux_build_script, r"(?mi)^\s*docker push\b")

    def test_release_workflow_only_runs_for_master(self) -> None:
        self.assertRegex(
            self.release_workflow,
            r"(?ms)^on:\n\s+push:\n\s+branches:\s*\[master\]\n\s+workflow_dispatch:",
        )
        self.assertNotIn("pull_request:", self.release_workflow)
        publish_job = self.release_workflow[
            self.release_workflow.index("  publish:") : self.release_workflow.index("  deploy-dev:")
        ]
        self.assertIn("if: github.ref == 'refs/heads/master'", publish_job)
        self.assertEqual(publish_job.count("name: Publish master images on Windows runner"), 1)
        self.assertNotIn("name: Build on Windows runner", publish_job)

    def test_release_runs_are_never_cancelled_mid_deployment(self) -> None:
        self.assertRegex(
            self.release_workflow,
            r"(?ms)^concurrency:\n\s+group: fusion-api-windows-ci-.*\n\s+cancel-in-progress: false$",
        )
        self.assertIn("cancel-in-progress: true", self.pr_workflow)

    def test_all_checkouts_disable_credential_persistence(self) -> None:
        checkout_without_credentials = f"uses: {CHECKOUT_ACTION}\n        with:\n          persist-credentials: false"
        self.assertEqual(self.pr_workflow.count(checkout_without_credentials), 1)
        self.assertEqual(self.release_workflow.count(checkout_without_credentials), 2)
        self.assertEqual(self.pr_workflow.count(f"uses: {CHECKOUT_ACTION}"), 1)
        self.assertEqual(self.release_workflow.count(f"uses: {CHECKOUT_ACTION}"), 2)

    def test_active_workflows_pin_external_actions_to_full_commit_sha(self) -> None:
        uses_key_pattern = re.compile(r"^\s*(?:-\s*)?uses\s*:")
        uses_value_pattern = re.compile(r"""^\s*(?:-\s*)?uses\s*:\s*(['"]?)([^'"#\s]+)\1(?:\s+#\s*(.*\S))?\s*$""")
        external_action_count = 0

        for action_document in github_action_documents():
            workflow = action_document.read_text(encoding="utf-8")
            for line_number, line in enumerate(workflow.splitlines(), start=1):
                if uses_key_pattern.match(line) is None:
                    continue
                action = uses_value_pattern.match(line)
                self.assertIsNotNone(
                    action,
                    f"{action_document.relative_to(ROOT)}:{line_number} 的 uses 语法未纳入安全校验",
                )
                reference = action.group(2)
                if reference.startswith("./"):
                    continue
                external_action_count += 1
                self.assertRegex(
                    reference,
                    r"^[^@\s]+@[0-9a-f]{40}$",
                    f"{action_document.name} 的 {reference} 必须锁定完整 commit SHA",
                )
                self.assertRegex(
                    action.group(3) or "",
                    r"^v\d",
                    f"{action_document.name} 的 {reference} 应保留版本注释",
                )
        self.assertGreater(external_action_count, 0, "仓库应至少包含一个外部 Action")

    def test_windows_publish_removes_only_isolated_docker_credentials(self) -> None:
        self.assertIn("$env:RUNNER_TEMP", self.cleanup_script)
        self.assertIn("$env:DOCKER_CONFIG", self.cleanup_script)
        self.assertIn('".docker-*"', self.cleanup_script)
        self.assertIn(
            "Remove-Item -LiteralPath $dockerConfigPath -Recurse -Force",
            self.cleanup_script,
        )
        self.assertLess(
            self.cleanup_script.index("$env:RUNNER_TEMP"),
            self.cleanup_script.index("Remove-Item -LiteralPath"),
        )

    def test_linux_deploy_isolates_and_removes_docker_credentials(self) -> None:
        deploy_job = self.release_workflow[self.release_workflow.index("  deploy-dev:") :]
        configure_step = "Configure Docker credential directory"
        login_step = "Login to ACR"
        cleanup_step = "Cleanup Docker credential directory"
        self.assertLess(deploy_job.index(configure_step), deploy_job.index(login_step))
        self.assertIn(
            'docker_config="${RUNNER_TEMP}/.docker-${GITHUB_RUN_ID}-${GITHUB_JOB}"',
            deploy_job,
        )
        self.assertIn('"DOCKER_CONFIG=${docker_config}" >> "${GITHUB_ENV}"', deploy_job)
        self.assertIn(f"- name: {cleanup_step}\n        if: always()", deploy_job)
        self.assertIn('"${RUNNER_TEMP}"/.docker-*', deploy_job)
        self.assertIn('rm -rf -- "${DOCKER_CONFIG}"', deploy_job)
        self.assertTrue(
            self.release_workflow.rstrip().endswith('rm -rf -- "${DOCKER_CONFIG}"'),
            "Docker 凭据目录清理必须是部署 workflow 的最后一步",
        )

    def test_release_publish_uses_dev_secrets_without_creating_deployment(self) -> None:
        publish_job = self.release_workflow[
            self.release_workflow.index("  publish:") : self.release_workflow.index("  deploy-dev:")
        ]
        self.assertRegex(
            publish_job,
            r"(?ms)^\s{4}environment:\n\s{6}name: dev\n\s{6}deployment: false$",
        )
        self.assertIn(f"uses: {LOGIN_ACTION}", publish_job)
        self.assertIn("username: ${{ secrets.ACR_USERNAME }}", publish_job)
        self.assertIn("password: ${{ secrets.ACR_PASSWORD }}", publish_job)
        self.assertIn("logout: false", publish_job)
        self.assertRegex(publish_job, r"(?m)^\s*docker push\b")

    def test_deploy_job_depends_on_release_build_and_uses_dev_environment(self) -> None:
        deploy_job = self.release_workflow[self.release_workflow.index("  deploy-dev:") :]
        self.assertIn("needs: publish", deploy_job)
        self.assertNotIn("needs.build.outputs", deploy_job)
        self.assertIn("needs.publish.outputs", deploy_job)
        self.assertRegex(deploy_job, r"(?m)^\s{4}environment: dev$")
        self.assertIn("Apply alembic migrations", deploy_job)
        self.assertIn("Run deployment smoke", deploy_job)
        self.assertIn("Push CI/CD metrics", deploy_job)
        self.assertIn("通知飞书(部署结果)", deploy_job)

    def test_deploy_verifies_running_container_images_before_health_and_smoke(self) -> None:
        deploy_job = self.release_workflow[self.release_workflow.index("  deploy-dev:") :]
        identity_step_name = "      - name: Verify deployed image identity"
        health_step_name = "      - name: Verify health"
        smoke_step_name = "      - name: Run deployment smoke"
        self.assertIn(identity_step_name, deploy_job)
        self.assertLess(deploy_job.index(identity_step_name), deploy_job.index(health_step_name))
        self.assertLess(deploy_job.index(identity_step_name), deploy_job.index(smoke_step_name))

        identity_step = deploy_job[deploy_job.index(identity_step_name) : deploy_job.index(health_step_name)]
        active_shell = "\n".join(line for line in identity_step.splitlines() if not line.lstrip().startswith("#"))
        expected_commands = (
            'expected_api_image="${IMAGE_NAME}:${GITHUB_SHA}"',
            "actual_api_image=\"$(docker inspect fusion-api --format '{{.Config.Image}}')\"",
            'if [ "${actual_api_image}" != "${expected_api_image}" ]; then',
            'expected_adapter_image="${FLYAI_ADAPTER_IMAGE_NAME}:${GITHUB_SHA}"',
            "actual_adapter_image=\"$(docker inspect fusion-flyai-adapter --format '{{.Config.Image}}')\"",
            'if [ "${actual_adapter_image}" != "${expected_adapter_image}" ]; then',
        )
        for command in expected_commands:
            self.assertRegex(active_shell, rf"(?m)^\s*{re.escape(command)}\s*$")
        self.assertGreaterEqual(active_shell.count("exit 1"), 2)

    def test_deploy_does_not_reference_or_inherit_unconfigured_context7_secret(self) -> None:
        self.assert_context7_has_no_secret_or_runner_fallback(self.release_workflow)

        compose_heredoc = "          cat > docker-compose.fusion-api-ghcr.yml <<'EOF'"
        forged_workflow = self.release_workflow.replace('          export CONTEXT7_API_KEY=""\n', "")
        forged_workflow = forged_workflow.replace("          unset CONTEXT7_API_KEY\n", "")
        forged_workflow = forged_workflow.replace(
            compose_heredoc,
            f"{compose_heredoc}\n          x-context7-contract-decoy: 'export CONTEXT7_API_KEY=\"\"'",
        )

        with self.assertRaises(AssertionError):
            self.assert_context7_has_no_secret_or_runner_fallback(forged_workflow)

    def test_release_keeps_buildkit_cache_governance(self) -> None:
        shared_script = ".github/scripts/windows-cleanup.ps1"
        self.assertIn(shared_script, self.release_workflow)
        self.assertIn('"--max-used-space", "10gb"', self.cleanup_script)
        self.assertIn('"--reserved-space", "4gb"', self.cleanup_script)
        self.assertIn('"--min-free-space", "30gb"', self.cleanup_script)


if __name__ == "__main__":
    unittest.main()
