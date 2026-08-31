import copy
import re
import stat
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PR_WORKFLOW = ROOT / ".github" / "workflows" / "pr-ci.yml"
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "deploy.yml"
RELEASE_SAFETY_MANIFEST = ROOT / ".github" / "release-safety.yml"
RELEASE_SAFETY_CONTRACT = ROOT / ".github" / "scripts" / "release-safety-contract.sh"
DRIFT_AUDIT_WORKFLOW = ROOT / ".github" / "workflows" / "baseline-drift-audit.yml"
CLEANUP_SCRIPT = ROOT / ".github" / "scripts" / "windows-cleanup.ps1"
BUILD_SCRIPT = ROOT / ".github" / "scripts" / "windows-build-and-test.ps1"
LINUX_BUILD_SCRIPT = ROOT / ".github" / "scripts" / "linux-build-and-test.sh"
README = ROOT / "README.md"
CI_REQUIREMENTS = ROOT / "requirements-ci.txt"
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


def workflow_step(job: dict, name: str) -> dict:
    matches = [step for step in job["steps"] if step.get("name") == name]
    if len(matches) != 1:
        raise AssertionError(f"workflow step {name!r} 应且只能出现一次，实际 {len(matches)} 次")
    return matches[0]


def normalized_condition(value: str) -> str:
    condition = value.strip()
    if condition.startswith("${{") and condition.endswith("}}"):
        condition = condition[3:-2].strip()
    return " ".join(condition.split())


def active_commands(script: str) -> str:
    executable_lines = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith(("echo ", "Write-Host ")):
            continue
        executable_lines.append(stripped)
    return "\n".join(executable_lines)


class CICDPermissionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pr_workflow = PR_WORKFLOW.read_text(encoding="utf-8")
        self.pr_document = yaml.safe_load(self.pr_workflow)
        self.release_workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")
        self.release_document = yaml.safe_load(self.release_workflow)
        self.release_safety_manifest = yaml.safe_load(RELEASE_SAFETY_MANIFEST.read_text(encoding="utf-8"))
        self.release_safety_contract = RELEASE_SAFETY_CONTRACT.read_text(encoding="utf-8")
        self.cleanup_script = CLEANUP_SCRIPT.read_text(encoding="utf-8")
        self.build_script = BUILD_SCRIPT.read_text(encoding="utf-8")
        self.linux_build_script = LINUX_BUILD_SCRIPT.read_text(encoding="utf-8")
        self.readme = README.read_text(encoding="utf-8")
        self.ci_requirements = CI_REQUIREMENTS.read_text(encoding="utf-8")

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

    def assert_deploy_job_guard(self, condition: str) -> None:
        self.assertEqual(
            normalized_condition(condition),
            "always() && github.ref == 'refs/heads/master' && "
            "needs.prepare.result == 'success' && "
            "((needs.prepare.outputs.rollback_requested == 'true' && "
            "needs.publish.result == 'skipped') || "
            "(needs.prepare.outputs.rollback_requested != 'true' && "
            "needs.publish.result == 'success'))",
        )

    def assert_rollback_step_guard(self, condition: str) -> None:
        self.assertEqual(
            normalized_condition(condition),
            "failure() && steps.capture_rollback_target.outcome == 'success' && "
            "steps.deploy_candidate.outcome != 'skipped'",
        )

    def assert_predeploy_worker_restore_contract(self, step: dict) -> None:
        self.assertEqual(
            normalized_condition(step["if"]),
            "failure() && steps.capture_rollback_target.outcome == 'success' && "
            "steps.pause_model_management_worker.outcome == 'success' && "
            "steps.deploy_candidate.outcome == 'skipped'",
        )
        self.assertEqual(
            step["env"]["ROLLBACK_MODEL_MANAGEMENT_TIMER_ACTIVE"],
            "${{ steps.capture_rollback_target.outputs.model_management_timer_active }}",
        )
        commands = active_commands(step["run"])
        self.assertIn('if [ "${ROLLBACK_MODEL_MANAGEMENT_TIMER_ACTIVE}" = "true" ]; then', commands)
        self.assertIn("systemctl --user start fusion-litellm-model-management.timer", commands)
        self.assertIn("systemctl --user is-active --quiet fusion-litellm-model-management.timer", commands)

    def assert_cleanup_success_guard(self, step: dict) -> None:
        self.assertEqual(normalized_condition(step.get("if", "")), "success()")

    def assert_capture_contract(self, script: str) -> None:
        commands = active_commands(script)
        expected_commands = (
            "api_image_ref=\"$(docker inspect fusion-api --format '{{.Config.Image}}')\"",
            "api_image_id=\"$(docker inspect fusion-api --format '{{.Image}}')\"",
            "adapter_image_ref=\"$(docker inspect fusion-flyai-adapter --format '{{.Config.Image}}')\"",
            "adapter_image_id=\"$(docker inspect fusion-flyai-adapter --format '{{.Image}}')\"",
            'api_image_prefix="${IMAGE_NAME}:"',
            'case "${api_image_ref}" in',
            '"${api_image_prefix}"*) api_sha="${api_image_ref#"${api_image_prefix}"}" ;;',
            'if [[ ! "${api_sha}" =~ ^[0-9a-f]{40}$ ]]; then',
            'expected_adapter_ref="${FLYAI_ADAPTER_IMAGE_NAME}:${api_sha}"',
            'if [ "${adapter_image_ref}" != "${expected_adapter_ref}" ]; then',
            'if [[ ! "${api_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then',
            'if [[ ! "${adapter_image_id}" =~ ^sha256:[0-9a-f]{64}$ ]]; then',
        )
        for command in expected_commands:
            self.assertRegex(commands, rf"(?m)^{re.escape(command)}$")
        for output_name in ("api_image_ref", "api_image_id", "adapter_image_ref", "adapter_image_id"):
            self.assertEqual(script.count(f'"{output_name}=${{{output_name}}}"'), 1)

    def assert_rollback_restore_contract(self, step: dict) -> None:
        self.assert_rollback_step_guard(step["if"])
        self.assertNotIn("continue-on-error", step)
        self.assertEqual(
            step["env"]["ROLLBACK_API_IMAGE_REF"], "${{ steps.capture_rollback_target.outputs.api_image_ref }}"
        )
        self.assertEqual(
            step["env"]["ROLLBACK_API_IMAGE_ID"], "${{ steps.capture_rollback_target.outputs.api_image_id }}"
        )
        self.assertEqual(
            step["env"]["ROLLBACK_ADAPTER_IMAGE_REF"],
            "${{ steps.capture_rollback_target.outputs.adapter_image_ref }}",
        )
        self.assertEqual(
            step["env"]["ROLLBACK_ADAPTER_IMAGE_ID"],
            "${{ steps.capture_rollback_target.outputs.adapter_image_id }}",
        )
        commands = active_commands(step["run"])
        expected_commands = (
            "docker compose -f docker-compose.fusion-api-ghcr.yml up -d",
            "restored_api_ref=\"$(docker inspect fusion-api --format '{{.Config.Image}}')\"",
            "restored_api_id=\"$(docker inspect fusion-api --format '{{.Image}}')\"",
            "restored_adapter_ref=\"$(docker inspect fusion-flyai-adapter --format '{{.Config.Image}}')\"",
            "restored_adapter_id=\"$(docker inspect fusion-flyai-adapter --format '{{.Image}}')\"",
            'if [ "${restored_api_ref}" != "${ROLLBACK_API_IMAGE_REF}" ] \\',
            'if [ "${restored_adapter_ref}" != "${ROLLBACK_ADAPTER_IMAGE_REF}" ] \\',
        )
        for command in expected_commands:
            self.assertRegex(commands, rf"(?m)^{re.escape(command)}$")
        self.assertIn("docker exec fusion-api", commands)
        self.assertIn("http://127.0.0.1:8000/health", commands)
        self.assertIn("docker exec fusion-flyai-adapter", commands)
        self.assertIn("http://127.0.0.1:8080/health", commands)
        self.assertIn(
            'git -C "${GITHUB_WORKSPACE}" show "${rollback_api_sha}:scripts/deployment_smoke.py"',
            commands,
        )
        self.assertIn("| python3 - --base-url http://127.0.0.1:8002", commands)
        self.assertIn('git -C "${GITHUB_WORKSPACE}" cat-file -e "${rollback_api_sha}^{commit}"', commands)
        self.assertNotIn(
            'python3 "${GITHUB_WORKSPACE}/scripts/deployment_smoke.py" --base-url http://127.0.0.1:8002',
            commands,
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

        pr_steps = self.pr_document["jobs"]["build"]["steps"]
        contract_steps = [step for step in pr_steps if step.get("id") == "release_safety_contract"]
        self.assertEqual(len(contract_steps), 1)
        contract_step = contract_steps[0]
        self.assertEqual(contract_step["run"], ".github/scripts/release-safety-contract.sh")
        self.assertEqual(contract_step["id"], "release_safety_contract")
        self.assertNotIn("if", contract_step)
        self.assertNotIn("continue-on-error", contract_step)

        dependency_step = workflow_step(self.pr_document["jobs"]["build"], "Install release safety contract dependency")
        self.assertNotIn("id", dependency_step)
        self.assertNotIn("if", dependency_step)
        self.assertNotIn("continue-on-error", dependency_step)
        self.assertEqual(
            dependency_step["run"].strip(),
            "python3 -m pip install --disable-pip-version-check --no-cache-dir PyYAML==6.0.2",
        )

        docker_step = workflow_step(self.pr_document["jobs"]["build"], "Test and build Docker image")
        self.assertNotIn("id", docker_step)
        checkout_step = workflow_step(self.pr_document["jobs"]["build"], "Checkout")
        self.assertLess(pr_steps.index(checkout_step), pr_steps.index(dependency_step))
        self.assertLess(pr_steps.index(dependency_step), pr_steps.index(contract_step))
        self.assertLess(pr_steps.index(contract_step), pr_steps.index(docker_step))

        self.assertEqual(
            self.release_safety_contract.splitlines(),
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "exec python3 test/test_ci_cd_permission_boundary.py",
            ],
        )
        self.assertEqual(stat.S_IMODE(RELEASE_SAFETY_CONTRACT.stat().st_mode), 0o755)

    def test_release_safety_manifest_maps_real_workflow_roles(self) -> None:
        self.assertEqual(
            self.release_safety_manifest,
            {
                "version": "1",
                "workflow": ".github/workflows/deploy.yml",
                "contract_test": {
                    "path": ".github/scripts/release-safety-contract.sh",
                    "pr_step": "release_safety_contract",
                },
                "jobs": {
                    "prepare": "prepare",
                    "publish": "publish",
                    "deploy": "deploy-dev",
                    "finalize": "finalize",
                },
                "steps": {
                    "target": "deployment_request",
                    "target_job": "prepare",
                    "capture": "capture_rollback_target",
                    "migrations": ["apply_migrations"],
                    "candidate": "deploy_candidate",
                    "verify": ["verify_image_identity", "verify_health", "deployment_smoke"],
                    "rollback": "rollback_failed_deployment",
                    "cleanup": "cleanup_old_images",
                    "failure": None,
                    "finalize": "finalize_release_result",
                    "finalize_failure": None,
                },
                "needs": {
                    "publish": ["prepare"],
                    "deploy": ["prepare", "publish"],
                    "finalize": ["prepare", "publish", "deploy"],
                },
                "conditions": {
                    "prepare": None,
                    "publish": (
                        "github.ref == 'refs/heads/master' && needs.prepare.outputs.rollback_requested != 'true'"
                    ),
                    "deploy": (
                        "always() && github.ref == 'refs/heads/master' && "
                        "needs.prepare.result == 'success' && "
                        "((needs.prepare.outputs.rollback_requested == 'true' && "
                        "needs.publish.result == 'skipped') || "
                        "(needs.prepare.outputs.rollback_requested != 'true' && "
                        "needs.publish.result == 'success'))"
                    ),
                    "migration": "needs.prepare.outputs.rollback_requested != 'true'",
                    "rollback": (
                        "failure() && steps.capture_rollback_target.outcome == 'success' && "
                        "steps.deploy_candidate.outcome != 'skipped'"
                    ),
                    "cleanup": "success()",
                    "failure": None,
                    "finalize": "always() && github.ref == 'refs/heads/master'",
                    "finalize_failure": None,
                },
            },
        )

        manifest = self.release_safety_manifest
        jobs = self.release_document["jobs"]
        for job_id in manifest["jobs"].values():
            if job_id is not None:
                self.assertIn(job_id, jobs)

        prepare_step = workflow_step(jobs[manifest["jobs"]["prepare"]], "Validate deployment request")
        self.assertEqual(prepare_step["id"], manifest["steps"]["target"])

        deploy_steps = jobs[manifest["jobs"]["deploy"]]["steps"]
        deploy_step_ids = [step.get("id") for step in deploy_steps if step.get("id")]
        expected_deploy_ids = [
            manifest["steps"]["capture"],
            *manifest["steps"]["migrations"],
            manifest["steps"]["candidate"],
            *manifest["steps"]["verify"],
            manifest["steps"]["rollback"],
            manifest["steps"]["cleanup"],
        ]
        for step_id in expected_deploy_ids:
            self.assertEqual(deploy_step_ids.count(step_id), 1)

        finalize_step = workflow_step(jobs[manifest["jobs"]["finalize"]], "Finalize release result")
        self.assertEqual(finalize_step["id"], manifest["steps"]["finalize"])

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

    def test_windows_publish_requires_preconfigured_docker_engine(self) -> None:
        publish_job = self.release_document["jobs"]["publish"]
        verify_step = workflow_step(publish_job, "Verify Docker access")
        script = verify_step["run"]

        self.assertEqual(
            verify_step["shell"],
            'powershell -NoProfile -ExecutionPolicy Bypass -Command ". \'{0}\'"',
        )
        self.assertEqual(script.count("docker version"), 1)
        self.assertEqual(script.count("docker ps"), 1)
        self.assertIn("if ($LASTEXITCODE -ne 0)", script)
        self.assertIn("exit $LASTEXITCODE", script)

        for forbidden_command in (
            "Docker Desktop.exe",
            "Start-Process",
            "Start-Service",
            "RUNNER_TRACKING_ID",
            "Start-Sleep",
        ):
            with self.subTest(forbidden_command=forbidden_command):
                self.assertNotIn(forbidden_command, script)

    def test_windows_powershell_51_inline_docker_check_is_ascii_only(self) -> None:
        publish_job = self.release_document["jobs"]["publish"]
        verify_step = workflow_step(publish_job, "Verify Docker access")

        self.assertTrue(
            verify_step["run"].isascii(),
            "Windows PowerShell 5.1 会错误解码 GitHub Actions 生成的无 BOM UTF-8 临时脚本",
        )

    def test_non_ascii_windows_powershell_files_have_utf8_bom(self) -> None:
        for script_path in (BUILD_SCRIPT, CLEANUP_SCRIPT):
            raw_script = script_path.read_bytes()
            if raw_script.isascii():
                continue

            with self.subTest(script=script_path.name):
                self.assertTrue(
                    raw_script.startswith(b"\xef\xbb\xbf"),
                    "含非 ASCII 字符的 Windows PowerShell 5.1 脚本必须使用 UTF-8 BOM",
                )
                raw_script.decode("utf-8-sig")

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
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        deploy_job_text = self.release_workflow[self.release_workflow.index("  deploy-dev:") :]
        configure_step = "Configure Docker credential directory"
        login_step = "Login to ACR"
        cleanup_step = "Cleanup Docker credential directory"
        step_names = [step.get("name") for step in deploy_job["steps"]]
        self.assertLess(step_names.index(configure_step), step_names.index(login_step))
        self.assertIn(
            'docker_config="${RUNNER_TEMP}/.docker-${GITHUB_RUN_ID}-${GITHUB_JOB}"',
            deploy_job_text,
        )
        self.assertIn('"DOCKER_CONFIG=${docker_config}" >> "${GITHUB_ENV}"', deploy_job_text)
        cleanup = workflow_step(deploy_job, cleanup_step)
        self.assertEqual(normalized_condition(cleanup["if"]), "always()")
        self.assertIn('"${RUNNER_TEMP}"/.docker-*', cleanup["run"])
        self.assertIn('rm -rf -- "${DOCKER_CONFIG}"', cleanup["run"])
        self.assertEqual(step_names[-1], cleanup_step, "Docker 凭据目录清理必须是部署 job 的最后一步")

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
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        self.assertEqual(deploy_job["needs"], ["prepare", "publish"])
        self.assertEqual(deploy_job["environment"], "dev")
        self.assertEqual(deploy_job["env"]["DEPLOY_TARGET_SHA"], "${{ needs.prepare.outputs.target_sha }}")
        self.assertNotIn("needs.build.outputs", self.release_workflow)
        step_names = [step.get("name") for step in deploy_job["steps"]]
        for step_name in (
            "Apply alembic migrations",
            "Run deployment smoke",
            "Push CI/CD metrics",
            "通知飞书(部署结果)",
        ):
            self.assertIn(step_name, step_names)

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
            'expected_api_image="${IMAGE_NAME}:${DEPLOY_TARGET_SHA}"',
            "actual_api_image=\"$(docker inspect fusion-api --format '{{.Config.Image}}')\"",
            'if [ "${actual_api_image}" != "${expected_api_image}" ]; then',
            "actual_api_id=\"$(docker inspect fusion-api --format '{{.Image}}')\"",
            'if [ "${actual_api_id}" != "${expected_api_id}" ]; then',
            'expected_adapter_image="${FLYAI_ADAPTER_IMAGE_NAME}:${DEPLOY_TARGET_SHA}"',
            "actual_adapter_image=\"$(docker inspect fusion-flyai-adapter --format '{{.Config.Image}}')\"",
            'if [ "${actual_adapter_image}" != "${expected_adapter_image}" ]; then',
            "actual_adapter_id=\"$(docker inspect fusion-flyai-adapter --format '{{.Image}}')\"",
            'if [ "${actual_adapter_id}" != "${expected_adapter_id}" ]; then',
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

    def test_manual_rollback_request_is_strict_and_skips_entire_publish_job(self) -> None:
        self.assertRegex(self.ci_requirements.lower(), r"(?m)^pyyaml==6\.0\.2$")
        workflow_dispatch = self.release_document[True]["workflow_dispatch"]
        self.assertEqual(
            workflow_dispatch["inputs"],
            {
                "rollback_sha": {
                    "description": "要恢复的已发布镜像 SHA（40 位小写 Git SHA；留空表示正常发布）",
                    "required": False,
                    "type": "string",
                },
                "rollback_reason": {
                    "description": "手动回滚原因（填写 rollback_sha 时必填）",
                    "required": False,
                    "type": "string",
                },
            },
        )
        self.assertNotRegex(self.release_workflow, r"\$\{\{\s*inputs\.rollback_")

        jobs = self.release_document["jobs"]
        prepare_job = jobs["prepare"]
        self.assertEqual(prepare_job["runs-on"], "ubuntu-latest")
        request_step = workflow_step(prepare_job, "Validate deployment request")
        self.assertEqual(request_step["id"], "deployment_request")
        self.assertEqual(request_step["env"]["EVENT_NAME"], "${{ github.event_name }}")
        self.assertEqual(
            request_step["env"]["REQUESTED_ROLLBACK_SHA"],
            "${{ github.event.inputs.rollback_sha }}",
        )
        self.assertEqual(
            request_step["env"]["REQUESTED_ROLLBACK_REASON"],
            "${{ github.event.inputs.rollback_reason }}",
        )
        request_commands = active_commands(request_step["run"])
        self.assertIn('[[ ! "${rollback_sha}" =~ ^[0-9a-f]{40}$ ]]', request_commands)
        self.assertIn('[[ ! "${rollback_reason}" =~ [^[:space:]] ]]', request_commands)
        self.assertIn('target_sha="${GITHUB_SHA}"', request_commands)
        self.assertIn('rollback_requested="true"', request_commands)

        publish_job = jobs["publish"]
        self.assertEqual(publish_job["needs"], "prepare")
        self.assertEqual(
            normalized_condition(publish_job["if"]),
            "github.ref == 'refs/heads/master' && needs.prepare.outputs.rollback_requested != 'true'",
        )
        for step_name in ("Test and build Docker image", "Login to ACR", "Push Docker image"):
            self.assertNotIn("if", workflow_step(publish_job, step_name))

        deploy_job = jobs["deploy-dev"]
        migration_step = workflow_step(deploy_job, "Apply alembic migrations")
        self.assertEqual(
            normalized_condition(migration_step["if"]),
            "needs.prepare.outputs.rollback_requested != 'true'",
        )
        self.assertIn("${DEPLOY_TARGET_SHA}", active_commands(migration_step["run"]))

    def test_windows_offline_does_not_block_manual_rollback_or_enable_normal_deploy(self) -> None:
        jobs = self.release_document["jobs"]
        deploy_job = jobs["deploy-dev"]
        self.assertEqual(deploy_job["needs"], ["prepare", "publish"])
        self.assert_deploy_job_guard(deploy_job["if"])

        with self.assertRaises(AssertionError):
            self.assert_deploy_job_guard(f"{deploy_job['if']} || success()")

        finalize_job = jobs["finalize"]
        self.assertEqual(finalize_job["needs"], ["prepare", "publish", "deploy-dev"])
        self.assertEqual(finalize_job["runs-on"], "ubuntu-latest")
        self.assertEqual(
            normalized_condition(finalize_job["if"]),
            "always() && github.ref == 'refs/heads/master'",
        )
        finalize_step = workflow_step(finalize_job, "Finalize release result")
        self.assertEqual(
            finalize_step["env"],
            {
                "PREPARE_RESULT": "${{ needs.prepare.result }}",
                "PUBLISH_RESULT": "${{ needs.publish.result }}",
                "DEPLOY_RESULT": "${{ needs.deploy-dev.result }}",
                "ROLLBACK_REQUESTED": "${{ needs.prepare.outputs.rollback_requested }}",
            },
        )
        finalize_commands = active_commands(finalize_step["run"])
        self.assertIn('if [ "${PREPARE_RESULT}" != "success" ]; then', finalize_commands)
        self.assertIn('if [ "${ROLLBACK_REQUESTED}" = "true" ]; then', finalize_commands)
        self.assertIn('[ "${PUBLISH_RESULT}" = "skipped" ]', finalize_commands)
        self.assertIn('[ "${PUBLISH_RESULT}" = "success" ]', finalize_commands)
        self.assertGreaterEqual(finalize_commands.count('[ "${DEPLOY_RESULT}" = "success" ]'), 2)
        self.assertGreaterEqual(finalize_commands.count("exit 1"), 3)

    def test_deploy_captures_strict_complete_rollback_target_before_any_mutation(self) -> None:
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        step_names = [step.get("name") for step in deploy_job["steps"]]
        self.assertLess(
            step_names.index("Capture current deployment for rollback"),
            step_names.index("Apply alembic migrations"),
        )
        self.assertLess(
            step_names.index("Capture current deployment for rollback"),
            step_names.index("Pull and restart fusion-api"),
        )
        capture_step = workflow_step(deploy_job, "Capture current deployment for rollback")
        self.assertEqual(capture_step["id"], "capture_rollback_target")
        self.assert_capture_contract(capture_step["run"])

        forged_scripts = (
            capture_step["run"].replace(
                "api_image_ref=\"$(docker inspect fusion-api --format '{{.Config.Image}}')\"",
                "# api_image_ref=\"$(docker inspect fusion-api --format '{{.Config.Image}}')\"",
            ),
            capture_step["run"].replace(
                "adapter_image_id=\"$(docker inspect fusion-flyai-adapter --format '{{.Image}}')\"",
                "echo \"adapter_image_id=$(docker inspect fusion-flyai-adapter --format '{{.Image}}')\"",
            ),
            capture_step["run"].replace(
                'if [ "${adapter_image_ref}" != "${expected_adapter_ref}" ]; then',
                '# if [ "${adapter_image_ref}" != "${expected_adapter_ref}" ]; then',
            ),
        )
        for forged_script in forged_scripts:
            with self.assertRaises(AssertionError):
                self.assert_capture_contract(forged_script)

    def test_failed_candidate_rolls_back_both_images_without_masking_failure(self) -> None:
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        candidate_step = workflow_step(deploy_job, "Pull and restart fusion-api")
        self.assertEqual(candidate_step["id"], "deploy_candidate")
        rollback_step = workflow_step(deploy_job, "Roll back failed deployment")
        self.assert_rollback_restore_contract(rollback_step)

        forged_guard = copy.deepcopy(rollback_step)
        forged_guard["if"] = f"{rollback_step['if']} || success()"
        with self.assertRaises(AssertionError):
            self.assert_rollback_restore_contract(forged_guard)

        forged_compare = copy.deepcopy(rollback_step)
        forged_compare["run"] = forged_compare["run"].replace(
            'if [ "${restored_api_ref}" != "${ROLLBACK_API_IMAGE_REF}" ] \\',
            '# if [ "${restored_api_ref}" != "${ROLLBACK_API_IMAGE_REF}" ] \\',
        )
        with self.assertRaises(AssertionError):
            self.assert_rollback_restore_contract(forged_compare)

    def test_predeploy_failure_only_restores_paused_worker_timer(self) -> None:
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        restore_step = workflow_step(deploy_job, "Restore model management worker after pre-deploy failure")
        rollback_step = workflow_step(deploy_job, "Roll back failed deployment")

        self.assert_predeploy_worker_restore_contract(restore_step)
        self.assertLess(deploy_job["steps"].index(restore_step), deploy_job["steps"].index(rollback_step))

    def test_cleanup_only_runs_after_success_and_rollback_never_downgrades_database(self) -> None:
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        cleanup_step = workflow_step(deploy_job, "Cleanup old images")
        self.assert_cleanup_success_guard(cleanup_step)

        forged_cleanup = copy.deepcopy(cleanup_step)
        forged_cleanup.pop("if")
        forged_cleanup["run"] = "# if: success()\n" + forged_cleanup["run"]
        with self.assertRaises(AssertionError):
            self.assert_cleanup_success_guard(forged_cleanup)

        rollback_step = workflow_step(deploy_job, "Roll back failed deployment")
        self.assertNotIn("alembic downgrade", self.release_workflow)
        self.assertNotIn("alembic upgrade", active_commands(rollback_step["run"]))
        self.assertIn("expand/contract", self.readme)
        self.assertIn("镜像回滚绝不执行 `alembic downgrade`", self.readme)
        self.assertIn("ACR 中的 SHA 标签继续作为手动回滚来源", self.readme)

    def test_release_evidence_uses_actual_deploy_target_sha(self) -> None:
        deploy_job = self.release_document["jobs"]["deploy-dev"]
        self.assertEqual(deploy_job["env"]["DEPLOY_TARGET_SHA"], "${{ needs.prepare.outputs.target_sha }}")
        metrics_step = workflow_step(deploy_job, "Push CI/CD metrics")
        metrics_commands = active_commands(metrics_step["run"])
        self.assertIn('"${IMAGE_NAME}:${DEPLOY_TARGET_SHA}"', metrics_commands)
        self.assertIn('"${DEPLOY_TARGET_SHA}"', metrics_commands)
        self.assertNotIn("github.sha", metrics_commands)
        self.assertNotIn("GITHUB_SHA", metrics_commands)

        notification_step = workflow_step(deploy_job, "通知飞书(部署结果)")
        self.assertEqual(notification_step["env"]["TARGET_SHA"], "${{ needs.prepare.outputs.target_sha }}")
        self.assertIn(
            'sha = os.environ.get("TARGET_SHA", "")[:7]',
            active_commands(notification_step["run"]),
        )


if __name__ == "__main__":
    unittest.main()
