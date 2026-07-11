import unittest
from pathlib import Path


class DeployAuthConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.workflow = (root / ".github" / "workflows" / "deploy.yml").read_text(encoding="utf-8")
        self.ci_requirements = (root / "requirements-ci.txt").read_text(encoding="utf-8")

    def test_deploy_dev_overrides_auth_internal_base_to_docker_dns(self):
        self.assertIn(
            'export AUTH_SERVICE_INTERNAL_BASE_URL="http://auth-service:8100"',
            self.workflow,
        )

    def test_deploy_health_checks_auth_jwks_from_running_container(self):
        self.assertIn("RESOLVED_AUTH_SERVICE_JWKS_URL", self.workflow)
        self.assertIn("auth JWKS ok", self.workflow)

    def test_deploy_runs_api_surface_smoke_after_health(self):
        self.assertIn("Run deployment smoke", self.workflow)
        self.assertIn("python3 scripts/deployment_smoke.py --base-url http://127.0.0.1:8002", self.workflow)
        self.assertIn("/api/models/", self.workflow)

    def test_deploy_mounts_persistent_file_storage(self):
        self.assertIn("mkdir -p ./fusion-api/storage/files", self.workflow)
        self.assertIn("./fusion-api/storage/files:/app/storage/files", self.workflow)
        self.assertIn("FILE_STORAGE_PATH=/app/storage/files", self.workflow)

    def test_deploy_preserves_container_files_before_recreate(self):
        self.assertIn("tar -C /app/storage/files -cf - .", self.workflow)
        self.assertIn("tar -C ./fusion-api/storage/files -xf -", self.workflow)
        self.assertNotIn("docker cp fusion-api:/app/storage/files/.", self.workflow)

    def test_deploy_verifies_persistent_file_storage_after_restart(self):
        self.assertIn("file storage backend ok", self.workflow)
        self.assertIn('settings.FILE_STORAGE_PATH != "/app/storage/files"', self.workflow)
        self.assertIn("OSS_ACCESS_KEY_SECRET", self.workflow)
        self.assertIn("DIRECT_UPLOAD_STALE_SECONDS=${DIRECT_UPLOAD_STALE_SECONDS:-1800}", self.workflow)
        self.assertIn("Access-Control-Request-Method", self.workflow)
        self.assertIn("oss cors PUT method is not allowed", self.workflow)

    def test_ci_runs_tests_with_process_timeout_and_verbose_output(self):
        self.assertIn(
            "timeout 270s python -u -m unittest discover -s test -t . -v",
            self.workflow,
        )

    def test_ci_lint_install_has_process_and_network_timeouts(self):
        self.assertIn("ruff==", self.ci_requirements)
        self.assertIn(
            "timeout 300s python -m pip install --default-timeout=30 --no-cache-dir -r requirements-ci.txt",
            self.workflow,
        )

    def test_windows_acr_login_uses_docker_login_action(self):
        self.assertIn("uses: docker/login-action@v3", self.workflow)
        self.assertIn("registry: ${{ env.REGISTRY }}", self.workflow)
        self.assertIn("username: ${{ secrets.ACR_USERNAME }}", self.workflow)
        self.assertIn("password: ${{ secrets.ACR_PASSWORD }}", self.workflow)

    def test_linux_acr_login_uses_password_stdin_without_shell_echo(self):
        self.assertIn(
            """printf '%s' "${ACR_PASSWORD}" | docker login""",
            self.workflow,
        )
