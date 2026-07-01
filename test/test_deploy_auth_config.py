import unittest
from pathlib import Path


class DeployAuthConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.workflow = (root / ".github" / "workflows" / "deploy.yml").read_text()

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

    def test_ci_runs_tests_with_process_timeout_and_verbose_output(self):
        self.assertIn(
            "timeout 270s python -u -m unittest discover -s test -t . -v",
            self.workflow,
        )

    def test_ci_lint_install_has_process_and_network_timeouts(self):
        self.assertIn(
            "timeout 180s python -m pip install --default-timeout=30 --no-cache-dir ruff",
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
