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

    def test_ci_runs_tests_with_process_timeout_and_verbose_output(self):
        self.assertIn(
            "timeout 270s python -u -m unittest discover -s test -t . -v",
            self.workflow,
        )

    def test_acr_login_uses_password_stdin_without_cmd_password_expansion(self):
        self.assertIn(
            "$env:ACR_PASSWORD | docker login $env:REGISTRY -u $env:ACR_USERNAME --password-stdin",
            self.workflow,
        )
        self.assertIn(
            """printf '%s' "${ACR_PASSWORD}" | docker login""",
            self.workflow,
        )
