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
