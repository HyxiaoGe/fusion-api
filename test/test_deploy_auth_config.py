import unittest
from pathlib import Path


class DeployAuthConfigTests(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[1]
        self.app_config = (root / "app" / "core" / "config.py").read_text(encoding="utf-8")
        self.env_example = (root / ".env.example").read_text(encoding="utf-8")
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

    def test_deploy_passes_mcp_policy_and_credentials_without_erasing_server_env(self):
        self.assertIn("DEPLOY_DASHSCOPE_API_KEY: ${{ secrets.DASHSCOPE_API_KEY }}", self.workflow)
        self.assertIn("DEPLOY_AMAP_MCP_API_KEY: ${{ secrets.AMAP_MCP_API_KEY }}", self.workflow)
        self.assertIn(
            'export DASHSCOPE_API_KEY="${DEPLOY_DASHSCOPE_API_KEY:-${DASHSCOPE_API_KEY:-}}"',
            self.workflow,
        )
        self.assertIn(
            'export AMAP_MCP_API_KEY="${DEPLOY_AMAP_MCP_API_KEY:-${AMAP_MCP_API_KEY:-}}"',
            self.workflow,
        )
        self.assertIn(
            'export MCP_ALLOWED_HOSTS="${DEPLOY_MCP_ALLOWED_HOSTS:-${MCP_ALLOWED_HOSTS:-learn.microsoft.com,dashscope.aliyuncs.com,mcp.amap.com}}"',
            self.workflow,
        )
        for variable in (
            "MCP_ALLOWED_HOSTS",
            "MCP_ALLOWED_CREDENTIAL_REFS",
            "MCP_CONNECT_TIMEOUT_SECONDS",
            "MCP_CALL_TIMEOUT_SECONDS",
            "MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS",
            "MCP_ADMIN_OPERATION_TIMEOUT_SECONDS",
            "MCP_MAX_DISCOVERY_PAGES",
            "MCP_MAX_DISCOVERED_TOOLS",
            "MCP_MAX_TOOL_DESCRIPTION_CHARS",
            "MCP_MAX_TOOL_SCHEMA_BYTES",
            "MCP_MAX_RESPONSE_BYTES",
            "DASHSCOPE_API_KEY",
            "AMAP_MCP_API_KEY",
        ):
            self.assertIn(f"- {variable}=${{{variable}", self.workflow)
        self.assertNotIn("echo ${DASHSCOPE_API_KEY}", self.workflow)
        self.assertNotIn("echo ${AMAP_MCP_API_KEY}", self.workflow)

    def test_mcp_timeout_defaults_keep_client_budget_inside_admin_request_budget(self):
        self.assertIn("MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS=12", self.env_example)
        self.assertIn("MCP_ADMIN_OPERATION_TIMEOUT_SECONDS=35", self.env_example)
        self.assertIn("MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS:-12", self.workflow)
        self.assertIn("MCP_ADMIN_OPERATION_TIMEOUT_SECONDS:-35", self.workflow)

    def test_mcp_default_hosts_include_public_no_auth_acceptance_server(self):
        default_hosts = "learn.microsoft.com,dashscope.aliyuncs.com,mcp.amap.com"
        self.assertIn(f'"{default_hosts}"', self.app_config)
        self.assertIn(f"MCP_ALLOWED_HOSTS={default_hosts}", self.env_example)
        self.assertIn(f"MCP_ALLOWED_HOSTS:-{default_hosts}", self.workflow)
