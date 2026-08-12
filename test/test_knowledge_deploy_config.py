import json
import os
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class KnowledgeDeployConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = Path(".github/workflows/deploy.yml").read_text(encoding="utf-8")
        cls.milvus_compose = Path("ops/knowledge/milvus-compose.yml").read_text(encoding="utf-8")
        cls.fusion_override = Path("ops/knowledge/fusion-compose.override.yml").read_text(encoding="utf-8")

    def _run_rollback_snapshot(self, containers):
        marker = 'docker inspect "${rollback_containers[@]}" | python3 -c \'\n'
        script = self.workflow.split(marker, 1)[1].split("\n          '\n", 1)[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollback.env"
            completed = subprocess.run(
                [sys.executable, "-c", textwrap.dedent(script)],
                input=json.dumps(containers),
                text=True,
                capture_output=True,
                check=True,
                env={**os.environ, "ROLLBACK_KNOWLEDGE_CONFIG_FILE": str(path)},
            )
            contents = path.read_text(encoding="utf-8")
            mode = stat.S_IMODE(path.stat().st_mode)
            syntax = subprocess.run(["bash", "-n", str(path)], text=True, capture_output=True, check=True)
        return completed, contents, mode, syntax

    def test_real_acceptance_stack_pins_private_authenticated_milvus(self):
        self.assertIn("milvusdb/milvus:v2.6.21", self.milvus_compose)
        self.assertIn('COMMON_SECURITY_AUTHORIZATIONENABLED: "true"', self.milvus_compose)
        self.assertIn('"127.0.0.1:19530:19530"', self.milvus_compose)
        self.assertIn("name: fusion_knowledge_milvus", self.milvus_compose)

    def test_worker_is_independent_and_shares_no_port(self):
        worker = self.fusion_override.split("knowledge-worker:", 1)[1]
        self.assertIn('["python", "-m", "scripts.run_knowledge_worker"]', worker)
        self.assertIn("./storage/files:/app/storage/files", worker)
        self.assertIn("fusion-knowledge-milvus", worker)
        self.assertNotIn("ports:", worker)

    def test_deploy_tracks_worker_identity_health_and_rollback(self):
        required = (
            "fusion-knowledge-worker",
            "knowledge_worker_image_id",
            "knowledge_worker_supported",
            "ROLLBACK_KNOWLEDGE_WORKER_EXISTED",
            "--profile knowledge-worker up -d",
            "--health-max-age-seconds 120",
            "DEPLOY_MILVUS_PASSWORD: ${{ secrets.MILVUS_PASSWORD }}",
            "DEPLOY_KNOWLEDGE_BASE_ENABLED: ${{ vars.KNOWLEDGE_BASE_ENABLED || 'false' }}",
        )
        for marker in required:
            self.assertIn(marker, self.workflow)
        self.assertNotIn("MILVUS_BOOTSTRAP_PASSWORD", self.workflow)

    def test_automatic_rollback_restores_secure_knowledge_configuration_snapshot(self):
        capture = self.workflow.split("- name: Capture current deployment for rollback", 1)[1].split(
            "- name: Login to ACR", 1
        )[0]
        rollback = self.workflow.split("- name: Roll back failed deployment", 1)[1].split(
            "- name: Cleanup old images", 1
        )[0]
        cleanup = self.workflow.split("- name: Cleanup Docker credential directory", 1)[1]
        github_output = capture.split("          {\n            printf", 1)[1]

        self.assertIn("ROLLBACK_KNOWLEDGE_CONFIG_FILE", capture)
        self.assertIn("os.O_EXCL", capture)
        self.assertIn("0o600", capture)
        self.assertIn('"MILVUS_PASSWORD"', capture)
        for name in (
            "KNOWLEDGE_MAX_BASES_PER_USER",
            "KNOWLEDGE_WORKER_RETRY_MAX_SECONDS",
            "KNOWLEDGE_WORKER_HEALTH_FILE",
            "MILVUS_TIMEOUT_SECONDS",
            "MILVUS_DOCKER_NETWORK",
        ):
            self.assertIn(f'"{name}"', capture)
            self.assertGreaterEqual(self.workflow.count(f"- {name}=${{{name}}}"), 2)
        self.assertNotIn("knowledge_base_enabled=", capture)
        self.assertNotIn("MILVUS_PASSWORD=${", capture)
        self.assertNotIn("KNOWLEDGE_", github_output)
        self.assertNotIn("MILVUS_", github_output)
        self.assertIn('source "${ROLLBACK_KNOWLEDGE_CONFIG_FILE}"', rollback)
        self.assertNotIn("DEPLOY_MILVUS_PASSWORD", rollback)
        self.assertNotIn("DEPLOY_KNOWLEDGE_EMBEDDING_MODEL", rollback)
        self.assertIn('rm -f -- "${rollback_knowledge_config_file}"', cleanup)

    def test_rollback_snapshot_supports_old_image_without_knowledge_environment(self):
        completed, contents, mode, syntax = self._run_rollback_snapshot(
            [
                {
                    "Config": {"Env": ["PATH=/usr/local/bin"]},
                    "NetworkSettings": {
                        "Networks": {
                            "postgres_default": {},
                            "middleware_default": {},
                            "fusion-prompthub": {},
                            "fusion-flyai": {},
                        }
                    },
                }
            ]
        )

        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(syntax.stdout, "")
        self.assertEqual(syntax.stderr, "")
        self.assertEqual(mode, 0o600)
        self.assertIn("export KNOWLEDGE_BASE_ENABLED=false", contents)
        self.assertIn("export MILVUS_PASSWORD=''", contents)

    def test_rollback_snapshot_preserves_enabled_worker_secret_without_logging(self):
        secret = "旧密码含空格和'引号"
        knowledge_environment = [
            "KNOWLEDGE_BASE_ENABLED=true",
            "KNOWLEDGE_EMBEDDING_MODEL=text-embedding-v4",
            "KNOWLEDGE_EMBEDDING_REVISION=2026-08-12",
            "MILVUS_URI=http://standalone:19530",
            "MILVUS_USERNAME=fusion_app",
            f"MILVUS_PASSWORD={secret}",
            "MILVUS_DATABASE=fusion_knowledge",
        ]
        container = {
            "Config": {"Env": knowledge_environment},
            "NetworkSettings": {
                "Networks": {
                    "postgres_default": {},
                    "middleware_default": {},
                    "fusion_knowledge_milvus": {},
                }
            },
        }

        completed, contents, mode, syntax = self._run_rollback_snapshot([container, container])

        self.assertEqual(completed.stdout, "")
        self.assertEqual(completed.stderr, "")
        self.assertEqual(syntax.stdout, "")
        self.assertEqual(syntax.stderr, "")
        self.assertNotIn(secret, completed.stdout + completed.stderr)
        self.assertEqual(mode, 0o600)
        self.assertIn("export KNOWLEDGE_BASE_ENABLED=true", contents)
        self.assertIn("export MILVUS_PASSWORD=", contents)
        self.assertIn("fusion_knowledge_milvus", contents)


if __name__ == "__main__":
    unittest.main()
