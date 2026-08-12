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

    def _run_candidate_knowledge_validation(self, **overrides):
        marker = "validated_revision_routes=\"$(python3 - <<'PY'\n"
        script = self.workflow.split(marker, 1)[1].split("\n          PY\n", 1)[0]
        environment = {
            "KNOWLEDGE_BASE_ENABLED": "true",
            "KNOWLEDGE_MAX_FILE_SIZE": "10485760",
            "KNOWLEDGE_CHUNKER_VERSION": "chunker-v2",
            "KNOWLEDGE_CHUNK_SIZE": "1200",
            "KNOWLEDGE_CHUNK_OVERLAP": "200",
            "KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT": "10000",
            "KNOWLEDGE_EMBEDDING_BATCH_SIZE": "32",
            "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS": "30",
            "KNOWLEDGE_SEARCH_MAX_PROFILES": "8",
            "KNOWLEDGE_WORKER_POLL_SECONDS": "2",
            "KNOWLEDGE_WORKER_RETRY_BASE_SECONDS": "5",
            "KNOWLEDGE_WORKER_RETRY_MAX_SECONDS": "300",
            "MILVUS_TIMEOUT_SECONDS": "10",
            "KNOWLEDGE_EMBEDDING_MODEL": "embedding-v1",
            "KNOWLEDGE_EMBEDDING_REVISION": "embedding-r1",
            "KNOWLEDGE_EMBEDDING_REVISION_ROUTES": json.dumps(
                {"embedding-v1@embedding-r1": "embedding-v1-r1-immutable"}
            ),
            "CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES": "{}",
        }
        environment.update(overrides)
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(script)],
            text=True,
            capture_output=True,
            env={**os.environ, **environment},
        )

    def _run_revision_routes_assignment(self, value: str | None) -> subprocess.CompletedProcess[str]:
        lines = self.workflow.splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if 'if [ -n "${DEPLOY_KNOWLEDGE_EMBEDDING_REVISION_ROUTES:-}" ]; then' in line
        )
        end = next(index for index in range(start, len(lines)) if lines[index].strip() == "fi")
        script = textwrap.dedent("\n".join(lines[start : end + 1]))
        script += "\nprintf '%s' \"${KNOWLEDGE_EMBEDDING_REVISION_ROUTES}\"\n"
        environment = dict(os.environ)
        if value is None:
            environment.pop("DEPLOY_KNOWLEDGE_EMBEDDING_REVISION_ROUTES", None)
        else:
            environment["DEPLOY_KNOWLEDGE_EMBEDDING_REVISION_ROUTES"] = value
        return subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            env=environment,
        )

    def _run_chunker_version_assignment(
        self,
        *,
        deploy_value: str | None,
        inherited_value: str | None,
    ) -> subprocess.CompletedProcess[str]:
        line = next(
            line
            for line in self.workflow.splitlines()
            if line.strip().startswith('export KNOWLEDGE_CHUNKER_VERSION="${DEPLOY_')
        )
        script = textwrap.dedent(line)
        script += "\nprintf '%s' \"${KNOWLEDGE_CHUNKER_VERSION}\"\n"
        environment = dict(os.environ)
        for name, value in (
            ("DEPLOY_KNOWLEDGE_CHUNKER_VERSION", deploy_value),
            ("KNOWLEDGE_CHUNKER_VERSION", inherited_value),
        ):
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        return subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            env=environment,
        )

    def _run_current_revision_routes_from_snapshot(self, contents: str) -> subprocess.CompletedProcess[str]:
        lines = self.workflow.splitlines()
        start = next(
            index for index, line in enumerate(lines) if line.strip() == "# BEGIN CURRENT_KNOWLEDGE_REVISION_ROUTES"
        )
        end = next(
            index
            for index in range(start + 1, len(lines))
            if lines[index].strip() == "# END CURRENT_KNOWLEDGE_REVISION_ROUTES"
        )
        script = textwrap.dedent("\n".join(lines[start + 1 : end]))
        script += "\nprintf '%s' \"${CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES}\"\n"
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                **os.environ,
                "RUNNER_TEMP": directory,
                "GITHUB_RUN_ID": "42",
                "GITHUB_JOB": "deploy",
                "GITHUB_RUN_ATTEMPT": "1",
            }
            snapshot = Path(directory) / "fusion-knowledge-rollback-42-deploy-1.env"
            snapshot.write_text(contents, encoding="utf-8")
            snapshot.chmod(0o600)
            return subprocess.run(
                ["bash", "-c", script],
                text=True,
                capture_output=True,
                env=environment,
            )

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
            "DEPLOY_KNOWLEDGE_EMBEDDING_REVISION_ROUTES: ${{ vars.KNOWLEDGE_EMBEDDING_REVISION_ROUTES }}",
            "DEPLOY_KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS: ${{ vars.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS || '30' }}",
            "DEPLOY_KNOWLEDGE_MAX_FILE_SIZE: ${{ vars.KNOWLEDGE_MAX_FILE_SIZE || '10485760' }}",
            "DEPLOY_KNOWLEDGE_CHUNKER_VERSION: ${{ vars.KNOWLEDGE_CHUNKER_VERSION || 'chunker-v2' }}",
            "DEPLOY_KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT: ${{ vars.KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT || '10000' }}",
            "DEPLOY_KNOWLEDGE_SEARCH_MAX_PROFILES: ${{ vars.KNOWLEDGE_SEARCH_MAX_PROFILES || '8' }}",
            "DEPLOY_KNOWLEDGE_WORKER_POLL_SECONDS: ${{ vars.KNOWLEDGE_WORKER_POLL_SECONDS || '2' }}",
            "DEPLOY_MILVUS_TIMEOUT_SECONDS: ${{ vars.MILVUS_TIMEOUT_SECONDS || '10' }}",
            "knowledge deployment bounds invalid",
            "knowledge chunker version invalid",
            "knowledge embedding revision route missing",
            "knowledge embedding revision routes history changed",
        )
        for marker in required:
            self.assertIn(marker, self.workflow)
        self.assertNotIn("MILVUS_BOOTSTRAP_PASSWORD", self.workflow)

    def test_expand_only_migration_runs_before_cleanup_capable_worker_restart(self):
        migration_step = self.workflow.index("- name: Apply alembic migrations")
        deploy_step = self.workflow.index("- name: Pull and restart fusion-api")
        worker_restart = self.workflow.index("--profile knowledge-worker up -d")

        self.assertLess(migration_step, deploy_step)
        self.assertLess(deploy_step, worker_restart)
        self.assertIn("alembic upgrade head", self.workflow[migration_step:deploy_step])

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
            "KNOWLEDGE_CHUNKER_VERSION",
            "KNOWLEDGE_EMBEDDING_REVISION_ROUTES",
            "KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS",
            "KNOWLEDGE_EMBEDDING_BATCH_SIZE",
            "KNOWLEDGE_SEARCH_MAX_PROFILES",
            "KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT",
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
        self.assertNotIn("knowledge embedding revision routes history changed", rollback)
        self.assertNotIn("knowledge chunker version invalid", rollback)
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
        self.assertIn("export KNOWLEDGE_CHUNKER_VERSION=chunker-v1", contents)
        self.assertIn("export MILVUS_PASSWORD=''", contents)

    def test_rollback_snapshot_preserves_enabled_worker_secret_without_logging(self):
        secret = "旧密码含空格和'引号"
        knowledge_environment = [
            "KNOWLEDGE_BASE_ENABLED=true",
            "KNOWLEDGE_EMBEDDING_MODEL=text-embedding-v4",
            "KNOWLEDGE_EMBEDDING_REVISION=2026-08-12",
            "KNOWLEDGE_CHUNKER_VERSION=chunker-v2",
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
        self.assertIn("export KNOWLEDGE_CHUNKER_VERSION=chunker-v2", contents)
        self.assertIn("export MILVUS_PASSWORD=", contents)
        self.assertIn("fusion_knowledge_milvus", contents)

    def test_current_registry_bash_read_supports_pre41_and_enabled_snapshots(self):
        _, old_contents, _, _ = self._run_rollback_snapshot(
            [
                {
                    "Config": {"Env": ["PATH=/usr/local/bin"]},
                    "NetworkSettings": {"Networks": {"postgres_default": {}}},
                }
            ]
        )
        configured_routes = json.dumps(
            {
                "embedding-v0@embedding-r0": "embedding-v0-r0-immutable",
                "embedding-v1@embedding-r1": "embedding-v1-r1-immutable",
            }
        )
        knowledge_environment = [
            "KNOWLEDGE_BASE_ENABLED=true",
            "KNOWLEDGE_EMBEDDING_MODEL=embedding-v1",
            "KNOWLEDGE_EMBEDDING_REVISION=embedding-r1",
            f"KNOWLEDGE_EMBEDDING_REVISION_ROUTES={configured_routes}",
            "MILVUS_URI=http://standalone:19530",
            "MILVUS_USERNAME=fusion_app",
            "MILVUS_PASSWORD=不能输出的旧密码",
            "MILVUS_DATABASE=fusion_knowledge",
        ]
        container = {
            "Config": {"Env": knowledge_environment},
            "NetworkSettings": {"Networks": {"fusion_knowledge_milvus": {}}},
        }
        _, configured_contents, _, _ = self._run_rollback_snapshot([container, container])

        old = self._run_current_revision_routes_from_snapshot(old_contents)
        configured = self._run_current_revision_routes_from_snapshot(configured_contents)

        self.assertEqual(old.returncode, 0, old.stderr)
        self.assertEqual(old.stdout, "{}")
        self.assertEqual(old.stderr, "")
        self.assertEqual(configured.returncode, 0, configured.stderr)
        self.assertEqual(json.loads(configured.stdout), json.loads(configured_routes))
        self.assertNotIn("不能输出的旧密码", configured.stdout + configured.stderr)
        pre41_candidate = self._run_candidate_knowledge_validation(
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES=old.stdout,
        )
        history_deletion = self._run_candidate_knowledge_validation(
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES=configured.stdout,
        )
        self.assertEqual(pre41_candidate.returncode, 0, pre41_candidate.stderr)
        self.assertNotEqual(history_deletion.returncode, 0)
        self.assertIn("knowledge embedding revision routes history changed", history_deletion.stderr)

    def test_candidate_deploy_validation_rejects_bounds_and_missing_revision_route(self):
        valid = self._run_candidate_knowledge_validation()
        invalid_size = self._run_candidate_knowledge_validation(KNOWLEDGE_MAX_FILE_SIZE="0")
        legacy_chunker = self._run_candidate_knowledge_validation(KNOWLEDGE_CHUNKER_VERSION="chunker-v1")
        invalid_chunks = self._run_candidate_knowledge_validation(KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT="10001")
        invalid_batch = self._run_candidate_knowledge_validation(KNOWLEDGE_EMBEDDING_BATCH_SIZE="129")
        invalid_profiles = self._run_candidate_knowledge_validation(KNOWLEDGE_SEARCH_MAX_PROFILES="17")
        invalid_poll = self._run_candidate_knowledge_validation(KNOWLEDGE_WORKER_POLL_SECONDS="61")
        invalid_retry_base = self._run_candidate_knowledge_validation(KNOWLEDGE_WORKER_RETRY_BASE_SECONDS="0")
        invalid_retry_order = self._run_candidate_knowledge_validation(
            KNOWLEDGE_WORKER_RETRY_BASE_SECONDS="301",
            KNOWLEDGE_WORKER_RETRY_MAX_SECONDS="300",
        )
        invalid_retry_max = self._run_candidate_knowledge_validation(KNOWLEDGE_WORKER_RETRY_MAX_SECONDS="3601")
        invalid_milvus_timeout = self._run_candidate_knowledge_validation(MILVUS_TIMEOUT_SECONDS="61")
        missing_route = self._run_candidate_knowledge_validation(KNOWLEDGE_EMBEDDING_REVISION_ROUTES="{}")
        missing_current_route = self._run_candidate_knowledge_validation(
            KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps({"embedding-v0@embedding-r0": "embedding-v0-r0-immutable"})
        )

        self.assertEqual(valid.returncode, 0)
        self.assertEqual(valid.stdout, '{"embedding-v1@embedding-r1":"embedding-v1-r1-immutable"}')
        self.assertNotEqual(legacy_chunker.returncode, 0)
        self.assertIn("knowledge chunker version invalid", legacy_chunker.stderr)
        for completed in (
            invalid_size,
            invalid_chunks,
            invalid_batch,
            invalid_profiles,
            invalid_poll,
            invalid_retry_base,
            invalid_retry_order,
            invalid_retry_max,
            invalid_milvus_timeout,
        ):
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("knowledge deployment bounds invalid", completed.stderr)
        self.assertNotEqual(missing_route.returncode, 0)
        self.assertIn("knowledge embedding revision routes invalid", missing_route.stderr)
        self.assertNotEqual(missing_current_route.returncode, 0)
        self.assertIn("knowledge embedding revision route missing", missing_current_route.stderr)

    def test_compose_injects_search_and_document_resource_limits_into_api_and_worker(self):
        for name in (
            "KNOWLEDGE_MAX_CHUNKS_PER_DOCUMENT",
            "KNOWLEDGE_SEARCH_MAX_PROFILES",
        ):
            self.assertEqual(self.workflow.count(f"- {name}=${{{name}}}"), 2)
            self.assertEqual(self.fusion_override.count(f"- {name}=${{{name}:-"), 2)

        self.assertEqual(self.workflow.count("- KNOWLEDGE_CHUNKER_VERSION=${KNOWLEDGE_CHUNKER_VERSION}"), 2)
        self.assertEqual(
            self.fusion_override.count("- KNOWLEDGE_CHUNKER_VERSION=${KNOWLEDGE_CHUNKER_VERSION:-chunker-v2}"),
            2,
        )

    def test_candidate_revision_registry_is_append_only_against_current_deployment(self):
        historical = {"embedding-v0@embedding-r0": "embedding-v0-r0-immutable"}
        appended = {
            **historical,
            "embedding-v1@embedding-r1": "embedding-v1-r1-immutable",
        }
        pre41 = self._run_candidate_knowledge_validation(
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES="{}",
        )
        append_only = self._run_candidate_knowledge_validation(
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(historical),
            KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(appended),
        )
        deleted = self._run_candidate_knowledge_validation(
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(historical),
        )
        rewritten = self._run_candidate_knowledge_validation(
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(historical),
            KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(
                {
                    "embedding-v0@embedding-r0": "embedding-v0-r0-rewritten",
                    "embedding-v1@embedding-r1": "embedding-v1-r1-immutable",
                }
            ),
        )
        disabled_deletion = self._run_candidate_knowledge_validation(
            KNOWLEDGE_BASE_ENABLED="false",
            CURRENT_KNOWLEDGE_EMBEDDING_REVISION_ROUTES=json.dumps(historical),
            KNOWLEDGE_EMBEDDING_REVISION_ROUTES="{}",
        )

        self.assertEqual(pre41.returncode, 0, pre41.stderr)
        self.assertEqual(append_only.returncode, 0, append_only.stderr)
        self.assertEqual(json.loads(append_only.stdout), appended)
        for completed in (deleted, rewritten, disabled_deletion):
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("knowledge embedding revision routes history changed", completed.stderr)

    def test_revision_routes_bash_assignment_preserves_json_and_defaults_empty_value(self):
        configured_json = '{"embedding-v1@embedding-r1":"embedding-v1-r1-immutable"}'
        configured = self._run_revision_routes_assignment(configured_json)
        empty = self._run_revision_routes_assignment("")
        missing = self._run_revision_routes_assignment(None)

        for completed in (configured, empty, missing):
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, "")
        self.assertEqual(configured.stdout, configured_json)
        self.assertEqual(empty.stdout, "{}")
        self.assertEqual(missing.stdout, "{}")

    def test_candidate_chunker_assignment_ignores_legacy_dotenv_value(self):
        inherited_legacy = self._run_chunker_version_assignment(
            deploy_value=None,
            inherited_value="chunker-v1",
        )
        configured = self._run_chunker_version_assignment(
            deploy_value="chunker-v2",
            inherited_value="chunker-v1",
        )

        for completed in (inherited_legacy, configured):
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(completed.stderr, "")
            self.assertEqual(completed.stdout, "chunker-v2")


if __name__ == "__main__":
    unittest.main()
