import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import run_litellm_governance_unit as runner


class LiteLLMGovernanceUnitRunnerTests(unittest.TestCase):
    def _files(self, root: Path) -> tuple[Path, Path, Path]:
        proxy_env = root / "proxy.env"
        proxy_env.write_text(
            "\n".join(
                [
                    "LITELLM_MASTER_KEY=master",
                    "MOONSHOT_API_KEY=moonshot",
                    "PYTHONPATH=/tmp/attacker",
                    "LD_PRELOAD=/tmp/attacker.so",
                ]
            ),
            encoding="utf-8",
        )
        governance_env = root / "governance.env"
        governance_env.write_text(
            "\n".join(
                [
                    "LITELLM_BASE_URL=http://127.0.0.1:4000",
                    "LITELLM_CANDIDATE_KEY=candidate",
                    "LITELLM_VIRTUAL_KEY=virtual",
                    "FUSION_MODEL_MANAGEMENT_BASE_URL=http://127.0.0.1:8002",
                    "LITELLM_MODEL_ADMISSION_WORKER_TOKEN=worker-token",
                    "LITELLM_GOVERNANCE_MAX_AGE_SECONDS=7200",
                ]
            ),
            encoding="utf-8",
        )
        registry = root / "registry.json"
        registry.write_text(
            json.dumps(
                {
                    "providers": [
                        {"api_key_env": "MOONSHOT_API_KEY"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        for path in (proxy_env, governance_env, registry):
            path.chmod(0o600)
        return proxy_env, governance_env, registry

    def test_build_environment_only_loads_allowlisted_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proxy_env, governance_env, registry = self._files(Path(temp_dir))

            execute_env, issues, registry_text = runner.build_exec_environment(
                proxy_env=proxy_env,
                governance_env=governance_env,
                registry=registry,
                required_env_names=[
                    "LITELLM_MASTER_KEY",
                    "LITELLM_CANDIDATE_KEY",
                    "LITELLM_VIRTUAL_KEY",
                    "LITELLM_MODEL_ADMISSION_WORKER_TOKEN",
                    "LITELLM_GOVERNANCE_MAX_AGE_SECONDS",
                    "FUSION_MODEL_MANAGEMENT_BASE_URL",
                ],
                inherited={
                    "HOME": "/home/test",
                    "PATH": "/usr/bin:/bin",
                    "PYTHONPATH": "/manager/attacker",
                    "LD_PRELOAD": "/manager/attacker.so",
                },
            )

        self.assertEqual(issues, [])
        self.assertIsNotNone(registry_text)
        self.assertEqual(execute_env["LITELLM_MASTER_KEY"], "master")
        self.assertEqual(execute_env["MOONSHOT_API_KEY"], "moonshot")
        self.assertEqual(execute_env["LITELLM_CANDIDATE_KEY"], "candidate")
        self.assertEqual(execute_env["LITELLM_VIRTUAL_KEY"], "virtual")
        self.assertEqual(
            execute_env["FUSION_MODEL_MANAGEMENT_BASE_URL"],
            "http://127.0.0.1:8002",
        )
        self.assertEqual(execute_env["LITELLM_MODEL_ADMISSION_WORKER_TOKEN"], "worker-token")
        self.assertEqual(execute_env["LITELLM_GOVERNANCE_MAX_AGE_SECONDS"], "7200")
        self.assertNotIn("PYTHONPATH", execute_env)
        self.assertNotIn("LD_PRELOAD", execute_env)
        self.assertEqual(execute_env["PYTHONNOUSERSITE"], "1")

    def test_read_only_governance_process_does_not_receive_worker_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proxy_env, governance_env, registry = self._files(Path(temp_dir))

            execute_env, issues, _ = runner.build_exec_environment(
                proxy_env=proxy_env,
                governance_env=governance_env,
                registry=registry,
                required_env_names=["LITELLM_MASTER_KEY", "LITELLM_CANDIDATE_KEY"],
                inherited={"HOME": "/home/test", "PATH": "/usr/bin:/bin"},
            )

        self.assertEqual(issues, [])
        self.assertNotIn("LITELLM_VIRTUAL_KEY", execute_env)
        self.assertNotIn("LITELLM_MODEL_ADMISSION_WORKER_TOKEN", execute_env)
        self.assertNotIn("LITELLM_GOVERNANCE_MAX_AGE_SECONDS", execute_env)
        self.assertNotIn("FUSION_MODEL_MANAGEMENT_BASE_URL", execute_env)

    def test_required_environment_cannot_expand_managed_allowlist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proxy_env, governance_env, registry = self._files(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "未受管变量"):
                runner.build_exec_environment(
                    proxy_env=proxy_env,
                    governance_env=governance_env,
                    registry=registry,
                    required_env_names=["PYTHONPATH"],
                    inherited={},
                )

    def test_governance_env_cannot_duplicate_upstream_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proxy_env, governance_env, registry = self._files(root)
            governance_env.write_text(
                "LITELLM_MASTER_KEY=shadow\nLITELLM_CANDIDATE_KEY=candidate\n",
                encoding="utf-8",
            )

            _, issues, _ = runner.build_exec_environment(
                proxy_env=proxy_env,
                governance_env=governance_env,
                registry=registry,
                required_env_names=[],
                inherited={},
            )

        self.assertIn(
            "governance_env_duplicates_upstream_credentials",
            issues,
        )

    def test_registry_cannot_turn_dangerous_environment_into_provider_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proxy_env, governance_env, registry = self._files(root)
            registry.write_text(
                json.dumps({"providers": [{"api_key_env": "PYTHONPATH"}]}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "api_key_env"):
                runner.build_exec_environment(
                    proxy_env=proxy_env,
                    governance_env=governance_env,
                    registry=registry,
                    required_env_names=[],
                    inherited={},
                )

    def test_secure_reader_rejects_symlink_and_open_permissions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            proxy_env, governance_env, registry = self._files(root)
            link = root / "proxy-link.env"
            os.symlink(proxy_env, link)

            with self.assertRaises(OSError):
                runner.build_exec_environment(
                    proxy_env=link,
                    governance_env=governance_env,
                    registry=registry,
                    required_env_names=[],
                    inherited={},
                )

            proxy_env.chmod(0o640)
            with self.assertRaisesRegex(ValueError, "权限"):
                runner.build_exec_environment(
                    proxy_env=proxy_env,
                    governance_env=governance_env,
                    registry=registry,
                    required_env_names=[],
                    inherited={},
                )

    def test_main_checks_files_before_exec_and_passes_minimal_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            proxy_env, governance_env, _ = self._files(Path(temp_dir))
            with (
                patch.object(
                    runner,
                    "runtime_status",
                    return_value={"healthy": True, "issues": []},
                ),
                patch.object(os, "execve", side_effect=RuntimeError("executed")) as execve,
            ):
                exit_code = runner.main(
                    [
                        "--proxy-env",
                        str(proxy_env),
                        "--governance-env",
                        str(governance_env),
                        "--require-env",
                        "LITELLM_MASTER_KEY",
                        "--",
                        "/usr/bin/true",
                    ]
                )

        command, args, execute_env = execve.call_args.args
        self.assertEqual(exit_code, 1)
        self.assertEqual(command, "/usr/bin/true")
        self.assertEqual(args, ["/usr/bin/true"])
        self.assertNotIn("PYTHONPATH", execute_env)

    @unittest.skipUnless(
        hasattr(os, "memfd_create") and hasattr(runner.fcntl, "F_ADD_SEALS"),
        "需要 Linux sealed memfd",
    )
    def test_registry_snapshot_is_repeatably_readable_and_not_the_source_path(self):
        descriptor, path = runner._materialize_registry_snapshot('{"providers":[]}')
        try:
            self.assertEqual(Path(path).read_text(encoding="utf-8"), '{"providers":[]}')
            self.assertEqual(Path(path).read_text(encoding="utf-8"), '{"providers":[]}')
            self.assertTrue(os.get_inheritable(descriptor))
            with self.assertRaises(OSError):
                os.write(descriptor, b"tamper")
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    unittest.main()
