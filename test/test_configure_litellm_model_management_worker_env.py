import os
import stat
import tempfile
import unittest
from pathlib import Path

from scripts.configure_litellm_model_management_worker_env import update_worker_environment


class ConfigureModelManagementWorkerEnvTests(unittest.TestCase):
    def test_updates_only_managed_values_atomically_and_keeps_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "litellm-governance.env"
            path.write_text(
                "# 治理配置\nLITELLM_BASE_URL=http://127.0.0.1:4000\nLITELLM_VIRTUAL_KEY=old\n",
                encoding="utf-8",
            )
            path.chmod(0o600)

            updated = update_worker_environment(
                path,
                virtual_key="virtual-key",
                worker_token="worker-token",
                fusion_base_url="http://127.0.0.1:8002",
                governance_max_age_seconds=7200,
            )

            self.assertEqual(
                updated,
                {
                    "FUSION_MODEL_MANAGEMENT_BASE_URL",
                    "LITELLM_GOVERNANCE_MAX_AGE_SECONDS",
                    "LITELLM_MODEL_ADMISSION_WORKER_TOKEN",
                    "LITELLM_VIRTUAL_KEY",
                },
            )
            content = path.read_text(encoding="utf-8")
            self.assertIn("# 治理配置", content)
            self.assertIn("LITELLM_BASE_URL=http://127.0.0.1:4000", content)
            self.assertEqual(content.count("LITELLM_VIRTUAL_KEY="), 1)
            self.assertIn("LITELLM_VIRTUAL_KEY=virtual-key", content)
            self.assertIn("LITELLM_MODEL_ADMISSION_WORKER_TOKEN=worker-token", content)
            self.assertIn("FUSION_MODEL_MANAGEMENT_BASE_URL=http://127.0.0.1:8002", content)
            self.assertIn("LITELLM_GOVERNANCE_MAX_AGE_SECONDS=7200", content)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_rejects_insecure_or_symlink_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "litellm-governance.env"
            path.write_text("LITELLM_BASE_URL=http://127.0.0.1:4000\n", encoding="utf-8")
            path.chmod(0o644)

            with self.assertRaises(ValueError):
                update_worker_environment(
                    path,
                    virtual_key="virtual-key",
                    worker_token="worker-token",
                    fusion_base_url="http://127.0.0.1:8002",
                    governance_max_age_seconds=7200,
                )

            path.chmod(0o600)
            symlink = root / "linked.env"
            os.symlink(path, symlink)
            with self.assertRaises(OSError):
                update_worker_environment(
                    symlink,
                    virtual_key="virtual-key",
                    worker_token="worker-token",
                    fusion_base_url="http://127.0.0.1:8002",
                    governance_max_age_seconds=7200,
                )


if __name__ == "__main__":
    unittest.main()
