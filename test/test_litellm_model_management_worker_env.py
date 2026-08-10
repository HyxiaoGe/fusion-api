import hashlib
import os
import subprocess
import sys
import unittest
from pathlib import Path

from scripts.check_litellm_model_management_worker_env import validate_worker_environment

ROOT = Path(__file__).resolve().parents[1]


class ModelManagementWorkerEnvTests(unittest.TestCase):
    def test_validates_api_and_worker_share_the_same_token(self):
        token = "worker-secret"

        issues = validate_worker_environment(
            {
                "FUSION_MODEL_MANAGEMENT_BASE_URL": "http://127.0.0.1:8002",
                "LITELLM_MODEL_ADMISSION_WORKER_TOKEN": token,
                "LITELLM_VIRTUAL_KEY": "virtual-key",
                "LITELLM_GOVERNANCE_MAX_AGE_SECONDS": "7200",
            },
            expected_token_sha256=hashlib.sha256(token.encode()).hexdigest(),
            expected_base_url="http://127.0.0.1:8002",
            expected_governance_max_age_seconds=7200,
        )

        self.assertEqual(issues, [])

    def test_rejects_mismatched_token_without_printing_secret(self):
        token = "worker-secret"
        environment = os.environ.copy()
        environment.update(
            {
                "FUSION_MODEL_MANAGEMENT_BASE_URL": "http://127.0.0.1:8002",
                "LITELLM_MODEL_ADMISSION_WORKER_TOKEN": token,
                "LITELLM_VIRTUAL_KEY": "virtual-key",
                "LITELLM_GOVERNANCE_MAX_AGE_SECONDS": "7200",
            }
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.check_litellm_model_management_worker_env",
                "--expected-token-sha256",
                "0" * 64,
                "--expected-base-url",
                "http://127.0.0.1:8002",
                "--expected-governance-max-age-seconds",
                "7200",
            ],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("worker_token_mismatch", result.stdout)
        self.assertNotIn(token, result.stdout)


if __name__ == "__main__":
    unittest.main()
