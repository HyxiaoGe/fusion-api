import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import check_litellm_governance_runtime as checker


class LiteLLMGovernanceRuntimeTests(unittest.TestCase):
    def test_current_test_runtime_satisfies_contract(self):
        status = checker.runtime_status()

        self.assertTrue(status["healthy"])
        self.assertEqual(status["httpx"], "0.28.1")

    def test_missing_httpx_fails_closed(self):
        with patch.object(checker.importlib, "import_module", side_effect=ImportError):
            status = checker.runtime_status()

        self.assertFalse(status["healthy"])
        self.assertIn("httpx_missing", status["issues"])

    def test_secret_files_must_be_owned_regular_and_private(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            secret = Path(temp_dir) / "governance.env"
            secret.write_text("KEY=value\n", encoding="utf-8")
            secret.chmod(0o600)

            healthy = checker.runtime_status(secret_files=[secret])
            secret.chmod(0o640)
            too_open = checker.runtime_status(secret_files=[secret])

        self.assertTrue(healthy["healthy"])
        self.assertEqual(healthy["secret_files_checked"], 1)
        self.assertIn("secret_file_permissions_too_open", too_open["issues"])

    def test_missing_and_symlink_secret_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "target"
            target.write_text("KEY=value\n", encoding="utf-8")
            target.chmod(0o600)
            link = root / "link"
            os.symlink(target, link)

            status = checker.runtime_status(secret_files=[root / "missing", link])

        self.assertFalse(status["healthy"])
        self.assertIn("secret_file_missing", status["issues"])
        self.assertIn("secret_file_not_regular", status["issues"])


if __name__ == "__main__":
    unittest.main()
