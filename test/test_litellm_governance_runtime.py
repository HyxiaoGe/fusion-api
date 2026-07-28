import unittest
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


if __name__ == "__main__":
    unittest.main()
