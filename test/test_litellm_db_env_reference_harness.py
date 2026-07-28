import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_litellm_db_env_reference.sh"


class LiteLLMDbEnvReferenceHarnessTests(unittest.TestCase):
    def test_harness_is_isolated_fixed_digest_and_uses_only_fake_provider_key(self):
        content = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("v1.93.0@sha256:", content)
        self.assertIn("fake-provider-secret", content)
        self.assertIn("env-reference-ok", content)
        self.assertIn("docker network create", content)
        self.assertIn("docker rm -f", content)
        self.assertNotIn("MOONSHOT_API_KEY", content)
        self.assertNotIn("DEEPSEEK_API_KEY", content)
        self.assertNotIn("--network host", content)

    def test_harness_restarts_proxy_before_completion(self):
        content = SCRIPT.read_text(encoding="utf-8")
        register_index = content.index("http://litellm:4000/model/new")
        restart_index = content.index('docker rm -f "$PROXY"', register_index)
        completion_index = content.index("http://litellm:4000/chat/completions")

        self.assertLess(register_index, restart_index)
        self.assertLess(restart_index, completion_index)


if __name__ == "__main__":
    unittest.main()
