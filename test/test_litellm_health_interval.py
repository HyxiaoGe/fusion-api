"""litellm_health 探测间隔解析测试。

/health 会对 LiteLLM DB 里每个模型各打一次真实 completion，其中 qwen 等 reasoning
模型每次生成数百 reasoning token（enable_thinking/max_tokens 都压不掉），频繁探测
在服务商侧产生真实费用。默认间隔应放宽到 30min，并支持 env 即时覆盖。
详见 https://github.com/HyxiaoGe/fusion-api/issues/10
"""

import os
import unittest

from app.ai import litellm_health


class ResolveRefreshIntervalTests(unittest.TestCase):
    def setUp(self):
        self._saved = os.environ.pop("LITELLM_HEALTH_INTERVAL_SECONDS", None)

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("LITELLM_HEALTH_INTERVAL_SECONDS", None)
        else:
            os.environ["LITELLM_HEALTH_INTERVAL_SECONDS"] = self._saved

    def test_default_refresh_interval_is_30min(self):
        """未设 env 时默认 1800s（30min），而不是过去的 300s。"""
        os.environ.pop("LITELLM_HEALTH_INTERVAL_SECONDS", None)
        self.assertEqual(litellm_health._resolve_refresh_interval(), 1800.0)

    def test_refresh_interval_env_override(self):
        """运维可用 env 即时覆盖（改 env + 重启即可，无需改代码）。"""
        os.environ["LITELLM_HEALTH_INTERVAL_SECONDS"] = "600"
        self.assertEqual(litellm_health._resolve_refresh_interval(), 600.0)


if __name__ == "__main__":
    unittest.main()
