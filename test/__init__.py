"""测试 package 入口 — 统一设 env fallback。

unittest discover 加载 test/ 下任何模块前都会先 import 本文件，
这里 setdefault 所有 app/core/config.py::Settings 必填字段，让单个 test 文件
不用关心 env 设置（Docker 容器跑 prod 时 env 已经齐全，setdefault 不会覆盖）。

历史教训：之前没有这层兜底时，按 alphabetical 顺序最早加载的 test 文件
（services.agent.test_session_cache、test_agent_logger）会触发 Settings()
在 DATABASE_URL=None 下 pydantic ValidationError，导致整个文件 _FailedTest。
"""
import os

os.environ.setdefault("DATABASE_URL", "sqlite:///./fusion-test.db")
os.environ.setdefault("SERVER_HOST", "http://test.local:8002")
os.environ.setdefault("FRONTEND_URL", "http://test.local:3000")
os.environ.setdefault("AUTH_SERVICE_BASE_URL", "http://auth.test:8100")
os.environ.setdefault("AUTH_SERVICE_CLIENT_ID", "test-client")
os.environ.setdefault("AUTH_SERVICE_JWKS_URL", "http://auth.test:8100/.well-known/jwks.json")
