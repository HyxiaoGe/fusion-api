import asyncio
import os
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"

from app.db.mcp_server_repository import McpServerRepository  # noqa: E402
from app.db.models import McpServer  # noqa: E402
from app.schemas.mcp import McpServerStatusRequest, McpServerUpdate  # noqa: E402
from app.services.mcp.server_service import McpServerService  # noqa: E402

NOW = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


class CallbackClientManager:
    def __init__(self, *, test_callback=None, refresh_callback=None):
        self.test_callback = test_callback
        self.refresh_callback = refresh_callback

    def validate_configuration(self, _config):
        return None

    async def test_connection(self, _config):
        if self.test_callback:
            self.test_callback()

    async def list_tools(self, _config):
        if self.refresh_callback:
            self.refresh_callback()
        return [{"name": "search", "description": "搜索", "input_schema": {"type": "object"}}]


class McpConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "mcp-concurrency.db"
        self.engine = create_engine(f"sqlite:///{database_path}")
        McpServer.__table__.create(self.engine)
        session_factory = sessionmaker(bind=self.engine)
        self.db1 = session_factory()
        self.db2 = session_factory()
        self.repo1 = McpServerRepository(self.db1)
        self.repo2 = McpServerRepository(self.db2)

    def tearDown(self):
        self.db1.close()
        self.db2.close()
        self.engine.dispose()
        self.temp_dir.cleanup()

    def _create_server(self, **overrides):
        values = {
            "name": "并发测试",
            "provider": "aliyun",
            "endpoint_url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            "transport": "streamable_http",
            "auth_type": "bearer",
            "auth_name": None,
            "credential_ref": "DASHSCOPE_API_KEY",
            "is_enabled": False,
            "allowed_tools": [],
            "discovered_tools": [],
            "health_status": "disabled",
        }
        values.update(overrides)
        return self.repo1.create(values)

    def test_stale_test_result_cannot_overwrite_concurrent_endpoint_change(self):
        row = self._create_server()
        manager2 = CallbackClientManager()
        service2 = McpServerService(self.repo2, manager2, clock=lambda: NOW)

        def change_endpoint():
            service2.update_server(
                row.id,
                McpServerUpdate(endpoint_url="https://dashscope.aliyuncs.com/api/v1/mcps/Another/mcp"),
            )

        service1 = McpServerService(
            self.repo1,
            CallbackClientManager(test_callback=change_endpoint),
            clock=lambda: NOW,
        )

        result = asyncio.run(service1.test_server(row.id))

        self.assertEqual(result.endpoint_url, "https://dashscope.aliyuncs.com/api/v1/mcps/Another/mcp")
        self.assertEqual(result.health_status, "disabled")
        self.assertEqual(result.config_version, 2)
        self.assertIsNone(result.last_checked_at)

    def test_stale_refresh_result_cannot_reenable_concurrently_disabled_server(self):
        row = self._create_server(is_enabled=True, health_status="unknown")
        service2 = McpServerService(self.repo2, CallbackClientManager(), clock=lambda: NOW)

        def disable_server():
            service2.set_status(row.id, McpServerStatusRequest(is_enabled=False))

        service1 = McpServerService(
            self.repo1,
            CallbackClientManager(refresh_callback=disable_server),
            clock=lambda: NOW,
        )

        result = asyncio.run(service1.refresh_tools(row.id))

        self.assertFalse(result.is_enabled)
        self.assertEqual(result.health_status, "disabled")
        self.assertEqual(result.config_version, 2)
        self.assertEqual(result.discovered_tools, [])
        self.assertIsNone(result.last_checked_at)


if __name__ == "__main__":
    unittest.main()
