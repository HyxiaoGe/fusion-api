import asyncio
import os
import unittest
from datetime import UTC, datetime
from types import SimpleNamespace

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"

from app.schemas.mcp import McpServerCreate, McpServerStatusRequest, McpServerUpdate  # noqa: E402
from app.schemas.response import ApiException  # noqa: E402
from app.services.mcp.client import McpClientError  # noqa: E402
from app.services.mcp.server_service import McpServerService  # noqa: E402

NOW = datetime(2026, 7, 16, 8, 0, tzinfo=UTC)


class FakeRepository:
    def __init__(self, rows=None):
        self.rows = {row.id: row for row in rows or []}

    def list_all(self):
        return list(self.rows.values())

    def get(self, server_id):
        return self.rows.get(server_id)

    def get_by_name(self, name):
        return next((row for row in self.rows.values() if row.name == name), None)

    def create(self, values):
        row = SimpleNamespace(id="server-new", config_version=1, **values, created_at=NOW, updated_at=NOW)
        self.rows[row.id] = row
        return row

    def update(self, row, values):
        for key, value in values.items():
            setattr(row, key, value)
        row.config_version += 1
        row.updated_at = NOW
        return row

    def update_if_version(self, server_id, expected_version, values):
        row = self.rows.get(server_id)
        if row is None or row.config_version != expected_version:
            return None
        return self.update(row, values)


class FakeClientManager:
    def __init__(self, *, tools=None, test_error=None, refresh_error=None):
        self.tools = tools or []
        self.test_error = test_error
        self.refresh_error = refresh_error
        self.validated = []
        self.tested = []
        self.refreshed = []

    def validate_configuration(self, config):
        self.validated.append(config)

    async def test_connection(self, config):
        self.tested.append(config)
        if self.test_error:
            raise self.test_error

    async def list_tools(self, config):
        self.refreshed.append(config)
        if self.refresh_error:
            raise self.refresh_error
        return self.tools


def build_row(**overrides):
    values = {
        "id": "server-1",
        "name": "百炼搜索",
        "provider": "aliyun",
        "endpoint_url": "https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
        "transport": "streamable_http",
        "auth_type": "bearer",
        "auth_name": None,
        "credential_ref": "DASHSCOPE_API_KEY",
        "config_version": 1,
        "is_enabled": False,
        "allowed_tools": ["search"],
        "discovered_tools": [
            {"name": "search", "description": "搜索", "input_schema": {"type": "object"}},
            {"name": "read", "description": "读取", "input_schema": {"type": "object"}},
        ],
        "health_status": "disabled",
        "last_checked_at": None,
        "last_error_code": None,
        "last_error_message": None,
        "created_at": NOW,
        "updated_at": NOW,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class McpServerServiceTests(unittest.TestCase):
    def test_create_starts_disabled_and_rejects_allowed_tools_before_discovery(self):
        service = McpServerService(FakeRepository(), FakeClientManager(), clock=lambda: NOW)
        request = McpServerCreate(
            name="百炼搜索",
            provider="aliyun",
            endpoint_url="https://dashscope.aliyuncs.com/api/v1/mcps/WebSearch/mcp",
            transport="streamable_http",
            auth_type="bearer",
            credential_ref="DASHSCOPE_API_KEY",
            allowed_tools=[],
        )

        row = service.create_server(request)

        self.assertFalse(row.is_enabled)
        self.assertEqual(row.health_status, "disabled")
        self.assertEqual(row.discovered_tools, [])

        with self.assertRaises(ApiException) as raised:
            service.create_server(request.model_copy(update={"name": "非法预授权", "allowed_tools": ["search"]}))
        self.assertEqual(raised.exception.status_code, 400)

    def test_update_connection_identity_clears_stale_discovery_and_authorization(self):
        row = build_row(health_status="healthy", last_checked_at=NOW)
        service = McpServerService(FakeRepository([row]), FakeClientManager(), clock=lambda: NOW)

        updated = service.update_server(
            row.id,
            McpServerUpdate(endpoint_url="https://dashscope.aliyuncs.com/api/v1/mcps/Another/mcp"),
        )

        self.assertEqual(updated.discovered_tools, [])
        self.assertEqual(updated.allowed_tools, [])
        self.assertEqual(updated.health_status, "disabled")
        self.assertIsNone(updated.last_checked_at)
        self.assertIsNone(updated.last_error_code)

    def test_name_only_update_preserves_discovery_and_allowed_tools(self):
        row = build_row()
        service = McpServerService(FakeRepository([row]), FakeClientManager(), clock=lambda: NOW)

        updated = service.update_server(row.id, McpServerUpdate(name="新名称"))

        self.assertEqual(updated.allowed_tools, ["search"])
        self.assertEqual(len(updated.discovered_tools), 2)

    def test_patch_can_explicitly_clear_auth_fields_when_switching_to_none(self):
        row = build_row(auth_type="header", auth_name="X-API-Key")
        service = McpServerService(FakeRepository([row]), FakeClientManager(), clock=lambda: NOW)

        updated = service.update_server(
            row.id,
            McpServerUpdate(auth_type="none", auth_name=None, credential_ref=None),
        )

        self.assertEqual(updated.auth_type, "none")
        self.assertIsNone(updated.auth_name)
        self.assertIsNone(updated.credential_ref)
        self.assertEqual(updated.allowed_tools, [])
        self.assertEqual(updated.discovered_tools, [])

    def test_allowed_tools_must_be_discovered_subset(self):
        row = build_row()
        service = McpServerService(FakeRepository([row]), FakeClientManager(), clock=lambda: NOW)

        with self.assertRaises(ApiException) as raised:
            service.update_server(row.id, McpServerUpdate(allowed_tools=["unknown"]))

        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("已发现工具", raised.exception.message)

    def test_refresh_saves_normalized_snapshot_and_intersects_existing_allowlist(self):
        row = build_row(allowed_tools=["search", "removed"])
        tools = [
            {"name": "search", "description": "搜索", "input_schema": {"type": "object"}},
            {"name": "new", "description": None, "input_schema": {"type": "object"}},
        ]
        service = McpServerService(FakeRepository([row]), FakeClientManager(tools=tools), clock=lambda: NOW)

        refreshed = asyncio.run(service.refresh_tools(row.id))

        self.assertEqual(refreshed.discovered_tools, tools)
        self.assertEqual(refreshed.allowed_tools, ["search"])
        self.assertEqual(refreshed.health_status, "healthy")
        self.assertEqual(refreshed.last_checked_at, NOW)
        self.assertIsNone(refreshed.last_error_message)

    def test_remote_test_failure_is_persisted_as_unhealthy_without_raw_secret(self):
        secret = "test-secret"
        row = build_row()
        client = FakeClientManager(test_error=McpClientError("auth_failed", "MCP 服务鉴权失败"))
        service = McpServerService(FakeRepository([row]), client, clock=lambda: NOW)

        tested = asyncio.run(service.test_server(row.id))

        self.assertEqual(tested.health_status, "unhealthy")
        self.assertEqual(tested.last_error_code, "auth_failed")
        self.assertEqual(tested.last_error_message, "MCP 服务鉴权失败")
        self.assertNotIn(secret, tested.last_error_message)

    def test_remote_refresh_failure_returns_persisted_unhealthy_snapshot(self):
        row = build_row(health_status="healthy")
        client = FakeClientManager(refresh_error=McpClientError("connect_timeout", "连接 MCP 服务超时"))
        service = McpServerService(FakeRepository([row]), client, clock=lambda: NOW)

        refreshed = asyncio.run(service.refresh_tools(row.id))

        self.assertEqual(refreshed.health_status, "unhealthy")
        self.assertEqual(refreshed.last_checked_at, NOW)
        self.assertEqual(refreshed.last_error_code, "connect_timeout")
        self.assertEqual(refreshed.last_error_message, "连接 MCP 服务超时")
        self.assertEqual(len(refreshed.discovered_tools), 2)

    def test_status_endpoint_state_transitions_are_explicit(self):
        row = build_row(is_enabled=False, health_status="disabled")
        service = McpServerService(FakeRepository([row]), FakeClientManager(), clock=lambda: NOW)

        enabled = service.set_status(row.id, McpServerStatusRequest(is_enabled=True))
        self.assertTrue(enabled.is_enabled)
        self.assertEqual(enabled.health_status, "unknown")

        disabled = service.set_status(row.id, McpServerStatusRequest(is_enabled=False))
        self.assertFalse(disabled.is_enabled)
        self.assertEqual(disabled.health_status, "disabled")


if __name__ == "__main__":
    unittest.main()
