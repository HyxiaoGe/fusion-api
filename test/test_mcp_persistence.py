import os
import unittest
from importlib.metadata import version

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ["DATABASE_URL"] = "sqlite:///./fusion-test.db"

from app.db.mcp_server_repository import McpServerRepository  # noqa: E402
from app.db.models import McpServer  # noqa: E402


class McpPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        McpServer.__table__.create(self.engine)
        self.db = sessionmaker(bind=self.engine)()
        self.repository = McpServerRepository(self.db)

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def test_repository_persists_reference_and_snapshots_without_secret_value(self):
        row = self.repository.create(
            {
                "name": "百炼搜索",
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
        )

        self.assertEqual(self.repository.get(row.id).credential_ref, "DASHSCOPE_API_KEY")
        self.assertFalse(hasattr(row, "credential"))
        self.assertFalse(hasattr(row, "secret"))

        updated = self.repository.update(
            row,
            {
                "discovered_tools": [{"name": "search", "description": "搜索", "input_schema": {"type": "object"}}],
                "allowed_tools": ["search"],
                "health_status": "healthy",
            },
        )
        self.assertEqual(updated.allowed_tools, ["search"])
        self.assertEqual(updated.config_version, 2)
        self.assertEqual(self.repository.list_all()[0].health_status, "healthy")

        stale_update = self.repository.update_if_version(
            row.id,
            expected_version=1,
            values={"health_status": "unhealthy"},
        )
        self.assertIsNone(stale_update)
        self.assertEqual(self.repository.get(row.id).health_status, "healthy")

    def test_runtime_dependency_versions_are_exactly_compatible_pins(self):
        self.assertEqual(version("mcp"), "1.28.1")
        self.assertEqual(version("sse-starlette"), "2.4.1")


if __name__ == "__main__":
    unittest.main()
