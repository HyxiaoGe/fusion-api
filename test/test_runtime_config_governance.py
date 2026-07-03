import unittest
from types import SimpleNamespace


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def order_by(self, *args, **kwargs):
        return self

    def filter(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return self.rows

    def first(self):
        return self.rows[0] if self.rows else None


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False
        self.committed = False
        self.refreshed = None
        self.added = None

    def query(self, model):
        return _FakeQuery(self.rows)

    def add(self, row):
        self.added = row
        self.rows.append(row)

    def commit(self):
        self.committed = True

    def refresh(self, row):
        self.refreshed = row

    def close(self):
        self.closed = True


class RuntimeConfigGovernanceTests(unittest.TestCase):
    def test_build_runtime_config_snapshot_reports_entries_and_effective_versions(self):
        from app.services.runtime_config_governance import build_runtime_config_snapshot

        rows = [
            SimpleNamespace(
                id="bad-row",
                namespace="prompt_template",
                key="generate_title",
                version="2026-07-03.bad",
                payload={"template": ""},
                is_active=True,
                description="坏配置",
                created_at=None,
                updated_at=None,
            ),
            SimpleNamespace(
                id="good-row",
                namespace="prompt_template",
                key="generate_title",
                version="2026-07-02.good",
                payload={"template": "有效模板"},
                is_active=True,
                description="好配置",
                created_at=None,
                updated_at=None,
            ),
        ]
        session = _FakeSession(rows)

        snapshot = build_runtime_config_snapshot(session_factory=lambda: session)

        self.assertTrue(session.closed)
        self.assertEqual(len(snapshot["entries"]), 2)
        self.assertFalse(snapshot["entries"][0]["valid"])
        self.assertEqual(snapshot["entries"][0]["issues"], ["template 必须是非空字符串"])
        effective = {
            (item["namespace"], item["key"]): item
            for item in snapshot["effective"]
            if (item["namespace"], item["key"]) == ("prompt_template", "generate_title")
        }
        self.assertEqual(effective[("prompt_template", "generate_title")]["version"], "2026-07-02.good")
        self.assertEqual(effective[("prompt_template", "generate_title")]["source"], "db")
        self.assertEqual(effective[("prompt_template", "generate_title")]["skipped_versions"], ["2026-07-03.bad"])

    def test_set_runtime_config_entry_active_updates_row_and_clears_cache(self):
        from app.services.runtime_config_governance import set_runtime_config_entry_active

        row = SimpleNamespace(
            id="row-1",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-03.bad",
            payload={"template": ""},
            is_active=True,
            description="坏配置",
            created_at=None,
            updated_at=None,
        )
        session = _FakeSession([row])

        result = set_runtime_config_entry_active("row-1", False, session_factory=lambda: session)

        self.assertFalse(row.is_active)
        self.assertTrue(session.committed)
        self.assertIs(session.refreshed, row)
        self.assertTrue(session.closed)
        self.assertEqual(result["id"], "row-1")
        self.assertFalse(result["is_active"])
        self.assertFalse(result["valid"])

    def test_create_runtime_config_entry_validates_and_creates_inactive_version(self):
        from app.services.runtime_config_governance import create_runtime_config_entry

        session = _FakeSession([])

        result = create_runtime_config_entry(
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-03.safe-write",
            payload={"template": "请为对话生成一个短标题"},
            description="安全写入测试",
            session_factory=lambda: session,
        )

        self.assertTrue(session.committed)
        self.assertIs(session.refreshed, session.added)
        self.assertTrue(session.closed)
        self.assertEqual(session.added.namespace, "prompt_template")
        self.assertEqual(session.added.key, "generate_title")
        self.assertEqual(session.added.version, "2026-07-03.safe-write")
        self.assertFalse(session.added.is_active)
        self.assertEqual(session.added.description, "安全写入测试")
        self.assertEqual(result["version"], "2026-07-03.safe-write")
        self.assertFalse(result["is_active"])
        self.assertTrue(result["valid"])

    def test_create_runtime_config_entry_rejects_invalid_payload_without_writing(self):
        from app.schemas.response import ApiException
        from app.services.runtime_config_governance import create_runtime_config_entry

        session = _FakeSession([])

        with self.assertRaises(ApiException) as raised:
            create_runtime_config_entry(
                namespace="prompt_template",
                key="generate_title",
                version="2026-07-03.bad",
                payload={"template": ""},
                session_factory=lambda: session,
            )

        self.assertEqual(raised.exception.code, "INVALID_PARAM")
        self.assertIn("template 必须是非空字符串", raised.exception.message)
        self.assertIsNone(session.added)
        self.assertFalse(session.committed)
        self.assertTrue(session.closed)

    def test_create_runtime_config_entry_rejects_duplicate_version(self):
        from app.schemas.response import ApiException
        from app.services.runtime_config_governance import create_runtime_config_entry

        existing = SimpleNamespace(
            id="row-1",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-03.same",
            payload={"template": "旧版本"},
            is_active=False,
            description="旧版本",
            created_at=None,
            updated_at=None,
        )
        session = _FakeSession([existing])

        with self.assertRaises(ApiException) as raised:
            create_runtime_config_entry(
                namespace="prompt_template",
                key="generate_title",
                version="2026-07-03.same",
                payload={"template": "新版本"},
                session_factory=lambda: session,
            )

        self.assertEqual(raised.exception.code, "CONFLICT")
        self.assertIsNone(session.added)
        self.assertFalse(session.committed)
        self.assertTrue(session.closed)

    def test_activate_runtime_config_entry_enforces_single_active_version(self):
        from app.services.runtime_config_governance import activate_runtime_config_entry

        target = SimpleNamespace(
            id="target",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-03.target",
            payload={"template": "目标版本"},
            is_active=False,
            description="目标版本",
            created_at=None,
            updated_at=None,
        )
        old_active = SimpleNamespace(
            id="old",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-02.old",
            payload={"template": "旧版本"},
            is_active=True,
            description="旧版本",
            created_at=None,
            updated_at=None,
        )
        session = _FakeSession([target, old_active])

        result = activate_runtime_config_entry("target", session_factory=lambda: session)

        self.assertTrue(target.is_active)
        self.assertFalse(old_active.is_active)
        self.assertTrue(session.committed)
        self.assertIs(session.refreshed, target)
        self.assertTrue(session.closed)
        self.assertEqual(result["id"], "target")
        self.assertTrue(result["is_active"])

    def test_activate_runtime_config_entry_rejects_invalid_target_without_changes(self):
        from app.schemas.response import ApiException
        from app.services.runtime_config_governance import activate_runtime_config_entry

        target = SimpleNamespace(
            id="target",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-03.bad",
            payload={"template": ""},
            is_active=False,
            description="坏版本",
            created_at=None,
            updated_at=None,
        )
        old_active = SimpleNamespace(
            id="old",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-02.old",
            payload={"template": "旧版本"},
            is_active=True,
            description="旧版本",
            created_at=None,
            updated_at=None,
        )
        session = _FakeSession([target, old_active])

        with self.assertRaises(ApiException) as raised:
            activate_runtime_config_entry("target", session_factory=lambda: session)

        self.assertEqual(raised.exception.code, "INVALID_PARAM")
        self.assertIn("template 必须是非空字符串", raised.exception.message)
        self.assertFalse(target.is_active)
        self.assertTrue(old_active.is_active)
        self.assertFalse(session.committed)
        self.assertTrue(session.closed)

    def test_set_runtime_config_entry_active_true_uses_safe_activation(self):
        from app.services.runtime_config_governance import set_runtime_config_entry_active

        target = SimpleNamespace(
            id="target",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-03.target",
            payload={"template": "目标版本"},
            is_active=False,
            description="目标版本",
            created_at=None,
            updated_at=None,
        )
        old_active = SimpleNamespace(
            id="old",
            namespace="prompt_template",
            key="generate_title",
            version="2026-07-02.old",
            payload={"template": "旧版本"},
            is_active=True,
            description="旧版本",
            created_at=None,
            updated_at=None,
        )
        session = _FakeSession([target, old_active])

        result = set_runtime_config_entry_active("target", True, session_factory=lambda: session)

        self.assertTrue(target.is_active)
        self.assertFalse(old_active.is_active)
        self.assertTrue(session.committed)
        self.assertEqual(result["id"], "target")


if __name__ == "__main__":
    unittest.main()
