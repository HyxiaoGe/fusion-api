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

    def query(self, model):
        return _FakeQuery(self.rows)

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


if __name__ == "__main__":
    unittest.main()
