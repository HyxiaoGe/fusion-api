import unittest
from types import SimpleNamespace
from uuid import UUID


class _FakeQuery:
    def __init__(self, row):
        self.row = row

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def first(self):
        return self.row


class _FakeSession:
    def __init__(self, row):
        self.row = row
        self.closed = False

    def query(self, model):
        return _FakeQuery(self.row)

    def close(self):
        self.closed = True


class RuntimeConfigServiceTests(unittest.TestCase):
    def setUp(self):
        from app.services.runtime_config_service import clear_runtime_config_cache

        clear_runtime_config_cache()

    def tearDown(self):
        from app.services.runtime_config_service import clear_runtime_config_cache

        clear_runtime_config_cache()

    def test_default_seed_rows_cover_strategy_presentation_and_prompts(self):
        from app.services.runtime_config_defaults import DEFAULT_PROMPT_TEMPLATES, iter_default_runtime_config_seed_rows

        rows = list(iter_default_runtime_config_seed_rows())
        row_keys = {(row["namespace"], row["key"]) for row in rows}

        self.assertIn(("agent_strategy", "default"), row_keys)
        self.assertIn(("model_presentation", "default"), row_keys)
        for prompt_key in DEFAULT_PROMPT_TEMPLATES:
            self.assertIn(("prompt_template", prompt_key), row_keys)
        self.assertEqual(len(rows), 2 + len(DEFAULT_PROMPT_TEMPLATES))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            UUID(row["id"])
        self.assertTrue(all(row["version"] == "2026-07-02.v1" for row in rows))
        self.assertTrue(all(row["is_active"] is True for row in rows))

    def test_get_runtime_config_payload_returns_default_when_db_unavailable(self):
        from app.services.runtime_config_service import clear_runtime_config_cache, get_runtime_config_payload

        clear_runtime_config_cache()

        def failing_session_factory():
            raise RuntimeError("db unavailable")

        payload, meta = get_runtime_config_payload(
            "agent_strategy",
            "default",
            {"search": {"standard_budget": {"requested_count": 5}}},
            session_factory=failing_session_factory,
        )

        self.assertEqual(payload, {"search": {"standard_budget": {"requested_count": 5}}})
        self.assertEqual(meta["source"], "default")
        self.assertEqual(meta["namespace"], "agent_strategy")
        self.assertEqual(meta["key"], "default")

    def test_get_runtime_config_payload_deep_merges_active_payload(self):
        from app.services.runtime_config_service import clear_runtime_config_cache, get_runtime_config_payload

        clear_runtime_config_cache()
        row = SimpleNamespace(
            payload={"search": {"standard_budget": {"requested_count": 7}}},
            version="2026-07-02.test",
        )
        session = _FakeSession(row)

        payload, meta = get_runtime_config_payload(
            "agent_strategy",
            "default",
            {
                "search": {
                    "standard_budget": {
                        "requested_count": 5,
                        "context_source_limit": 5,
                    },
                    "thresholds": {"duplicate_search": 0.82},
                }
            },
            session_factory=lambda: session,
        )

        self.assertEqual(payload["search"]["standard_budget"]["requested_count"], 7)
        self.assertEqual(payload["search"]["standard_budget"]["context_source_limit"], 5)
        self.assertEqual(payload["search"]["thresholds"]["duplicate_search"], 0.82)
        self.assertEqual(meta["source"], "db")
        self.assertEqual(meta["version"], "2026-07-02.test")
        self.assertTrue(session.closed)

    def test_get_runtime_config_payload_ignores_non_dict_payload(self):
        from app.services.runtime_config_service import clear_runtime_config_cache, get_runtime_config_payload

        clear_runtime_config_cache()
        row = SimpleNamespace(payload=["invalid"], version="bad")

        payload, meta = get_runtime_config_payload(
            "model_presentation",
            "default",
            {"weights": {"base": 40}},
            session_factory=lambda: _FakeSession(row),
        )

        self.assertEqual(payload, {"weights": {"base": 40}})
        self.assertEqual(meta["source"], "default")

    def test_get_runtime_config_payload_uses_cache_until_cleared(self):
        from app.services.runtime_config_service import clear_runtime_config_cache, get_runtime_config_payload

        clear_runtime_config_cache()
        first = SimpleNamespace(payload={"value": 1}, version="v1")
        second = SimpleNamespace(payload={"value": 2}, version="v2")
        calls = {"count": 0}

        def session_factory():
            calls["count"] += 1
            return _FakeSession(first if calls["count"] == 1 else second)

        payload, meta = get_runtime_config_payload(
            "agent_strategy", "default", {"value": 0}, session_factory=session_factory
        )
        cached_payload, cached_meta = get_runtime_config_payload(
            "agent_strategy",
            "default",
            {"value": 0},
            session_factory=session_factory,
        )

        self.assertEqual(payload["value"], 1)
        self.assertEqual(cached_payload["value"], 1)
        self.assertEqual(meta["version"], "v1")
        self.assertEqual(cached_meta["version"], "v1")
        self.assertEqual(calls["count"], 1)

        clear_runtime_config_cache()
        refreshed_payload, refreshed_meta = get_runtime_config_payload(
            "agent_strategy",
            "default",
            {"value": 0},
            session_factory=session_factory,
        )

        self.assertEqual(refreshed_payload["value"], 2)
        self.assertEqual(refreshed_meta["version"], "v2")


if __name__ == "__main__":
    unittest.main()
