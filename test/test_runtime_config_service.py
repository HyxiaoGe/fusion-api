import unittest
from types import SimpleNamespace
from uuid import UUID


class _FakeQuery:
    def __init__(self, rows):
        if rows is None:
            self.rows = []
        elif isinstance(rows, list):
            self.rows = rows
        else:
            self.rows = [rows]

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def first(self):
        return self.rows[0] if self.rows else None

    def all(self):
        return self.rows


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.closed = False

    def query(self, model):
        return _FakeQuery(self.rows)

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
        self.assertIn(("ui_prompt_catalog", "home"), row_keys)
        for prompt_key in DEFAULT_PROMPT_TEMPLATES:
            self.assertIn(("prompt_template", prompt_key), row_keys)
        self.assertEqual(len(rows), 3 + len(DEFAULT_PROMPT_TEMPLATES))
        self.assertEqual(len({row["id"] for row in rows}), len(rows))
        for row in rows:
            UUID(row["id"])
        versions = {(row["namespace"], row["key"]): row["version"] for row in rows}
        self.assertEqual(versions[("ui_prompt_catalog", "home")], "2026-07-14.v1")
        self.assertTrue(
            all(
                version == "2026-07-02.v1"
                for key, version in versions.items()
                if key != ("ui_prompt_catalog", "home")
            )
        )
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

    def test_get_runtime_config_payload_skips_invalid_latest_active_candidate(self):
        from app.services.runtime_config_service import get_runtime_config_payload

        invalid_latest = SimpleNamespace(payload={"template": 123}, version="2026-07-03.bad")
        valid_previous = SimpleNamespace(payload={"template": "有效模板"}, version="2026-07-02.good")
        session = _FakeSession([invalid_latest, valid_previous])

        payload, meta = get_runtime_config_payload(
            "prompt_template",
            "generate_title",
            {"template": "默认模板"},
            session_factory=lambda: session,
        )

        self.assertEqual(payload, {"template": "有效模板"})
        self.assertEqual(meta["source"], "db")
        self.assertEqual(meta["version"], "2026-07-02.good")
        self.assertEqual(meta["skipped_versions"], ["2026-07-03.bad"])
        self.assertTrue(any("template" in issue for issue in meta["validation_warnings"]["2026-07-03.bad"]))
        self.assertTrue(session.closed)

    def test_get_runtime_config_payload_returns_default_when_all_candidates_invalid(self):
        from app.services.runtime_config_defaults import DEFAULT_MODEL_PRESENTATION_CONFIG
        from app.services.runtime_config_service import get_runtime_config_payload

        invalid_latest = SimpleNamespace(payload={"weights": "bad"}, version="2026-07-03.bad")

        payload, meta = get_runtime_config_payload(
            "model_presentation",
            "default",
            DEFAULT_MODEL_PRESENTATION_CONFIG,
            session_factory=lambda: _FakeSession([invalid_latest]),
        )

        self.assertEqual(payload, DEFAULT_MODEL_PRESENTATION_CONFIG)
        self.assertEqual(meta["source"], "default")
        self.assertEqual(meta["skipped_versions"], ["2026-07-03.bad"])
        self.assertIn("2026-07-03.bad", meta["validation_warnings"])

    def test_validate_runtime_config_payload_reports_domain_specific_issues(self):
        from app.core.runtime_config_schema import validate_runtime_config_payload

        prompt_result = validate_runtime_config_payload("prompt_template", "generate_title", {"template": ""})
        strategy_result = validate_runtime_config_payload("agent_strategy", "default", {"search": {}})
        presentation_result = validate_runtime_config_payload(
            "model_presentation",
            "default",
            {"weights": {}, "levels": {}, "copy": {}},
        )
        catalog_result = validate_runtime_config_payload(
            "ui_prompt_catalog",
            "home",
            {"items": [{"id": "broken", "kind": "starter"}]},
        )

        self.assertFalse(prompt_result.valid)
        self.assertIn("template 必须是非空字符串", prompt_result.issues)
        self.assertFalse(strategy_result.valid)
        self.assertTrue(any("model_runtime" in issue for issue in strategy_result.issues))
        self.assertFalse(presentation_result.valid)
        self.assertTrue(any("weights.base" in issue for issue in presentation_result.issues))
        self.assertFalse(catalog_result.valid)
        self.assertTrue(any("items[0].title" in issue for issue in catalog_result.issues))


if __name__ == "__main__":
    unittest.main()
