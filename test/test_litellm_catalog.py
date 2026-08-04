import unittest
from unittest.mock import Mock, patch

from app.ai import litellm_catalog


class LiteLLMCatalogTests(unittest.TestCase):
    def tearDown(self):
        litellm_catalog.invalidate()

    def test_normalize_capabilities_adds_agent_tools_from_function_calling(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "deepseek-chat",
            {"functionCalling": True, "vision": False},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertTrue(capabilities["agentTools"])
        self.assertTrue(capabilities["searchCapable"])
        self.assertTrue(capabilities["webSearch"])

    def test_normalize_capabilities_disables_known_non_agent_models_by_default(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "qwen-vl-max",
            {"functionCalling": True, "vision": True},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])
        self.assertFalse(capabilities["webSearch"])

    def test_normalize_capabilities_respects_explicit_agent_tools_false(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "future-model",
            {"functionCalling": True, "agentTools": False},
        )

        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])

    def test_normalize_capabilities_never_enables_agent_tools_without_function_calling(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "metadata-mismatch",
            {"functionCalling": False, "agentTools": True, "webSearch": True},
        )

        self.assertFalse(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])
        self.assertFalse(capabilities["webSearch"])

    def test_normalize_capabilities_preserves_vision_independent_of_search_tools(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "vision-only",
            {"functionCalling": False, "vision": True},
        )

        self.assertTrue(capabilities["vision"])
        self.assertFalse(capabilities["searchCapable"])

    def test_normalize_capabilities_uses_runtime_disabled_agent_tool_aliases(self):
        capabilities = litellm_catalog.normalize_capabilities(
            "deepseek-chat",
            {"functionCalling": True},
            agent_tools_disabled_aliases={"deepseek-chat"},
        )

        self.assertTrue(capabilities["functionCalling"])
        self.assertFalse(capabilities["agentTools"])
        self.assertFalse(capabilities["searchCapable"])

    def test_initial_failed_fetch_uses_short_backoff_before_retrying(self):
        litellm_catalog.invalidate()
        with (
            patch.object(litellm_catalog, "_fetch_catalog", return_value={}) as fetch,
            patch.object(litellm_catalog, "_read_catalog_generation", return_value="0"),
            patch.object(litellm_catalog.time, "monotonic", side_effect=[100.0, 101.0, 131.0]),
        ):
            self.assertEqual(litellm_catalog.list_aliases(), {})
            self.assertEqual(
                litellm_catalog.get_cache_status(),
                {"availability": "degraded", "has_cache": False},
            )
            self.assertEqual(litellm_catalog.list_aliases(), {})
            self.assertEqual(litellm_catalog.list_aliases(), {})

        self.assertEqual(fetch.call_count, 2)

    def test_stale_catalog_survives_failed_fetch_without_retrying_each_request(self):
        stale = {"deepseek-chat": {"db_model": True, "metadata": {}}}
        litellm_catalog._cache_payload = stale
        litellm_catalog._cache_loaded_at = 1.0
        litellm_catalog._cache_generation = "0"
        with (
            patch.object(litellm_catalog, "_fetch_catalog", return_value={}) as fetch,
            patch.object(litellm_catalog, "_read_catalog_generation", return_value="0"),
            patch.object(litellm_catalog.time, "monotonic", side_effect=[100.0, 101.0]),
        ):
            self.assertEqual(litellm_catalog.list_aliases(), stale)
            self.assertEqual(
                litellm_catalog.get_cache_status(),
                {"availability": "degraded", "has_cache": True},
            )
            self.assertEqual(litellm_catalog.list_aliases(), stale)

        fetch.assert_called_once_with()

    def test_redis_generation_change_invalidates_preheated_process_cache(self):
        litellm_catalog._cache_payload = {"old-model": {"db_model": True, "metadata": {}}}
        litellm_catalog._cache_loaded_at = 100.0
        litellm_catalog._cache_generation = "41"
        refreshed = {"new-model": {"db_model": True, "metadata": {}}}

        with (
            patch.object(litellm_catalog, "_read_catalog_generation", return_value="42"),
            patch.object(litellm_catalog, "_fetch_catalog", return_value=refreshed) as fetch,
            patch.object(litellm_catalog.time, "monotonic", return_value=101.0),
        ):
            self.assertEqual(litellm_catalog.list_aliases(), refreshed)

        fetch.assert_called_once_with()
        self.assertEqual(litellm_catalog._cache_generation, "42")

    def test_first_generation_seen_invalidates_unversioned_cache(self):
        litellm_catalog._cache_payload = {"old-model": {"db_model": True, "metadata": {}}}
        litellm_catalog._cache_loaded_at = 100.0
        litellm_catalog._cache_generation = None
        refreshed = {"new-model": {"db_model": True, "metadata": {}}}

        with (
            patch.object(litellm_catalog, "_read_catalog_generation", return_value="5"),
            patch.object(litellm_catalog, "_fetch_catalog", return_value=refreshed),
            patch.object(litellm_catalog.time, "monotonic", return_value=101.0),
        ):
            self.assertEqual(litellm_catalog.list_aliases(), refreshed)

        self.assertEqual(litellm_catalog._cache_generation, "5")

    def test_bump_generation_invalidates_local_cache(self):
        redis_client = Mock()
        redis_client.incr.return_value = 7
        litellm_catalog._cache_payload = {"old-model": {"db_model": True, "metadata": {}}}

        with patch.object(litellm_catalog, "_get_generation_redis", return_value=redis_client):
            generation = litellm_catalog.bump_generation()

        self.assertEqual(generation, "7")
        self.assertIsNone(litellm_catalog._cache_payload)
        self.assertEqual(litellm_catalog._cache_generation, "7")


if __name__ == "__main__":
    unittest.main()
