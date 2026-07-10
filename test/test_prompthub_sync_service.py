import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from test.test_prompt_bundle import _published_bundle


class _FakeQuery:
    def __init__(self, rows):
        self.rows = rows

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def limit(self, *args, **kwargs):
        return self

    def all(self):
        return list(self.rows)


class _FakeSession:
    def __init__(self, rows=None):
        self.rows = list(rows or [])
        self.added = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False

    def query(self, model):
        return _FakeQuery(self.rows)

    def add(self, row):
        self.rows.append(row)
        self.added.append(row)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        self.closed = True

    def get_bind(self):
        return SimpleNamespace(dialect=SimpleNamespace(name="sqlite"))


class _CommitFailingSession(_FakeSession):
    def commit(self):
        raise RuntimeError("commit failed")


class PromptHubSyncServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_disabled_does_not_call_prompthub_or_database(self):
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        client = SimpleNamespace(fetch_published_bundle=AsyncMock())
        session_factory = unittest.mock.Mock()

        result = await sync_prompthub_bundle(
            mode="disabled",
            client=client,
            session_factory=session_factory,
        )

        self.assertEqual(result["status"], "disabled")
        client.fetch_published_bundle.assert_not_awaited()
        session_factory.assert_not_called()

    async def test_shadow_persists_validated_lkg_inactive_and_reports_zero_diff(self):
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        session = _FakeSession()
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="shadow",
            client=client,
            session_factory=lambda: session,
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["revision"], "b" * 64)
        self.assertEqual(result["changed_prompt_keys"], [])
        self.assertEqual(len(session.added), 1)
        self.assertEqual(session.added[0].namespace, "prompt_bundle")
        self.assertEqual(session.added[0].key, "fusion")
        self.assertFalse(session.added[0].is_active)
        self.assertEqual(session.commits, 1)

    async def test_apply_atomically_activates_bundle_and_deactivates_old_revision(self):
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        old = SimpleNamespace(
            id="old",
            namespace="prompt_bundle",
            key="fusion",
            version="a" * 64,
            payload={"revision": "a" * 64},
            is_active=True,
        )
        session = _FakeSession([old])
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="apply",
            client=client,
            session_factory=lambda: session,
        )

        self.assertEqual(result["status"], "success")
        self.assertFalse(old.is_active)
        self.assertTrue(session.added[0].is_active)
        self.assertEqual(session.commits, 1)

    async def test_same_revision_is_idempotent(self):
        from app.core.prompt_bundle import validate_published_bundle
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        existing = SimpleNamespace(
            id="same",
            namespace="prompt_bundle",
            key="fusion",
            version="b" * 64,
            payload=validate_published_bundle(_published_bundle()),
            is_active=True,
        )
        session = _FakeSession([existing])
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="apply",
            client=client,
            session_factory=lambda: session,
        )

        self.assertTrue(result["idempotent"])
        self.assertEqual(session.added, [])
        self.assertEqual(session.commits, 0)

    async def test_shadow_makes_same_revision_inactive_after_apply_rollback(self):
        from app.core.prompt_bundle import validate_published_bundle
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        existing = SimpleNamespace(
            id="same",
            namespace="prompt_bundle",
            key="fusion",
            version="b" * 64,
            payload=validate_published_bundle(_published_bundle()),
            is_active=True,
        )
        session = _FakeSession([existing])
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="shadow",
            client=client,
            session_factory=lambda: session,
        )

        self.assertFalse(result["idempotent"])
        self.assertFalse(result["active"])
        self.assertFalse(existing.is_active)
        self.assertEqual(session.commits, 1)

    async def test_apply_activates_inactive_same_revision_created_by_shadow(self):
        from app.core.prompt_bundle import validate_published_bundle
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        old = SimpleNamespace(
            id="old",
            namespace="prompt_bundle",
            key="fusion",
            version="a" * 64,
            payload={"revision": "a" * 64},
            is_active=True,
        )
        shadow_lkg = SimpleNamespace(
            id="shadow",
            namespace="prompt_bundle",
            key="fusion",
            version="b" * 64,
            payload=validate_published_bundle(_published_bundle()),
            is_active=False,
        )
        session = _FakeSession([old, shadow_lkg])
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="apply",
            client=client,
            session_factory=lambda: session,
        )

        self.assertFalse(result["idempotent"])
        self.assertTrue(result["active"])
        self.assertFalse(old.is_active)
        self.assertTrue(shadow_lkg.is_active)
        self.assertEqual(session.commits, 1)

    async def test_apply_rejects_corrupted_inactive_same_revision_and_preserves_old_active(self):
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        old = SimpleNamespace(
            id="old",
            namespace="prompt_bundle",
            key="fusion",
            version="a" * 64,
            payload={"revision": "a" * 64},
            is_active=True,
        )
        corrupted = SimpleNamespace(
            id="corrupted",
            namespace="prompt_bundle",
            key="fusion",
            version="b" * 64,
            payload={"schema_version": 0, "revision": "b" * 64},
            is_active=False,
        )
        session = _FakeSession([old, corrupted])
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="apply",
            client=client,
            session_factory=lambda: session,
        )

        self.assertEqual(result["status"], "error")
        self.assertTrue(old.is_active)
        self.assertFalse(corrupted.is_active)
        self.assertEqual(session.commits, 0)

    async def test_apply_rolls_back_when_commit_fails(self):
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        session = _CommitFailingSession()
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))

        result = await sync_prompthub_bundle(
            mode="apply",
            client=client,
            session_factory=lambda: session,
        )

        self.assertEqual(result["status"], "error")
        self.assertEqual(session.rollbacks, 1)

    async def test_successful_write_clears_prompt_bundle_cache(self):
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        session = _FakeSession()
        client = SimpleNamespace(fetch_published_bundle=AsyncMock(return_value=_published_bundle()))
        with patch("app.services.prompthub_sync_service.clear_prompt_bundle_cache") as clear_cache:
            result = await sync_prompthub_bundle(
                mode="apply",
                client=client,
                session_factory=lambda: session,
            )

        self.assertEqual(result["status"], "success")
        clear_cache.assert_called_once_with()

    async def test_fetch_or_validation_failure_preserves_existing_lkg(self):
        from app.services.external.prompthub_client import PromptHubClientError
        from app.services.prompthub_sync_service import sync_prompthub_bundle

        existing = SimpleNamespace(
            id="old",
            namespace="prompt_bundle",
            key="fusion",
            version="a" * 64,
            payload={"revision": "a" * 64},
            is_active=True,
        )
        invalid_bundle = _published_bundle()
        invalid_bundle = SimpleNamespace(**{**vars(invalid_bundle), "prompts": invalid_bundle.prompts[:-1]})
        failures = [PromptHubClientError("timeout", "请求超时"), invalid_bundle]

        for failure in failures:
            with self.subTest(failure=type(failure).__name__):
                session = _FakeSession([existing])
                fetch = (
                    AsyncMock(side_effect=failure)
                    if isinstance(failure, Exception)
                    else AsyncMock(return_value=failure)
                )
                client = SimpleNamespace(fetch_published_bundle=fetch)

                result = await sync_prompthub_bundle(
                    mode="apply",
                    client=client,
                    session_factory=lambda: session,
                )

                self.assertEqual(result["status"], "error")
                self.assertTrue(existing.is_active)
                self.assertEqual(session.added, [])
                self.assertEqual(session.commits, 0)

    async def test_best_effort_wrapper_never_raises(self):
        from app.services.prompthub_sync_service import run_prompthub_sync_best_effort

        with patch(
            "app.services.prompthub_sync_service.sync_prompthub_bundle",
            new=AsyncMock(side_effect=RuntimeError("boom")),
        ):
            result = await run_prompthub_sync_best_effort()

        self.assertEqual(result["status"], "error")
        self.assertNotIn("api_key", result)


class PromptHubSyncDiagnosticsTests(unittest.TestCase):
    def test_admin_snapshot_contains_read_only_sync_diagnostics_without_secret(self):
        from app.services.runtime_config_governance import build_runtime_config_snapshot

        session = _FakeSession()
        diagnostics = {
            "mode": "shadow",
            "status": "success",
            "revision": "b" * 64,
            "last_success_at": "2026-07-10T00:00:00+00:00",
            "last_error": None,
        }
        with patch(
            "app.services.runtime_config_governance.get_prompthub_sync_diagnostics",
            return_value=diagnostics,
            create=True,
        ):
            snapshot = build_runtime_config_snapshot(session_factory=lambda: session)

        self.assertEqual(snapshot["prompt_sync"], diagnostics)
        self.assertNotIn("api_key", snapshot["prompt_sync"])


if __name__ == "__main__":
    unittest.main()
