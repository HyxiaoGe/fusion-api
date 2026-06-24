import unittest
from unittest.mock import AsyncMock, patch


class LifespanCleanupTests(unittest.IsolatedAsyncioTestCase):
    async def test_lifespan_shutdown_closes_litellm_async_clients(self):
        import main

        with (
            patch("main.init_redis", new=AsyncMock()),
            patch("main.init_storage", new=AsyncMock()),
            patch("main.start_scheduler", new=AsyncMock()),
            patch("main.litellm_health.start", new=AsyncMock()),
            patch("main.litellm_health.stop", new=AsyncMock()),
            patch("main.stop_scheduler", new=AsyncMock()),
            patch("main.close_redis", new=AsyncMock()),
            patch("main.litellm_cleanup.close_async_clients", new=AsyncMock()) as close_litellm,
        ):
            async with main.lifespan(main.app):
                pass

        close_litellm.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
