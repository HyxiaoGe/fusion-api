import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from scripts.run_knowledge_worker import _refresh_health, _run_cleanup_loop, _run_once_work, run_worker


class KnowledgeWorkerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_persistent_cleanup_loop_does_not_block_index_worker(self):
        cleanup_started = asyncio.Event()
        cleanup_release = asyncio.Event()
        stop_event = asyncio.Event()

        async def blocked_cleanup():
            cleanup_started.set()
            await cleanup_release.wait()
            stop_event.set()
            return True

        cleanup_worker = MagicMock()
        cleanup_worker.run_once = AsyncMock(side_effect=blocked_cleanup)
        index_worker = MagicMock()
        index_worker.run_once = AsyncMock(return_value=True)
        processed = 0

        def on_processed():
            nonlocal processed
            processed += 1

        cleanup_loop = asyncio.create_task(
            _run_cleanup_loop(
                stop_event,
                cleanup_worker,
                on_processed=on_processed,
                poll_seconds=0.001,
            )
        )
        await cleanup_started.wait()
        index_handled = await asyncio.wait_for(index_worker.run_once(), timeout=0.1)
        cleanup_release.set()
        await cleanup_loop

        self.assertTrue(index_handled)
        self.assertEqual(processed, 1)
        cleanup_worker.run_once.assert_awaited_once()
        index_worker.run_once.assert_awaited_once()

    async def test_once_mode_processes_at_most_one_task_across_queues(self):
        cleanup_worker = MagicMock()
        cleanup_worker.run_once = AsyncMock(side_effect=[True, False])
        index_worker = MagicMock()
        index_worker.run_once = AsyncMock(return_value=True)

        cleanup_count = await _run_once_work(cleanup_worker, index_worker)
        index_count = await _run_once_work(cleanup_worker, index_worker)

        self.assertEqual(cleanup_count, 1)
        self.assertEqual(index_count, 1)
        self.assertEqual(index_worker.run_once.await_count, 1)

    async def test_dependency_health_debounces_failures_and_recovers(self):
        stop_event = asyncio.Event()
        outcomes = [RuntimeError("瞬时失败"), RuntimeError("再次失败"), RuntimeError("持续失败"), None]

        async def vector_health() -> None:
            outcome = outcomes.pop(0)
            if not outcomes:
                stop_event.set()
            if outcome is not None:
                raise outcome

        with patch("scripts.run_knowledge_worker._write_health") as write_health:
            await _refresh_health(
                stop_event,
                worker_id="worker-1",
                processed=lambda: 7,
                vector_health=vector_health,
                interval_seconds=0.001,
            )

        self.assertEqual(
            [call.kwargs["status"] for call in write_health.call_args_list],
            ["running", "running", "unhealthy", "running"],
        )
        self.assertEqual(
            [call.kwargs["error_code"] for call in write_health.call_args_list],
            [None, None, "KNOWLEDGE_VECTOR_UNAVAILABLE", None],
        )
        self.assertTrue(all(call.kwargs["processed"] == 7 for call in write_health.call_args_list))

    async def test_disabled_feature_still_processes_storage_cleanup_queue(self):
        cleanup_worker = MagicMock()
        cleanup_worker.run_once = AsyncMock(return_value=True)

        with (
            patch("scripts.run_knowledge_worker.settings.KNOWLEDGE_BASE_ENABLED", False),
            patch("scripts.run_knowledge_worker.init_storage", new=AsyncMock()) as init_storage,
            patch("scripts.run_knowledge_worker.KnowledgeStorageCleanupWorker", return_value=cleanup_worker),
            patch("scripts.run_knowledge_worker._write_health") as write_health,
        ):
            result = await run_worker(once=True)

        self.assertEqual(result, 0)
        init_storage.assert_awaited_once_with()
        cleanup_worker.run_once.assert_awaited_once_with()
        self.assertEqual(write_health.call_args_list[-1].kwargs["status"], "disabled")
        self.assertEqual(write_health.call_args_list[-1].kwargs["processed"], 1)

    async def test_disabled_feature_rejects_invalid_cleanup_worker_configuration(self):
        with (
            patch("scripts.run_knowledge_worker.settings.KNOWLEDGE_BASE_ENABLED", False),
            patch("scripts.run_knowledge_worker.settings.KNOWLEDGE_WORKER_POLL_SECONDS", 0),
            patch("scripts.run_knowledge_worker.init_storage", new=AsyncMock()) as init_storage,
        ):
            result = await run_worker(once=True)

        self.assertEqual(result, 2)
        init_storage.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
