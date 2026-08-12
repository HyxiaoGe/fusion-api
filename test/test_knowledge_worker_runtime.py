import asyncio
import unittest
from unittest.mock import patch

from scripts.run_knowledge_worker import _refresh_health


class KnowledgeWorkerRuntimeTests(unittest.IsolatedAsyncioTestCase):
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


if __name__ == "__main__":
    unittest.main()
