import unittest
from unittest.mock import Mock, patch

from app.services import scheduler_service
from app.services.agent.trajectory_reconciliation import (
    TrajectoryReconciliationResult,
    reconcile_trajectory_best_effort,
)


class FakeScheduler:
    def __init__(self) -> None:
        self.jobs = []
        self.started = 0
        self.running = False

    def add_job(self, func, **kwargs):
        self.jobs.append((func, kwargs))

    def start(self):
        self.started += 1
        self.running = True

    def shutdown(self, wait=False):
        self.running = False


class TrajectorySchedulerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        scheduler_service._scheduler = None

    def tearDown(self) -> None:
        scheduler_service._scheduler = None

    async def test_start_scheduler_registers_one_idempotent_trajectory_job(self):
        fake = FakeScheduler()
        with (
            patch.object(scheduler_service, "AsyncIOScheduler", return_value=fake) as constructor,
            patch.object(scheduler_service.settings, "PROMPTHUB_SYNC_MODE", "off"),
        ):
            await scheduler_service.start_scheduler()
            await scheduler_service.start_scheduler()

        constructor.assert_called_once_with()
        self.assertEqual(fake.started, 1)
        trajectory_jobs = [job for job in fake.jobs if job[1]["id"] == "reconcile_trajectory_ledger"]
        self.assertEqual(len(trajectory_jobs), 1)
        func, options = trajectory_jobs[0]
        self.assertIs(func, reconcile_trajectory_best_effort)
        self.assertEqual(options["trigger"].interval.total_seconds(), 60)
        self.assertTrue(options["replace_existing"])
        self.assertEqual(options["max_instances"], 1)
        self.assertTrue(options["coalesce"])

    async def test_best_effort_failure_is_swallowed_without_sensitive_exception_text(self):
        logger = Mock()

        def broken_factory():
            raise RuntimeError("postgresql://user:secret@db/private")

        result = await reconcile_trajectory_best_effort(
            session_factory=broken_factory,
            logger=logger,
        )

        self.assertEqual(result, TrajectoryReconciliationResult())
        logger.error.assert_called_once()
        rendered_log = " ".join(str(item) for item in logger.error.call_args.args)
        self.assertNotIn("secret", rendered_log)
        self.assertNotIn("postgresql", rendered_log)


if __name__ == "__main__":
    unittest.main()
