import asyncio
import unittest

from app.services import task_manager


class TaskManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        task_manager._tasks.clear()

    async def asyncTearDown(self):
        for task, _task_id in task_manager._tasks.values():
            task.cancel()
        await asyncio.sleep(0)
        task_manager._tasks.clear()

    async def test_cancel_task_rejects_replaced_task_id(self):
        task = asyncio.create_task(asyncio.Event().wait())
        task_manager.register_task("conv-1", task, "task-new")

        self.assertFalse(task_manager.cancel_task("conv-1", "task-old"))
        self.assertFalse(task.cancelled())
        self.assertIs(task_manager.get_task("conv-1"), task)

        self.assertTrue(task_manager.cancel_task("conv-1", "task-new"))
        with self.assertRaises(asyncio.CancelledError):
            await task


if __name__ == "__main__":
    unittest.main()
