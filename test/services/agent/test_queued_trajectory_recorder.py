import asyncio
import unittest
from collections.abc import Mapping
from typing import Any

from app.services.agent.queued_trajectory_recorder import QueuedTrajectoryRecorder
from app.services.stream.tool_executor import AgentEventCompositeWriter


def _event(sequence: int) -> dict[str, Any]:
    return {
        "type": "run_started",
        "run_id": "run-1",
        "sequence": sequence,
    }


class _RecordingInner:
    def __init__(self) -> None:
        self.run_id = "run-1"
        self.conversation_id = "conv-1"
        self.message_id = "msg-1"
        self.record_started = asyncio.Event()
        self.release_recording = asyncio.Event()
        self.block_recording = False
        self.record_error: Exception | None = None
        self.writes: list[int] = []
        self.finalize_calls: list[int] = []
        self.timeline: list[tuple[str, int]] = []
        self._degraded_reason: str | None = None
        self._pending_terminal_reconciliation: object | None = None

    @property
    def degraded_reason(self) -> str | None:
        return self._degraded_reason

    @property
    def pending_terminal_reconciliation(self) -> object | None:
        return self._pending_terminal_reconciliation

    def degraded_latch(self, run_id: str | None = None) -> bool:
        return (run_id is None or run_id == self.run_id) and self.degraded_reason is not None

    def _mark_degraded(self, reason: str) -> bool:
        if self._degraded_reason is None:
            self._degraded_reason = reason
        return True

    async def record_chunk(
        self,
        _conversation_id: str,
        _chunk_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        self.record_started.set()
        if self.block_recording:
            await self.release_recording.wait()
        if self.record_error is not None:
            raise self.record_error
        sequence = int(payload["sequence"])
        self.writes.append(sequence)
        self.timeline.append(("record", sequence))

    async def finalize(self, expected_last_sequence: int) -> None:
        self.finalize_calls.append(expected_last_sequence)
        self.timeline.append(("finalize", expected_last_sequence))


class _RedisWriter:
    def __init__(self) -> None:
        self.writes: list[int] = []

    async def append_chunk(
        self,
        _conversation_id: str,
        _task_id: str,
        _chunk_type: str,
        payload: dict[str, Any],
    ) -> None:
        self.writes.append(int(payload["sequence"]))


class QueuedTrajectoryRecorderTests(unittest.IsolatedAsyncioTestCase):
    async def test_record_chunk_and_composite_writer_do_not_wait_for_slow_inner(self):
        inner = _RecordingInner()
        inner.block_recording = True
        recorder = QueuedTrajectoryRecorder(inner)

        await asyncio.wait_for(
            recorder.record_chunk("conv-1", "agent_event", _event(0)),
            timeout=0.1,
        )
        await asyncio.wait_for(inner.record_started.wait(), timeout=0.1)

        redis_writer = _RedisWriter()
        writer = AgentEventCompositeWriter(
            redis_writer=redis_writer,
            trajectory_recorder=recorder,
        )
        await asyncio.wait_for(
            writer.append_chunk("conv-1", "task-1", "agent_event", _event(1)),
            timeout=0.1,
        )

        self.assertEqual(redis_writer.writes, [1])
        self.assertEqual(inner.writes, [])
        inner.release_recording.set()
        await recorder.finalize(1)
        self.assertEqual(inner.writes, [0, 1])

    async def test_finalize_flushes_accepted_events_in_order_before_inner_finalize(self):
        inner = _RecordingInner()
        recorder = QueuedTrajectoryRecorder(inner)

        for sequence in range(3):
            await recorder.record_chunk("conv-1", "agent_event", _event(sequence))
        await recorder.finalize(2)
        await recorder.record_chunk("conv-1", "agent_event", _event(3))

        self.assertEqual(inner.writes, [0, 1, 2])
        self.assertEqual(inner.finalize_calls, [2])
        self.assertEqual(
            inner.timeline,
            [("record", 0), ("record", 1), ("record", 2), ("finalize", 2)],
        )

    async def test_full_queue_degrades_without_blocking_admission(self):
        inner = _RecordingInner()
        inner.block_recording = True
        recorder = QueuedTrajectoryRecorder(inner, queue_size=1)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await asyncio.wait_for(inner.record_started.wait(), timeout=0.1)
        await recorder.record_chunk("conv-1", "agent_event", _event(1))
        await asyncio.wait_for(
            recorder.record_chunk("conv-1", "agent_event", _event(2)),
            timeout=0.1,
        )

        self.assertEqual(inner.degraded_reason, "admission_full")
        inner.release_recording.set()
        await recorder.finalize(2)
        self.assertEqual(inner.writes, [0])
        self.assertEqual(inner.finalize_calls, [2])

    async def test_worker_failure_degrades_and_finalize_still_closes_inner(self):
        inner = _RecordingInner()
        inner.record_error = RuntimeError("数据库写入失败")
        recorder = QueuedTrajectoryRecorder(inner)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await recorder.finalize(0)

        self.assertEqual(inner.degraded_reason, "write_failed")
        self.assertEqual(inner.finalize_calls, [0])

    async def test_flush_timeout_degrades_cancels_worker_and_leaks_no_task(self):
        tasks_before = set(asyncio.all_tasks())
        inner = _RecordingInner()
        inner.block_recording = True
        recorder = QueuedTrajectoryRecorder(inner, flush_timeout_seconds=0.01)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await asyncio.wait_for(inner.record_started.wait(), timeout=0.1)
        await recorder.finalize(0)
        await asyncio.sleep(0)

        self.assertEqual(inner.degraded_reason, "recorder_timeout")
        self.assertEqual(inner.finalize_calls, [0])
        self.assertEqual(
            [task for task in asyncio.all_tasks() - tasks_before if not task.done()],
            [],
        )

    async def test_cancelled_flush_degrades_cleans_worker_and_reraises(self):
        tasks_before = set(asyncio.all_tasks())
        inner = _RecordingInner()
        inner.block_recording = True
        recorder = QueuedTrajectoryRecorder(inner, flush_timeout_seconds=1.0)

        await recorder.record_chunk("conv-1", "agent_event", _event(0))
        await asyncio.wait_for(inner.record_started.wait(), timeout=0.1)
        finalize_task = asyncio.create_task(recorder.finalize(0))
        await asyncio.sleep(0)
        finalize_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await finalize_task
        await asyncio.sleep(0)

        self.assertEqual(inner.degraded_reason, "recorder_cancelled")
        self.assertEqual(inner.finalize_calls, [0])
        self.assertEqual(
            [task for task in asyncio.all_tasks() - tasks_before if not task.done()],
            [],
        )

    async def test_filters_non_trajectory_chunks_and_passes_through_public_state(self):
        inner = _RecordingInner()
        inner._pending_terminal_reconciliation = object()
        recorder = QueuedTrajectoryRecorder(inner)

        await recorder.record_chunk("other", "agent_event", _event(0))
        await recorder.record_chunk("conv-1", "content", _event(1))

        self.assertIs(recorder.inner, inner)
        self.assertEqual(recorder.run_id, "run-1")
        self.assertEqual(recorder.conversation_id, "conv-1")
        self.assertEqual(recorder.message_id, "msg-1")
        self.assertIsNone(recorder.degraded_reason)
        self.assertFalse(recorder.degraded_latch())
        self.assertIs(
            recorder.pending_terminal_reconciliation,
            inner.pending_terminal_reconciliation,
        )
        inner._mark_degraded("prior_failure")
        await recorder.record_chunk("conv-1", "agent_event", _event(2))
        self.assertEqual(recorder.degraded_reason, "prior_failure")
        self.assertTrue(recorder.degraded_latch())
        self.assertEqual(inner.writes, [])


if __name__ == "__main__":
    unittest.main()
