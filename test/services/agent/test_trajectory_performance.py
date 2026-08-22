import asyncio
import concurrent.futures
import threading
import time
import unittest

from app.services.agent.queued_trajectory_recorder import QueuedTrajectoryRecorder
from app.services.agent.trajectory_recorder import TrajectoryRecorder
from app.services.stream.tool_executor import AgentEventCompositeWriter
from scripts.trajectory_p0_baseline import run_trajectory_stub_baseline


def _event(run_id: str, sequence: int) -> dict:
    return {
        "schema_version": 1,
        "type": "step_started",
        "run_id": run_id,
        "parent_run_id": None,
        "step_id": f"step-{sequence}",
        "parent_step_id": None,
        "tool_call_id": None,
        "sequence": sequence,
        "trace_id": "trace-performance",
        "ts": 1_700_000_000.0 + sequence,
        "step_number": sequence + 1,
    }


def _percentile_ms(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(len(ordered) * percentile + 0.999999) - 1))
    return ordered[index] * 1000


class _RedisWriterStub:
    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls

    async def append_chunk(self, conversation_id, task_id, chunk_type, payload):
        if self.calls is not None:
            self.calls.append("redis")


class _ProgressSinkStub:
    def __init__(self, calls: list[str] | None = None) -> None:
        self.calls = calls

    def record_chunk(self, conversation_id, chunk_type, payload):
        if self.calls is not None:
            self.calls.append("progress")


class TrajectoryPerformanceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_tracked_runner_finalizes_each_unique_run_without_leaking_tasks(self):
        tasks_before = set(asyncio.all_tasks())
        result = await run_trajectory_stub_baseline(rounds=3, samples_per_path=5)
        await asyncio.sleep(0)

        self.assertEqual(result["rounds"], 3)
        self.assertEqual(result["samples_per_path"], 5)
        self.assertEqual(len(result["measurements"]), 3)
        self.assertEqual(len({item["run_id"] for item in result["measurements"]}), 3)
        for measurement in result["measurements"]:
            self.assertEqual(
                set(measurement),
                {"round", "run_id", "finalized_sequence", "baseline_stub", "trajectory_stub"},
            )
            self.assertEqual(measurement["finalized_sequence"], 4)
            for path in ("baseline_stub", "trajectory_stub"):
                self.assertEqual(set(measurement[path]), {"p50_ms", "p95_ms", "p99_ms"})
                self.assertLessEqual(measurement[path]["p50_ms"], measurement[path]["p95_ms"])
                self.assertLessEqual(measurement[path]["p95_ms"], measurement[path]["p99_ms"])
            self.assertLessEqual(measurement["trajectory_stub"]["p95_ms"], 5.0)
            self.assertLessEqual(measurement["trajectory_stub"]["p99_ms"], 15.0)
        self.assertEqual(
            [
                task
                for task in asyncio.all_tasks() - tasks_before
                if task.get_name().startswith("trajectory-recorder-")
            ],
            [],
        )

    async def test_shared_admission_caps_simultaneous_workers_and_sessions_at_four(self):
        semaphore = threading.BoundedSemaphore(4)
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=12)
        release = threading.Event()
        all_workers_started = threading.Event()
        state_lock = threading.Lock()
        active_workers = 0
        max_workers = 0
        opened_sessions = 0
        max_sessions = 0

        def write_event(_payload):
            nonlocal active_workers, max_workers, opened_sessions, max_sessions
            with state_lock:
                active_workers += 1
                opened_sessions += 1
                max_workers = max(max_workers, active_workers)
                max_sessions = max(max_sessions, opened_sessions)
                if active_workers == 4:
                    all_workers_started.set()
            release.wait(timeout=2)
            with state_lock:
                active_workers -= 1
                opened_sessions -= 1

        recorders = []
        for index in range(12):
            recorder = TrajectoryRecorder(
                run_id=f"run-{index}",
                conversation_id="conv-1",
                message_id=f"msg-{index}",
                executor=executor,
                semaphore=semaphore,
            )
            recorder._write_event = write_event
            recorders.append(recorder)

        tasks = [
            asyncio.create_task(recorder.record_chunk("conv-1", "agent_event", _event(recorder.run_id, 0)))
            for recorder in recorders
        ]
        self.assertTrue(await asyncio.to_thread(all_workers_started.wait, 1))
        await asyncio.sleep(0.02)

        self.assertEqual(max_workers, 4)
        self.assertEqual(max_sessions, 4)
        self.assertLessEqual(opened_sessions, 4)
        self.assertEqual(sum(recorder.degraded_reason == "admission_full" for recorder in recorders), 8)

        release.set()
        await asyncio.gather(*tasks)
        permits = [semaphore.acquire(blocking=False) for _ in range(5)]
        self.assertEqual(permits, [True, True, True, True, False])
        for _ in range(4):
            semaphore.release()
        executor.shutdown(wait=True)

    async def test_failed_queued_sink_does_not_change_visible_redis_and_progress_order(self):
        calls: list[str] = []
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        inner = TrajectoryRecorder(
            run_id="run-failed",
            conversation_id="conv-1",
            message_id="msg-1",
            executor=executor,
            semaphore=threading.BoundedSemaphore(4),
        )

        def failed_write(_payload):
            calls.append("trajectory_started")
            raise RuntimeError("stub 写入失败")

        async def capture_finalize(_expected_last_sequence: int) -> None:
            calls.append("trajectory_finalized")

        inner._write_event = failed_write
        inner._run_finalize_isolated = capture_finalize
        recorder = QueuedTrajectoryRecorder(inner)
        writer = AgentEventCompositeWriter(
            redis_writer=_RedisWriterStub(calls),
            recorder=_ProgressSinkStub(calls),
            trajectory_recorder=recorder,
        )

        await writer.append_chunk(
            "conv-1",
            "task-1",
            "agent_event",
            _event("run-failed", 0),
        )
        self.assertEqual(calls, ["redis", "progress"])
        await recorder.finalize(0)

        self.assertEqual(calls[:3], ["redis", "progress", "trajectory_started"])
        self.assertEqual(calls[-1], "trajectory_finalized")
        self.assertEqual(recorder.degraded_reason, "write_failed")
        executor.shutdown(wait=True)

    async def test_production_queue_admission_meets_gate_with_twenty_ms_inner_write(self):
        tasks_before = set(asyncio.all_tasks())
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        inner = TrajectoryRecorder(
            run_id="run-slow",
            conversation_id="conv-1",
            message_id="msg-1",
            executor=executor,
            semaphore=threading.BoundedSemaphore(4),
        )
        finalized_sequences: list[int] = []

        async def capture_finalize(expected_last_sequence: int) -> None:
            finalized_sequences.append(expected_last_sequence)

        inner._write_event = lambda _payload: time.sleep(0.02)
        inner._run_finalize_isolated = capture_finalize
        recorder = QueuedTrajectoryRecorder(inner)
        writer = AgentEventCompositeWriter(
            redis_writer=_RedisWriterStub(),
            recorder=_ProgressSinkStub(),
            trajectory_recorder=recorder,
        )

        samples: list[float] = []
        for sequence in range(250):
            started = time.perf_counter()
            await writer.append_chunk(
                "conv-1",
                "task-1",
                "agent_event",
                _event("run-slow", sequence),
            )
            samples.append(time.perf_counter() - started)
        await recorder.finalize(249)
        await asyncio.sleep(0)

        p50 = _percentile_ms(samples, 0.50)
        p95 = _percentile_ms(samples, 0.95)
        p99 = _percentile_ms(samples, 0.99)
        self.assertLessEqual(p50, p95)
        self.assertLessEqual(p95, p99)
        self.assertLessEqual(p95, 5.0)
        self.assertLessEqual(p99, 15.0)
        self.assertEqual(finalized_sequences, [249])
        self.assertEqual(
            [
                task
                for task in asyncio.all_tasks() - tasks_before
                if task.get_name().startswith("trajectory-recorder-")
            ],
            [],
        )
        executor.shutdown(wait=True)


if __name__ == "__main__":
    unittest.main()
