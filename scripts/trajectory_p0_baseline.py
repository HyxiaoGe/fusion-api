"""复现 Trajectory P0 的本机 stub 延迟样本。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.agent.trajectory_recorder import TrajectoryRecorder  # noqa: E402


def _event(run_id: str, sequence: int) -> dict[str, Any]:
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
    async def append_chunk(self, conversation_id, task_id, chunk_type, payload):
        return None


class _ProgressSinkStub:
    def record_chunk(self, conversation_id, chunk_type, payload):
        return None


class _CompositeWriterStub:
    """只复现 Redis → progress → trajectory 的确定性基线路径。"""

    def __init__(self, *, trajectory_recorder: TrajectoryRecorder | None = None) -> None:
        self.redis_writer = _RedisWriterStub()
        self.progress_recorder = _ProgressSinkStub()
        self.trajectory_recorder = trajectory_recorder

    async def append_chunk(self, conversation_id, task_id, chunk_type, payload):
        await self.redis_writer.append_chunk(conversation_id, task_id, chunk_type, payload)
        self.progress_recorder.record_chunk(conversation_id, chunk_type, payload)
        if self.trajectory_recorder is not None:
            await self.trajectory_recorder.record_chunk(conversation_id, chunk_type, payload)


async def _sample_path(writer: _CompositeWriterStub, *, count: int) -> dict[str, float]:
    samples: list[float] = []
    for sequence in range(count):
        started = time.perf_counter()
        await writer.append_chunk(
            "conv-1",
            "task-1",
            "agent_event",
            _event("run-fast", sequence),
        )
        samples.append(time.perf_counter() - started)
    return {
        "p50_ms": round(_percentile_ms(samples, 0.50), 4),
        "p95_ms": round(_percentile_ms(samples, 0.95), 4),
        "p99_ms": round(_percentile_ms(samples, 0.99), 4),
    }


async def run_trajectory_stub_baseline(
    *,
    rounds: int = 3,
    samples_per_path: int = 500,
) -> dict[str, Any]:
    """返回多轮 baseline/TrajectoryRecorder fast-path 的分位数。"""
    if rounds <= 0 or samples_per_path <= 0:
        raise ValueError("rounds 与 samples_per_path 必须大于 0")

    measurements: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        try:
            recorder = TrajectoryRecorder(
                run_id="run-fast",
                conversation_id="conv-1",
                message_id="msg-1",
                executor=executor,
                semaphore=threading.BoundedSemaphore(4),
            )
            recorder._write_event = lambda _payload: None
            baseline_writer = _CompositeWriterStub()
            trajectory_writer = _CompositeWriterStub(
                trajectory_recorder=recorder,
            )
            measurements.append(
                {
                    "round": round_index,
                    "baseline_stub": await _sample_path(
                        baseline_writer,
                        count=samples_per_path,
                    ),
                    "trajectory_stub": await _sample_path(
                        trajectory_writer,
                        count=samples_per_path,
                    ),
                }
            )
        finally:
            executor.shutdown(wait=True)

    return {
        "rounds": rounds,
        "samples_per_path": samples_per_path,
        "measurements": measurements,
    }


def main() -> int:
    result = asyncio.run(run_trajectory_stub_baseline())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
