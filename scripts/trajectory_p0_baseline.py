"""测量 Trajectory P0 生产队列包装器的本机接纳延迟。

这里的同步核心使用可控 stub 写入，不连接真实数据库，因此结果不能冒充 dev PostgreSQL 基线。
"""

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
from uuid import uuid4

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from app.services.agent.queued_trajectory_recorder import QueuedTrajectoryRecorder  # noqa: E402
from app.services.agent.trajectory_recorder import TrajectoryRecorder  # noqa: E402
from app.services.stream.tool_executor import AgentEventCompositeWriter  # noqa: E402


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


async def _sample_path(
    writer: AgentEventCompositeWriter,
    *,
    count: int,
    run_id: str,
) -> dict[str, float]:
    samples: list[float] = []
    for sequence in range(count):
        started = time.perf_counter()
        await writer.append_chunk(
            "conv-1",
            "task-1",
            "agent_event",
            _event(run_id, sequence),
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
    samples_per_path: int = 250,
    write_delay_seconds: float = 0.02,
) -> dict[str, Any]:
    """返回多轮 baseline/生产队列包装器接纳边界的本机 stub 分位数。"""
    if rounds <= 0 or samples_per_path <= 0 or write_delay_seconds < 0:
        raise ValueError("rounds、samples_per_path 必须大于 0，write_delay_seconds 不得小于 0")

    measurements: list[dict[str, Any]] = []
    for round_index in range(1, rounds + 1):
        run_id = f"run-local-{round_index}-{uuid4()}"
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        queued_recorder: QueuedTrajectoryRecorder | None = None
        finalized_sequences: list[int] = []
        try:
            recorder = TrajectoryRecorder(
                run_id=run_id,
                conversation_id="conv-1",
                message_id=f"msg-{round_index}",
                executor=executor,
                semaphore=threading.BoundedSemaphore(4),
            )
            recorder._write_event = lambda _payload: time.sleep(write_delay_seconds)

            async def _capture_finalize(expected_last_sequence: int) -> None:
                finalized_sequences.append(expected_last_sequence)

            recorder._run_finalize_isolated = _capture_finalize
            queued_recorder = QueuedTrajectoryRecorder(recorder)
            baseline_writer = AgentEventCompositeWriter(
                redis_writer=_RedisWriterStub(),
                recorder=_ProgressSinkStub(),
            )
            trajectory_writer = AgentEventCompositeWriter(
                redis_writer=_RedisWriterStub(),
                recorder=_ProgressSinkStub(),
                trajectory_recorder=queued_recorder,
            )
            baseline_stub = await _sample_path(
                baseline_writer,
                count=samples_per_path,
                run_id=run_id,
            )
            trajectory_stub = await _sample_path(
                trajectory_writer,
                count=samples_per_path,
                run_id=run_id,
            )
            await queued_recorder.finalize(samples_per_path - 1)
            measurements.append(
                {
                    "round": round_index,
                    "run_id": run_id,
                    "finalized_sequence": finalized_sequences[-1],
                    "baseline_stub": baseline_stub,
                    "trajectory_stub": trajectory_stub,
                }
            )
        finally:
            if queued_recorder is not None:
                await queued_recorder.finalize(samples_per_path - 1)
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
