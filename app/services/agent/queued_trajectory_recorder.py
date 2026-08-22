"""每个 run 单消费者的有界异步轨迹接纳器。"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from app.services.agent.trajectory_recorder import (
    TrajectoryRecorder,
    TrajectoryTerminalReconciliation,
)

TRAJECTORY_QUEUE_SIZE = 1000
TRAJECTORY_FLUSH_TIMEOUT_SECONDS = 10.0

_QueueItem = tuple[str, str, Mapping[str, Any]]


class QueuedTrajectoryRecorder:
    """将同步账本核心包装为非阻塞接纳、顺序写入的 per-run sink。"""

    def __init__(
        self,
        inner: TrajectoryRecorder,
        *,
        queue_size: int = TRAJECTORY_QUEUE_SIZE,
        flush_timeout_seconds: float = TRAJECTORY_FLUSH_TIMEOUT_SECONDS,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("trajectory queue_size 必须大于 0")
        if flush_timeout_seconds <= 0:
            raise ValueError("trajectory flush_timeout_seconds 必须大于 0")
        self._inner = inner
        self._queue: asyncio.Queue[_QueueItem | None] = asyncio.Queue(maxsize=queue_size)
        self._flush_timeout_seconds = flush_timeout_seconds
        self._consumer_task: asyncio.Task[None] | None = None
        self._worker_error: Exception | None = None
        self._closed = False

    @property
    def inner(self) -> TrajectoryRecorder:
        return self._inner

    @property
    def run_id(self) -> str:
        return self._inner.run_id

    @property
    def conversation_id(self) -> str:
        return self._inner.conversation_id

    @property
    def message_id(self) -> str | None:
        return self._inner.message_id

    @property
    def degraded_reason(self) -> str | None:
        return self._inner.degraded_reason

    @property
    def pending_terminal_reconciliation(self) -> TrajectoryTerminalReconciliation | None:
        return self._inner.pending_terminal_reconciliation

    def degraded_latch(self, run_id: str | None = None) -> bool:
        return self._inner.degraded_latch(run_id)

    async def record_chunk(
        self,
        conversation_id: str,
        chunk_type: str,
        payload: Mapping[str, Any],
    ) -> None:
        if conversation_id != self.conversation_id or chunk_type != "agent_event":
            return
        if self._closed or self._inner.degraded_latch(self.run_id):
            return
        try:
            self._queue.put_nowait((conversation_id, chunk_type, payload))
        except asyncio.QueueFull:
            self._inner._mark_degraded("admission_full")
            return
        if self._consumer_task is None:
            self._consumer_task = asyncio.create_task(
                self._consume(),
                name=f"trajectory-recorder-{self.run_id}",
            )

    async def finalize(self, expected_last_sequence: int) -> None:
        if self._closed:
            return
        self._closed = True
        cancelled_error: asyncio.CancelledError | None = None

        if self._consumer_task is not None:
            flush_task = asyncio.create_task(
                self._flush_consumer(),
                name=f"trajectory-recorder-flush-{self.run_id}",
            )
            try:
                await asyncio.wait_for(
                    asyncio.shield(flush_task),
                    timeout=self._flush_timeout_seconds,
                )
            except asyncio.CancelledError as error:
                self._inner._mark_degraded("recorder_cancelled")
                await self._cancel_background(flush_task)
                cancelled_error = error
            except TimeoutError:
                self._inner._mark_degraded("recorder_timeout")
                await self._cancel_background(flush_task)
            except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
                self._worker_error = self._worker_error or error
                self._inner._mark_degraded("write_failed")
                await self._cancel_background(flush_task)

        if cancelled_error is not None:
            try:
                await self._inner.finalize(expected_last_sequence)
            finally:
                raise cancelled_error

        try:
            await self._inner.finalize(expected_last_sequence)
        except asyncio.CancelledError:
            self._inner._mark_degraded("recorder_cancelled")
            raise

    async def _consume(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item is None:
                    return
                conversation_id, chunk_type, payload = item
                if self._inner.degraded_latch(self.run_id):
                    continue
                try:
                    await self._inner.record_chunk(conversation_id, chunk_type, payload)
                except asyncio.CancelledError:
                    raise
                except Exception as error:  # noqa: BLE001 — auxiliary sink 必须 fail-open
                    self._worker_error = error
                    self._inner._mark_degraded("write_failed")
            finally:
                self._queue.task_done()

    async def _flush_consumer(self) -> None:
        await self._queue.put(None)
        if self._consumer_task is not None:
            await self._consumer_task
        if self._worker_error is not None:
            self._inner._mark_degraded("write_failed")

    async def _cancel_background(self, flush_task: asyncio.Task[None]) -> None:
        tasks = [flush_task]
        if self._consumer_task is not None:
            tasks.append(self._consumer_task)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
