"""独立运行知识库持久任务 Worker。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import socket
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Awaitable, Callable

from sqlalchemy import text

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.services.knowledge.storage_cleanup_worker import KnowledgeStorageCleanupWorker
from app.services.knowledge.worker import KnowledgeWorker
from app.services.storage import init_storage

_DEPENDENCY_HEALTH_FAILURE_THRESHOLD = 3


def _health_path() -> Path:
    return Path(settings.KNOWLEDGE_WORKER_HEALTH_FILE)


def _write_health(*, status: str, worker_id: str, processed: int, error_code: str | None = None) -> None:
    path = _health_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "status": status,
        "worker_id": worker_id,
        "processed": processed,
        "updated_at": datetime.now(UTC).isoformat(),
        "error_code": error_code,
    }
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def check_health(max_age_seconds: int) -> int:
    try:
        payload = json.loads(_health_path().read_text(encoding="utf-8"))
        updated_at = datetime.fromisoformat(str(payload["updated_at"]))
        age = (datetime.now(UTC) - updated_at).total_seconds()
        if payload.get("status") not in {"running", "disabled"} or age > max_age_seconds:
            return 1
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
        return 0
    except Exception:
        return 1


async def _refresh_health(
    stop_event: asyncio.Event,
    *,
    worker_id: str,
    processed: Callable[[], int],
    vector_health: Callable[[], Awaitable[None]],
    interval_seconds: float | None = None,
) -> None:
    interval = (
        max(1, min(settings.KNOWLEDGE_WORKER_HEARTBEAT_SECONDS, 30)) if interval_seconds is None else interval_seconds
    )
    consecutive_failures = 0
    dependency_unhealthy = False
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            try:
                await vector_health()
            except Exception:
                consecutive_failures += 1
                became_unhealthy = consecutive_failures >= _DEPENDENCY_HEALTH_FAILURE_THRESHOLD
                if became_unhealthy and not dependency_unhealthy:
                    logger.warning(
                        "知识库 Worker 持续无法连接 Milvus，连续失败次数=%d",
                        consecutive_failures,
                    )
                dependency_unhealthy = became_unhealthy
            else:
                if dependency_unhealthy:
                    logger.info("知识库 Worker 已恢复 Milvus 连接")
                consecutive_failures = 0
                dependency_unhealthy = False
            _write_health(
                status="unhealthy" if dependency_unhealthy else "running",
                worker_id=worker_id,
                processed=processed(),
                error_code="KNOWLEDGE_VECTOR_UNAVAILABLE" if dependency_unhealthy else None,
            )


async def _run_once_work(
    cleanup_worker: KnowledgeStorageCleanupWorker,
    worker: KnowledgeWorker,
) -> int:
    """单次模式跨两类队列总计最多处理一个任务。"""
    cleanup_handled = await cleanup_worker.run_once()
    if cleanup_handled:
        return 1
    index_handled = await worker.run_once()
    return int(index_handled)


async def _run_cleanup_loop(
    stop_event: asyncio.Event,
    cleanup_worker: KnowledgeStorageCleanupWorker,
    *,
    on_processed: Callable[[], None],
    poll_seconds: float | None = None,
) -> None:
    """独立清理循环避免不可取消的对象删除阻塞索引任务队列。"""
    interval = settings.KNOWLEDGE_WORKER_POLL_SECONDS if poll_seconds is None else poll_seconds
    while not stop_event.is_set():
        try:
            handled = await cleanup_worker.run_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("知识库孤立对象清理循环异常")
            handled = False
        if handled:
            on_processed()
            continue
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except TimeoutError:
            pass


async def run_worker(*, once: bool) -> int:
    if not settings.KNOWLEDGE_BASE_ENABLED:
        worker_id = f"{socket.gethostname()}-{os.getpid()}"
        _write_health(status="disabled", worker_id=worker_id, processed=0)
        if once:
            return 0
        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, stop_event.set)
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=60)
            except TimeoutError:
                _write_health(status="disabled", worker_id=worker_id, processed=0)
        return 0
    try:
        settings.validate_knowledge_base_configuration()
    except ValueError as exc:
        logger.error("知识库 Worker 配置无效: %s", exc)
        return 2
    await init_storage()
    worker = KnowledgeWorker(SessionLocal, worker_id=f"{socket.gethostname()}-{os.getpid()}")
    cleanup_worker = KnowledgeStorageCleanupWorker(
        SessionLocal,
        worker_id=f"{worker.worker_id}-storage-cleanup",
    )
    await worker.vector_store.health()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_event.set)
    processed = 0

    def record_cleanup() -> None:
        nonlocal processed
        processed += 1

    _write_health(status="running", worker_id=worker.worker_id, processed=processed)
    health_refresh = asyncio.create_task(
        _refresh_health(
            stop_event,
            worker_id=worker.worker_id,
            processed=lambda: processed,
            vector_health=worker.vector_store.health,
        )
    )
    cleanup_loop = None
    if not once:
        cleanup_loop = asyncio.create_task(
            _run_cleanup_loop(
                stop_event,
                cleanup_worker,
                on_processed=record_cleanup,
            )
        )
    try:
        if once:
            processed += await _run_once_work(cleanup_worker, worker)
            return 0
        while not stop_event.is_set():
            handled = await worker.run_once()
            if handled:
                processed += 1
            else:
                try:
                    await asyncio.wait_for(stop_event.wait(), timeout=settings.KNOWLEDGE_WORKER_POLL_SECONDS)
                except TimeoutError:
                    pass
        return 0
    finally:
        stop_event.set()
        # cleanup 删除不可取消；保持任务和数据库行锁直到 SDK 真正返回。
        if cleanup_loop is not None:
            with suppress(asyncio.CancelledError):
                await cleanup_loop
        health_refresh.cancel()
        with suppress(asyncio.CancelledError):
            await health_refresh
        _write_health(status="stopped", worker_id=worker.worker_id, processed=processed)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行 Fusion 知识库 Worker")
    parser.add_argument("--once", action="store_true", help="最多领取并处理一个任务后退出")
    parser.add_argument("--healthcheck", action="store_true", help="检查 Worker 健康状态文件和数据库")
    parser.add_argument("--health-max-age-seconds", type=int, default=120)
    args = parser.parse_args()
    if args.healthcheck:
        return check_health(args.health_max_age_seconds)
    return asyncio.run(run_worker(once=args.once))


if __name__ == "__main__":
    raise SystemExit(main())
