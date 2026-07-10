"""
定时任务调度服务

使用 APScheduler 管理定时任务。
在 lifespan startup 时启动，shutdown 时关闭。
"""

from datetime import UTC, datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import settings
from app.core.logger import app_logger as logger

_scheduler: AsyncIOScheduler | None = None


async def start_scheduler() -> None:
    """启动定时任务调度器"""
    global _scheduler

    from app.services.prompt_examples_service import refresh_prompt_examples
    from app.services.prompthub_sync_service import run_prompthub_sync_best_effort

    _scheduler = AsyncIOScheduler()

    # 每 12 小时刷新示例问题
    _scheduler.add_job(
        refresh_prompt_examples,
        trigger=IntervalTrigger(hours=12),
        id="refresh_prompt_examples",
        replace_existing=True,
    )

    if settings.PROMPTHUB_SYNC_MODE in {"shadow", "apply"}:
        startup_options = {"next_run_time": datetime.now(UTC)} if settings.PROMPTHUB_SYNC_ON_STARTUP else {}
        _scheduler.add_job(
            run_prompthub_sync_best_effort,
            trigger=IntervalTrigger(seconds=settings.PROMPTHUB_SYNC_INTERVAL_SECONDS),
            id="sync_prompthub_bundle",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
            **startup_options,
        )

    # provider 健康追踪已迁移到 LiteLLM Proxy，本进程不再做探活

    _scheduler.start()
    logger.info("定时任务调度器已启动")


async def stop_scheduler() -> None:
    """关闭定时任务调度器"""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("定时任务调度器已关闭")
