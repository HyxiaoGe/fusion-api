"""
定时任务调度服务

使用 APScheduler 管理定时任务。
在 lifespan startup 时启动，shutdown 时关闭。
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.logger import app_logger as logger

_scheduler: AsyncIOScheduler | None = None


async def start_scheduler() -> None:
    """启动定时任务调度器"""
    global _scheduler

    from app.services.prompt_examples_service import refresh_prompt_examples
    from app.services.provider_probe import probe_offline_providers

    _scheduler = AsyncIOScheduler()

    # 每 12 小时刷新示例问题
    _scheduler.add_job(
        refresh_prompt_examples,
        trigger=IntervalTrigger(hours=12),
        id="refresh_prompt_examples",
        replace_existing=True,
    )

    # 每 30 min 探活 offline provider（job 内部按 offline_reason 决定是否到期，
    # tos_blocked 24h、其他 30min — 详见 provider_probe.PROBE_INTERVALS_MINUTES）
    _scheduler.add_job(
        probe_offline_providers,
        trigger=IntervalTrigger(minutes=30),
        id="probe_offline_providers",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info("定时任务调度器已启动")


async def stop_scheduler() -> None:
    """关闭定时任务调度器"""
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("定时任务调度器已关闭")
