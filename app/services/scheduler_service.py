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

    _scheduler = AsyncIOScheduler()

    # 每 12 小时刷新示例问题
    _scheduler.add_job(
        refresh_prompt_examples,
        trigger=IntervalTrigger(hours=12),
        id="refresh_prompt_examples",
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
