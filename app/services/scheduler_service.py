from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.services.hot_topic_service import HotTopicService
from app.db.database import SessionLocal
from app.db.repositories import ScheduledTaskRepository

class SchedulerService:
    """调度服务，管理定时任务"""
    
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        
    def start(self):
        """启动调度器"""
        # 每小时更新一次热点数据
        self.scheduler.add_job(
            self._check_and_run_tasks,
            CronTrigger(minute=0),  # 每小时整点执行
            id="check_and_run_tasks",
            replace_existing=True
        )
        
        # 启动时也立即执行一次
        self.scheduler.add_job(
            self._check_and_run_tasks,
            id="initial_check_tasks",
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("调度器已启动")
        
    async def _check_and_run_tasks(self):
        """检查并执行所有到期的定时任务"""
        db = SessionLocal()
        try:
            # 获取所有活跃任务
            task_repo = ScheduledTaskRepository(db)
            tasks = task_repo.get_all_active_tasks()
            
            if not tasks:
                logger.info("未发现任何定时任务，执行初始化...")
                # 初始化热点话题任务
                hot_topic_service = HotTopicService(db)
                # 初始化成功后，强制执行一次更新
                await hot_topic_service.update_hot_topics(force=True)
            else:
                for task in tasks:
                    # 检查每个任务是否应该执行
                    if task_repo.should_run_task(task.name):
                        logger.info(f"准备执行定时任务: {task.name}")
                        await self._run_task(task.name, db)
        except Exception as e:
            logger.error(f"检查定时任务失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            db.close()
            
    async def _run_task(self, task_name: str, db: Session):
        """执行特定的定时任务"""
        try:
            # 使用任务名称关联到相应的服务和方法
            task_handlers = {
                HotTopicService.TASK_NAME: self._run_hot_topic_task,
                HotTopicService.LOCAL_DOCS_TASK_NAME: self._run_local_docs_task,
            }

            # 查找对应的处理器
            if task_name in task_handlers:
                handler = task_handlers[task_name]
                await handler(db)
            else:
                logger.warning(f"未知的定时任务类型: {task_name}")
        except Exception as e:
            logger.error(f"执行定时任务 {task_name} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())

    async def _run_hot_topic_task(self, db: Session):
        """执行热点话题更新任务"""
        service = HotTopicService(db)
        await service.update_hot_topics(True)

    async def _run_local_docs_task(self, db: Session):
        """执行本地文档更新任务"""
        service = HotTopicService(db)
        await service.update_local_docs(True)
