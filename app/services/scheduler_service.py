from app.core.logger import app_logger as logger

class SchedulerService:
    """调度服务占位实现。"""

    def start(self):
        """当前聊天精简模式下禁用后台调度。"""
        logger.info("调度器已禁用：当前运行模式只保留聊天核心能力")
