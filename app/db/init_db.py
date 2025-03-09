from app.core.logger import app_logger
from app.db.database import engine, Base


def init_db():
    """初始化数据库表结构"""
    try:
        app_logger.info("正在创建数据库表...")
        Base.metadata.create_all(bind=engine)
        app_logger.info("数据库表创建成功！")
    except Exception as e:
        app_logger.error(f"创建数据库表失败: {e}")
        raise
