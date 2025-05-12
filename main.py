from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.core.config import settings
from app.api import chat, settings as settings_api, prompts, files, hot_topics, scheduled_tasks, web_search, models, credentials
from app.core.logger import app_logger
from app.db.init_db import init_db
from app.db.database import SessionLocal
from app.ai.llm_manager import llm_manager
from app.services.scheduler_service import SchedulerService
from app.core.function_manager import init_function_registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_logger.info("应用启动中...")
        init_db()
        app_logger.info("数据库初始化完成")

        # 初始化LLM管理器的数据库连接
        db = SessionLocal()
        llm_manager.db = db
        app_logger.info("LLM管理器数据库初始化完成")
        
        # 初始化函数注册表
        init_function_registry()

        # 启动调度器
        scheduler = SchedulerService()
        scheduler.start()
        app_logger.info("调度器启动完成")
        yield
        # 应用关闭时，关闭数据库连接
        if llm_manager.db:
            llm_manager.db.close()
    except Exception as e:
        app_logger.error(f"应用启动失败: {e}")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

# 输出启动日志
app_logger.info(f"正在启动 {settings.APP_NAME} v{settings.APP_VERSION}")

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境可以设置为"*"，生产环境应该限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(settings_api.router, prefix="/api/settings", tags=["settings"])
app.include_router(prompts.router, prefix="/api/prompts", tags=["prompts"])
app.include_router(hot_topics.router, prefix="/api/topics", tags=["topics"])
app.include_router(scheduled_tasks.router, prefix="/api/scheduled-tasks", tags=["scheduled-tasks"])
app.include_router(web_search.router, prefix="/api/web_search", tags=["web_search"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(credentials.router, prefix="/api/credentials", tags=["credentials"])

if __name__ == "__main__":
    import uvicorn
    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )