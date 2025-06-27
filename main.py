from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import asyncio
import time
from app.core.config import settings
from app.api import chat, settings as settings_api, prompts, files, hot_topics, scheduled_tasks, web_search, models, credentials, rss, auth, users
from app.core.logger import app_logger
from app.db.init_db import init_db
from app.db.database import SessionLocal
from app.services.scheduler_service import SchedulerService
from app.core.function_manager import init_function_registry


# 超时中间件
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int = 10):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                call_next(request), 
                timeout=self.timeout_seconds
            )
            # 添加处理时间头
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except asyncio.TimeoutError:
            process_time = time.time() - start_time
            return JSONResponse(
                status_code=408,
                content={
                    "detail": f"请求超时，处理时间超过{self.timeout_seconds}秒",
                    "timeout_seconds": self.timeout_seconds,
                    "process_time": process_time
                },
                headers={"X-Process-Time": str(process_time)}
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app_logger.info("应用启动中...")
        init_db()
        app_logger.info("数据库初始化完成")
        
        # 初始化函数注册表
        init_function_registry()

        # 启动调度器
        scheduler = SchedulerService()
        scheduler.start()
        app_logger.info("调度器启动完成")
        yield
        # 应用关闭时的清理工作
        app_logger.info("应用关闭中...")
    except Exception as e:
        app_logger.error(f"应用启动失败: {e}")

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

# 输出启动日志
app_logger.info(f"正在启动 {settings.APP_NAME} v{settings.APP_VERSION}")

# 添加 Session 中间件
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# 添加超时中间件
app.add_middleware(TimeoutMiddleware, timeout_seconds=10)

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
app.include_router(rss.router, prefix="/api/rss", tags=["rss"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(users.router, prefix="/api/users", tags=["users"])

if __name__ == "__main__":
    import uvicorn
    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )