from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from contextlib import asynccontextmanager
import asyncio
import time
from app.core.config import settings
from app.api import chat, files, models, auth
from app.core.logger import app_logger
from app.db.init_db import init_db
from app.db.database import SessionLocal



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
    allow_origins=settings.RESOLVED_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 健康检查端点（Railway需要）
@app.get("/health")
async def health_check():
    """健康检查端点，Railway用来判断应用是否正常运行"""
    try:
        from datetime import datetime
        from sqlalchemy import text
        
        # 简单的数据库连接测试
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected",
            "service": "fusion-api",
            "version": settings.APP_VERSION
        }
    except Exception as e:
        app_logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "service": "fusion-api"
        }

# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

if __name__ == "__main__":
    import uvicorn
    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
