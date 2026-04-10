import asyncio
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.api import auth, chat, files, models, prompts
from app.core.config import settings
from app.core.logger import app_logger
from app.core.redis import close_redis, init_redis
from app.db.database import SessionLocal
from app.db.init_db import init_db
from app.schemas.response import ApiException, generate_request_id
from app.services.scheduler_service import start_scheduler, stop_scheduler
from app.services.storage import init_storage


# 超时中间件
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int = 10):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        try:
            response = await asyncio.wait_for(call_next(request), timeout=self.timeout_seconds)
            # 添加处理时间头
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except asyncio.TimeoutError:
            process_time = time.time() - start_time
            return JSONResponse(
                status_code=408,
                content={
                    "code": "REQUEST_TIMEOUT",
                    "message": f"请求超时，处理时间超过{self.timeout_seconds}秒",
                    "data": None,
                    "request_id": getattr(request.state, "request_id", generate_request_id()),
                },
                headers={"X-Process-Time": str(process_time)},
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("应用启动中...")
    try:
        init_db()
        app_logger.info("数据库初始化完成")
    except Exception as e:
        app_logger.error(f"数据库初始化失败: {e}")
    await init_redis()
    await init_storage()
    app_logger.info(f"存储后端初始化完成: {settings.STORAGE_BACKEND}")
    await start_scheduler()

    yield

    await stop_scheduler()
    await close_redis()
    app_logger.info("应用关闭完成")


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

# 输出启动日志
app_logger.info(f"正在启动 {settings.APP_NAME} v{settings.APP_VERSION}")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = generate_request_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


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
            "version": settings.APP_VERSION,
        }
    except Exception as e:
        app_logger.error(f"健康检查失败: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "service": "fusion-api",
        }


@app.exception_handler(ApiException)
async def api_exception_handler(request: Request, exc: ApiException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": exc.code,
            "message": exc.message,
            "data": None,
            "request_id": getattr(request.state, "request_id", generate_request_id()),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    code_map = {
        400: "INVALID_PARAM",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        500: "INTERNAL_ERROR",
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "code": code_map.get(exc.status_code, "INTERNAL_ERROR"),
            "message": str(exc.detail),
            "data": None,
            "request_id": getattr(request.state, "request_id", generate_request_id()),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    app_logger.exception("未预期的异常")
    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "服务器内部错误",
            "data": None,
            "request_id": getattr(request.state, "request_id", generate_request_id()),
        },
    )


# 注册路由
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(files.router, prefix="/api/files", tags=["files"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(prompts.router, prefix="/api/prompts", tags=["prompts"])

if __name__ == "__main__":
    import uvicorn

    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
