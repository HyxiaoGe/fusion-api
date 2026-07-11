import asyncio
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.ai import litellm_cleanup, litellm_health
from app.api import admin, admin_audit, auth, chat, files, models, prompts
from app.core.config import settings
from app.core.logger import app_logger
from app.core.redis import close_redis, get_redis_pool, init_redis
from app.db.database import SessionLocal
from app.schemas.response import ApiException, generate_request_id
from app.services.scheduler_service import start_scheduler, stop_scheduler
from app.services.storage import init_storage

ASIA_SHANGHAI = timezone(timedelta(hours=8))


# 超时中间件
class TimeoutMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, timeout_seconds: int = 10, route_timeouts: dict[tuple[str, str], int] | None = None):
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
        self.route_timeouts = {
            ("POST", "/api/files/upload"): settings.FILE_UPLOAD_TIMEOUT_SECONDS,
            ("POST", "/api/files/upload/complete"): settings.FILE_UPLOAD_TIMEOUT_SECONDS,
        }
        if route_timeouts:
            self.route_timeouts.update(route_timeouts)

    def _resolve_timeout_seconds(self, request: Request) -> int:
        key = (request.method.upper(), request.scope.get("path", ""))
        return self.route_timeouts.get(key, self.timeout_seconds)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        timeout_seconds = self._resolve_timeout_seconds(request)
        try:
            response = await asyncio.wait_for(call_next(request), timeout=timeout_seconds)
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
                    "message": f"请求超时，处理时间超过{timeout_seconds}秒",
                    "data": None,
                    "request_id": getattr(request.state, "request_id", generate_request_id()),
                },
                headers={"X-Process-Time": str(process_time)},
            )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_logger.info("应用启动中...")
    await init_redis()
    await init_storage()
    app_logger.info(f"存储后端初始化完成: {settings.STORAGE_BACKEND}")
    await start_scheduler()
    await litellm_health.start()

    yield

    await litellm_health.stop()
    await litellm_cleanup.close_async_clients()
    await stop_scheduler()
    await close_redis()
    app_logger.info("应用关闭完成")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.ENABLE_DOCS else None,
    redoc_url="/redoc" if settings.ENABLE_DOCS else None,
    openapi_url="/openapi.json" if settings.ENABLE_DOCS else None,
)

# 输出启动日志
app_logger.info(f"正在启动 {settings.APP_NAME} v{settings.APP_VERSION}")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = generate_request_id()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.middleware("http")
async def prevent_admin_audit_caching(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/admin/audit"):
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Pragma"] = "no-cache"
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
    """就绪检查：数据库和 Redis 都可用时才允许实例接流量。"""
    database_status = "unavailable"
    redis_status = "unavailable"
    errors: list[str] = []
    db = None
    try:
        from sqlalchemy import text

        db = SessionLocal()
        db.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception as e:
        errors.append(f"database: {e}")
    finally:
        if db is not None:
            db.close()

    try:
        redis = get_redis_pool()
        if redis is None:
            raise RuntimeError("连接池未初始化")
        await redis.ping()
        redis_status = "connected"
    except Exception as e:
        errors.append(f"redis: {e}")

    payload = {
        "status": "healthy" if not errors else "unhealthy",
        "timestamp": datetime.now(ASIA_SHANGHAI).isoformat(),
        "database": database_status,
        "redis": redis_status,
        "service": "fusion-api",
        "version": settings.APP_VERSION,
    }
    if errors:
        app_logger.error(f"就绪检查失败: {'; '.join(errors)}")
        payload["error"] = "; ".join(errors)
        return JSONResponse(status_code=503, content=payload)

    return payload


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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """兼容旧代码中抛出的 ValueError，统一转为 400"""
    return JSONResponse(
        status_code=400,
        content={
            "code": "INVALID_PARAM",
            "message": str(exc),
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
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(admin_audit.router, prefix="/api/admin/audit", tags=["admin-audit"])

if __name__ == "__main__":
    import uvicorn

    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
