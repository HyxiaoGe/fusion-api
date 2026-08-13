import asyncio
import math
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.ai import litellm_cleanup, litellm_health
from app.api import (
    admin,
    admin_audit,
    admin_mcp,
    admin_model_management,
    auth,
    chat,
    files,
    knowledge_bases,
    models,
    prompts,
)
from app.core.config import (
    KNOWLEDGE_MILVUS_FILTER_TERM_BATCH_SIZE,
    KNOWLEDGE_SEARCH_PROFILE_CONCURRENCY,
    settings,
)
from app.core.logger import app_logger
from app.core.redis import close_redis, get_redis_pool, init_redis
from app.db.database import SessionLocal
from app.schemas.knowledge import KNOWLEDGE_SEARCH_MAX_BASES_PER_REQUEST
from app.schemas.response import ApiException, generate_request_id
from app.services.knowledge.storage_upload_guard import shutdown_storage_upload_lifecycles
from app.services.mcp.runtime import get_mcp_client_manager
from app.services.scheduler_service import start_scheduler, stop_scheduler
from app.services.storage import init_storage
from app.services.suggested_question_worker import (
    recover_pending_suggested_questions,
    stop_suggested_question_workers,
)

ASIA_SHANGHAI = timezone(timedelta(hours=8))
MULTIPART_OVERHEAD_BYTES = 1024 * 1024


class RequestBodyTooLarge(MultiPartException):
    """multipart 解析前请求体已超过有界限制。"""


class UploadBodyLimitMiddleware:
    """在 Starlette multipart 解析和临时文件写入前限制上传请求体。"""

    def __init__(self, app: ASGIApp):
        self.app = app

    @staticmethod
    def _body_limit(scope: Scope) -> int | None:
        if scope.get("type") != "http" or str(scope.get("method", "")).upper() != "POST":
            return None
        path = str(scope.get("path", ""))
        if path == "/api/files/upload":
            return settings.MAX_FILE_SIZE * 5 + MULTIPART_OVERHEAD_BYTES
        segments = path.strip("/").split("/")
        if len(segments) == 4 and segments[:2] == ["api", "knowledge-bases"] and segments[3] == "documents":
            return settings.KNOWLEDGE_MAX_FILE_SIZE + MULTIPART_OVERHEAD_BYTES
        return None

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        limit = self._body_limit(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return
        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        try:
            content_length = int(headers.get(b"content-length", b"0"))
        except ValueError:
            content_length = 0
        if content_length > limit:
            await self._reject(scope, receive, send)
            return

        received = 0
        response_started = False
        body_too_large = False

        async def limited_receive() -> Message:
            nonlocal body_too_large, received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > limit:
                    body_too_large = True
                    raise RequestBodyTooLarge("上传请求体超过文件大小上限")
            return message

        async def tracked_send(message: Message) -> None:
            nonlocal response_started
            # Starlette 会先关闭 multipart 临时文件，再把 MultiPartException
            # 转为 400。超限标记由本中间件持有，因此丢弃该内部响应并统一返回 413。
            if body_too_large:
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except RequestBodyTooLarge:
            pass
        if body_too_large and not response_started:
            await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: Scope, receive: Receive, send: Send) -> None:
        state = scope.get("state") or {}
        request_id = state.get("request_id") if isinstance(state, dict) else getattr(state, "request_id", None)
        request_id = request_id or generate_request_id()
        response = JSONResponse(
            status_code=413,
            content={
                "code": "FILE_TOO_LARGE",
                "message": "上传请求体超过文件大小上限",
                "data": None,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )
        await response(scope, receive, send)


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

    def _resolve_timeout_seconds(self, request: Request) -> float:
        key = (request.method.upper(), request.scope.get("path", ""))
        path = key[1]
        if key == ("POST", "/api/knowledge-bases/search"):
            profile_batches = math.ceil(settings.KNOWLEDGE_SEARCH_MAX_PROFILES / KNOWLEDGE_SEARCH_PROFILE_CONCURRENCY)
            max_index_versions = settings.KNOWLEDGE_MAX_DOCUMENTS_PER_BASE * KNOWLEDGE_SEARCH_MAX_BASES_PER_REQUEST
            version_batches = math.ceil(max_index_versions / KNOWLEDGE_MILVUS_FILTER_TERM_BATCH_SIZE)
            return (
                profile_batches
                * (
                    settings.KNOWLEDGE_EMBEDDING_TIMEOUT_SECONDS
                    + (version_batches + 1) * settings.MILVUS_TIMEOUT_SECONDS
                )
                + 5
            )
        path_segments = path.strip("/").split("/")
        if (
            key[0] == "POST"
            and len(path_segments) == 4
            and path_segments[:2] == ["api", "knowledge-bases"]
            and path_segments[3] == "documents"
        ):
            return settings.FILE_UPLOAD_TIMEOUT_SECONDS
        if path.startswith("/api/admin/mcp/servers/") and path.endswith(("/test", "/tools/refresh")):
            return settings.MCP_ADMIN_OPERATION_TIMEOUT_SECONDS
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
    await recover_pending_suggested_questions()

    yield

    # 对象存储 SDK 在线程中不可取消。先等待上传 lifecycle 完成真实 RPC、intent
    # finalizer 和续租收敛，再关闭其余外部客户端；硬终止则由持久 cleanup 接管。
    await shutdown_storage_upload_lifecycles()
    await stop_suggested_question_workers()
    await litellm_health.stop()
    await litellm_cleanup.close_async_clients()
    await get_mcp_client_manager().close()
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
    if request.url.path.startswith(("/api/admin", "/api/models", "/api/internal/model-management")):
        response.headers["Cache-Control"] = "private, no-store"
        response.headers["Pragma"] = "no-cache"
    return response


# 添加 Session 中间件
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

# 添加超时中间件
app.add_middleware(TimeoutMiddleware, timeout_seconds=10)

# 在 multipart 解析/临时文件写入前拒绝超限请求
app.add_middleware(UploadBodyLimitMiddleware)

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


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(part) for part in first.get("loc", ()) if part != "body")
    message = str(first.get("msg", "请求参数校验失败"))
    return JSONResponse(
        status_code=422,
        content={
            "code": "INVALID_PARAM",
            "message": f"{location}: {message}" if location else message,
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
app.include_router(knowledge_bases.router, prefix="/api/knowledge-bases", tags=["knowledge-bases"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(prompts.router, prefix="/api/prompts", tags=["prompts"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])
app.include_router(admin_audit.router, prefix="/api/admin/audit", tags=["admin-audit"])
app.include_router(admin_mcp.router, prefix="/api/admin/mcp", tags=["admin-mcp"])
app.include_router(
    admin_model_management.router,
    prefix="/api/admin/model-management",
    tags=["admin-model-management"],
)
app.include_router(
    admin_model_management.internal_router,
    prefix="/api/internal/model-management",
    tags=["internal-model-management"],
)

if __name__ == "__main__":
    import uvicorn

    app_logger.info("使用 uvicorn 启动服务器")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
