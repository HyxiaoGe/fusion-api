import os
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "AI桌面聊天应用"
    APP_VERSION: str = "0.1.1"

    ENABLE_VECTOR_EMBEDDINGS: bool = False

    CHROMA_URL: str = os.getenv("CHROMA_URL", "http://localhost:8001")

    # 模型配置
    DEFAULT_MODEL: str = "qwen"  # 默认使用的模型

    # 数据库配置
    DATABASE_URL: str = os.getenv("DATABASE_URL")

    # Redis 配置
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    REDIS_STREAM_TTL: int = 300  # 流状态缓存 TTL（秒）

    # 文件存储配置
    FILE_STORAGE_PATH: str = os.getenv("FILE_STORAGE_PATH", "./storage/files")
    FILE_STORAGE_KEY_PREFIX: str = os.getenv("FILE_STORAGE_KEY_PREFIX", "files/v1")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    FILE_UPLOAD_TIMEOUT_SECONDS: int = int(os.getenv("FILE_UPLOAD_TIMEOUT_SECONDS", "60"))
    DIRECT_UPLOAD_STALE_SECONDS: int = int(os.getenv("DIRECT_UPLOAD_STALE_SECONDS", "1800"))
    ALLOWED_FILE_TYPES: List[str] = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/bmp",
        "image/webp",
        "image/heic",
        "image/heif",
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain",
        "text/markdown",
        "text/csv",
    ]

    # 存储后端配置（"local"、"minio" 或 "oss"）
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "fusion-files")
    MINIO_USE_SSL: bool = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
    MINIO_PRESIGN_EXPIRES: int = int(os.getenv("MINIO_PRESIGN_EXPIRES", "3600"))
    OSS_ENDPOINT: str = os.getenv("OSS_ENDPOINT", "")
    OSS_ACCESS_KEY_ID: str = os.getenv("OSS_ACCESS_KEY_ID", "")
    OSS_ACCESS_KEY_SECRET: str = os.getenv("OSS_ACCESS_KEY_SECRET", "")
    OSS_BUCKET: str = os.getenv("OSS_BUCKET", "")
    OSS_USE_SSL: bool = os.getenv("OSS_USE_SSL", "true").lower() == "true"

    # Github OAuth
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None

    # Google OAuth
    GOOGLE_CLIENT_ID: Optional[str] = None
    GOOGLE_CLIENT_SECRET: Optional[str] = None

    # 前端基础URL，回调路径会自动拼接
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    FRONTEND_AUTH_CALLBACK_PATH: str = "/auth/callback"
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "")

    AUTH_SERVICE_BASE_URL: str = os.getenv("AUTH_SERVICE_BASE_URL", "http://localhost:8100")
    AUTH_SERVICE_CLIENT_ID: Optional[str] = os.getenv("AUTH_SERVICE_CLIENT_ID")
    AUTH_SERVICE_JWKS_URL: Optional[str] = os.getenv("AUTH_SERVICE_JWKS_URL")
    # 服务端内网直连地址（可选）。设置后 JWKS 抓取与 userinfo 走内网，绕开公网域名经
    # Cloudflare tunnel 的回环（实测 1-3s → 内网 3-8ms）。issuer/audience 仍用公网
    # AUTH_SERVICE_BASE_URL 校验，不受影响。
    AUTH_SERVICE_INTERNAL_BASE_URL: Optional[str] = os.getenv("AUTH_SERVICE_INTERNAL_BASE_URL")

    @property
    def FRONTEND_AUTH_CALLBACK_URL(self) -> str:
        """动态生成前端OAuth回调URL"""
        return f"{self.FRONTEND_URL.rstrip('/')}{self.FRONTEND_AUTH_CALLBACK_PATH}"

    @property
    def RESOLVED_CORS_ORIGINS(self) -> List[str]:
        configured = [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]
        origins = configured or [self.FRONTEND_URL.rstrip("/")]
        # Keep order stable while de-duplicating.
        return list(dict.fromkeys(origins))

    @property
    def RESOLVED_AUTH_SERVICE_JWKS_URL(self) -> str:
        # 内网 base 优先（绕开公网域名的 CF tunnel 回环）；否则沿用显式 JWKS_URL；
        # 最后回退公网 base 拼接。
        if self.AUTH_SERVICE_INTERNAL_BASE_URL:
            return f"{self.AUTH_SERVICE_INTERNAL_BASE_URL.rstrip('/')}/.well-known/jwks.json"
        return self.AUTH_SERVICE_JWKS_URL or f"{self.AUTH_SERVICE_BASE_URL.rstrip('/')}/.well-known/jwks.json"

    @property
    def AUTH_SERVICE_USERINFO_URL(self) -> str:
        # userinfo 是服务端调用，优先内网直连；issuer 校验仍用公网 BASE_URL（见 security.py）。
        base = (self.AUTH_SERVICE_INTERNAL_BASE_URL or self.AUTH_SERVICE_BASE_URL).rstrip("/")
        return f"{base}/auth/userinfo"

    @property
    def RESOLVED_MCP_ALLOWED_HOSTS(self) -> List[str]:
        return list(dict.fromkeys(host.strip().lower() for host in self.MCP_ALLOWED_HOSTS.split(",") if host.strip()))

    @property
    def RESOLVED_MCP_ALLOWED_CREDENTIAL_REFS(self) -> List[str]:
        return list(dict.fromkeys(ref.strip() for ref in self.MCP_ALLOWED_CREDENTIAL_REFS.split(",") if ref.strip()))

    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_secret_key")
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_NAME: str = "localhost"
    SERVER_HOST: str = os.getenv("SERVER_HOST", "http://localhost:8000")
    PROJECT_NAME: str = "Fusion API"
    SENTRY_DSN: Optional[str] = None

    # Moonshot (Kimi) — 用于 $web_search 生成动态示例问题
    MOONSHOT_API_KEY: Optional[str] = os.getenv("MOONSHOT_API_KEY")

    # 搜索服务地址（容器间通过 Docker 内网访问）
    SEARCH_SERVICE_URL: str = os.getenv("SEARCH_SERVICE_URL", "http://search-service:8080")

    # 网页读取服务地址
    READER_SERVICE_URL: str = os.getenv("READER_SERVICE_URL", "http://reader-service:8091")
    # reader-service 内部默认等待 Jina 10s；调用方保留更长余量，避免冷抓取或文档站读取提前断开。
    READER_SERVICE_TIMEOUT: float = float(os.getenv("READER_SERVICE_TIMEOUT", "20"))

    # 远程 MCP Client：仅允许精确配置的 HTTPS 主机和凭证环境变量引用。
    MCP_ALLOWED_HOSTS: str = os.getenv(
        "MCP_ALLOWED_HOSTS",
        "learn.microsoft.com,dashscope.aliyuncs.com,mcp.amap.com",
    )
    MCP_ALLOWED_CREDENTIAL_REFS: str = os.getenv(
        "MCP_ALLOWED_CREDENTIAL_REFS",
        "DASHSCOPE_API_KEY,AMAP_MCP_API_KEY",
    )
    MCP_CONNECT_TIMEOUT_SECONDS: float = float(os.getenv("MCP_CONNECT_TIMEOUT_SECONDS", "5"))
    MCP_CALL_TIMEOUT_SECONDS: float = float(os.getenv("MCP_CALL_TIMEOUT_SECONDS", "15"))
    MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS: float = float(os.getenv("MCP_IDEMPOTENT_TOTAL_TIMEOUT_SECONDS", "12"))
    MCP_ADMIN_OPERATION_TIMEOUT_SECONDS: int = int(os.getenv("MCP_ADMIN_OPERATION_TIMEOUT_SECONDS", "35"))
    MCP_MAX_DISCOVERY_PAGES: int = int(os.getenv("MCP_MAX_DISCOVERY_PAGES", "5"))
    MCP_MAX_DISCOVERED_TOOLS: int = int(os.getenv("MCP_MAX_DISCOVERED_TOOLS", "50"))
    MCP_MAX_TOOL_DESCRIPTION_CHARS: int = int(os.getenv("MCP_MAX_TOOL_DESCRIPTION_CHARS", "2000"))
    MCP_MAX_TOOL_SCHEMA_BYTES: int = int(os.getenv("MCP_MAX_TOOL_SCHEMA_BYTES", "32768"))
    MCP_MAX_RESPONSE_BYTES: int = int(os.getenv("MCP_MAX_RESPONSE_BYTES", "262144"))
    MCP_MAX_TOOL_CALLS_PER_SERVER_PER_RUN: int = int(os.getenv("MCP_MAX_TOOL_CALLS_PER_SERVER_PER_RUN", "8"))
    MCP_SERVER_CIRCUIT_FAILURE_THRESHOLD: int = int(os.getenv("MCP_SERVER_CIRCUIT_FAILURE_THRESHOLD", "3"))
    MCP_SERVER_CIRCUIT_COOLDOWN_SECONDS: float = float(os.getenv("MCP_SERVER_CIRCUIT_COOLDOWN_SECONDS", "30"))

    # PromptHub bundle 后台同步；disabled 不发出请求，聊天热路径始终只读本地 LKG。
    PROMPTHUB_SYNC_MODE: str = os.getenv("PROMPTHUB_SYNC_MODE", "disabled").lower()
    PROMPTHUB_BASE_URL: str = os.getenv("PROMPTHUB_BASE_URL", "")
    PROMPTHUB_API_KEY: str = os.getenv("PROMPTHUB_API_KEY", "")
    PROMPTHUB_PROJECT_SLUG: str = os.getenv("PROMPTHUB_PROJECT_SLUG", "fusion")
    PROMPTHUB_REQUEST_TIMEOUT_SECONDS: float = float(os.getenv("PROMPTHUB_REQUEST_TIMEOUT_SECONDS", "3"))
    PROMPTHUB_SYNC_INTERVAL_SECONDS: int = int(os.getenv("PROMPTHUB_SYNC_INTERVAL_SECONDS", "300"))
    PROMPTHUB_SYNC_ON_STARTUP: bool = os.getenv("PROMPTHUB_SYNC_ON_STARTUP", "true").lower() == "true"

    # 文档站点开关（生产关掉减少攻击面）
    ENABLE_DOCS: bool = True

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
