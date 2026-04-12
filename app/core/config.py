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
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
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

    # 存储后端配置（"local" 或 "minio"）
    STORAGE_BACKEND: str = os.getenv("STORAGE_BACKEND", "local")
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "")
    MINIO_BUCKET: str = os.getenv("MINIO_BUCKET", "fusion-files")
    MINIO_USE_SSL: bool = os.getenv("MINIO_USE_SSL", "false").lower() == "true"
    MINIO_PRESIGN_EXPIRES: int = int(os.getenv("MINIO_PRESIGN_EXPIRES", "3600"))

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
        return self.AUTH_SERVICE_JWKS_URL or f"{self.AUTH_SERVICE_BASE_URL.rstrip('/')}/.well-known/jwks.json"

    @property
    def AUTH_SERVICE_USERINFO_URL(self) -> str:
        return f"{self.AUTH_SERVICE_BASE_URL.rstrip('/')}/auth/userinfo"

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
    READER_SERVICE_URL: str = os.getenv("READER_SERVICE_URL", "http://reader-service:8090")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
