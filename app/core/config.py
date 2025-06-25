import os
from typing import Dict, List, Optional, Any

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

    SEARCH_API_KEY: Optional[str] = None
    SEARCH_API_ENDPOINT: str = "https://www.searchapi.io/api/v1/search"
    BING_API_ENDPOINT: Optional[str] = "https://api.bing.microsoft.com/v7.0/search"
    ENABLE_WEB_SEARCH: bool = True
    
    WEAVIATE_URL: str = os.getenv("WEAVIATE_URL", "http://localhost:8080")

    # 文件存储配置
    FILE_STORAGE_PATH: str = os.getenv("FILE_STORAGE_PATH", "./storage/files")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [
        "image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp",
        "application/pdf",
        "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain", "text/markdown", "text/csv"
    ]

    # Github OAuth
    GITHUB_CLIENT_ID: Optional[str] = None
    GITHUB_CLIENT_SECRET: Optional[str] = None
    FRONTEND_AUTH_CALLBACK_URL: str = "http://localhost:3000/auth/callback"

    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "a_secret_key")
    # 60 minutes * 24 hours * 8 days = 8 days
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8
    SERVER_NAME: str = "localhost"
    SERVER_HOST: str = "http://localhost:8000"
    PROJECT_NAME: str = "Fusion API"
    SENTRY_DSN: Optional[str] = None

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
