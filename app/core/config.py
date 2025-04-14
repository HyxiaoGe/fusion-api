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
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://fusion:fusion123!!@localhost:5432/fusion"
    )

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

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
