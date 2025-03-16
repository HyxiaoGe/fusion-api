from pydantic_settings import BaseSettings
import os
from typing import Dict, List, Optional, Any


class Settings(BaseSettings):
    APP_NAME: str = "AI桌面聊天应用"
    APP_VERSION: str = "0.1.0"

    # API密钥配置
    WENXIN_API_KEY: Optional[str] = None
    WENXIN_SECRET_KEY: Optional[str] = None
    QWEN_API_KEY: Optional[str] = None
    # CLAUDE_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    CHROMA_URL: str = os.getenv("CHROMA_URL", "http://localhost:8001")

    # 模型配置
    DEFAULT_MODEL: str = "qwen"  # 默认使用的模型

    # 数据库配置
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://fusion:fusion123!!@localhost:5432/fusion"
    )

    # 文件存储配置
    FILE_STORAGE_PATH: str = os.getenv("FILE_STORAGE_PATH", "./storage/files")
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [
        "image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp",
        "application/pdf",
        "application/msword", "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "text/plain", "text/markdown", "text/csv"
    ]

    # 模型文件支持配置
    MODEL_FILE_SUPPORT: Dict[str, Dict[str, Any]] = {
        "wenxin": {
            "supported_files": ["image/jpeg", "image/png", "image/gif", "image/bmp", "application/pdf"],
            "max_files": 5
        },
        "qwen": {
            "supported_files": ["image/jpeg", "image/png", "image/gif", "image/bmp", "image/webp"],
            "max_files": 5
        },
        "claude": {
            "supported_files": ["image/jpeg", "image/png", "image/gif", "image/webp"],
            "max_files": 5
        },
        "deepseek": {
            "supported_files": [],
            "max_files": 0
        }
    }

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()