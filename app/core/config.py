from pydantic_settings import BaseSettings
import os
from typing import Dict, List, Optional


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

    # 模型配置
    DEFAULT_MODEL: str = "qwen"  # 默认使用的模型

    # 数据库配置
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://fusion:fusion123!!@localhost:5432/fusion"
    )

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()