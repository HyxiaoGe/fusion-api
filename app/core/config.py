from pydantic_settings import BaseSettings
import os
from typing import Dict, List, Optional


class Settings(BaseSettings):
    APP_NAME: str = "AI桌面聊天应用"
    APP_VERSION: str = "0.1.0"

    # API密钥配置
    WENXIN_API_KEY: Optional[str] = None
    WENXIN_SECRET_KEY: Optional[str] = None
    QIANWEN_API_KEY: Optional[str] = None
    CLAUDE_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    OPENAI_API_KEY: Optional[str] = None

    # 模型配置
    DEFAULT_MODEL: str = "deepseek"  # 默认使用的模型

    # 数据库配置
    DATABASE_URL: str = "sqlite:///./chat_app.db"

    # Redis配置（可选）
    REDIS_HOST: Optional[str] = None
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_USE_SSL: bool = False

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()