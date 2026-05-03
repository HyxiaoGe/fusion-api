from typing import Optional

from pydantic import BaseModel, Field


class Token(BaseModel):
    access_token: str
    token_type: str


class User(BaseModel):
    id: str
    username: str
    email: Optional[str] = None
    nickname: Optional[str] = None
    avatar: Optional[str] = None
    mobile: Optional[str] = None

    class Config:
        from_attributes = True


class UserSettingsUpdate(BaseModel):
    """用户个性化设置更新请求"""

    system_prompt: str = Field(..., max_length=1000, description="用户自定义 AI 个性化提示词")
