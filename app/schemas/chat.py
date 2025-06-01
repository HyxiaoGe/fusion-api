from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.constants import MessageRoles, MessageTypes


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["user", "assistant", "system"]  # 限制为标准角色
    type: Literal[
        "user_query", 
        "assistant_content", 
        "reasoning_content", 
        "function_call", 
        "function_result", 
        "web_search", 
        "hot_topics"
    ]  # 限制为预定义的消息类型
    content: str
    duration: int = Field(0, description="处理耗时(毫秒)")  # 默认为0毫秒
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    provider: str
    model: str
    message: str
    conversation_id: Optional[str] = None
    topic_id: Optional[str] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None
    file_ids: Optional[List[str]] = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    model: str
    message: Message
    conversation_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    reasoning: Optional[str] = None
    suggested_questions: Optional[List[str]] = None

class SuggestedQuestionsRequest(BaseModel):
    conversation_id: str
    options: Optional[Dict[str, Any]] = None

class SuggestedQuestionsResponse(BaseModel):
    questions: List[str]
    conversation_id: str

class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    provider: str
    model: str
    title: str
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


# 添加到现有文件末尾
class TitleGenerationRequest(BaseModel):
    message: Optional[str] = None
    conversation_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class TitleGenerationResponse(BaseModel):
    title: str
    conversation_id: Optional[str] = None
