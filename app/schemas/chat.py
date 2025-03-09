from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import uuid4, UUID


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # user 或 assistant
    content: str
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    model: str
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False
    options: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    model: str
    message: Message
    conversation_id: str
    created_at: datetime = Field(default_factory=datetime.now)


class Conversation(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    messages: List[Message] = []
    model: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        from_attributes = True

# 添加到现有文件末尾
class TitleGenerationRequest(BaseModel):
    model: str
    message: Optional[str] = None
    conversation_id: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class TitleGenerationResponse(BaseModel):
    title: str
    conversation_id: Optional[str] = None