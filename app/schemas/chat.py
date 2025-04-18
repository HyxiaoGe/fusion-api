from datetime import datetime
from typing import List, Optional, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    role: str  # user 或 assistant
    content: str
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
