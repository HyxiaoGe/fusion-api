from datetime import datetime
from typing import Optional, Dict, Any

from pydantic import BaseModel


class FileCreate(BaseModel):
    filename: str
    original_filename: str
    mimetype: str
    size: int
    path: str
    status: str = "pending"
    processing_result: Optional[Dict[str, Any]] = None


class FileResponse(BaseModel):
    id: str
    filename: str
    original_filename: str
    mimetype: str
    size: int
    status: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ConversationFileResponse(BaseModel):
    conversation_id: str
    file_id: str
    file: FileResponse
    created_at: datetime

    class Config:
        from_attributes = True
