from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class TaskBase(BaseModel):
    name: str
    description: Optional[str] = None
    status: str
    interval: int
    last_run_at: Optional[datetime] = None
    next_run_at: Optional[datetime] = None

class TaskResponse(TaskBase):
    id: str
    created_at: datetime
    updated_at: datetime
    task_data: Optional[Dict[str, Any]] = None
    
    class Config:
        from_attributes = True

class TaskUpdateRequest(BaseModel):
    description: Optional[str] = None
    status: Optional[str] = None
    interval: Optional[int] = None