from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class HotTopicResponse(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    source: str
    category: Optional[str] = None
    url: Optional[str] = None
    published_at: Optional[datetime] = None
    created_at: datetime
    view_count: int = 0
    
    class Config:
        from_attributes = True