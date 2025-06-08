from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class RssSourceBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    url: str = Field(..., min_length=10, max_length=512)
    description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=50)
    is_enabled: bool = True
    filter_apply: Optional[str] = Field(None, max_length=50)
    filter_type: Optional[str] = Field(None, max_length=50)
    filter_rule: Optional[str] = Field(None, max_length=500)

class RssSourceCreate(RssSourceBase):
    pass

class RssSourceUpdate(RssSourceBase):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    url: Optional[str] = Field(None, min_length=10, max_length=512)
    is_enabled: Optional[bool] = None

class RssSourceResponse(RssSourceBase):
    id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True 