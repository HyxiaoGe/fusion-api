"""用户凭证 Pydantic schemas"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserCredentialInfo(BaseModel):
    provider_id: str
    api_key_masked: str
    is_active: bool
    last_error_kind: Optional[str] = None
    last_error_message: Optional[str] = None
    last_failure_at: Optional[datetime] = None
    consecutive_failures: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class UserCredentialUpsert(BaseModel):
    api_key: str = Field(..., min_length=1, max_length=512)
    is_active: bool = True


class UserCredentialTestRequest(BaseModel):
    api_key: Optional[str] = Field(None, max_length=512)


class UserCredentialTestResult(BaseModel):
    valid: bool
    reason: Optional[str] = None
    message: Optional[str] = None
