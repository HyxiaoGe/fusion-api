"""
API 层依赖注入工厂

所有 Service / Repository 实例通过这里的工厂函数创建，
路由层通过 Depends() 注入，不再手动 new。
"""

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user  # noqa: F401 — re-export
from app.db.database import get_db  # noqa: F401 — re-export
from app.db.repositories import (
    ModelCredentialRepository,
    ModelSourceRepository,
    ProviderRepository,
)
from app.services.chat_service import ChatService
from app.services.file_service import FileService
from app.services.user_memory_service import UserMemoryService


def get_chat_service(db: Session = Depends(get_db)) -> ChatService:
    return ChatService(db)


def get_file_service(db: Session = Depends(get_db)) -> FileService:
    return FileService(db)


def get_user_memory_service(db: Session = Depends(get_db)) -> UserMemoryService:
    return UserMemoryService(db)


def get_provider_repo(db: Session = Depends(get_db)) -> ProviderRepository:
    return ProviderRepository(db)


def get_model_source_repo(db: Session = Depends(get_db)) -> ModelSourceRepository:
    return ModelSourceRepository(db)


def get_model_credential_repo(db: Session = Depends(get_db)) -> ModelCredentialRepository:
    return ModelCredentialRepository(db)
