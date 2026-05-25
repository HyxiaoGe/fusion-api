"""
API 层依赖注入工厂

所有 Service / Repository 实例通过这里的工厂函数创建，
路由层通过 Depends() 注入，不再手动 new。
"""

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user  # noqa: F401 — re-export
from app.db.database import get_db  # noqa: F401 — re-export
from app.db.models import User as UserModel
from app.db.repositories import UserRepository
from app.services.chat_service import ChatService
from app.services.file_service import FileService


def get_chat_service(db: Session = Depends(get_db)) -> ChatService:
    return ChatService(db)


def get_file_service(db: Session = Depends(get_db)) -> FileService:
    return FileService(db)


def get_user_repo(db: Session = Depends(get_db)) -> UserRepository:
    return UserRepository(db)


def get_current_admin_user(
    current_user: UserModel = Depends(get_current_user),
) -> UserModel:
    """要求 is_superuser=True，否则 403。用于 admin 专属端点。"""
    if not getattr(current_user, "is_superuser", False):
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return current_user
