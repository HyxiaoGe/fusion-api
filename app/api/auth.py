from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session

from app.core import security
from app.db.database import get_db
from app.db.models import User as UserModel
from app.schemas.auth import UserSettingsUpdate
from app.schemas.response import success

router = APIRouter()


@router.get("/me")
async def read_current_user(request: Request, current_user: UserModel = Depends(security.get_current_user)):
    user_data = {
        "id": current_user.id,
        "username": current_user.username,
        "nickname": current_user.nickname,
        "avatar": current_user.avatar,
        "system_prompt": current_user.system_prompt or "",
    }
    return success(data=user_data, request_id=request.state.request_id)


@router.patch("/me")
async def update_current_user(
    body: UserSettingsUpdate,
    request: Request,
    current_user: UserModel = Depends(security.get_current_user),
    db: Session = Depends(get_db),
):
    """更新当前用户的个性化设置（system_prompt）"""
    current_user.system_prompt = body.system_prompt
    db.commit()
    db.refresh(current_user)
    return success(
        data={"system_prompt": current_user.system_prompt},
        request_id=request.state.request_id,
    )
