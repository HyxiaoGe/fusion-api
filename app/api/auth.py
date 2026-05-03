from fastapi import APIRouter, Depends, Request

from app.api.deps import get_user_repo
from app.core import security
from app.db.models import User as UserModel
from app.db.repositories import UserRepository
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
    user_repo: UserRepository = Depends(get_user_repo),
):
    """更新当前用户的个性化设置（system_prompt）"""
    updated = user_repo.update_system_prompt(current_user, body.system_prompt)
    return success(
        data={"system_prompt": updated.system_prompt},
        request_id=request.state.request_id,
    )
