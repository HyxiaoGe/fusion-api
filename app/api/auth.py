from fastapi import APIRouter, Depends, Request

from app.core import security
from app.db.models import User as UserModel
from app.schemas.response import success

router = APIRouter()


@router.get("/me")
async def read_current_user(request: Request, current_user: UserModel = Depends(security.get_current_user)):
    user_data = {
        "id": current_user.id,
        "username": current_user.username,
        "nickname": current_user.nickname,
        "avatar": current_user.avatar,
    }
    return success(data=user_data, request_id=request.state.request_id)
