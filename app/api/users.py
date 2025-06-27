from fastapi import APIRouter, Depends
from app.core.security import get_current_user
from app.db.models import User as UserModel
from app.schemas.auth import User as UserSchema

router = APIRouter()


@router.get("/profile", response_model=UserSchema)
async def read_users_profile(current_user: UserModel = Depends(get_current_user)):
    return current_user 