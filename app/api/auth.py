from fastapi import APIRouter, Depends

from app.core import security
from app.db.models import User as UserModel
from app.schemas.auth import User as UserSchema

router = APIRouter()


@router.get("/me", response_model=UserSchema)
async def read_current_user(current_user: UserModel = Depends(security.get_current_user)):
    return current_user
