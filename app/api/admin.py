"""管理员专用端点（需 is_superuser=true）"""

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.api.deps import get_current_admin_user, get_provider_repo
from app.db.database import get_db
from app.db.models import User as UserModel
from app.db.repositories import ProviderRepository
from app.schemas.response import success
from app.services.provider_health import ProviderHealthService

router = APIRouter()


@router.post("/providers/{provider_id}/recover")
async def recover_provider(
    provider_id: str,
    request: Request,
    admin: UserModel = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    provider_repo: ProviderRepository = Depends(get_provider_repo),
):
    """admin 手动恢复离线 provider：重置 status / consecutive_failures / offline_*"""
    if not provider_repo.get_by_id(provider_id):
        raise HTTPException(status_code=404, detail=f"未知 provider: {provider_id}")
    ProviderHealthService(db).manual_recover(provider_id, by_user_id=admin.id)
    return success(data={"recovered": True, "provider_id": provider_id}, request_id=request.state.request_id)
