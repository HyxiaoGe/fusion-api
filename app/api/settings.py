from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from app.db.database import get_db
from app.core.config import settings
from app.db.repositories import SettingRepository

router = APIRouter()

@router.get("/")
def get_settings(db: Session = Depends(get_db)):
    """获取应用程序当前设置"""
    repo = SettingRepository(db=db)
    app_settings = {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "default_model": settings.DEFAULT_MODEL
    }

    user_settings = repo.get("user_settings")
    if user_settings:
        app_settings.update(user_settings)

    return app_settings


@router.post("/")
def update_settings(settings_data: Dict[str, Any], db: Session = Depends(get_db)):
    """更新应用程序设置"""
    repo = SettingRepository(db)
    success = repo.set("user_settings", settings_data)

    if not success:
        raise HTTPException(status_code=500, detail="设置更新失败")

    return {"status": "success", "message": "设置已更新"}