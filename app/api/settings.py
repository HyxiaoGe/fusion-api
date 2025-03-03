from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any
from app.db.database import get_db
from app.core.config import settings

router = APIRouter()

@router.get("/")
def get_settings(db: Session = Depends(get_db)):
    """获取应用程序当前设置"""
    # 这里可以从数据库或配置中获取设置信息
    return {
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "default_model": settings.DEFAULT_MODEL,
        # 不要返回敏感信息如API密钥
    }

@router.post("/")
def update_settings(settings_data: Dict[str, Any], db: Session = Depends(get_db)):
    """更新应用程序设置"""
    # 在实际应用中，你需要实现更新逻辑
    # 例如保存到数据库或更新配置文件
    return {"status": "success", "message": "设置已更新"}