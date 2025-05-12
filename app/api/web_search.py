from fastapi import APIRouter, Query, Depends
from app.services.web_search_service import WebSearchService
from sqlalchemy.orm import Session
from app.db.database import get_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/")
async def search_web(
    query: str = Query(..., description="搜索查询文本"),
    limit: int = Query(10, description="返回结果数量限制", ge=1, le=20),
    db: Session = Depends(get_db)
):
    """执行网络搜索"""
    try:
        search_service = WebSearchService()
        results = await search_service.search(query, limit)

        if not results:
            return {"status": "empty", "message": "未找到相关结果", "results": []}
        
        return {"status": "success", "message": "搜索成功", "results": results}
    except Exception as e:
        logger.error(f"网络搜索失败: {e}")
        return {"status": "error", "message": "搜索失败", "results": []}

