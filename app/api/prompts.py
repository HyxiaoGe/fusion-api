# app/api/prompts.py
from fastapi import APIRouter, Query

from app.services.prompt_examples_service import get_prompt_examples

router = APIRouter()


@router.get("/examples")
async def get_examples(
    limit: int = Query(default=8, ge=1, le=50, description="返回数量，传大值可获取全量池"),
):
    """获取动态示例问题（无需鉴权，供首页展示）"""
    return await get_prompt_examples(limit)
