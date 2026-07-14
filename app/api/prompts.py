# app/api/prompts.py
from fastapi import APIRouter, Query, Request

from app.schemas.response import success
from app.services.prompt_catalog_service import get_home_prompt_catalog
from app.services.prompt_examples_service import get_prompt_examples

router = APIRouter()


@router.get("/templates")
async def get_templates(request: Request):
    """获取首页任务卡和系统提示词模板目录。"""

    return success(data=get_home_prompt_catalog(), request_id=request.state.request_id)


@router.get("/examples")
async def get_examples(
    request: Request,
    limit: int = Query(default=8, ge=1, le=50, description="返回数量，传大值可获取全量池"),
):
    """获取动态示例问题（无需鉴权，供首页展示）"""
    data = await get_prompt_examples(limit)
    return success(data=data, request_id=request.state.request_id)
