from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_current_admin_user
from app.db.models import User as UserModel
from app.schemas.response import success
from app.services.external import search_usage_client
from app.services.external.search_usage_client import SearchUsageClientError

router = APIRouter()


@router.get("/search-usage")
async def get_search_usage(_admin: UserModel = Depends(get_current_admin_user)):
    try:
        firecrawl_usage = await search_usage_client.get_firecrawl_usage()
        firecrawl_historical = await search_usage_client.get_firecrawl_historical_usage()
    except SearchUsageClientError as exc:
        raise HTTPException(status_code=502, detail="联网用量查询失败") from exc

    return success(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "providers": [
                {"provider": "firecrawl", "official_usage": True},
                {"provider": "brave", "official_usage": False},
            ],
            "firecrawl": firecrawl_usage,
            "historical": firecrawl_historical,
        }
    )
