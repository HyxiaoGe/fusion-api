"""
话题聚合摘要相关的数据模型
"""

from typing import List, Optional
from datetime import date as DateType
from pydantic import BaseModel, Field


class DigestResponse(BaseModel):
    """话题摘要响应"""
    id: str = Field(..., description="摘要ID")
    category: str = Field(..., description="分类")
    cluster_title: str = Field(..., description="聚类标题")
    cluster_summary: Optional[str] = Field(None, description="聚类摘要")
    key_points: List[str] = Field(default_factory=list, description="关键要点")
    topic_count: int = Field(..., description="话题数量")
    heat_score: float = Field(..., description="热度分数")
    view_count: int = Field(..., description="查看次数")


class DigestListResponse(BaseModel):
    """摘要列表响应"""
    date: DateType = Field(..., description="日期")
    total: int = Field(..., description="摘要总数")
    digests: List[DigestResponse] = Field(..., description="摘要列表")