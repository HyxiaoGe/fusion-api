"""
向量数据库服务

用于处理RSS数据的向量化存储和检索
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import hashlib

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    MatchValue
)
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.db.models import HotTopic
from app.db.repositories import HotTopicRepository


class VectorService:
    """向量数据库服务"""
    
    def __init__(self, db: Session):
        self.db = db
        self.hot_topic_repo = HotTopicRepository(db)
        
        # 初始化 Qdrant 客户端 - 支持本地和云端部署
        qdrant_url = os.getenv("QDRANT_URL")
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if qdrant_url:
            # 云端部署使用URL和API Key
            logger.info(f"连接Qdrant Cloud: {qdrant_url}")
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
            )
        else:
            # 本地部署使用host+port
            qdrant_host = os.getenv("QDRANT_HOST", "fusion-qdrant")
            logger.info(f"连接本地Qdrant: {qdrant_host}:6333")
            self.client = QdrantClient(
                host=qdrant_host,
                port=6333,
            )
        
        # 初始化 Embedding 模型
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="text-embedding-3-small"  # 或使用其他模型
        )
        
        # 集合名称
        self.collection_name = "hot_topics"
        
        # 确保集合存在
        self._ensure_collection_exists()
    
    def _ensure_collection_exists(self):
        """确保向量集合存在"""
        try:
            collections = self.client.get_collections().collections
            if not any(col.name == self.collection_name for col in collections):
                # 创建集合
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=1536,  # OpenAI embedding 维度
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"创建向量集合: {self.collection_name}")
        except Exception as e:
            logger.error(f"检查/创建集合失败: {e}")
    
    def _generate_vector_id(self, hot_topic_id: str) -> str:
        """生成向量ID"""
        return hashlib.md5(hot_topic_id.encode()).hexdigest()
    
    def _prepare_text_for_embedding(self, hot_topic: HotTopic) -> str:
        """准备用于嵌入的文本"""
        # 组合标题、描述和其他相关信息
        parts = [
            f"标题: {hot_topic.title}",
            f"描述: {hot_topic.description or ''}",
            f"来源: {hot_topic.source}",
            f"分类: {hot_topic.category or '未分类'}"
        ]
        return "\n".join(parts)
    
    async def index_hot_topic(self, hot_topic: HotTopic):
        """将单个热点话题索引到向量数据库"""
        try:
            # 准备文本
            text = self._prepare_text_for_embedding(hot_topic)
            
            # 生成向量
            embedding = await self.embeddings.aembed_query(text)
            
            # 准备元数据
            payload = {
                "hot_topic_id": hot_topic.id,
                "title": hot_topic.title,
                "description": hot_topic.description,
                "source": hot_topic.source,
                "category": hot_topic.category or "未分类",
                "url": hot_topic.url,
                "published_at": hot_topic.published_at.isoformat() if hot_topic.published_at else None,
                "created_at": hot_topic.created_at.isoformat(),
                "view_count": hot_topic.view_count
            }
            
            # 创建点
            point = PointStruct(
                id=self._generate_vector_id(hot_topic.id),
                vector=embedding,
                payload=payload
            )
            
            # 插入到向量数据库
            self.client.upsert(
                collection_name=self.collection_name,
                points=[point]
            )
            
            logger.info(f"成功索引热点话题: {hot_topic.title}")
            
        except Exception as e:
            logger.error(f"索引热点话题失败: {e}")
    
    async def batch_index_hot_topics(self, limit: int = 100):
        """批量索引热点话题"""
        try:
            # 获取最近的热点话题
            hot_topics = self.hot_topic_repo.get_recent_topics(days=30, limit=limit)
            
            # 分批处理，避免速率限制
            batch_size = 20
            for i in range(0, len(hot_topics), batch_size):
                batch = hot_topics[i:i + batch_size]
                points = []
                
                for hot_topic in batch:
                    try:
                        text = self._prepare_text_for_embedding(hot_topic)
                        embedding = await self.embeddings.aembed_query(text)
                        
                        payload = {
                            "hot_topic_id": hot_topic.id,
                            "title": hot_topic.title,
                            "description": hot_topic.description,
                            "source": hot_topic.source,
                            "category": hot_topic.category or "未分类",
                            "url": hot_topic.url,
                            "published_at": hot_topic.published_at.isoformat() if hot_topic.published_at else None,
                            "created_at": hot_topic.created_at.isoformat(),
                            "view_count": hot_topic.view_count
                        }
                        
                        point = PointStruct(
                            id=self._generate_vector_id(hot_topic.id),
                            vector=embedding,
                            payload=payload
                        )
                        points.append(point)
                        
                        # 添加延迟避免速率限制
                        import asyncio
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        logger.error(f"处理话题失败 {hot_topic.title}: {e}")
                        continue
                
                # 批量插入
                if points:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True
                    )
                    logger.info(f"成功批量索引 {len(points)} 个热点话题（批次 {i//batch_size + 1}）")
                    
                # 批次间延迟
                if i + batch_size < len(hot_topics):
                    await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"批量索引失败: {e}")
    
    async def search_similar_topics(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """搜索相似的热点话题"""
        try:
            # 生成查询向量
            query_embedding = await self.embeddings.aembed_query(query)
            
            # 搜索
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=limit
            )
            
            # 格式化结果
            similar_topics = []
            for result in results:
                similar_topics.append({
                    "score": result.score,
                    "hot_topic_id": result.payload["hot_topic_id"],
                    "title": result.payload["title"],
                    "description": result.payload["description"],
                    "source": result.payload["source"],
                    "category": result.payload["category"],
                    "url": result.payload["url"],
                    "published_at": result.payload["published_at"]
                })
            
            return similar_topics
            
        except Exception as e:
            logger.error(f"搜索相似话题失败: {e}")
            return []
    
    async def find_duplicate_topics(self, hot_topic: HotTopic, threshold: float = 0.9) -> List[str]:
        """查找重复的话题"""
        try:
            text = self._prepare_text_for_embedding(hot_topic)
            embedding = await self.embeddings.aembed_query(text)
            
            # 搜索高相似度的内容
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=5,
                score_threshold=threshold
            )
            
            # 排除自己
            duplicate_ids = []
            for result in results:
                if result.payload["hot_topic_id"] != hot_topic.id:
                    duplicate_ids.append(result.payload["hot_topic_id"])
            
            return duplicate_ids
            
        except Exception as e:
            logger.error(f"查找重复话题失败: {e}")
            return []
    
    async def analyze_topic_trends(self, days: int = 7) -> Dict[str, Any]:
        """分析话题趋势"""
        try:
            # 获取指定天数内的所有向量
            # 这里可以实现聚类算法来发现热点趋势
            # 简化版本：按分类统计
            
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="published_at",
                        range=Range(
                            gte=(datetime.now() - timedelta(days=days)).isoformat()
                        )
                    )
                ]
            )
            
            # 获取所有符合条件的点
            results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=1000
            )
            
            # 统计分析
            category_counts = {}
            source_counts = {}
            
            for point in results[0]:
                category = point.payload.get("category", "未分类")
                source = point.payload.get("source", "未知")
                
                category_counts[category] = category_counts.get(category, 0) + 1
                source_counts[source] = source_counts.get(source, 0) + 1
            
            return {
                "total_topics": len(results[0]),
                "category_distribution": category_counts,
                "source_distribution": source_counts,
                "time_range_days": days
            }
            
        except Exception as e:
            logger.error(f"分析话题趋势失败: {e}")
            return {}