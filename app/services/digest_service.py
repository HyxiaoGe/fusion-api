"""
每日话题聚合服务

生成每日热点话题的聚合摘要
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.core.logger import app_logger as logger
from app.db.models import HotTopic, DailyTopicDigest, TopicClusterItem
from app.db.repositories import HotTopicRepository
from app.services.vector_service import VectorService
from app.ai.llm_manager import llm_manager


class DigestService:
    """话题聚合服务"""
    
    # 预定义的分类
    CATEGORIES = {
        "AI": ["AI", "人工智能", "机器学习", "深度学习", "GPT", "大模型", "算法"],
        "科技": ["科技", "技术", "互联网", "软件", "硬件", "手机", "电脑"],
        "财经": ["财经", "金融", "股票", "投资", "经济", "市场", "货币"],
        "社会": ["社会", "民生", "教育", "医疗", "就业", "房价"],
        "其他": []  # 默认分类
    }
    
    def __init__(self, db: Session):
        self.db = db
        self.hot_topic_repo = HotTopicRepository(db)
        self.vector_service = VectorService(db)
    
    def _categorize_topic(self, topic: HotTopic) -> str:
        """根据标题和描述对话题进行分类"""
        text = f"{topic.title} {topic.description or ''} {topic.category or ''}".lower()
        
        for category, keywords in self.CATEGORIES.items():
            if category == "其他":
                continue
            for keyword in keywords:
                if keyword.lower() in text:
                    return category
        
        return "其他"
    
    async def generate_daily_digest(self, target_date: Optional[date] = None) -> List[DailyTopicDigest]:
        """生成指定日期的话题聚合摘要"""
        if not target_date:
            target_date = date.today()
        
        logger.info(f"开始生成 {target_date} 的话题聚合摘要")
        
        # 获取当天的所有话题
        start_time = datetime.combine(target_date, datetime.min.time())
        end_time = datetime.combine(target_date, datetime.max.time())
        
        topics = self.db.query(HotTopic).filter(
            and_(
                HotTopic.created_at >= start_time,
                HotTopic.created_at < end_time
            )
        ).all()
        
        if not topics:
            logger.info(f"{target_date} 没有话题数据")
            return []
        
        logger.info(f"找到 {len(topics)} 个话题，开始分类和聚类")
        
        # 按类别分组
        categorized_topics = defaultdict(list)
        for topic in topics:
            category = self._categorize_topic(topic)
            categorized_topics[category].append(topic)
        
        # 对每个类别进行聚类
        digests = []
        for category, category_topics in categorized_topics.items():
            if len(category_topics) < 3:  # 话题太少不进行聚类
                continue
            
            logger.info(f"处理 {category} 类别，共 {len(category_topics)} 个话题")
            
            try:
                digest = await self._cluster_and_create_digest(
                    category, category_topics, target_date
                )
                if digest:
                    digests.append(digest)
            except Exception as e:
                logger.error(f"处理 {category} 类别时出错: {e}")
        
        # 保存到数据库
        for digest in digests:
            self.db.add(digest)
        
        self.db.commit()
        logger.info(f"成功生成 {len(digests)} 个话题聚合")
        
        return digests
    
    async def _cluster_and_create_digest(
        self, 
        category: str, 
        topics: List[HotTopic], 
        target_date: date
    ) -> Optional[DailyTopicDigest]:
        """对话题进行聚类并创建摘要"""
        # 获取话题的向量
        embeddings = []
        valid_topics = []
        
        for topic in topics:
            try:
                # 从向量数据库获取向量
                text = self.vector_service._prepare_text_for_embedding(topic)
                embedding = await self.vector_service.embeddings.aembed_query(text)
                embeddings.append(embedding)
                valid_topics.append(topic)
            except Exception as e:
                logger.error(f"获取话题向量失败: {topic.title}, 错误: {e}")
        
        if len(valid_topics) < 3:
            return None
        
        # 延迟导入以避免启动时的依赖问题
        try:
            import numpy as np
            from sklearn.cluster import DBSCAN
        except ImportError:
            logger.error("缺少必要的依赖：numpy 或 sklearn")
            return None
        
        # 使用 DBSCAN 进行聚类
        embeddings_array = np.array(embeddings)
        clustering = DBSCAN(eps=0.3, min_samples=2, metric='cosine').fit(embeddings_array)
        
        # 找到最大的聚类
        labels = clustering.labels_
        unique_labels = set(labels)
        unique_labels.discard(-1)  # 移除噪声点
        
        if not unique_labels:
            # 如果没有聚类，使用所有话题
            cluster_topics = valid_topics
            cluster_embeddings = embeddings_array
        else:
            # 找到最大的聚类
            label_counts = {label: np.sum(labels == label) for label in unique_labels}
            largest_label = max(label_counts, key=label_counts.get)
            
            # 获取该聚类的话题
            cluster_indices = np.where(labels == largest_label)[0]
            cluster_topics = [valid_topics[i] for i in cluster_indices]
            cluster_embeddings = embeddings_array[cluster_indices]
        
        # 计算聚类中心
        cluster_center = np.mean(cluster_embeddings, axis=0)
        
        # 生成聚类标题和摘要
        cluster_title, cluster_summary, key_points = await self._generate_cluster_summary(
            cluster_topics, category
        )
        
        # 计算热度分数
        heat_score = sum(topic.view_count for topic in cluster_topics) / len(cluster_topics)
        
        # 创建摘要记录
        digest = DailyTopicDigest(
            date=target_date,
            category=category,
            cluster_title=cluster_title,
            cluster_summary=cluster_summary,
            key_points=key_points,
            topic_ids=[topic.id for topic in cluster_topics],
            topic_count=len(cluster_topics),
            cluster_vector=cluster_center.tolist(),
            heat_score=float(heat_score),
            display_order=self._get_category_order(category)
        )
        
        return digest
    
    async def _generate_cluster_summary(
        self, 
        topics: List[HotTopic], 
        category: str
    ) -> tuple[str, str, List[str]]:
        """使用 LLM 生成聚类的标题和摘要"""
        # 准备话题列表
        topic_list = "\n".join([
            f"- {topic.title}: {topic.description or '无描述'}"
            for topic in topics[:10]  # 最多使用10个话题
        ])
        
        prompt = f"""
请分析以下{category}领域的相关话题，生成一个总结性的标题、摘要和关键要点。

话题列表：
{topic_list}

请返回以下格式的内容：
标题：[一个吸引人的总结性标题，15字以内]
摘要：[简要概括这些话题的共同主题和重要性，50字以内]
要点：
1. [第一个关键要点]
2. [第二个关键要点]
3. [第三个关键要点]
"""
        
        try:
            llm = llm_manager.get_default_model()
            response = await llm.ainvoke(prompt)
            
            # 解析响应
            lines = response.content.strip().split('\n')
            title = "今日热点"
            summary = ""
            key_points = []
            
            for line in lines:
                if line.startswith("标题："):
                    title = line.replace("标题：", "").strip()
                elif line.startswith("摘要："):
                    summary = line.replace("摘要：", "").strip()
                elif line.strip().startswith(("1.", "2.", "3.")):
                    point = line.strip()[2:].strip()
                    if point:
                        key_points.append(point)
            
            return title, summary, key_points[:3]  # 最多3个要点
            
        except Exception as e:
            logger.error(f"生成摘要失败: {e}")
            # 返回默认值
            return (
                f"{category}领域热点",
                f"今日{category}领域共有{len(topics)}个相关话题",
                [topic.title for topic in topics[:3]]
            )
    
    def _get_category_order(self, category: str) -> int:
        """获取分类的显示顺序"""
        order_map = {
            "AI": 1,
            "科技": 2,
            "财经": 3,
            "社会": 4,
            "其他": 99
        }
        return order_map.get(category, 50)
    
    def get_daily_digests(self, target_date: Optional[date] = None) -> List[DailyTopicDigest]:
        """获取指定日期的话题摘要"""
        if not target_date:
            target_date = date.today()
        
        return self.db.query(DailyTopicDigest).filter(
            DailyTopicDigest.date == target_date
        ).order_by(DailyTopicDigest.display_order).all()
    
    def increment_digest_view_count(self, digest_id: str) -> bool:
        """增加摘要的查看次数"""
        try:
            digest = self.db.query(DailyTopicDigest).filter(
                DailyTopicDigest.id == digest_id
            ).first()
            
            if digest:
                digest.view_count += 1
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"增加查看次数失败: {e}")
            return False