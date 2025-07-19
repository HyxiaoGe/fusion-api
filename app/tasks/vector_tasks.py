"""
向量化任务

定期将RSS数据向量化并存储到向量数据库
"""

import asyncio
from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.db.database import SessionLocal
from app.services.vector_service import VectorService
from app.services.digest_service import DigestService
from app.db.repositories import HotTopicRepository


class VectorTasks:
    """向量化相关的定时任务"""
    
    @staticmethod
    async def vectorize_new_topics():
        """向量化新的热点话题"""
        db = SessionLocal()
        try:
            vector_service = VectorService(db)
            hot_topic_repo = HotTopicRepository(db)
            
            # 获取最近1小时内的新话题
            one_hour_ago = datetime.now() - timedelta(hours=1)
            new_topics = hot_topic_repo.get_topics_after(one_hour_ago)
            
            logger.info(f"发现 {len(new_topics)} 个新话题需要向量化")
            
            # 逐个向量化
            for topic in new_topics:
                await vector_service.index_hot_topic(topic)
                await asyncio.sleep(0.1)  # 避免请求过快
            
            logger.info("新话题向量化完成")
            
        except Exception as e:
            logger.error(f"向量化新话题失败: {e}")
        finally:
            db.close()
    
    @staticmethod
    async def update_vector_indices():
        """更新向量索引（处理更新的内容）"""
        db = SessionLocal()
        try:
            vector_service = VectorService(db)
            hot_topic_repo = HotTopicRepository(db)
            
            # 获取最近更新的话题
            one_day_ago = datetime.now() - timedelta(days=1)
            updated_topics = hot_topic_repo.get_updated_topics_after(one_day_ago)
            
            logger.info(f"发现 {len(updated_topics)} 个话题需要更新向量")
            
            for topic in updated_topics:
                await vector_service.index_hot_topic(topic)
                await asyncio.sleep(0.1)
                
            logger.info("向量索引更新完成")
            
        except Exception as e:
            logger.error(f"更新向量索引失败: {e}")
        finally:
            db.close()
    
    @staticmethod
    async def deduplicate_topics():
        """去重任务：查找并标记重复的话题"""
        db = SessionLocal()
        try:
            vector_service = VectorService(db)
            hot_topic_repo = HotTopicRepository(db)
            
            # 获取最近的话题
            recent_topics = hot_topic_repo.get_recent_topics(days=1, limit=100)
            
            duplicate_count = 0
            for topic in recent_topics:
                duplicate_ids = await vector_service.find_duplicate_topics(topic, threshold=0.95)
                if duplicate_ids:
                    logger.info(f"发现重复话题: {topic.title} -> {len(duplicate_ids)} 个重复")
                    duplicate_count += 1
                    # 这里可以添加标记或删除重复话题的逻辑
            
            logger.info(f"去重任务完成，发现 {duplicate_count} 个重复话题")
            
        except Exception as e:
            logger.error(f"去重任务失败: {e}")
        finally:
            db.close()
    
    @staticmethod
    async def generate_daily_trends():
        """生成每日趋势分析"""
        db = SessionLocal()
        try:
            vector_service = VectorService(db)
            
            # 分析最近7天的趋势
            trends = await vector_service.analyze_topic_trends(days=7)
            
            logger.info(f"趋势分析完成: {trends}")
            
            # 这里可以将趋势数据保存到数据库或发送报告
            
        except Exception as e:
            logger.error(f"生成趋势分析失败: {e}")
        finally:
            db.close()
    
    @staticmethod
    async def initial_batch_index():
        """初始批量索引（首次运行时使用）"""
        db = SessionLocal()
        try:
            vector_service = VectorService(db)
            
            logger.info("开始初始批量索引...")
            await vector_service.batch_index_hot_topics(limit=1000)
            logger.info("初始批量索引完成")
            
        except Exception as e:
            logger.error(f"初始批量索引失败: {e}")
        finally:
            db.close()
    
    @staticmethod
    async def generate_daily_digest():
        """生成每日话题摘要"""
        db = SessionLocal()
        try:
            digest_service = DigestService(db)
            
            # 生成今天的摘要
            today = date.today()
            logger.info(f"开始生成 {today} 的话题摘要")
            
            digests = await digest_service.generate_daily_digest(today)
            
            logger.info(f"成功生成 {len(digests)} 个话题摘要")
            
        except Exception as e:
            logger.error(f"生成每日摘要失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
        finally:
            db.close()
    
    @staticmethod
    async def cleanup_old_digests(days_to_keep: int = 30):
        """清理旧的摘要数据"""
        db = SessionLocal()
        try:
            from app.db.models import DailyTopicDigest
            
            cutoff_date = date.today() - timedelta(days=days_to_keep)
            
            # 删除旧的摘要
            deleted = db.query(DailyTopicDigest).filter(
                DailyTopicDigest.date < cutoff_date
            ).delete()
            
            db.commit()
            
            logger.info(f"清理了 {deleted} 个旧的话题摘要")
            
        except Exception as e:
            db.rollback()
            logger.error(f"清理旧摘要失败: {e}")
        finally:
            db.close()