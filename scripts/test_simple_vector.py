#!/usr/bin/env python3
"""
简单测试向量插入
"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.database import SessionLocal
from app.services.vector_service import VectorService
from app.core.logger import app_logger as logger


async def main():
    """主函数"""
    logger.info("开始简单向量测试...")
    
    db = SessionLocal()
    try:
        vector_service = VectorService(db)
        
        # 获取一个话题
        from app.db.models import HotTopic
        topic = db.query(HotTopic).first()
        
        if topic:
            logger.info(f"找到话题: {topic.title}")
            
            # 尝试向量化并插入
            await vector_service.index_hot_topic(topic)
            
            # 检查集合状态
            collection_info = vector_service.client.get_collection(vector_service.collection_name)
            logger.info(f"向量数量: {collection_info.points_count}")
        else:
            logger.error("没有找到话题数据")
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db.close()


if __name__ == "__main__":
    asyncio.run(main())