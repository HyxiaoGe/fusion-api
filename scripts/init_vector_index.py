#!/usr/bin/env python3
"""
初始化向量索引脚本
"""

import asyncio
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.tasks.vector_tasks import VectorTasks
from app.core.logger import app_logger as logger


async def main():
    """主函数"""
    logger.info("开始执行向量索引初始化...")
    
    try:
        # 1. 执行初始批量索引
        logger.info("步骤 1: 执行初始批量索引")
        await VectorTasks.initial_batch_index()
        
        # 2. 向量化新的话题
        logger.info("步骤 2: 向量化最近的新话题")
        await VectorTasks.vectorize_new_topics()
        
        # 3. 生成今日摘要
        logger.info("步骤 3: 生成今日话题摘要")
        await VectorTasks.generate_daily_digest()
        
        logger.info("向量索引初始化完成！")
        
    except Exception as e:
        logger.error(f"向量索引初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())