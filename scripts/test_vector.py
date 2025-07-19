#!/usr/bin/env python3
"""
测试向量化功能
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
    logger.info("开始测试向量化功能...")
    
    try:
        # 执行批量索引
        logger.info("执行批量索引...")
        await VectorTasks.initial_batch_index()
        
        logger.info("向量化测试完成！")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())