import asyncio
from datetime import datetime
from typing import Dict, Any, List

from app.core.logger import app_logger as logger
from app.services.hot_topic_service import HotTopicService

async def hot_topics_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    热点话题函数处理器
    
    参数:
        args: 函数参数，包含:
            - category: 类别 (可选)
            - limit: 返回结果数量 (可选，默认10)
            - topic_id: 特定话题ID (可选)
        context: 上下文信息，包含数据库连接
        
    返回:
        热点话题结果
    """
    try:
        # 提取参数
        category = args.get("category")
        limit = int(args.get("limit", 10))
        topic_id = args.get("topic_id")
        
        # 获取数据库连接
        db = context.get("db")
        if not db:
            return {"error": "数据库连接未提供"}
            
        # 创建热点话题服务
        hot_topic_service = HotTopicService(db)
        
        # 如果提供了特定话题ID
        if topic_id:
            logger.info(f"获取特定热点话题: {topic_id}")
            topic = hot_topic_service.get_topic_by_id(topic_id)
            
            if not topic:
                return {"error": f"话题不存在: {topic_id}"}
                
            # 增加浏览计数
            hot_topic_service.increment_view_count(topic_id)
            
            return {
                "id": topic.id,
                "title": topic.title,
                "description": topic.description,
                "source": topic.source,
                "category": topic.category,
                "url": topic.url,
                "published_at": topic.published_at.isoformat() if topic.published_at else None,
                "view_count": topic.view_count
            }
        
        # 获取热点话题列表
        logger.info(f"获取热点话题列表，类别: {category or '全部'}, 限制: {limit}")
        topics = hot_topic_service.get_hot_topics(category=category, limit=limit)
        
        # 格式化返回结果
        formatted_topics = []
        for topic in topics:
            formatted_topics.append({
                "id": topic.id,
                "title": topic.title,
                "description": topic.description[:100] + "..." if topic.description and len(topic.description) > 100 else topic.description,
                "source": topic.source,
                "category": topic.category,
                "url": topic.url,
                "published_at": topic.published_at.isoformat() if topic.published_at else None,
                "view_count": topic.view_count
            })
        
        return {
            "topics": formatted_topics,
            "count": len(formatted_topics),
            "category": category,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"热点话题处理器出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"获取热点话题失败: {str(e)}"}