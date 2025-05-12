import asyncio
from datetime import datetime
from typing import Dict, Any

from app.services.web_search_service import WebSearchService
from app.core.logger import app_logger as logger

# 注意：使用搜索功能时需要使用"今天"、"昨天"、"本周"等词语而不是具体日期
# 不直接引用超过25个单词的内容
# 不复制或翻译歌词
# 不评论响应的合法性

async def web_search_handler(args: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """
    网络搜索函数处理器
    
    参数:
        args: 函数参数，包含:
            - query: 搜索查询文本
            - limit: 返回结果数量 (可选，默认10)
        context: 上下文信息
        
    返回:
        搜索结果
    """
    try:
        # 提取参数
        query = args.get("query")
        if not query:
            return {"error": "搜索查询不能为空"}
            
        limit = int(args.get("limit", 10))
        
        logger.info(f"执行网络搜索: {query}, 限制: {limit}")
        
        # 执行搜索
        search_service = WebSearchService()
        results = await search_service.search(query, limit)
        
        # 格式化返回结果
        formatted_results = []
        for result in results:
            formatted_results.append({
                "title": result.get("title", ""),
                "snippet": result.get("snippet", ""),
                "link": result.get("link", ""),
                "source": result.get("source", "")
            })
        
        return {
            "query": query,
            "results": formatted_results,
            "result_count": len(formatted_results),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"网络搜索处理器出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {"error": f"搜索失败: {str(e)}"}