import httpx
import logging
import asyncio
from typing import List, Dict, Any
from app.core.config import settings

logger = logging.getLogger(__name__)

class WebSearchService:
    """网络搜素服务 - 提供互联网搜索功能"""

    def __init__(self):
        self.api_key = settings.SEARCH_API_KEY
        self.search_endpoint = settings.SEARCH_API_ENDPOINT
        self.max_retries = 3
        self.retry_delay = 2  # 重试间隔秒数
        

    async def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """执行网络搜索
        
        Args:
            query: 搜索查询文本
            limit: 返回结果数量限制
            
        Returns:
            搜索结果列表，每个结果包含标题、摘要、URL等
        """
        try:
            logger.info(f"执行网络搜索: query='{query}', limit={limit}")

            # 构建请求参数
            params = {
                "api_key": self.api_key,
                "q": query,
                "num": limit,
                "engine": "google",
                "safe": "active"
            }

            # 重试机制
            for attempt in range(self.max_retries):
                try:
                    # 发送请求
                    async with httpx.AsyncClient(timeout=60.0) as client:
                        response = await client.get(self.search_endpoint, params=params)
                        
                        if response.status_code != 200:
                            logger.error(f"搜索API返回错误: {response.status_code}, {response.text}")
                            return []
                        
                        # 解析响应
                        results = response.json()
                        
                        print(f"results: {results}")

                        # 解析响应
                        webpages = results.get("organic_results", {})
                        
                        # 提取搜索结果
                        formatted_results = []
                        for page in webpages:
                            formatted_results.append({
                                "title": page.get("title", ""),
                                "snippet": page.get("snippet", ""),
                                "link": page.get("link", ""),
                                "source": page.get("source", "")
                            })

                        logger.info(f"搜索成功，找到 {len(formatted_results)} 个结果")
                        return formatted_results
                        
                except httpx.ReadTimeout:
                    if attempt < self.max_retries - 1:
                        retry_wait = self.retry_delay * (attempt + 1)
                        logger.warning(f"请求超时，{retry_wait}秒后重试 (尝试 {attempt+1}/{self.max_retries})")
                        await asyncio.sleep(retry_wait)
                    else:
                        logger.error(f"请求超时，已达到最大重试次数 {self.max_retries}")
                        raise
                        
        except Exception as e:
            logger.error(f"网络搜索失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
                    
