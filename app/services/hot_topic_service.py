import feedparser
import asyncio
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.db.models import HotTopic
from app.db.repositories import HotTopicRepository, ScheduledTaskRepository

class HotTopicService:
    """热点话题服务，负责从RSS获取热点数据并定期更新"""

    # 任务名称常量
    TASK_NAME = "rss_hot_topics_update"

    def __init__(self, db: Session):
        self.db = db
        self.repo = HotTopicRepository(db)
        self.task_repo = ScheduledTaskRepository(db)
        # RSS源配置
        self.rss_sources = [
            {
                "url": "https://rsshub.rssforever.com/chaping/newsflash",
                "name": "差评快讯",
                "category": "科技"
            },
            {
                "url": "https://rsshub.rssforever.com/36kr/motif/327686782977",
                "name": "36氪创投",
                "category": "AI"
            },
            {
                "url": "https://feeds.feedburner.com/ruanyifeng",
                "name": "科技爱好者周刊",
                "category": "weekly"
            },
            {
                "url": "https://www.ithome.com/rss/",
                "name": "IT之家",
                "category": "科技"
            },
        ]


    def _init_task(self):
        """初始化定时任务记录"""
        try:
            task = self.task_repo.get_task_by_name(self.TASK_NAME)
            if not task:
                # 创建新任务记录
                task_data = {
                    "name": self.TASK_NAME,
                    "description": "从RSS源获取最新热点话题",
                    "status": "active",
                    "interval": 3600,  # 默认1小时
                    "task_data": {
                        "processed_urls": []
                    }
                }
                self.task_repo.create_task(task_data)
                logger.info(f"已创建定时任务: {self.TASK_NAME}")
        except Exception as e:
            logger.error(f"初始化定时任务失败: {e}")


    async def update_hot_topics(self, force: bool):
        """
        更新所有热点话题数据
        
        参数:
            force: 是否强制更新，忽略时间间隔限制
            
        返回:
            int: 新增的热点话题数量
        """
        print(f"force: {force}")
        # 首先确保任务存在
        task = self.task_repo.get_task_by_name(self.TASK_NAME)
        if not task:
            logger.info(f"任务 {self.TASK_NAME} 不存在，创建新任务...")
            # 创建新任务记录
            task_data = {
                "name": self.TASK_NAME,
                "description": "从RSS源获取最新热点话题",
                "status": "active",
                "interval": 3600,  # 默认1小时
                "task_data": {
                    "processed_urls": []
                }
            }
            try:
                task = self.task_repo.create_task(task_data)
                logger.info(f"已创建定时任务: {self.TASK_NAME}, ID={task.id}")
            except Exception as e:
                logger.error(f"创建定时任务失败: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # 检查是否应该执行更新
        if not force and task and not self.task_repo.should_run_task(self.TASK_NAME):
            logger.info(f"定时任务 {self.TASK_NAME} 还未到执行时间，跳过本次更新")
            return 0
            
        logger.info("开始更新热点话题数据")
        
        # 获取任务数据或初始化空集合
        if not task or not task.task_data:
            processed_urls = set()
        else:
            processed_urls = set(task.task_data.get("processed_urls", []))
        
        new_processed_urls = set()

        for source in self.rss_sources:
            try:
                source_urls = await self._process_rss_source(source, processed_urls)
                new_processed_urls.update(source_urls)
            except Exception as e:
                logger.error(f"处理RSS源 {source['name']} 失败: {e}")
                
        # 更新已处理的URL列表
        processed_urls.update(new_processed_urls)
        
        # 限制URL列表大小
        if len(processed_urls) > 1000:
            processed_urls = set(list(processed_urls)[-1000:])
        
        # 更新任务数据
        task_data = {
            "processed_urls": list(processed_urls),
            "last_update_count": len(new_processed_urls)
        }
        
        # 更新任务执行时间和数据
        self.task_repo.update_last_run(self.TASK_NAME, task_data)
                
        # 清理过期数据
        # self._clean_expired_topics()
        logger.info(f"热点话题数据更新完成，新增 {len(new_processed_urls)} 条")

        return len(new_processed_urls)

    async def _process_rss_source(self, source: Dict[str, str], processed_urls: Set[str]) -> Set[str]:
        """
        处理单个RSS源的数据
        
        返回:
            新处理的URL集合
        """
        logger.info(f"正在处理RSS源: {source['name']}")
        new_processed_urls = set()
        
        try:
            # 获取RSS数据
            feed = feedparser.parse(source["url"])
            
            if not feed.entries:
                logger.warning(f"RSS源 {source['name']} 没有条目")
                return new_processed_urls
                
            # 处理每个条目
            for entry in feed.entries[:20]:  # 限制每次处理的条目数
                # 提取标题和链接
                title = entry.title if hasattr(entry, "title") else "无标题"
                link = entry.link if hasattr(entry, "link") else None
                
                if not link:
                    logger.warning(f"跳过无链接条目: {title}")
                    continue
                
                # 检查URL是否已处理
                if link in processed_urls:
                    logger.debug(f"跳过已处理URL: {link}")
                    continue
                
                # 检查数据库中是否存在
                if self.repo.exists_by_url(link):
                    new_processed_urls.add(link)  # 标记为已处理
                    continue
                    
                # 提取发布时间
                published = None
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    published = datetime(*entry.published_parsed[:6])
                
                # 提取描述
                description = None
                if hasattr(entry, "description"):
                    description = entry.description
                    description = self._clean_html(description)
                
                # 创建新的热点话题
                hot_topic = HotTopic(
                    title=title,
                    description=description,
                    source=source["name"],
                    category=source["category"],
                    url=link,
                    published_at=published,
                )
                
                # 保存到数据库
                self.repo.create(hot_topic)
                new_processed_urls.add(link)  # 标记为已处理
                logger.info(f"添加新热点: {title}")

            return new_processed_urls
                
        except Exception as e:
            logger.error(f"处理RSS源 {source['name']} 失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return new_processed_urls

    def _clean_html(self, html_content: str) -> str:
        """清理HTML内容"""
        if not html_content:
            return ""
            
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup(['script', 'style', 'img', 'a', 'video', 'audio', 'iframe', 'input']):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return text
        except Exception as e:
            logger.error(f"清理HTML内容失败: {e}")
            return html_content

    def _clean_expired_topics(self):
        """清理过期的热点话题（如7天前的数据）"""
        cutoff_date = datetime.now() - timedelta(days=7)
        deleted_count = self.repo.delete_before_date(cutoff_date)
        logger.info(f"清理了 {deleted_count} 条过期热点数据")

    def get_hot_topics(self, category: Optional[str] = None, limit: int = 10) -> List[HotTopic]:
        """获取热点话题列表"""
        return self.repo.get_hot_topics(category=category, limit=limit)
        
    def get_topic_by_id(self, topic_id: str) -> Optional[HotTopic]:
        """根据ID获取热点话题"""
        return self.repo.get_topic_by_id(topic_id)

    def increment_view_count(self, topic_id: str) -> bool:
        """增加热点的浏览计数"""
        return self.repo.increment_view_count(topic_id)