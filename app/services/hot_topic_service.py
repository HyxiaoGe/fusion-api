import feedparser
import asyncio
import re
import os
import hashlib
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logger import app_logger as logger
from app.db.models import HotTopic
from app.db.repositories import HotTopicRepository, ScheduledTaskRepository, RssSourceRepository

class HotTopicService:
    """热点话题服务，负责从RSS获取热点数据并定期更新"""

    # 任务名称常量
    TASK_NAME = "rss_hot_topics_update"
    LOCAL_DOCS_TASK_NAME = "local_docs_update"
    
    def __init__(self, db: Session):
        self.db = db
        self.repo = HotTopicRepository(db)
        self.task_repo = ScheduledTaskRepository(db)
        self.rss_repo = RssSourceRepository(db)
        # 项目根目录下的docs文件夹路径
        self.docs_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "docs")
        
        # 初始化任务
        self._init_tasks()
        # 初始化种子RSS数据
        self._seed_rss_sources()

    def _seed_rss_sources(self):
        """如果数据库为空，则植入初始的RSS源数据"""
        try:
            # 检查是否已存在全局RSS源
            existing_sources = self.rss_repo.get_all_enabled()
            if existing_sources:
                return

            logger.info("数据库中没有RSS源，开始植入初始数据...")
            
            initial_sources = [
                {
                    "url": "http://www.jintiankansha.me/rss/GE2TG7DCGM4DIYLFGJTGEMLEMQZWCZRRMRSDIYZQGNSDMOBQGFSDIOJYMVTGCOBRGU4WKNI=",
                    "name": "差评快讯",
                    "category": "微信公众平台",
                    "filter_apply": "title",
                    "filter_type": "include",
                    "filter_rule": "OpenAI|ChatGPT|Claude|GPT|Sora|AI|微软|谷歌|字节|千问|DeepSeek|混元|阿里|xAI|马斯克|Gemini|Anthropic|豆包|文心一言"
                },
                {
                    "url": "https://rsshub.rssforever.com/36kr/motif/327686782977",
                    "name": "36氪创投",
                    "category": "AI",
                    "filter_apply": "title",
                    "filter_type": "include",
                    "filter_rule": "OpenAI|ChatGPT|Claude|GPT|Sora|AI|微软|谷歌|字节|千问|DeepSeek|混元|阿里|xAI|马斯克|Gemini|Anthropic|豆包|文心一言"
                },
                {
                    "url": "https://rsshub.app/36kr/hot-list",
                    "name": "36氪 - 24小时热榜",
                    "category": "热榜"
                },
                {
                    "url": "https://rsshub.app/sspai/matrix",
                    "name": "少数派",
                    "category": "热榜"
                },
                {
                    "url": "https://rsshub.app/woshipm/popular",
                    "name": "人人都是产品经理",
                    "category": "热榜"
                },
                {
                    "url": "https://feeds.feedburner.com/ruanyifeng",
                    "name": "科技爱好者周刊",
                    "category": "weekly"
                },
                {
                    "url": "http://www.jintiankansha.me/rss/GMZTSOJTGZ6DOMBUHE2WGZTGMI2TKMRQG4YWGMRVGQYTSZJYGJRTENBYGYZTKMRWHAYTGMLDMM3Q====",
                    "name": "知危",
                    "category": "微信公众平台"
                },
                {
                    "url": "https://www.ithome.com/rss/",
                    "name": "IT之家",
                    "category": "科技",
                    "filter_apply": "title",
                    "filter_type": "include",
                    "filter_rule": "OpenAI|ChatGPT|Claude|GPT|Sora|AI|微软|谷歌|字节|千问|DeepSeek|混元|阿里|xAI|马斯克|Gemini|Anthropic|豆包|文心一言"
                },
                {
                    "url": "https://rsshub.app/aibase/news",
                    "name": "AIbase",
                    "category": "AI"
                },
                {
                    "url": "https://rsshub.app/huxiu/article",
                    "name": "虎嗅网",
                    "category": "热榜"
                },
            ]

            for source_data in initial_sources:
                from app.schemas.rss import RssSourceCreate
                source_create = RssSourceCreate(**source_data)
                self.rss_repo.create(source_create, user_id=None)
            
            logger.info(f"成功植入 {len(initial_sources)} 条初始RSS源数据。")

        except Exception as e:
            logger.error(f"植入初始RSS源数据失败: {e}")

    def _init_tasks(self):
        """初始化所有定时任务记录"""
        self._init_rss_task()
        self._init_docs_task()

    def _init_rss_task(self):
        """初始化RSS定时任务记录"""
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
            logger.error(f"初始化RSS定时任务失败: {e}")

    def _init_docs_task(self):
        """初始化本地文档定时任务记录"""
        try:
            task = self.task_repo.get_task_by_name(self.LOCAL_DOCS_TASK_NAME)
            if not task:
                # 创建新任务记录
                task_data = {
                    "name": self.LOCAL_DOCS_TASK_NAME,
                    "description": "从项目docs目录获取热点话题",
                    "status": "active",
                    "interval": 3600,  # 与RSS任务相同，默认1小时
                    "task_data": {
                        "processed_files": []
                    }
                }
                self.task_repo.create_task(task_data)
                logger.info(f"已创建定时任务: {self.LOCAL_DOCS_TASK_NAME}")
        except Exception as e:
            logger.error(f"初始化本地文档定时任务失败: {e}")

    def _filter_entry(self, entry, filter_apply, filter_type, filter_rule):
        """
        根据筛选条件过滤RSS条目
        
        参数:
            entry: RSS条目
            filter_apply: 应用过滤的字段（title, description 或 link）
            filter_type: 过滤类型（include, exclude, regex match, regex not match）
            filter_rule: 过滤规则（关键词或正则表达式）
            
        返回:
            bool: 条目是否通过过滤
        """
        if not filter_apply or not filter_type or not filter_rule:
            return True
            
        # 根据filter_apply获取要筛选的文本
        if filter_apply == 'title':
            text = entry.title if hasattr(entry, "title") else ""
        elif filter_apply == 'description':
            if hasattr(entry, "description"):
                text = entry.description
            elif hasattr(entry, "summary"):
                text = entry.summary
            else:
                text = ""
        elif filter_apply == 'link':
            text = entry.link if hasattr(entry, "link") else ""
        elif filter_apply == 'article':
            # 尝试获取文章内容
            if hasattr(entry, "content") and entry.content:
                if isinstance(entry.content, list) and entry.content:
                    text = entry.content[0].value
                else:
                    text = str(entry.content)
            elif hasattr(entry, "summary"):
                text = entry.summary
            elif hasattr(entry, "description"):
                text = entry.description
            else:
                text = ""
        elif filter_apply == 'content':
            text = entry.content if hasattr(entry, "content") else ""
        else:
            logger.warning(f"不支持的filter_apply类型: {filter_apply}")
            return True
            
        # 根据filter_type进行过滤
        try:
            if filter_type == 'include':
                return bool(re.search(filter_rule, text))
            elif filter_type == 'exclude':
                return not bool(re.search(filter_rule, text))
            elif filter_type == 'regex match':
                return bool(re.search(filter_rule, text))
            elif filter_type == 'regex not match':
                return not bool(re.search(filter_rule, text))
            else:
                logger.warning(f"不支持的filter_type类型: {filter_type}")
                return True
        except Exception as e:
            logger.error(f"应用过滤规则时出错: {e}")
            return True

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

        rss_sources = self.rss_repo.get_all_enabled()
        logger.info(f"从数据库加载了 {len(rss_sources)} 个启用的RSS源进行处理。")

        for source in rss_sources:
            try:
                source_dict = source.to_dict()
                source_urls = await self._process_rss_source(source_dict, processed_urls)
                new_processed_urls.update(source_urls)
                
                # 更新RSS源的检查时间
                self.rss_repo.touch(source.id)
                logger.debug(f"已更新RSS源 '{source.name}' 的 updated_at 时间戳。")
            except Exception as e:
                logger.error(f"处理RSS源 {source.name} 失败: {e}")
                
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
            feed = await asyncio.to_thread(feedparser.parse, source["url"])
            
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
                
                # 应用过滤规则
                filter_apply = source.get("filter_apply")
                filter_type = source.get("filter_type") 
                filter_rule = source.get("filter_rule")
                
                if filter_apply and filter_type and filter_rule:
                    if not self._filter_entry(entry, filter_apply, filter_type, filter_rule):
                        logger.info(f"条目被过滤: {title}")
                        new_processed_urls.add(link)  # 标记为已处理，避免重复处理
                        continue
                    else:
                        logger.info(f"条目通过过滤: {title}")
                    
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
        """清理HTML内容和Markdown图片链接"""
        if not html_content:
            return ""
            
        try:
            # 清理Markdown格式的图片链接
            # 匹配 ![任意文本](图片URL) 格式
            html_content = re.sub(r'!\[.*?\]\(.*?\)', '', html_content)
            
            # 清理Markdown中的纯图片URL链接格式
            # 匹配 ![](图片URL) 格式
            html_content = re.sub(r'!\[\]\(.*?\)', '', html_content)
            
            # 清理可能包含的其他Markdown语法
            html_content = re.sub(r'\[.*?\]\(.*?\)', '', html_content)  # 清理链接 [text](url)
            
            soup = BeautifulSoup(html_content, "html.parser")
            for element in soup(['script', 'style', 'img', 'a', 'video', 'audio', 'iframe', 'input']):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            
            # 移除多余的空行
            text = re.sub(r'\n\s*\n', '\n\n', text)
            
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
 
    async def update_local_docs(self, force: bool):
        """
        更新本地文档数据
        
        参数:
            force: 是否强制更新，忽略时间间隔限制
            
        返回:
            int: 新增的热点话题数量
        """
        # 确保任务存在
        task = self.task_repo.get_task_by_name(self.LOCAL_DOCS_TASK_NAME)
        if not task:
            self._init_docs_task()
            task = self.task_repo.get_task_by_name(self.LOCAL_DOCS_TASK_NAME)
            if not task:
                logger.error(f"无法创建或获取本地文档任务 {self.LOCAL_DOCS_TASK_NAME}")
                return 0
        
        # 检查是否应该执行更新
        if not force and task and not self.task_repo.should_run_task(self.LOCAL_DOCS_TASK_NAME):
            logger.info(f"定时任务 {self.LOCAL_DOCS_TASK_NAME} 还未到执行时间，跳过本次更新")
            return 0
            
        logger.info("开始更新本地文档数据")
        
        # 获取任务数据或初始化空集合
        if not task or not task.task_data:
            processed_files = set()
        else:
            processed_files = set(task.task_data.get("processed_files", []))
        
        new_processed_files = set()

        try:
            processed = await self._process_docs_folder(processed_files)
            new_processed_files.update(processed)
        except Exception as e:
            logger.error(f"处理本地文档失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
                
        # 更新已处理的文件列表
        processed_files.update(new_processed_files)
        
        # 限制文件列表大小
        if len(processed_files) > 1000:
            processed_files = set(list(processed_files)[-1000:])
        
        # 更新任务数据
        task_data = {
            "processed_files": list(processed_files),
            "last_update_count": len(new_processed_files)
        }
        
        # 更新任务执行时间和数据
        self.task_repo.update_last_run(self.LOCAL_DOCS_TASK_NAME, task_data)
                
        logger.info(f"本地文档数据更新完成，新增 {len(new_processed_files)} 条")

        return len(new_processed_files)
        
    async def _process_docs_folder(self, processed_files: Set[str]) -> Set[str]:
        """
        处理项目docs目录下的所有文档
        
        返回:
            新处理的文件标识集合
        """
        logger.info(f"正在处理本地文档: {self.docs_folder}")
        new_processed_files = set()
        
        if not os.path.exists(self.docs_folder):
            logger.error(f"文档路径不存在: {self.docs_folder}")
            return new_processed_files
            
        # 遍历文件夹中的所有文件
        for root, dirs, files in os.walk(self.docs_folder):
            for file in files:
                # 只处理文本文件
                if not file.lower().endswith(('.txt', '.md', '.html', '.htm', '.doc', '.docx')):
                    continue
                    
                file_path = os.path.join(root, file)
                file_hash = self._get_file_hash(file_path)
                
                # 检查文件是否已处理
                if file_hash in processed_files:
                    logger.debug(f"跳过已处理文件: {file_path}")
                    continue
                
                # 读取文件内容
                try:
                    content = self._read_file_content(file_path)
                    if not content:
                        logger.warning(f"文件内容为空: {file_path}")
                        new_processed_files.add(file_hash)
                        continue
                    
                    # 清理HTML和图片标签
                    content = self._clean_html(content)
                except Exception as e:
                    logger.error(f"读取文件 {file_path} 失败: {e}")
                    continue
                
                # 提取文件标题
                title = os.path.splitext(file)[0]
                
                # 提取文件修改时间作为发布时间
                published = datetime.fromtimestamp(os.path.getmtime(file_path))
                                
                # 确定分类（可以基于文件夹名称）
                folder_name = os.path.basename(root)
                source = folder_name if folder_name != "docs" else "未分类"
                
                # 生成唯一URL
                url = f"file://{file_path}"
                
                # 创建新的热点话题
                hot_topic = HotTopic(
                    title=title,
                    description=content,
                    source=source,
                    category="微信公众平台",
                    url=url,
                    published_at=published,
                )
                
                # 保存到数据库
                self.repo.create(hot_topic)
                new_processed_files.add(file_hash)
                logger.info(f"添加新热点: {title}")

        return new_processed_files
            
    def _read_file_content(self, file_path: str) -> str:
        """读取文件内容"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except UnicodeDecodeError:
            try:
                # 尝试使用GBK编码
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                return content
            except:
                logger.error(f"无法读取文件内容: {file_path}")
                return ""
                
    def _get_file_hash(self, file_path: str) -> str:
        """获取文件的MD5哈希值和修改时间组合，用于标识文件是否被修改"""
        try:
            mtime = os.path.getmtime(file_path)
            file_size = os.path.getsize(file_path)
            return f"{file_path}_{mtime}_{file_size}"
        except Exception as e:
            logger.error(f"获取文件哈希失败: {e}")
            return file_path  # 如果获取失败，返回文件路径作为标识