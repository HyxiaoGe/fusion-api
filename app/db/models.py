import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, JSON, Boolean, Float, UniqueConstraint
from sqlalchemy.orm import relationship, backref

from app.db.database import Base


def get_china_time():
    return datetime.now(timezone(timedelta(hours=8)))


class User(Base):
    __tablename__ = 'users'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    nickname = Column(String, nullable=True)
    avatar = Column(String, nullable=True)
    mobile = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    social_accounts = relationship('SocialAccount', back_populates='user', cascade='all, delete-orphan')
    conversations = relationship('Conversation', back_populates='user', cascade='all, delete-orphan')
    files = relationship('File', back_populates='user', cascade='all, delete-orphan')
    rss_sources = relationship('RssSource', back_populates='user', cascade='all, delete-orphan')
    prompt_templates = relationship('PromptTemplate', back_populates='user', cascade='all, delete-orphan')


class SocialAccount(Base):
    __tablename__ = 'social_accounts'
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    provider = Column(String, nullable=False)
    provider_user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship('User', back_populates='social_accounts')

    __table_args__ = (UniqueConstraint('provider', 'provider_user_id', name='uix_provider_user_id'),)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    title = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship('User', back_populates='conversations')
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at")
    files = relationship("ConversationFile", back_populates="conversation", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True)
    filename = Column(String, nullable=False)  # 存储的文件名
    original_filename = Column(String, nullable=False)  # 原始文件名
    mimetype = Column(String, nullable=False)  # 文件MIME类型
    size = Column(Integer, nullable=False)  # 文件大小(字节)
    path = Column(String, nullable=False)  # 存储路径
    status = Column(String, nullable=False, default="pending")  # 状态
    processing_result = Column(JSON, nullable=True)  # 处理结果
    parsed_content = Column(Text, nullable=True)  # 解析后的内容
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship('User', back_populates='files')


class ConversationFile(Base):
    __tablename__ = "conversation_files"

    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True)
    file_id = Column(String, ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime, default=get_china_time)

    # 关系
    conversation = relationship("Conversation", back_populates="files")
    file = relationship("File")


class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # 'user', 'assistant', 'system'
    type = Column(String, nullable=False)  # 'user_query', 'assistant_content', 'reasoning_content', 'function_call', 'function_result', 'web_search', 'hot_topics'
    content = Column(Text, nullable=False)
    turn_id = Column(String, nullable=True)  # 对话轮次ID，使用该轮对话中用户消息的ID
    duration = Column(Integer, nullable=False, default=0)  # 处理耗时(毫秒)，默认为0
    created_at = Column(DateTime, default=get_china_time)

    # 建立与Conversation的关系
    conversation = relationship("Conversation", back_populates="messages")


class HotTopic(Base):
    __tablename__ = "hot_topics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    source = Column(String, nullable=False)  # 来源，如"36氪"
    category = Column(String, nullable=True)  # 分类，如"科技"、"财经"
    url = Column(String, nullable=True)  # 原文链接
    published_at = Column(DateTime, nullable=True)  # 发布时间
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)
    view_count = Column(Integer, default=0)  # 浏览次数，用于排序


class RssSource(Base):
    __tablename__ = "rss_sources"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)
    name = Column(String, nullable=False, unique=True)
    url = Column(String, nullable=False, unique=True)
    description = Column(String, nullable=True)
    category = Column(String, nullable=True)
    is_enabled = Column(Boolean, default=True)
    
    # 过滤规则
    filter_apply = Column(String, nullable=True)  # 应用于: title, description, ...
    filter_type = Column(String, nullable=True)   # 过滤类型: include, exclude, ...
    filter_rule = Column(String, nullable=True)   # 规则: 关键字或正则表达式
    
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship('User', back_populates='rss_sources')
    
    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


class ScheduledTask(Base):
    __tablename__ = "scheduled_tasks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False, unique=True)  # 任务名称，如"rss_hot_topics_update"
    description = Column(Text, nullable=True)  # 任务描述
    status = Column(String, nullable=False, default="active")  # 任务状态：active, paused
    interval = Column(Integer, nullable=False)  # 间隔时间（秒）
    last_run_at = Column(DateTime, nullable=True)  # 上次执行时间
    next_run_at = Column(DateTime, nullable=True)  # 下次执行时间
    params = Column(JSON, nullable=True)  # 任务参数，以JSON格式存储
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)
    
    # 额外字段用于存储任务特定数据
    task_data = Column(JSON, nullable=True)  # 如已处理的URL等


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey('users.id', ondelete='CASCADE'), nullable=True, index=True)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship('User', back_populates='prompt_templates')


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)


class ModelSource(Base):
    """模型数据源表"""
    __tablename__ = "model_sources"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String, unique=True, index=True, nullable=False)
    name = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    knowledge_cutoff = Column(String, nullable=True)
    
    capabilities = Column(JSON, nullable=False, default={}) # 模型能力配置
    pricing = Column(JSON, nullable=False, default={}) # 模型定价信息
    
    auth_config = Column(JSON, nullable=False, default={})  # 认证配置模板
    model_configuration = Column(JSON, nullable=False, default={})  # 模型默认参数配置
    priority = Column(Integer, default=100)  # 优先级，默认100，数字越小优先级越高
    
    enabled = Column(Boolean, default=True)
    description = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    credentials = relationship("ModelCredential", back_populates="model_source", cascade="all, delete-orphan")

class ModelCredential(Base):
    __tablename__ = "model_credentials"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_id = Column(String, ForeignKey("model_sources.model_id", ondelete="CASCADE"), nullable=False)
    name = Column(String, nullable=False)  # 凭证名称，如"默认"、"测试环境"等
    is_default = Column(Boolean, default=False)  # 是否为默认凭证
    credentials = Column(JSON, nullable=False)  # 存储实际的认证信息
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)
    
    # 关联关系
    model_source = relationship("ModelSource", back_populates="credentials")
    
    # 联合唯一约束
    __table_args__ = (
        UniqueConstraint('model_id', 'name', name='uix_model_credential_name'),
    )