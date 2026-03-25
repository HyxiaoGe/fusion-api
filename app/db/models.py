import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, JSON, Boolean, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

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
    # model_id 对应 model_sources 表中的 model_id，支持中途切换模型
    model_id = Column(String, nullable=False)
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
    role = Column(String, nullable=False)  # 'user' | 'assistant'

    # content blocks 数组，结构示例：
    # 用户消息: [{"type": "text", "text": "..."}, {"type": "file", "file_id": "...", "filename": "...", "mime_type": "..."}]
    # AI 回复:  [{"type": "thinking", "id": "blk_001", "thinking": "..."}, {"type": "text", "id": "blk_002", "text": "..."}]
    content = Column(JSONB, nullable=False)

    # 仅 assistant 消息填充，记录实际生成该消息时使用的模型
    model_id = Column(String, nullable=True)

    # 仅 assistant 消息填充，记录本次请求的 token 消耗
    # 结构: {"input_tokens": 312, "output_tokens": 876}
    usage = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=get_china_time)

    conversation = relationship("Conversation", back_populates="messages")


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
