import os
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import JSON, Boolean, Column, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.database import Base

# 生产环境（PostgreSQL）使用 JSONB 以获得更好的查询性能；
# 测试环境（SQLite）回退到通用 JSON 类型
_db_url = os.getenv("DATABASE_URL", "")
if _db_url.startswith("postgresql"):
    from sqlalchemy.dialects.postgresql import JSONB
else:
    JSONB = JSON  # type: ignore[misc,assignment]


def get_china_time():
    return datetime.now(timezone(timedelta(hours=8)))


class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, index=True, nullable=False)
    nickname = Column(String, nullable=True)
    avatar = Column(String, nullable=True)
    mobile = Column(String, nullable=True)
    email = Column(String, unique=True, index=True, nullable=True)
    system_prompt = Column(Text, nullable=False, default="", server_default="")
    is_superuser = Column(Boolean, nullable=False, server_default="false", default=False)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    social_accounts = relationship("SocialAccount", back_populates="user", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")
    files = relationship("File", back_populates="user", cascade="all, delete-orphan")


class SocialAccount(Base):
    __tablename__ = "social_accounts"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String, nullable=False)
    provider_user_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship("User", back_populates="social_accounts")

    __table_args__ = (UniqueConstraint("provider", "provider_user_id", name="uix_provider_user_id"),)


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False)
    # model_id 对应 model_sources 表中的 model_id，支持中途切换模型
    model_id = Column(String, nullable=False)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", cascade="all, delete-orphan", order_by="Message.created_at"
    )
    files = relationship("ConversationFile", back_populates="conversation", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String, nullable=False)  # 存储的文件名
    original_filename = Column(String, nullable=False)  # 原始文件名
    mimetype = Column(String, nullable=False)  # 文件MIME类型
    size = Column(Integer, nullable=False)  # 文件大小(字节)
    path = Column(String, nullable=False)  # 存储路径（兼容旧数据）
    status = Column(String, nullable=False, default="pending")  # 状态
    processing_result = Column(JSON, nullable=True)  # 处理结果
    parsed_content = Column(Text, nullable=True)  # 解析后的内容

    # 图片相关字段（阶段2新增）
    storage_key = Column(String, nullable=True)  # 处理图的存储键
    thumbnail_key = Column(String, nullable=True)  # 缩略图的存储键
    storage_backend = Column(String, default="local")  # 存储后端标识（"local" 或 "minio"）
    width = Column(Integer, nullable=True)  # 图片宽度
    height = Column(Integer, nullable=True)  # 图片高度

    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    user = relationship("User", back_populates="files")


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

    # 仅 assistant 消息填充，持久化推荐问题（只保留最新一批）
    # 结构: ["问题1", "问题2", "问题3"]
    suggested_questions = Column(JSONB, nullable=True)

    created_at = Column(DateTime, default=get_china_time)

    conversation = relationship("Conversation", back_populates="messages")


class PromptExample(Base):
    """动态示例问题（由 Kimi $web_search 定时生成）"""

    __tablename__ = "prompt_examples"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    question = Column(String, nullable=False)
    category = Column(String, nullable=False)  # "news" | "tech" | "general"
    source = Column(String, default="kimi")  # 预留给未来其他来源
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=get_china_time)
    expires_at = Column(DateTime, nullable=True)


class ToolCallLog(Base):
    """工具调用统计日志"""

    __tablename__ = "tool_call_logs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    message_id = Column(String, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)

    tool_name = Column(String(50), nullable=False, index=True)
    status = Column(String(20), nullable=False)  # 'success' | 'failed' | 'degraded'
    error_message = Column(Text, nullable=True)
    duration_ms = Column(Integer, nullable=True)
    model_id = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)

    input_params = Column(JSONB, nullable=True)
    output_data = Column(JSONB, nullable=True)
    extra_metadata = Column("metadata", JSONB, nullable=True)

    trace_id = Column(String, nullable=True, index=True)
    step_number = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=get_china_time, index=True)


class AgentSession(Base):
    """Agent 执行会话记录 — 一次 Agent Loop 一条"""

    __tablename__ = "agent_sessions"

    id = Column(String, primary_key=True)  # = HTTP request_id
    conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_id = Column(String, nullable=True)
    user_id = Column(String, nullable=False, index=True)
    model_id = Column(String(100), nullable=False)
    provider = Column(String(50), nullable=False)
    run_config = Column("config", JSONB, nullable=True)

    total_steps = Column(Integer, default=0)
    total_tool_calls = Column(Integer, default=0)
    total_duration_ms = Column(Integer, nullable=True)

    status = Column(
        String(20), nullable=False
    )  # "running" | "completed" | "limit_reached" | "incomplete" | "error" | "interrupted"
    limit_reason = Column(String(30), nullable=True)  # "max_steps" | "max_tool_calls" | "timeout"
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, default=get_china_time, index=True)

    __table_args__ = (
        Index("ix_agent_sessions_conversation_message_created_at", "conversation_id", "message_id", "created_at"),
    )


class AgentProgressSnapshot(Base):
    """Agent 可读进度 compact snapshot — 按 run_id 保存最新折叠状态"""

    __tablename__ = "agent_progress_snapshots"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id = Column(String, ForeignKey("agent_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    conversation_id = Column(
        String,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    message_id = Column(String, nullable=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    protocol_version = Column(Integer, nullable=False, default=2)
    state = Column(JSONB, nullable=False)
    created_at = Column(DateTime, default=get_china_time, index=True)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time, index=True)

    __table_args__ = (
        UniqueConstraint("run_id", name="uq_agent_progress_snapshots_run_id"),
        Index("ix_agent_progress_message_updated", "conversation_id", "message_id", "updated_at"),
    )


class AgentStep(Base):
    """Agent 单步执行记录 — 每步工具调用完成后写入"""

    __tablename__ = "agent_steps"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    trace_id = Column(String, nullable=False, index=True)
    step_number = Column(Integer, nullable=False)

    status = Column(
        String(20), nullable=False, server_default="completed"
    )  # "running" | "completed" | "failed" | "interrupted"
    # 注：ORM 层无 default，新建必须显式传 status；server_default 仅用于 ALTER 时兜底已有历史行。

    tool_calls_count = Column(Integer, default=0)
    tool_names = Column(JSONB, nullable=True)  # ["web_search", "url_read"]
    duration_ms = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=get_china_time)

    __table_args__ = (UniqueConstraint("trace_id", "step_number", name="uq_trace_step"),)
