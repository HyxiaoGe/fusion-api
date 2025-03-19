import uuid
from datetime import datetime, timezone, timedelta

from sqlalchemy import Column, String, Integer, Text, ForeignKey, DateTime, JSON
from sqlalchemy.orm import relationship

from app.db.database import Base


def get_china_time():
    return datetime.now(timezone(timedelta(hours=8)))


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    provider = Column(String, nullable=False)
    model = Column(String, nullable=False)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)

    # 建立与Message的关系
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")


class File(Base):
    __tablename__ = "files"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)  # 存储的文件名
    original_filename = Column(String, nullable=False)  # 原始文件名
    mimetype = Column(String, nullable=False)  # 文件MIME类型
    size = Column(Integer, nullable=False)  # 文件大小(字节)
    path = Column(String, nullable=False)  # 存储路径
    status = Column(String, nullable=False, default="pending")  # 状态
    processing_result = Column(JSON, nullable=True)  # 处理结果
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)


class ConversationFile(Base):
    __tablename__ = "conversation_files"

    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True)
    file_id = Column(String, ForeignKey("files.id", ondelete="CASCADE"), primary_key=True)
    created_at = Column(DateTime, default=get_china_time)

    # 关系
    conversation = relationship("Conversation", back_populates="files")
    file = relationship("File")


Conversation.files = relationship("ConversationFile", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    role = Column(String, nullable=False)  # 'user' 或 'assistant'
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=get_china_time)

    # 建立与Conversation的关系
    conversation = relationship("Conversation", back_populates="messages")


class PromptTemplate(Base):
    __tablename__ = "prompt_templates"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    tags = Column(JSON, default=list)
    created_at = Column(DateTime, default=get_china_time)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)


class Setting(Base):
    __tablename__ = "settings"

    key = Column(String, primary_key=True)
    value = Column(JSON, nullable=False)
    updated_at = Column(DateTime, default=get_china_time, onupdate=get_china_time)
