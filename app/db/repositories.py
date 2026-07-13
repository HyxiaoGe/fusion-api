import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    AgentProgressSnapshot,
    AgentSession,
    ConversationFile,
    File,
    message_order_sequence,
)
from app.db.models import Conversation as ConversationModel
from app.db.models import Message as MessageModel
from app.db.models import SocialAccount as SocialAccountModel
from app.db.models import User as UserModel
from app.schemas.chat import (
    AgentRunSummary,
    Conversation,
    FileBlock,
    Message,
    SearchBlock,
    SearchSourceSummary,
    TextBlock,
    ThinkingBlock,
    UrlBlock,
    Usage,
)
from app.utils.time import utc_now

logger = logging.getLogger(__name__)


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, id: str) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.id == id).first()

    def get_by_username(self, username: str) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.username == username).first()

    def get_by_email(self, email: str) -> Optional[UserModel]:
        return self.db.query(UserModel).filter(UserModel.email == email).first()

    def build_unique_username(self, preferred: str, fallback_suffix: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_]+", "-", preferred).strip("-").lower()
        slug = slug or f"user-{fallback_suffix[:8]}"

        if not self.get_by_username(slug):
            return slug

        candidate = f"{slug}-{fallback_suffix[:8]}"
        if not self.get_by_username(candidate):
            return candidate

        counter = 2
        while self.get_by_username(f"{candidate}-{counter}"):
            counter += 1
        return f"{candidate}-{counter}"

    def create(self, obj_in: Dict[str, Any]) -> UserModel:
        db_obj = UserModel(**obj_in)
        self.db.add(db_obj)
        return db_obj

    def update_system_prompt(self, user: UserModel, system_prompt: str) -> UserModel:
        user.system_prompt = system_prompt
        self.db.commit()
        self.db.refresh(user)
        return user


class SocialAccountRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_provider(self, provider: str, provider_user_id: str) -> Optional[SocialAccountModel]:
        return (
            self.db.query(SocialAccountModel)
            .filter(SocialAccountModel.provider == provider, SocialAccountModel.provider_user_id == provider_user_id)
            .first()
        )

    def create(self, obj_in: Dict[str, Any]) -> SocialAccountModel:
        db_obj = SocialAccountModel(**obj_in)
        self.db.add(db_obj)
        return db_obj


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, conversation: Conversation) -> Conversation:
        """创建新的对话"""
        try:
            db_conversation = ConversationModel(
                id=conversation.id,
                user_id=conversation.user_id,
                title=conversation.title,
                model_id=conversation.model_id,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at,
            )

            # 添加消息（content blocks 序列化为 JSONB）
            for msg in conversation.messages:
                db_message = MessageModel(
                    id=msg.id,
                    sequence=msg.sequence,
                    role=msg.role,
                    content=[block.model_dump() for block in msg.content],
                    model_id=msg.model_id,
                    usage=msg.usage.model_dump() if msg.usage else None,
                    created_at=msg.created_at,
                )
                db_conversation.messages.append(db_message)

            self.db.add(db_conversation)
            return self._convert_to_schema(db_conversation)
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建对话失败: {e}")
            raise

    def update(self, conversation: Conversation) -> Conversation:
        """更新现有对话"""
        try:
            db_conversation = (
                self.db.query(ConversationModel)
                .filter(ConversationModel.id == conversation.id, ConversationModel.user_id == conversation.user_id)
                .first()
            )

            if not db_conversation:
                raise ValueError(f"找不到对话ID: {conversation.id} 或无权访问")

            # 仅更新对话元数据，不触碰消息（消息通过 create_message() 单独写入）
            db_conversation.title = conversation.title
            db_conversation.model_id = conversation.model_id
            db_conversation.updated_at = utc_now()

            self.db.flush()
            return self._convert_to_schema(db_conversation)
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新对话失败: {e}")
            raise

    def update_title(self, conversation_id: str, title: str) -> None:
        """仅更新会话标题"""
        self.db.query(ConversationModel).filter(ConversationModel.id == conversation_id).update(
            {"title": title, "updated_at": utc_now()}
        )
        self.db.flush()

    def delete(self, conversation_id: str, user_id: str) -> bool:
        """删除对话"""
        try:
            # 查找对话
            result = (
                self.db.query(ConversationModel)
                .filter(ConversationModel.id == conversation_id, ConversationModel.user_id == user_id)
                .delete()
            )

            self.db.commit()
            return result > 0
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除对话失败: {e}")
            return False

    def get_by_id(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """根据ID获取对话"""
        try:
            db_conversation = (
                self.db.query(ConversationModel)
                .filter(ConversationModel.id == conversation_id, ConversationModel.user_id == user_id)
                .first()
            )

            if not db_conversation:
                return None

            return self._convert_to_schema(db_conversation)
        except Exception as e:
            logger.error(f"获取对话失败: {e}")
            return None

    def get_all(self, user_id: str) -> List[Conversation]:
        """获取指定用户的所有对话"""
        try:
            db_conversations = (
                self.db.query(ConversationModel)
                .filter(ConversationModel.user_id == user_id)
                .order_by(ConversationModel.updated_at.desc())
                .all()
            )

            return [self._convert_to_schema(db_conv) for db_conv in db_conversations]
        except Exception as e:
            logger.error(f"获取所有对话失败: {e}")
            return []

    def get_paginated(self, user_id: str, page: int = 1, page_size: int = 20) -> Tuple[List[Conversation], int]:
        """分页获取对话列表（不包含消息内容）"""
        try:
            offset = (page - 1) * page_size
            query = self.db.query(ConversationModel).filter(ConversationModel.user_id == user_id)
            total = query.count()

            db_conversations = query.order_by(ConversationModel.updated_at.desc()).offset(offset).limit(page_size).all()

            conversations = []
            for db_conv in db_conversations:
                conversation = Conversation(
                    id=db_conv.id,
                    user_id=db_conv.user_id,
                    model_id=db_conv.model_id,
                    title=db_conv.title,
                    messages=[],
                    created_at=db_conv.created_at,
                    updated_at=db_conv.updated_at,
                )
                conversations.append(conversation)

            return conversations, total

        except Exception as e:
            logger.error(f"分页获取对话失败: {e}")
            return [], 0

    def get_metadata_by_ids(self, user_id: str, conversation_ids: List[str]) -> List[Conversation]:
        """按 ID 列表拉取对话元数据（不含 messages），仅返回属于当前用户的对话。

        用途：发完消息 / 重命名后只刷新已显示对话的标题等，避免重新拉取整个分页。
        """
        if not conversation_ids:
            return []
        try:
            db_conversations = (
                self.db.query(ConversationModel)
                .filter(
                    ConversationModel.user_id == user_id,
                    ConversationModel.id.in_(conversation_ids),
                )
                .all()
            )
            return [
                Conversation(
                    id=db_conv.id,
                    user_id=db_conv.user_id,
                    model_id=db_conv.model_id,
                    title=db_conv.title,
                    messages=[],
                    created_at=db_conv.created_at,
                    updated_at=db_conv.updated_at,
                )
                for db_conv in db_conversations
            ]
        except Exception as e:
            logger.error(f"按 ID 列表拉取对话元数据失败: {e}")
            return []

    def search_by_title(self, user_id: str, query: str, limit: int = 50) -> List[Conversation]:
        """按标题模糊搜索当前用户的对话，按 updated_at 倒序，限 limit 条。"""
        if not query or not query.strip():
            return []
        try:
            pattern = f"%{query.strip()}%"
            db_conversations = (
                self.db.query(ConversationModel)
                .filter(
                    ConversationModel.user_id == user_id,
                    ConversationModel.title.ilike(pattern),
                )
                .order_by(ConversationModel.updated_at.desc())
                .limit(limit)
                .all()
            )
            return [
                Conversation(
                    id=db_conv.id,
                    user_id=db_conv.user_id,
                    model_id=db_conv.model_id,
                    title=db_conv.title,
                    messages=[],
                    created_at=db_conv.created_at,
                    updated_at=db_conv.updated_at,
                )
                for db_conv in db_conversations
            ]
        except Exception as e:
            logger.error(f"按标题搜索对话失败: {e}")
            return []

    def create_message(self, message: Message, conversation_id: str) -> Message:
        """创建新消息并附加到现有对话"""
        try:
            db_message = MessageModel(
                id=message.id,
                conversation_id=conversation_id,
                sequence=message.sequence,
                role=message.role,
                content=[block.model_dump() for block in message.content],
                model_id=message.model_id,
                usage=message.usage.model_dump() if message.usage else None,
                created_at=message.created_at,
            )

            self.db.add(db_message)

            # 同步刷新 conversation.updated_at，让 sidebar 排序正确反映最近活跃对话
            db_conversation = self.db.query(ConversationModel).filter(ConversationModel.id == conversation_id).first()
            if db_conversation:
                db_conversation.updated_at = utc_now()

            self.db.flush()
            self.db.refresh(db_message)
            return self._convert_message_to_schema(db_message)
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建消息失败: {e}")
            raise

    def reserve_message_sequence_pair(self) -> tuple[int, int]:
        """用 PostgreSQL 全局 sequence 原子预留 user/assistant 顺序号。"""
        dialect_name = getattr(getattr(self.db.get_bind(), "dialect", None), "name", None)
        if dialect_name != "postgresql":
            raise RuntimeError("消息顺序号仅支持 PostgreSQL 数据库 sequence")
        user_sequence = int(self.db.execute(select(message_order_sequence.next_value())).scalar_one())
        return user_sequence, user_sequence + 1

    def get_message_by_id(self, message_id: str) -> Optional[Message]:
        """根据ID获取消息"""
        try:
            db_message = self.db.query(MessageModel).filter(MessageModel.id == message_id).first()
            if not db_message:
                return None
            return self._convert_message_to_schema(db_message)
        except Exception as e:
            logger.error(f"获取消息失败: {e}")
            return None

    def update_message(self, message_id: str, update_data: Dict[str, Any]) -> Optional[Message]:
        """更新消息内容"""
        try:
            db_message = self.db.query(MessageModel).filter(MessageModel.id == message_id).first()
            if db_message:
                for key, value in update_data.items():
                    # content blocks 和 usage 需要序列化为 dict 再写入 JSONB
                    if key == "content" and isinstance(value, list):
                        value = [block.model_dump() if hasattr(block, "model_dump") else block for block in value]
                    if key == "usage" and hasattr(value, "model_dump"):
                        value = value.model_dump()
                    setattr(db_message, key, value)
                self.db.flush()
                self.db.refresh(db_message)
                return self._convert_message_to_schema(db_message)
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新消息失败: {e}")
            return None

    def _convert_message_to_schema(
        self,
        db_message: MessageModel,
        *,
        agent_run: AgentRunSummary | None = None,
    ) -> Message:
        """将消息数据库模型转换为业务模型（JSONB → content blocks）"""
        content_blocks = []
        for block_data in db_message.content or []:
            block_type = block_data.get("type")
            if block_type == "text":
                content_blocks.append(TextBlock(**block_data))
            elif block_type == "thinking":
                content_blocks.append(ThinkingBlock(**block_data))
            elif block_type == "file":
                content_blocks.append(FileBlock(**block_data))
            elif block_type == "search":
                content_blocks.append(
                    SearchBlock(
                        type="search",
                        id=block_data.get("id", f"blk_{__import__('uuid').uuid4().hex[:12]}"),
                        query=block_data.get("query", ""),
                        tool_call_log_id=block_data.get("tool_call_log_id", ""),
                        sources=[SearchSourceSummary(**s) for s in block_data.get("sources", [])],
                        status=block_data.get("status", "success"),
                        error_message=block_data.get("error_message"),
                        source_count=block_data.get("source_count", 0),
                        source_refs=block_data.get("source_refs", []),
                        requested_provider=block_data.get("requested_provider"),
                        result_provider=block_data.get("result_provider"),
                        fallback_used=bool(block_data.get("fallback_used", False)),
                        provider_chain=block_data.get("provider_chain", []),
                        requested_count=block_data.get("requested_count"),
                        actual_count=block_data.get("actual_count"),
                        context_source_count=block_data.get("context_source_count"),
                        context_source_limit=block_data.get("context_source_limit"),
                        search_budget=block_data.get("search_budget"),
                        intent=block_data.get("intent"),
                        domains=block_data.get("domains", []),
                        recency_days=block_data.get("recency_days"),
                        budget_limited=bool(block_data.get("budget_limited", False)),
                    )
                )
            elif block_type == "url_read":
                content_blocks.append(UrlBlock(**block_data))
            # 未知类型跳过，保持前向兼容

        return Message(
            id=db_message.id,
            sequence=db_message.sequence,
            role=db_message.role,
            content=content_blocks,
            model_id=db_message.model_id,
            usage=Usage(**db_message.usage) if db_message.usage else None,
            suggested_questions=db_message.suggested_questions or None,
            agent_run=agent_run,
            created_at=db_message.created_at,
        )

    def get_last_assistant_message(self, conversation_id: str) -> Optional[MessageModel]:
        """获取会话中最后一条 assistant 消息的原始 DB 对象"""
        return (
            self.db.query(MessageModel)
            .filter(
                MessageModel.conversation_id == conversation_id,
                MessageModel.role == "assistant",
            )
            .order_by(
                MessageModel.sequence.desc().nullslast(),
                MessageModel.created_at.desc(),
                MessageModel.id.desc(),
            )
            .first()
        )

    def update_message_suggested_questions(self, message_id: str, questions: list[str]) -> None:
        """将推荐问题写回到指定消息"""
        self.db.query(MessageModel).filter(MessageModel.id == message_id).update({"suggested_questions": questions})
        self.db.flush()

    def _convert_to_schema(self, db_conversation: ConversationModel) -> Conversation:
        """将数据库模型转换为业务模型"""
        agent_runs = self._latest_agent_runs_for_messages(
            db_conversation.id,
            [msg.id for msg in db_conversation.messages if msg.role == "assistant"],
        )
        messages = [
            self._convert_message_to_schema(msg, agent_run=agent_runs.get(msg.id)) for msg in db_conversation.messages
        ]
        return Conversation(
            id=db_conversation.id,
            user_id=db_conversation.user_id,
            model_id=db_conversation.model_id,
            title=db_conversation.title,
            messages=messages,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
        )

    def _latest_agent_runs_for_messages(
        self,
        conversation_id: str,
        message_ids: list[str],
    ) -> dict[str, AgentRunSummary]:
        if not self.db or not message_ids:
            return {}

        rows = (
            self.db.query(AgentSession)
            .filter(
                AgentSession.conversation_id == conversation_id,
                AgentSession.message_id.in_(message_ids),
            )
            .order_by(AgentSession.message_id.asc(), AgentSession.created_at.desc(), AgentSession.id.desc())
            .all()
        )
        latest_rows_by_message_id: dict[str, AgentSession] = {}
        for row in rows:
            if not row.message_id or row.message_id in latest_rows_by_message_id:
                continue
            latest_rows_by_message_id[row.message_id] = row

        run_ids = [row.id for row in latest_rows_by_message_id.values()]
        snapshots = (
            self.db.query(AgentProgressSnapshot).filter(AgentProgressSnapshot.run_id.in_(run_ids)).all()
            if run_ids
            else []
        )
        snapshots_by_run_id = {snapshot.run_id: snapshot for snapshot in snapshots}

        latest_by_message_id: dict[str, AgentRunSummary] = {}
        for message_id, row in latest_rows_by_message_id.items():
            snapshot = snapshots_by_run_id.get(row.id)
            latest_by_message_id[message_id] = AgentRunSummary(
                run_id=row.id,
                status=row.status,
                config=row.run_config or {},
                total_steps=row.total_steps or 0,
                total_tool_calls=row.total_tool_calls or 0,
                limit_reason=row.limit_reason if row.status == "limit_reached" else None,
                progress=snapshot.state if snapshot else None,
            )
        return latest_by_message_id


class FileRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_file(self, file_data: Dict[str, Any]) -> File:
        """创建新文件记录"""
        try:
            db_file = File(
                id=file_data.get("id", str(uuid.uuid4())),
                user_id=file_data["user_id"],
                filename=file_data["filename"],
                original_filename=file_data["original_filename"],
                mimetype=file_data["mimetype"],
                size=file_data["size"],
                path=file_data["path"],
                status=file_data.get("status", "pending"),
                processing_result=file_data.get("processing_result"),
                parsed_content=file_data.get("parsed_content"),
                storage_key=file_data.get("storage_key"),
                thumbnail_key=file_data.get("thumbnail_key"),
                storage_backend=file_data.get("storage_backend", "local"),
                width=file_data.get("width"),
                height=file_data.get("height"),
            )
            self.db.add(db_file)
            self.db.commit()
            self.db.refresh(db_file)
            return db_file
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建文件记录失败: {e}")
            raise

    def link_file_to_conversation(self, conversation_id: str, file_id: str) -> bool:
        """关联文件到对话"""
        try:
            # 检查是否已存在关联
            existing = (
                self.db.query(ConversationFile)
                .filter(ConversationFile.conversation_id == conversation_id, ConversationFile.file_id == file_id)
                .first()
            )

            if existing:
                return True

            # 创建新关联
            conv_file = ConversationFile(conversation_id=conversation_id, file_id=file_id)
            self.db.add(conv_file)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"关联文件到对话失败: {e}")
            return False

    def get_parsed_file_content(self, file_ids: List[str]) -> Dict[str, str]:
        """获取多个文件的解析内容"""
        try:
            result = {}
            if not file_ids:
                return result

            # 查询指定ID的所有已处理文件
            files = self.db.query(File).filter(File.id.in_(file_ids), File.status == "processed").all()

            # 构建ID到内容的映射
            for file in files:
                if file.parsed_content:
                    result[file.id] = file.parsed_content

            return result
        except Exception as e:
            logger.error(f"获取文件解析内容失败: {e}")
            return {}

    def get_conversation_files(self, conversation_id: str) -> List[ConversationFile]:
        """获取对话关联的所有文件"""
        try:
            return (
                self.db.query(ConversationFile)
                .filter(ConversationFile.conversation_id == conversation_id)
                .options(joinedload(ConversationFile.file))
                .all()
            )
        except Exception as e:
            logger.error(f"获取对话文件失败: {e}")
            return []

    def is_file_linked_to_conversation(self, conversation_id: str, file_id: str) -> bool:
        """确认文件是否已经关联到指定对话。"""
        return (
            self.db.query(ConversationFile)
            .filter(ConversationFile.conversation_id == conversation_id, ConversationFile.file_id == file_id)
            .first()
            is not None
        )

    def count_conversation_files(self, conversation_id: str) -> int:
        """计算对话关联的文件数量"""
        try:
            return self.db.query(ConversationFile).filter(ConversationFile.conversation_id == conversation_id).count()
        except Exception as e:
            logger.error(f"计算对话文件数量失败: {e}")
            return 0

    def get_stale_uploading_files(self, conversation_id: str, cutoff: datetime) -> List[File]:
        """获取指定对话里超过直传窗口仍未完成的文件。"""
        try:
            return (
                self.db.query(File)
                .join(ConversationFile, ConversationFile.file_id == File.id)
                .filter(
                    ConversationFile.conversation_id == conversation_id,
                    File.status == "uploading",
                    File.created_at < cutoff,
                )
                .all()
            )
        except Exception as e:
            logger.error(f"查询过期上传文件失败: {e}")
            return []

    def get_file_by_id(self, file_id: str, user_id: Optional[str] = None) -> Optional[File]:
        """根据ID获取文件，可选择按user_id过滤"""
        query = self.db.query(File).filter(File.id == file_id)
        if user_id:
            query = query.filter(File.user_id == user_id)
        return query.first()

    def get_files_by_user_id(self, user_id: str) -> List[File]:
        """获取用户的所有文件"""
        return self.db.query(File).filter(File.user_id == user_id).all()

    def get_files_info(self, file_ids: List[str]) -> List[File]:
        """获取一组文件的信息"""
        return self.db.query(File).filter(File.id.in_(file_ids)).all()

    def get_file_paths(self, file_ids: List[str]) -> List[str]:
        """获取一组文件的存储路径"""
        return [row[0] for row in self.db.query(File.path).filter(File.id.in_(file_ids)).all()]

    def update_file(self, file_id: str, updates: Dict[str, Any]) -> bool:
        """更新文件信息"""
        try:
            result = self.db.query(File).filter(File.id == file_id).update(updates)
            if result > 0:
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新文件失败: {e}")
            return False

    def delete_file(self, file_id: str, user_id: str) -> bool:
        """删除文件记录"""
        try:
            file = self.db.query(File).filter(File.id == file_id, File.user_id == user_id).first()
            if not file:
                return False
            self.db.query(ConversationFile).filter(ConversationFile.file_id == file_id).delete()
            self.db.delete(file)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除文件失败: {e}")
            return False
