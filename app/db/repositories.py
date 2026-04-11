import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session, joinedload

from app.db.models import Conversation as ConversationModel
from app.db.models import ConversationFile, File, Memory, ModelCredential, ModelSource, Provider, get_china_time
from app.db.models import Message as MessageModel
from app.db.models import SocialAccount as SocialAccountModel
from app.db.models import User as UserModel
from app.schemas.chat import (
    Conversation,
    FileBlock,
    Message,
    SearchBlock,
    SearchSource,
    TextBlock,
    ThinkingBlock,
    Usage,
)
from app.schemas.models import (
    AuthConfig,
    AuthConfigField,
    ModelBasicInfo,
    ModelCapabilities,
    ModelConfigParam,
    ModelConfiguration,
    ModelCredentialInfo,
    ModelInfo,
    ModelPricing,
    ProviderInfo,
)

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
            db_conversation.updated_at = get_china_time()

            self.db.flush()
            return self._convert_to_schema(db_conversation)
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新对话失败: {e}")
            raise

    def update_title(self, conversation_id: str, title: str) -> None:
        """仅更新会话标题"""
        self.db.query(ConversationModel).filter(ConversationModel.id == conversation_id).update(
            {"title": title, "updated_at": get_china_time()}
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

    def create_message(self, message: Message, conversation_id: str) -> Message:
        """创建新消息并附加到现有对话"""
        try:
            db_message = MessageModel(
                id=message.id,
                conversation_id=conversation_id,
                role=message.role,
                content=[block.model_dump() for block in message.content],
                model_id=message.model_id,
                usage=message.usage.model_dump() if message.usage else None,
                created_at=message.created_at,
            )

            self.db.add(db_message)
            self.db.flush()
            self.db.refresh(db_message)
            return self._convert_message_to_schema(db_message)
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建消息失败: {e}")
            raise

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

    def _convert_message_to_schema(self, db_message: MessageModel) -> Message:
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
                        sources=[SearchSource(**s) for s in block_data.get("sources", [])],
                    )
                )
            # 未知类型跳过，保持前向兼容

        return Message(
            id=db_message.id,
            role=db_message.role,
            content=content_blocks,
            model_id=db_message.model_id,
            usage=Usage(**db_message.usage) if db_message.usage else None,
            suggested_questions=db_message.suggested_questions or None,
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
            .order_by(MessageModel.created_at.desc())
            .first()
        )

    def update_message_suggested_questions(self, message_id: str, questions: list[str]) -> None:
        """将推荐问题写回到指定消息"""
        self.db.query(MessageModel).filter(MessageModel.id == message_id).update({"suggested_questions": questions})
        self.db.flush()

    def _convert_to_schema(self, db_conversation: ConversationModel) -> Conversation:
        """将数据库模型转换为业务模型"""
        messages = [self._convert_message_to_schema(msg) for msg in db_conversation.messages]
        return Conversation(
            id=db_conversation.id,
            user_id=db_conversation.user_id,
            model_id=db_conversation.model_id,
            title=db_conversation.title,
            messages=messages,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at,
        )


class MemoryRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_active(self, user_id: str) -> List[Memory]:
        """获取用户所有活跃且未删除的记忆"""
        return (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id, Memory.is_active == True, Memory.is_deleted == False)
            .order_by(Memory.created_at)
            .all()
        )

    def get_all(self, user_id: str) -> List[Memory]:
        """获取用户所有未删除的记忆（含停用的）"""
        return (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id, Memory.is_deleted == False)
            .order_by(Memory.created_at.desc())
            .all()
        )

    def create(self, memory_data: Dict[str, Any]) -> Memory:
        """创建新记忆"""
        memory = Memory(**memory_data)
        self.db.add(memory)
        self.db.flush()
        self.db.refresh(memory)
        return memory

    def update_content(self, memory_id: str, user_id: str, content: str) -> Optional[Memory]:
        """更新记忆内容"""
        memory = (
            self.db.query(Memory)
            .filter(Memory.id == memory_id, Memory.user_id == user_id, Memory.is_deleted == False)
            .first()
        )
        if not memory:
            return None
        memory.content = content
        memory.updated_at = get_china_time()
        self.db.flush()
        return memory

    def toggle_active(self, memory_id: str, user_id: str, is_active: bool) -> Optional[Memory]:
        """切换记忆启用/停用状态"""
        memory = (
            self.db.query(Memory)
            .filter(Memory.id == memory_id, Memory.user_id == user_id, Memory.is_deleted == False)
            .first()
        )
        if not memory:
            return None
        memory.is_active = is_active
        memory.updated_at = get_china_time()
        self.db.flush()
        return memory

    def soft_delete(self, memory_id: str, user_id: str) -> bool:
        """软删除记忆"""
        result = (
            self.db.query(Memory)
            .filter(Memory.id == memory_id, Memory.user_id == user_id, Memory.is_deleted == False)
            .update({"is_deleted": True, "updated_at": get_china_time()})
        )
        self.db.flush()
        return result > 0

    def count_active(self, user_id: str) -> int:
        """统计用户活跃记忆数量"""
        return (
            self.db.query(Memory)
            .filter(Memory.user_id == user_id, Memory.is_active == True, Memory.is_deleted == False)
            .count()
        )


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

    def count_conversation_files(self, conversation_id: str) -> int:
        """计算对话关联的文件数量"""
        try:
            return self.db.query(ConversationFile).filter(ConversationFile.conversation_id == conversation_id).count()
        except Exception as e:
            logger.error(f"计算对话文件数量失败: {e}")
            return 0

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
            result = self.db.query(File).filter(File.id == file_id, File.user_id == user_id).delete()
            if result > 0:
                self.db.commit()
                return True
            return False
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除文件失败: {e}")
            return False


class ProviderRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, enabled: Optional[bool] = None) -> List[Provider]:
        """获取所有 provider"""
        query = self.db.query(Provider)
        if enabled is not None:
            query = query.filter(Provider.enabled == enabled)
        return query.order_by(Provider.priority, Provider.name).all()

    def get_by_id(self, provider_id: str) -> Optional[Provider]:
        """根据 ID 获取 provider"""
        return self.db.query(Provider).filter(Provider.id == provider_id).first()

    def create(self, data: Dict[str, Any]) -> Provider:
        """创建 provider"""
        now = get_china_time()
        provider = Provider(
            id=data["id"],
            name=data["name"],
            auth_config=data.get("auth_config", {}),
            litellm_prefix=data["litellm_prefix"],
            custom_base_url=data.get("custom_base_url", False),
            priority=data.get("priority", 100),
            enabled=data.get("enabled", True),
            created_at=now,
            updated_at=now,
        )
        self.db.add(provider)
        self.db.commit()
        self.db.refresh(provider)
        return provider

    def update(self, provider_id: str, update_data: Dict[str, Any]) -> Optional[Provider]:
        """更新 provider"""
        provider = self.get_by_id(provider_id)
        if not provider:
            return None
        for key, value in update_data.items():
            if hasattr(provider, key):
                setattr(provider, key, value)
        provider.updated_at = get_china_time()
        self.db.commit()
        self.db.refresh(provider)
        return provider

    def delete(self, provider_id: str) -> bool:
        """删除 provider"""
        provider = self.get_by_id(provider_id)
        if not provider:
            return False
        self.db.delete(provider)
        self.db.commit()
        return True

    def to_schema(self, provider: Provider, order: int = 0) -> ProviderInfo:
        """转换为 Pydantic Schema"""
        auth_config = None
        if provider.auth_config:
            auth_config = AuthConfig(**provider.auth_config)

        return ProviderInfo(
            id=provider.id,
            name=provider.name,
            auth_config=auth_config,
            litellm_prefix=provider.litellm_prefix,
            custom_base_url=provider.custom_base_url,
            priority=provider.priority,
            enabled=provider.enabled,
            order=order,
        )


class ModelSourceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(
        self, provider: Optional[str] = None, enabled: Optional[bool] = None, capability: Optional[str] = None
    ) -> List[ModelSource]:
        """获取所有模型数据源，支持筛选"""
        query = self.db.query(ModelSource)

        # 应用筛选条件
        if provider:
            query = query.filter(ModelSource.provider == provider)

        if enabled is not None:
            query = query.filter(ModelSource.enabled == enabled)

        if capability:
            # JSON查询，需根据具体数据库类型调整
            query = query.filter(ModelSource.capabilities[capability].as_boolean() == True)

        # 按优先级排序，数字小的排前面
        query = query.order_by(ModelSource.priority, ModelSource.name)

        return query.all()

    def get_by_id(self, model_id: str) -> Optional[ModelSource]:
        """根据模型ID获取模型数据源"""
        return self.db.query(ModelSource).filter(ModelSource.model_id == model_id).first()

    def create(self, model_data: Dict[str, Any]) -> ModelSource:
        """创建新的模型数据源"""
        now = get_china_time()

        model_source = ModelSource(
            model_id=model_data.get("modelId"),
            name=model_data.get("name"),
            provider=model_data.get("provider"),
            knowledge_cutoff=model_data.get("knowledgeCutoff"),
            capabilities=model_data.get("capabilities"),
            pricing=model_data.get("pricing"),
            model_configuration=model_data.get("model_configuration"),
            priority=model_data.get("priority", 100),
            enabled=model_data.get("enabled", True),
            description=model_data.get("description", ""),
            created_at=now,
            updated_at=now,
        )

        self.db.add(model_source)
        self.db.commit()
        self.db.refresh(model_source)
        return model_source

    def update(self, model_id: str, update_data: Dict[str, Any]) -> Optional[ModelSource]:
        """更新模型数据源"""
        model_source = self.get_by_id(model_id)
        if not model_source:
            return None

        for key, value in update_data.items():
            field_name = key
            if key == "modelId":
                field_name = "model_id"
            elif key == "knowledgeCutoff":
                field_name = "knowledge_cutoff"

            if hasattr(model_source, field_name):
                setattr(model_source, field_name, value)

        model_source.updated_at = get_china_time()
        self.db.commit()
        self.db.refresh(model_source)
        return model_source

    def delete(self, model_id: str) -> bool:
        """删除模型数据源"""
        model_source = self.get_by_id(model_id)
        if not model_source:
            return False

        self.db.delete(model_source)
        self.db.commit()
        return True

    def get_providers(self) -> list:
        """获取所有启用的 provider 列表"""
        provider_repo = ProviderRepository(self.db)
        providers = provider_repo.get_all(enabled=True)
        return [
            {
                "id": p.id,
                "name": p.name,
                "order": idx,
            }
            for idx, p in enumerate(providers, start=1)
        ]

    def to_basic_schema(self, model_source: ModelSource) -> ModelBasicInfo:
        """将数据库模型转换为基础Pydantic模型"""
        capabilities = ModelCapabilities(**model_source.capabilities)

        return ModelBasicInfo(
            modelId=model_source.model_id,
            name=model_source.name,
            provider=model_source.provider,
            knowledgeCutoff=model_source.knowledge_cutoff,
            capabilities=capabilities,
            priority=model_source.priority,
            enabled=model_source.enabled,
            description=model_source.description,
        )

    def to_full_schema(self, model_source: ModelSource) -> ModelInfo:
        """将数据库模型转换为完整 Pydantic 模型"""
        capabilities = ModelCapabilities(**model_source.capabilities)
        pricing = ModelPricing(**model_source.pricing)

        # auth_config 从关联的 provider 读取
        auth_config = None
        if model_source.provider_rel and model_source.provider_rel.auth_config:
            fields = []
            for field_data in model_source.provider_rel.auth_config.get("fields", []):
                fields.append(AuthConfigField(**field_data))
            auth_config = AuthConfig(
                fields=fields,
                auth_type=model_source.provider_rel.auth_config.get("auth_type", "api_key"),
            )

        model_configuration = None
        if model_source.model_configuration:
            params = []
            for param_data in model_source.model_configuration.get("params", []):
                params.append(ModelConfigParam(**param_data))
            model_configuration = ModelConfiguration(params=params)

        return ModelInfo(
            modelId=model_source.model_id,
            name=model_source.name,
            provider=model_source.provider,
            knowledgeCutoff=model_source.knowledge_cutoff,
            capabilities=capabilities,
            pricing=pricing,
            auth_config=auth_config,
            model_configuration=model_configuration,
            priority=model_source.priority,
            enabled=model_source.enabled,
            description=model_source.description,
        )

    def to_schema(self, model_source: ModelSource) -> ModelInfo:
        """将数据库模型转换为 Pydantic 模型"""
        capabilities = ModelCapabilities(**model_source.capabilities)
        pricing = ModelPricing(**model_source.pricing)

        auth_config = None
        if model_source.provider_rel and model_source.provider_rel.auth_config:
            auth_config = AuthConfig(**model_source.provider_rel.auth_config)

        model_configuration = None
        if model_source.model_configuration:
            model_configuration = ModelConfiguration(**model_source.model_configuration)

        return ModelInfo(
            modelId=model_source.model_id,
            name=model_source.name,
            provider=model_source.provider,
            knowledgeCutoff=model_source.knowledge_cutoff,
            capabilities=capabilities,
            pricing=pricing,
            auth_config=auth_config,
            model_configuration=model_configuration,
            enabled=model_source.enabled,
            description=model_source.description,
        )


class ModelCredentialRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, model_id: Optional[str] = None) -> List[ModelCredential]:
        """获取所有凭证或指定模型的凭证"""
        query = self.db.query(ModelCredential)
        if model_id:
            query = query.filter(ModelCredential.model_id == model_id)
        return query.all()

    def get_by_id(self, credential_id: int) -> Optional[ModelCredential]:
        """根据ID获取凭证"""
        return self.db.query(ModelCredential).filter(ModelCredential.id == credential_id).first()

    def get_default(self, model_id: str) -> Optional[ModelCredential]:
        """获取模型的默认凭证"""
        return (
            self.db.query(ModelCredential)
            .filter(ModelCredential.model_id == model_id, ModelCredential.is_default == True)
            .first()
        )

    def create(self, credential_data: Dict[str, Any]) -> ModelCredential:
        """创建新的凭证"""
        # 如果设置为默认凭证，先取消其他默认凭证
        if credential_data.get("is_default", False):
            self._reset_default_status(credential_data["model_id"])

        credential = ModelCredential(**credential_data)
        self.db.add(credential)
        self.db.commit()
        self.db.refresh(credential)
        return credential

    def update(self, credential_id: int, update_data: Dict[str, Any]) -> Optional[ModelCredential]:
        """更新凭证"""
        credential = self.get_by_id(credential_id)
        if not credential:
            return None

        # 如果设置为默认凭证，先取消其他默认凭证
        if update_data.get("is_default", False) and not credential.is_default:
            self._reset_default_status(credential.model_id)

        # 更新字段
        for key, value in update_data.items():
            if hasattr(credential, key):
                setattr(credential, key, value)

        credential.updated_at = get_china_time()
        self.db.commit()
        self.db.refresh(credential)
        return credential

    def delete(self, credential_id: int) -> bool:
        """删除凭证"""
        credential = self.get_by_id(credential_id)
        if not credential:
            return False

        # 如果是默认凭证，可能需要设置另一个凭证为默认
        was_default = credential.is_default
        model_id = credential.model_id

        self.db.delete(credential)
        self.db.commit()

        # 如果删除的是默认凭证，尝试设置另一个为默认
        if was_default:
            self._set_new_default(model_id)

        return True

    def _reset_default_status(self, model_id: str) -> None:
        """重置指定模型的所有凭证的默认状态"""
        self.db.query(ModelCredential).filter(
            ModelCredential.model_id == model_id, ModelCredential.is_default == True
        ).update({"is_default": False})
        self.db.commit()

    def _set_new_default(self, model_id: str) -> None:
        """设置一个新的默认凭证"""
        # 获取第一个可用的凭证并设置为默认
        credential = self.db.query(ModelCredential).filter(ModelCredential.model_id == model_id).first()

        if credential:
            credential.is_default = True
            self.db.commit()

    def to_schema(self, credential: ModelCredential) -> ModelCredentialInfo:
        """将数据库模型转换为Pydantic模型"""
        return ModelCredentialInfo(
            id=credential.id,
            model_id=credential.model_id,
            name=credential.name,
            is_default=credential.is_default,
            credentials=credential.credentials,
            created_at=credential.created_at,
            updated_at=credential.updated_at,
        )
