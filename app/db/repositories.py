import logging
import re
import uuid
from copy import deepcopy
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from app.db.models import Conversation as ConversationModel, get_china_time, File, ConversationFile
from app.db.models import Message as MessageModel
from app.db.models import ModelSource, ModelCredential
from app.db.models import User as UserModel, SocialAccount as SocialAccountModel
from app.schemas.chat import Conversation, Message, TextBlock, ThinkingBlock, FileBlock, SearchBlock, SearchSource, Usage
from app.schemas.models import ModelInfo, ModelCapabilities, ModelPricing, AuthConfig, ModelConfiguration, ModelConfigParam, ModelBasicInfo, AuthConfigField, ModelCredentialInfo
from app.schemas.auth import User as UserSchema

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
        return self.db.query(SocialAccountModel).filter(
            SocialAccountModel.provider == provider,
            SocialAccountModel.provider_user_id == provider_user_id
        ).first()

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
                updated_at=conversation.updated_at
            )

            # 添加消息（content blocks 序列化为 JSONB）
            for msg in conversation.messages:
                db_message = MessageModel(
                    id=msg.id,
                    role=msg.role,
                    content=[block.model_dump() for block in msg.content],
                    model_id=msg.model_id,
                    usage=msg.usage.model_dump() if msg.usage else None,
                    created_at=msg.created_at
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
            db_conversation = self.db.query(ConversationModel).filter(
                ConversationModel.id == conversation.id,
                ConversationModel.user_id == conversation.user_id
            ).first()

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
        self.db.query(ConversationModel)\
            .filter(ConversationModel.id == conversation_id)\
            .update({"title": title, "updated_at": get_china_time()})
        self.db.flush()

    def delete(self, conversation_id: str, user_id: str) -> bool:
        """删除对话"""
        try:
            # 查找对话
            result = self.db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id
            ).delete()

            self.db.commit()
            return result > 0
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除对话失败: {e}")
            return False

    def get_by_id(self, conversation_id: str, user_id: str) -> Optional[Conversation]:
        """根据ID获取对话"""
        try:
            db_conversation = self.db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id,
                ConversationModel.user_id == user_id
            ).first()

            if not db_conversation:
                return None

            return self._convert_to_schema(db_conversation)
        except Exception as e:
            logger.error(f"获取对话失败: {e}")
            return None

    def get_all(self, user_id: str) -> List[Conversation]:
        """获取指定用户的所有对话"""
        try:
            db_conversations = self.db.query(ConversationModel).filter(
                ConversationModel.user_id == user_id
            ).order_by(
                ConversationModel.updated_at.desc()
            ).all()

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

            db_conversations = (
                query
                .order_by(ConversationModel.updated_at.desc())
                .offset(offset)
                .limit(page_size)
                .all()
            )

            conversations = []
            for db_conv in db_conversations:
                conversation = Conversation(
                    id=db_conv.id,
                    user_id=db_conv.user_id,
                    model_id=db_conv.model_id,
                    title=db_conv.title,
                    messages=[],
                    created_at=db_conv.created_at,
                    updated_at=db_conv.updated_at
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
                        value = [block.model_dump() if hasattr(block, 'model_dump') else block for block in value]
                    if key == "usage" and hasattr(value, 'model_dump'):
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
        for block_data in (db_message.content or []):
            block_type = block_data.get("type")
            if block_type == "text":
                content_blocks.append(TextBlock(**block_data))
            elif block_type == "thinking":
                content_blocks.append(ThinkingBlock(**block_data))
            elif block_type == "file":
                content_blocks.append(FileBlock(**block_data))
            elif block_type == "search":
                content_blocks.append(SearchBlock(
                    type="search",
                    id=block_data.get("id", f"blk_{__import__('uuid').uuid4().hex[:12]}"),
                    query=block_data.get("query", ""),
                    sources=[SearchSource(**s) for s in block_data.get("sources", [])],
                ))
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

    def update_message_suggested_questions(
        self, message_id: str, questions: list[str]
    ) -> None:
        """将推荐问题写回到指定消息"""
        self.db.query(MessageModel).filter(MessageModel.id == message_id).update(
            {"suggested_questions": questions}
        )
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
            existing = self.db.query(ConversationFile).filter(
                ConversationFile.conversation_id == conversation_id,
                ConversationFile.file_id == file_id
            ).first()

            if existing:
                return True

            # 创建新关联
            conv_file = ConversationFile(
                conversation_id=conversation_id,
                file_id=file_id
            )
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
            files = self.db.query(File).filter(
                File.id.in_(file_ids),
                File.status == "processed"
            ).all()

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
            return self.db.query(ConversationFile).filter(
                ConversationFile.conversation_id == conversation_id
            ).options(joinedload(ConversationFile.file)).all()
        except Exception as e:
            logger.error(f"获取对话文件失败: {e}")
            return []

    def count_conversation_files(self, conversation_id: str) -> int:
        """计算对话关联的文件数量"""
        try:
            return self.db.query(ConversationFile).filter(
                ConversationFile.conversation_id == conversation_id
            ).count()
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


class ModelSourceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all(self, 
                provider: Optional[str] = None, 
                enabled: Optional[bool] = None,
                capability: Optional[str] = None) -> List[ModelSource]:
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

    def _get_provider_template(self, provider: Optional[str]) -> Optional[ModelSource]:
        """获取同 provider 的模板模型，用于继承认证和参数配置。"""
        if not provider:
            return None

        candidates = (
            self.db.query(ModelSource)
            .filter(ModelSource.provider == provider)
            .order_by(ModelSource.priority, ModelSource.id)
            .all()
        )

        for candidate in candidates:
            if candidate.auth_config or candidate.model_configuration:
                return candidate

        return None
    
    def create(self, model_data: Dict[str, Any]) -> ModelSource:
        """创建新的模型数据源"""
        now = get_china_time()
        provider_template = self._get_provider_template(model_data.get("provider"))

        auth_config = model_data.get("auth_config")
        if auth_config is None and provider_template and provider_template.auth_config:
            auth_config = deepcopy(provider_template.auth_config)

        model_configuration = model_data.get("model_configuration")
        if model_configuration is None and provider_template and provider_template.model_configuration:
            model_configuration = deepcopy(provider_template.model_configuration)
        
        # 将Pydantic模型转换为数据库模型
        model_source = ModelSource(
            model_id=model_data.get("modelId"),
            name=model_data.get("name"),
            provider=model_data.get("provider"),
            knowledge_cutoff=model_data.get("knowledgeCutoff"),
            capabilities=model_data.get("capabilities"),
            pricing=model_data.get("pricing"),
            auth_config=auth_config,
            model_configuration=model_configuration,
            priority=model_data.get("priority", 100),
            enabled=model_data.get("enabled", True),
            description=model_data.get("description", ""),
            created_at=now,
            updated_at=now
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
        
        # 更新属性
        for key, value in update_data.items():
            # 转换键名格式
            field_name = key
            if key == "modelId":
                field_name = "model_id"
            elif key == "knowledgeCutoff":
                field_name = "knowledge_cutoff"
            elif key == "auth_config":
                field_name = "auth_config"
            elif key == "model_configuration":
                field_name = "model_configuration"
                
            # 设置值
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
            description=model_source.description
        )
        
    def to_full_schema(self, model_source: ModelSource) -> ModelInfo:
        """将数据库模型转换为完整Pydantic模型"""
        capabilities = ModelCapabilities(**model_source.capabilities)
        pricing = ModelPricing(**model_source.pricing)
        
        auth_config = None
        if model_source.auth_config:
            fields = []
            for field_data in model_source.auth_config.get("fields", []):
                fields.append(AuthConfigField(**field_data))
            
            auth_config = AuthConfig(
                fields=fields,
                auth_type=model_source.auth_config.get("auth_type", "api_key")
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
            description=model_source.description
        )
    
    def to_schema(self, model_source: ModelSource) -> ModelInfo:
        """将数据库模型转换为Pydantic模型"""
        capabilities = ModelCapabilities(**model_source.capabilities)
        pricing = ModelPricing(**model_source.pricing)
        
        auth_config = None
        if model_source.auth_config:
            auth_config = AuthConfig(**model_source.auth_config)
            
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
            description=model_source.description
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
        return self.db.query(ModelCredential).filter(
            ModelCredential.model_id == model_id,
            ModelCredential.is_default == True
        ).first()
    
    def create(self, credential_data: Dict[str, Any]) -> ModelCredential:
        """创建新的凭证"""
        # 如果设置为默认凭证，先取消其他默认凭证
        if credential_data.get('is_default', False):
            self._reset_default_status(credential_data['model_id'])
            
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
        if update_data.get('is_default', False) and not credential.is_default:
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
            ModelCredential.model_id == model_id,
            ModelCredential.is_default == True
        ).update({'is_default': False})
        self.db.commit()
    
    def _set_new_default(self, model_id: str) -> None:
        """设置一个新的默认凭证"""
        # 获取第一个可用的凭证并设置为默认
        credential = self.db.query(ModelCredential).filter(
            ModelCredential.model_id == model_id
        ).first()
        
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
            updated_at=credential.updated_at
        )
