import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

from sqlalchemy import desc, func
from sqlalchemy.orm import Session, joinedload

from app.ai.llm_manager import get_model_display_name
from app.db.models import Conversation as ConversationModel, get_china_time, File, ConversationFile
from app.db.models import Message as MessageModel
from app.db.models import PromptTemplate as PromptTemplateModel
from app.db.models import Setting as SettingModel
from app.db.models import HotTopic as HotTopicModel
from app.db.models import ScheduledTask as ScheduledTaskModel
from app.db.models import ModelSource, ModelCredential
from app.schemas.chat import Conversation, Message
from app.schemas.prompts import PromptTemplate
from app.schemas.models import ModelInfo, ModelCapabilities, ModelPricing, AuthConfig, ModelConfiguration, ModelConfigParam, ModelBasicInfo, AuthConfigField, ModelCredentialInfo

logger = logging.getLogger(__name__)


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, conversation: Conversation) -> Conversation:
        """创建新的对话"""
        try:
            # 转换为数据库模型
            db_conversation = ConversationModel(
                id=conversation.id,
                title=conversation.title,
                provider=conversation.provider,
                model=conversation.model,
                created_at=conversation.created_at,
                updated_at=conversation.updated_at
            )

            # 添加消息
            for msg in conversation.messages:
                db_message = MessageModel(
                    id=msg.id,
                    role=msg.role,
                    content=msg.content,
                    created_at=msg.created_at
                )
                db_conversation.messages.append(db_message)

            # 保存到数据库
            self.db.add(db_conversation)
            self.db.commit()
            self.db.refresh(db_conversation)

            # 转换回业务模型
            return self._convert_to_schema(db_conversation)
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建对话失败: {e}")
            raise

    def update(self, conversation: Conversation) -> Conversation:
        """更新现有对话"""
        try:
            # 查找现有对话
            db_conversation = self.db.query(ConversationModel).filter(
                ConversationModel.id == conversation.id
            ).first()

            if not db_conversation:
                raise ValueError(f"找不到对话ID: {conversation.id}")

            # 更新对话属性
            db_conversation.title = conversation.title
            db_conversation.model = conversation.model
            db_conversation.updated_at = get_china_time()

            # 删除旧消息并添加新消息
            self.db.query(MessageModel).filter(
                MessageModel.conversation_id == conversation.id
            ).delete()

            for msg in conversation.messages:
                db_message = MessageModel(
                    id=msg.id,
                    conversation_id=conversation.id,
                    role=msg.role,
                    content=msg.content,
                    created_at=msg.created_at
                )
                self.db.add(db_message)

            # 提交更改
            self.db.commit()
            self.db.refresh(db_conversation)

            # 转换回业务模型
            return self._convert_to_schema(db_conversation)
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新对话失败: {e}")
            raise

    def delete(self, conversation_id: str) -> bool:
        """删除对话"""
        try:
            # 查找对话
            result = self.db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).delete()

            self.db.commit()
            return result > 0
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除对话失败: {e}")
            return False

    def get_by_id(self, conversation_id: str) -> Optional[Conversation]:
        """根据ID获取对话"""
        try:
            db_conversation = self.db.query(ConversationModel).filter(
                ConversationModel.id == conversation_id
            ).first()

            if not db_conversation:
                return None

            return self._convert_to_schema(db_conversation)
        except Exception as e:
            logger.error(f"获取对话失败: {e}")
            return None

    def get_all(self) -> List[Conversation]:
        """获取所有对话"""
        try:
            db_conversations = self.db.query(ConversationModel).order_by(
                ConversationModel.updated_at.desc()
            ).all()

            return [self._convert_to_schema(db_conv) for db_conv in db_conversations]
        except Exception as e:
            logger.error(f"获取所有对话失败: {e}")
            return []

    def get_paginated(self, page: int = 1, page_size: int = 20) -> Tuple[List[Conversation], int]:
        """分页获取对话列表（不包含消息内容）"""
        try:
            # 计算偏移量
            offset = (page - 1) * page_size
            
            # 获取总数
            total = self.db.query(ConversationModel).count()
            
            # 获取分页数据，不加载messages
            db_conversations = (
                self.db.query(ConversationModel)
                .order_by(ConversationModel.updated_at.desc())
                .offset(offset)
                .limit(page_size)
                .all()
            )
            
            # 转换为Conversation对象，但不包含messages
            conversations = []
            for db_conv in db_conversations:
                conversation = Conversation(
                    id=db_conv.id,
                    title=db_conv.title,
                    provider=get_model_display_name(db_conv.model),
                    model=db_conv.model,
                    messages=[],  # 空的messages数组
                    created_at=db_conv.created_at,
                    updated_at=db_conv.updated_at
                )
                conversations.append(conversation)
            
            return conversations, total
            
        except Exception as e:
            logger.error(f"分页获取对话失败: {e}")
            return [], 0

    def _convert_to_schema(self, db_conversation: ConversationModel) -> Conversation:
        """将数据库模型转换为业务模型"""
        messages = []
        for db_msg in db_conversation.messages:
            messages.append(Message(
                id=db_msg.id,
                role=db_msg.role,
                content=db_msg.content,
                created_at=db_msg.created_at
            ))

        return Conversation(
            id=db_conversation.id,
            title=db_conversation.title,
            provider=get_model_display_name(db_conversation.model),
            model=db_conversation.model,
            messages=messages,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at
        )


class FileRepository:
    def __init__(self, db: Session):
        self.db = db

    def create_file(self, file_data: Dict[str, Any]) -> File:
        """创建新文件记录"""
        try:
            db_file = File(
                id=file_data.get("id", str(uuid.uuid4())),
                filename=file_data["filename"],
                original_filename=file_data["original_filename"],
                mimetype=file_data["mimetype"],
                size=file_data["size"],
                path=file_data["path"],
                status=file_data.get("status", "pending"),
                processing_result=file_data.get("processing_result")
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

    def get_file_by_id(self, file_id: str) -> Optional[File]:
        """通过ID获取文件"""
        try:
            return self.db.query(File).filter(File.id == file_id).first()
        except Exception as e:
            logger.error(f"获取文件失败: {e}")
            return None

    def get_files_info(self, file_ids: List[str]) -> List[File]:
        """获取多个文件的信息"""
        try:
            return self.db.query(File).filter(File.id.in_(file_ids)).all()
        except Exception as e:
            logger.error(f"获取文件信息失败: {e}")
            return []

    def get_file_paths(self, file_ids: List[str]) -> List[str]:
        """获取多个文件的路径"""
        try:
            files = self.db.query(File).filter(File.id.in_(file_ids)).all()
            return [file.path for file in files]
        except Exception as e:
            logger.error(f"获取文件路径失败: {e}")
            return []

    def update_file(self, file_id: str, updates: Dict[str, Any]) -> bool:
        """更新文件信息"""
        try:
            result = self.db.query(File).filter(File.id == file_id).update(updates)
            self.db.commit()
            return result > 0
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新文件失败: {e}")
            return False

    def delete_file(self, file_id: str) -> bool:
        """删除文件记录"""
        try:
            self.db.query(File).filter(File.id == file_id).delete()
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除文件失败: {e}")
            return False


class HotTopicRepository:
    """热点话题数据仓库"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def create(self, hot_topic: HotTopicModel) -> HotTopicModel:
        """创建新的热点话题"""
        try:
            self.db.add(hot_topic)
            self.db.commit()
            self.db.refresh(hot_topic)
            return hot_topic
        except Exception as e:
            self.db.rollback()
            raise e
            
    def exists_by_url(self, url: str) -> bool:
        """检查指定URL的热点是否已存在"""
        if not url:
            return False
        return self.db.query(HotTopicModel).filter(HotTopicModel.url == url).first() is not None
        
    def get_hot_topics(self, category: Optional[str] = None, limit: int = 10) -> List[HotTopicModel]:
        """获取热点话题列表，可按分类筛选"""
        query = self.db.query(HotTopicModel)
        
        if category:
            query = query.filter(HotTopicModel.category == category)
            
        # 先按浏览次数排序，再按发布时间排序
        return query.order_by(desc(HotTopicModel.published_at)).limit(limit).all()
        
    def get_topic_by_id(self, topic_id: str) -> Optional[HotTopicModel]:
        """根据ID获取热点话题"""
        return self.db.query(HotTopicModel).filter(HotTopicModel.id == topic_id).first()
        
    def delete_before_date(self, date: datetime) -> int:
        """删除指定日期之前的热点话题"""
        try:
            result = self.db.query(HotTopicModel).filter(HotTopicModel.published_at < date).delete()
            self.db.commit()
            return result
        except Exception as e:
            self.db.rollback()
            raise e
            
    def increment_view_count(self, topic_id: str) -> bool:
        """增加热点的浏览计数"""
        try:
            topic = self.db.query(HotTopicModel).filter(HotTopicModel.id == topic_id).first()
            if not topic:
                return False
                
            topic.view_count += 1
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            return False


class ScheduledTaskRepository:
    """定时任务数据仓库"""
    
    def __init__(self, db: Session):
        self.db = db
        
    def get_task_by_name(self, name: str) -> Optional[ScheduledTaskModel]:
        """根据名称获取任务"""
        return self.db.query(ScheduledTaskModel).filter(ScheduledTaskModel.name == name).first()
        
    def get_all_active_tasks(self) -> List[ScheduledTaskModel]:
        """获取所有活跃的任务"""
        return self.db.query(ScheduledTaskModel).filter(ScheduledTaskModel.status == "active").all()
        
    def create_task(self, task_data: Dict[str, Any]) -> ScheduledTaskModel:
        """创建新任务"""
        try:
            task = ScheduledTaskModel(**task_data)
            self.db.add(task)
            self.db.commit()
            self.db.refresh(task)
            return task
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建定时任务失败: {e}")
            raise
            
    def update_task(self, name: str, update_data: Dict[str, Any]) -> bool:
        """更新任务信息"""
        try:
            task = self.get_task_by_name(name)
            if not task:
                return False
                
            for key, value in update_data.items():
                if hasattr(task, key):
                    setattr(task, key, value)
                    
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新定时任务失败: {e}")
            return False
            
    def update_last_run(self, name: str, new_task_data: Optional[Dict] = None) -> bool:
        """更新任务的最后执行时间和下次执行时间"""
        try:
            task = self.get_task_by_name(name)
            if not task:
                return False
                
            now = datetime.now()
            task.last_run_at = now
            
            # 计算下次执行时间
            if task.interval:
                from datetime import timedelta
                task.next_run_at = now + timedelta(seconds=task.interval)
                
            # 更新任务数据
            if new_task_data is not None:
                task.task_data = new_task_data
                
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新任务执行时间失败: {e}")
            return False
    
    def should_run_task(self, name: str) -> bool:
        """检查任务是否应该执行"""
        task = self.get_task_by_name(name)
        if not task or task.status != "active":
            return False
            
        if not task.next_run_at:
            return True
            
        return datetime.now() >= task.next_run_at


class PromptTemplateRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, prompt: PromptTemplate) -> PromptTemplate:
        """创建新的提示词模板"""
        try:
            db_prompt = PromptTemplateModel(
                id=prompt.id,
                title=prompt.title,
                content=prompt.content,
                tags=prompt.tags,
                created_at=prompt.created_at,
                updated_at=prompt.updated_at
            )

            self.db.add(db_prompt)
            self.db.commit()
            self.db.refresh(db_prompt)

            return PromptTemplate(
                id=db_prompt.id,
                title=db_prompt.title,
                content=db_prompt.content,
                tags=db_prompt.tags,
                created_at=db_prompt.created_at,
                updated_at=db_prompt.updated_at
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"创建提示词模板失败: {e}")
            raise

    def get_all(self) -> List[PromptTemplate]:
        """获取所有提示词模板"""
        try:
            db_prompts = self.db.query(PromptTemplateModel).all()

            return [
                PromptTemplate(
                    id=db_prompt.id,
                    title=db_prompt.title,
                    content=db_prompt.content,
                    tags=db_prompt.tags,
                    created_at=db_prompt.created_at,
                    updated_at=db_prompt.updated_at
                )
                for db_prompt in db_prompts
            ]
        except Exception as e:
            logger.error(f"获取所有提示词模板失败: {e}")
            return []
            
    def update(self, prompt_id: str, prompt_data: Dict) -> Optional[PromptTemplate]:
        """更新提示词模板"""
        try:
            db_prompt = self.db.query(PromptTemplateModel).filter(
                PromptTemplateModel.id == prompt_id
            ).first()
            
            if not db_prompt:
                return None
                
            # 更新字段
            for key, value in prompt_data.items():
                if hasattr(db_prompt, key):
                    setattr(db_prompt, key, value)
            
            db_prompt.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(db_prompt)
            
            return PromptTemplate(
                id=db_prompt.id,
                title=db_prompt.title,
                content=db_prompt.content,
                tags=db_prompt.tags,
                created_at=db_prompt.created_at,
                updated_at=db_prompt.updated_at
            )
        except Exception as e:
            self.db.rollback()
            logger.error(f"更新提示词模板失败: {e}")
            return None
            
    def delete(self, prompt_id: str) -> bool:
        """删除提示词模板"""
        try:
            db_prompt = self.db.query(PromptTemplateModel).filter(
                PromptTemplateModel.id == prompt_id
            ).first()
            
            if not db_prompt:
                return False
                
            self.db.delete(db_prompt)
            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"删除提示词模板失败: {e}")
            return False
            
    def load_to_prompt_manager(self) -> None:
        """将所有提示词模板加载到提示词管理器中"""
        try:
            from app.ai.prompts import prompt_manager
            
            # 获取所有提示词模板
            templates = self.get_all()
            
            # 加载到提示词管理器
            for template in templates:
                # 使用标题作为模板名称（转为snake_case）
                name = template.title.lower().replace(' ', '_')
                prompt_manager.add_template(name, template.content)
                
            logger.info(f"成功加载 {len(templates)} 个提示词模板到提示词管理器")
        except Exception as e:
            logger.error(f"加载提示词模板到提示词管理器失败: {e}")
            
    def get_by_id(self, prompt_id: str) -> Optional[PromptTemplate]:
        """根据ID获取提示词模板"""
        try:
            db_prompt = self.db.query(PromptTemplateModel).filter(
                PromptTemplateModel.id == prompt_id
            ).first()
            
            if not db_prompt:
                return None
                
            return PromptTemplate(
                id=db_prompt.id,
                title=db_prompt.title,
                content=db_prompt.content,
                tags=db_prompt.tags,
                created_at=db_prompt.created_at,
                updated_at=db_prompt.updated_at
            )
        except Exception as e:
            logger.error(f"获取提示词模板失败: {e}")
            return None


class SettingRepository:
    def __init__(self, db: Session):
        self.db = db

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取设置值"""
        try:
            db_setting = self.db.query(SettingModel).filter(
                SettingModel.key == key
            ).first()

            if not db_setting:
                return None

            return db_setting.value
        except Exception as e:
            logger.error(f"获取设置失败: {e}")
            return None

    def set(self, key: str, value: Dict[str, Any]) -> bool:
        """设置或更新设置值"""
        try:
            db_setting = self.db.query(SettingModel).filter(
                SettingModel.key == key
            ).first()

            if db_setting:
                db_setting.value = value
                db_setting.updated_at = datetime.utcnow()
            else:
                db_setting = SettingModel(
                    key=key,
                    value=value
                )
                self.db.add(db_setting)

            self.db.commit()
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"设置值失败: {e}")
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
    
    def create(self, model_data: Dict[str, Any]) -> ModelSource:
        """创建新的模型数据源"""
        now = get_china_time()
        
        # 将Pydantic模型转换为数据库模型
        model_source = ModelSource(
            model_id=model_data.get("modelId"),
            name=model_data.get("name"),
            provider=model_data.get("provider"),
            knowledge_cutoff=model_data.get("knowledgeCutoff"),
            capabilities=model_data.get("capabilities"),
            pricing=model_data.get("pricing"),
            auth_config=model_data.get("auth_config"),
            model_configuration=model_data.get("model_configuration"),
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