from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.db.models import Conversation as ConversationModel, get_china_time
from app.db.models import Message as MessageModel
from app.db.models import PromptTemplate as PromptTemplateModel
from app.db.models import Setting as SettingModel
from app.schemas.chat import Conversation, Message
from app.schemas.prompts import PromptTemplate
from datetime import datetime
import logging
from app.ai.llm_manager import get_model_display_name

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
            model=get_model_display_name(db_conversation.model),
            messages=messages,
            created_at=db_conversation.created_at,
            updated_at=db_conversation.updated_at
        )


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