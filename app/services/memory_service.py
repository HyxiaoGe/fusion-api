from sqlalchemy.orm import Session
from typing import List, Optional, Dict
from app.schemas.chat import Conversation
import json
import os


class MemoryService:
    """
    内存服务 - 管理对话历史和上下文记忆
    这里使用简单的文件存储实现，你也可以替换为数据库实现
    """

    def __init__(self, db: Session):
        self.db = db
        self.storage_dir = "./conversations"
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_conversation(self, conversation: Conversation) -> bool:
        """保存或更新对话"""
        try:
            file_path = os.path.join(self.storage_dir, f"{conversation.id}.json")
            with open(file_path, "w", encoding="utf-8") as f:
                # 转换为字典并序列化
                conv_dict = conversation.model_dump()
                # 转换datetime对象为字符串
                conv_dict["created_at"] = conv_dict["created_at"].isoformat()
                conv_dict["updated_at"] = conv_dict["updated_at"].isoformat()

                for msg in conv_dict["messages"]:
                    msg["created_at"] = msg["created_at"].isoformat()

                json.dump(conv_dict, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"保存对话失败: {e}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[Conversation]:
        """获取特定对话"""
        try:
            file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            if not os.path.exists(file_path):
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                conv_dict = json.load(f)
                # 转换回Conversation对象
                return Conversation.model_validate(conv_dict)
        except Exception as e:
            print(f"获取对话失败: {e}")
            return None

    def get_all_conversations(self) -> List[Conversation]:
        """获取所有对话"""
        conversations = []
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    conversation_id = filename[:-5]  # 去除.json后缀
                    conversation = self.get_conversation(conversation_id)
                    if conversation:
                        conversations.append(conversation)
            return conversations
        except Exception as e:
            print(f"获取所有对话失败: {e}")
            return []

    def delete_conversation(self, conversation_id: str) -> bool:
        """删除特定对话"""
        try:
            file_path = os.path.join(self.storage_dir, f"{conversation_id}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
                return True
            return False
        except Exception as e:
            print(f"删除对话失败: {e}")
            return False