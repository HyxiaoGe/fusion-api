import asyncio
import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from app.schemas.chat import Message, Conversation
from app.services.vector_service import VectorService


class MessageProcessor:
    def __init__(self, db):
        self.db = db
        self.vector_service = VectorService.get_instance(db)

    def prepare_chat_messages(self, chat_history):
        """准备发送给LLM的消息格式"""
        messages = []
        last_role = None
        
        for msg in chat_history:
            current_role = msg["role"]
            
            # 如果是 reasoning 角色的消息，跳过
            if current_role == "reasoning":
                continue
                
            # 检查是否有连续的角色
            if last_role and last_role == current_role:
                # 如果是连续的用户消息，合并内容
                if current_role == "user":
                    messages[-1].content += "\n" + msg["content"]
                    continue
                # 如果是连续的助手消息，跳过
                elif current_role == "assistant":
                    continue
            
            # 添加消息
            if current_role == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif current_role == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif current_role == "system":
                messages.append(SystemMessage(content=msg["content"]))
            
            last_role = current_role

        return messages

    def enhance_with_file_content(self, messages, message, file_contents):
        """使用文件内容增强消息"""
        if not file_contents:
            return messages
            
        # 将文件内容添加到消息中
        file_content_text = "\n\n".join([
            f"文件内容 ({i + 1}):\n{content}"
            for i, content in enumerate(file_contents.values())
        ])

        enhanced_message = f"""用户问题: {message}\n\n参考以下文件内容:\n{file_content_text}"""
        # 替换最后一条消息内容
        messages[-1] = HumanMessage(content=enhanced_message)
        
        return messages

    async def vectorize_message_async(self, message: Message, conversation_id: str):
        """异步向量化单条消息"""
        # 为了提高回复速度，暂时关闭
        return
        # try:
        #     # 线程池执行CPU密集型向量化操作
        #     loop = asyncio.get_event_loop()
        #     await loop.run_in_executor(None, self.vector_service.vectorize_message, message, conversation_id)
        # except Exception as e:
        #     logging.error(f"异步向量化消息失败: {e}")

    async def vectorize_messages_async(self, messages: List[Message], conversation_id: str):
        """异步向量化多条消息"""
        try:
            for message in messages:
                await self.vectorize_message_async(message, conversation_id)
        except Exception as e:
            logging.error(f"异步向量化多条消息失败: {e}") 