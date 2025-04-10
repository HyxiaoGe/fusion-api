from typing import List, Dict, Any, Optional

import weaviate
import uuid

from app.ai.embeddings.text_embedder import TextEmbedder
from app.core.config import settings
from app.core.logger import app_logger as logger


class WeaviateVectorStore:
    """基于Weaviate的向量存储，用于语义搜索"""

    def __init__(self):
        if not settings.ENABLE_VECTOR_EMBEDDINGS:
            logger.info("向量存储功能已禁用")
            self.embedder = None
            self.client = None
            return

        self.embedder = TextEmbedder.get_instance()

        try:
            # 连接到Weaviate服务
            self.weaviate_url = settings.WEAVIATE_URL
            logger.info(f"连接到Weaviate服务: {self.weaviate_url}")
            
            self.client = weaviate.Client(self.weaviate_url)
            logger.info("成功连接到Weaviate服务")
            
            # 确保必要的schema存在
            self._ensure_schema()
        except Exception as e:
            logger.error(f"连接Weaviate服务失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _ensure_schema(self):
        """确保必要的schema存在"""
        schema = self.client.schema.get()
        classes = schema.get("classes", [])
        
        # 检查Message类
        if not any(c["class"] == "Message" for c in classes):
            message_class = {
                "class": "Message",
                "vectorizer": "none",  # 不使用内置向量化器
                "properties": [
                    {"name": "conversation_id", "dataType": ["string"]},
                    {"name": "role", "dataType": ["string"]},
                    {"name": "content", "dataType": ["text"]},
                    {"name": "content_preview", "dataType": ["string"]}
                ]
            }
            self.client.schema.create_class(message_class)
            logger.info("创建Message schema成功")

        # 检查Conversation类
        if not any(c["class"] == "Conversation" for c in classes):
            conversation_class = {
                "class": "Conversation",
                "vectorizer": "none",  # 不使用内置向量化器
                "properties": [
                    {"name": "title", "dataType": ["string"]},
                    {"name": "model", "dataType": ["string"]},
                    {"name": "summary", "dataType": ["text"]},
                    {"name": "summary_preview", "dataType": ["string"]}
                ]
            }
            self.client.schema.create_class(conversation_class)
            logger.info("创建Conversation schema成功")

    def add_message(self, message_id: str, conversation_id: str, role: str, content: str) -> bool:
        """添加消息向量到存储"""
        if not message_id or not conversation_id or not content:
            logger.error("添加消息参数无效")
            return False

        try:
            # 计算嵌入向量
            embedding = self.embedder.embed_text(content)

            if not embedding or len(embedding) == 0:
                logger.error(f"消息 {message_id} 嵌入计算失败")
                return False

            # 生成确定性UUID
            uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, message_id)
            
            # 准备元数据
            data_object = {
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }

            # 添加或更新记录
            self.client.data_object.create(
                data_object=data_object,
                class_name="Message",
                uuid=str(uuid_obj),
                vector=embedding  # 显式提供向量
            )

            logger.info(f"成功添加消息向量: {message_id}")
            return True
        except Exception as e:
            logger.error(f"添加消息向量失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def search_messages(self, query: str, limit: int = 5,
                        conversation_id: Optional[str] = None,
                        threshold: float = 0.5) -> List[Dict[str, Any]]:
        """搜索与查询相关的消息"""
        if not query or query.strip() == "":
            logger.error("搜索查询不能为空")
            return []

        try:
            # 计算查询的嵌入向量
            query_embedding = self.embedder.embed_text(query)

            if not query_embedding or len(query_embedding) == 0:
                logger.error("查询嵌入计算失败")
                return []

            # 准备查询条件
            where_filter = None
            if conversation_id:
                where_filter = {
                    "path": ["conversation_id"],
                    "operator": "Equal",
                    "valueString": conversation_id
                }

            # 执行查询
            query_builder = (
                self.client.query
                .get("Message", ["conversation_id", "role", "content", "content_preview"])
                .with_additional(["id", "certainty"])
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": threshold  # 相似度阈值
                })
                .with_limit(limit)
            )

            # 添加过滤条件
            if where_filter:
                query_builder = query_builder.with_where(where_filter)

            # 执行查询
            result = query_builder.do()

            # 处理结果
            messages = []
            if "data" in result and "Get" in result["data"] and "Message" in result["data"]["Get"]:
                for item in result["data"]["Get"]["Message"]:
                    # 获取相似度分数
                    similarity = item.get("_additional", {}).get("certainty", 0)
                    
                    # 应用相似度阈值过滤
                    if similarity >= threshold:
                        messages.append({
                            "id": item["_additional"]["id"],
                            "conversation_id": item["conversation_id"],
                            "role": item["role"],
                            "content": item["content"],
                            "content_preview": item["content_preview"],
                            "similarity": similarity
                        })

            # 按相似度排序
            sorted_results = sorted(messages, key=lambda x: x["similarity"], reverse=True)
            logger.info(f"过滤后的结果数量: {len(sorted_results)}")
            return sorted_results
        except Exception as e:
            logger.error(f"搜索消息失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def search_conversations(self, query: str, limit: int = 5,
                             threshold: float = 0.3) -> List[Dict[str, Any]]:
        """搜索相关对话"""
        if not query:
            logger.error("搜索查询不能为空")
            return []

        try:
            # 计算查询的嵌入向量
            query_embedding = self.embedder.embed_text(query)

            if not query_embedding or len(query_embedding) == 0:
                logger.error("查询嵌入计算失败")
                return []

            # 执行查询
            result = (
                self.client.query
                .get("Conversation", ["title", "model", "summary", "summary_preview"])
                .with_additional(["id", "certainty"])
                .with_near_vector({
                    "vector": query_embedding,
                    "certainty": threshold
                })
                .with_limit(limit)
                .do()
            )

            # 处理结果
            conversations = []
            if "data" in result and "Get" in result["data"] and "Conversation" in result["data"]["Get"]:
                for item in result["data"]["Get"]["Conversation"]:
                    similarity = item.get("_additional", {}).get("certainty", 0)
                    
                    if similarity >= threshold:
                        conversations.append({
                            "id": item["_additional"]["id"],
                            "title": item.get("title", ""),
                            "model": item.get("model", ""),
                            "summary": item.get("summary", ""),
                            "summary_preview": item.get("summary_preview", ""),
                            "similarity": similarity
                        })

            # 按相似度排序
            sorted_results = sorted(conversations, key=lambda x: x["similarity"], reverse=True)
            logger.info(f"对话结果数量: {len(sorted_results)}")
            return sorted_results
        except Exception as e:
            logger.error(f"搜索对话失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_related_context(self, query: str, conversation_id: Optional[str] = None,
                            limit: int = 3, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """获取与查询相关的上下文信息，用于增强提示"""
        try:
            logger.info(f"开始获取相关上下文: query='{query}', conversation_id='{conversation_id}'")

            # 搜索相关消息
            results = self.search_messages(
                query=query,
                limit=limit,
                conversation_id=conversation_id,
                threshold=threshold
            )

            logger.info(f"相关上下文搜索结果数量: {len(results)}")
            if results:
                logger.info(f"最高相似度上下文: {results[0]['similarity']:.4f}")

            return results
        except Exception as e:
            logger.error(f"获取相关上下文失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def add_conversation_summary(self, conversation_id: str, title: str, summary: str, model: str) -> bool:
        """添加对话摘要向量"""
        try:
            # 计算嵌入向量
            embedding = self.embedder.embed_text(summary)

            if not embedding or len(embedding) == 0:
                logger.error(f"对话摘要 {conversation_id} 嵌入计算失败")
                return False

            # 生成UUID
            uuid_obj = uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id)
            
            # 添加对象
            self.client.data_object.create(
                data_object={
                    "title": title,
                    "model": model,
                    "summary": summary,
                    "summary_preview": summary[:100] + "..." if len(summary) > 100 else summary
                },
                class_name="Conversation",
                uuid=str(uuid_obj),
                vector=embedding
            )

            logger.info(f"成功添加对话摘要向量: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"添加对话摘要向量失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def delete_conversation_data(self, conversation_id: str) -> bool:
        """删除特定对话的所有向量数据"""
        try:
            # 删除对话摘要
            conversation_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, conversation_id))
            try:
                self.client.data_object.delete(
                    uuid=conversation_uuid,
                    class_name="Conversation"
                )
                logger.info(f"删除对话摘要成功: {conversation_id}")
            except Exception as e:
                logger.warning(f"删除对话摘要时出错 (可能不存在): {e}")
            
            # 删除对话中的所有消息
            # Weaviate的where过滤查询
            result = (
                self.client.query
                .get("Message", ["conversation_id"])
                .with_additional(["id"])
                .with_where({
                    "path": ["conversation_id"],
                    "operator": "Equal",
                    "valueString": conversation_id
                })
                .do()
            )
            
            # 删除找到的所有消息
            if "data" in result and "Get" in result["data"] and "Message" in result["data"]["Get"]:
                for message in result["data"]["Get"]["Message"]:
                    message_id = message["_additional"]["id"]
                    self.client.data_object.delete(
                        uuid=message_id,
                        class_name="Message"
                    )
            
            logger.info(f"成功删除对话数据: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"删除对话向量数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False