import os
from typing import List, Dict, Any, Optional

from langchain.vectorstores import Chroma

from app.ai.embeddings.text_embedder import TextEmbedder
from app.core.logger import app_logger as logger


class ChromaVectorStore:
    """增强版向量存储，用于管理消息和对话向量"""

    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embedder = TextEmbedder.get_instance()

        # 确保存储目录存在
        os.makedirs(persist_directory, exist_ok=True)

        # 初始化向量存储集合
        self.message_store = self._init_collection("message_vectors")
        self.conversation_store = self._init_collection("conversation_vectors")

    def _init_collection(self, collection_name: str) -> Chroma:
        """初始化指定集合的向量存储"""
        try:
            return Chroma(
                collection_name=collection_name,
                embedding_function=self.embedder.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception as e:
            logger.error(f"初始化集合 {collection_name} 失败: {e}")
            raise

    def add_message(self, message_id: str, conversation_id: str,
                    role: str, content: str) -> bool:
        """添加消息向量到存储"""
        try:
            # 检查是否已存在
            existing = self.message_store.get(
                where={"id": message_id}
            )

            if existing and len(existing['ids']) > 0:
                # 如果已存在，更新内容
                self.message_store.delete(
                    where={"id": message_id}
                )

            # 添加新向量
            self.message_store.add_texts(
                texts=[content],
                metadatas=[{
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "role": role,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content
                }],
                ids=[message_id]
            )

            # 持久化存储
            self.message_store.persist()
            return True
        except Exception as e:
            logger.error(f"添加消息向量失败: {e}")
            return False

    def add_conversation_summary(self, conversation_id: str,
                                 title: str, summary: str,
                                 model: str) -> bool:
        """添加对话摘要向量到存储"""
        try:
            # 检查是否已存在
            existing = self.conversation_store.get(
                where={"id": conversation_id}
            )

            if existing and len(existing['ids']) > 0:
                # 如果已存在，更新内容
                self.conversation_store.delete(
                    where={"id": conversation_id}
                )

            # 添加新向量
            self.conversation_store.add_texts(
                texts=[summary],
                metadatas=[{
                    "id": conversation_id,
                    "title": title,
                    "model": model,
                    "summary_preview": summary[:100] + "..." if len(summary) > 100 else summary
                }],
                ids=[conversation_id]
            )

            # 持久化存储
            self.conversation_store.persist()
            return True
        except Exception as e:
            logger.error(f"添加对话摘要向量失败: {e}")
            return False

    def search_messages(self, query: str, limit: int = 5,
                        conversation_id: Optional[str] = None,
                        threshold: float = 0.6) -> List[Dict[str, Any]]:
        """搜索相关消息"""
        try:
            filter_dict = {}
            if conversation_id:
                filter_dict["conversation_id"] = conversation_id

            results = self.message_store.similarity_search_with_score(
                query=query,
                k=limit * 2,  # 获取更多结果，以便后续过滤
                filter=filter_dict if filter_dict else None
            )

            # 过滤低相关性结果
            filtered_results = []
            for doc, score in results:
                # 分数越低表示越相似
                similarity = 1.0 - score
                if similarity >= threshold:
                    filtered_results.append({
                        "id": doc.metadata.get("id"),
                        "conversation_id": doc.metadata.get("conversation_id"),
                        "role": doc.metadata.get("role"),
                        "content": doc.page_content,
                        "content_preview": doc.metadata.get("content_preview"),
                        "similarity": similarity
                    })

            # 按相关性排序
            sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
            return sorted_results[:limit]
        except Exception as e:
            logger.error(f"搜索消息失败: {e}")
            return []

    def search_conversations(self, query: str, limit: int = 5,
                             threshold: float = 0.6) -> List[Dict[str, Any]]:
        """搜索相关对话"""
        try:
            results = self.conversation_store.similarity_search_with_score(
                query=query,
                k=limit * 2  # 获取更多结果，以便后续过滤
            )

            # 过滤低相关性结果
            filtered_results = []
            for doc, score in results:
                # 分数越低表示越相似
                similarity = 1.0 - score
                if similarity >= threshold:
                    filtered_results.append({
                        "id": doc.metadata.get("id"),
                        "title": doc.metadata.get("title"),
                        "model": doc.metadata.get("model"),
                        "summary": doc.page_content,
                        "summary_preview": doc.metadata.get("summary_preview"),
                        "similarity": similarity
                    })

            # 按相关性排序
            sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
            return sorted_results[:limit]
        except Exception as e:
            logger.error(f"搜索对话失败: {e}")
            return []

    def get_related_context(self, query: str, conversation_id: Optional[str] = None,
                            limit: int = 3, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """获取与查询相关的上下文信息，用于增强提示"""
        try:
            # 搜索相关消息
            related_messages = self.search_messages(
                query=query,
                limit=limit,
                conversation_id=conversation_id,
                threshold=threshold
            )

            return related_messages
        except Exception as e:
            logger.error(f"获取相关上下文失败: {e}")
            return []

    def delete_conversation_data(self, conversation_id: str) -> bool:
        """删除特定对话的所有向量数据"""
        try:
            # 删除对话摘要
            self.conversation_store.delete(
                where={"id": conversation_id}
            )

            # 删除对话中的所有消息
            self.message_store.delete(
                where={"conversation_id": conversation_id}
            )

            # 持久化更改
            self.conversation_store.persist()
            self.message_store.persist()

            return True
        except Exception as e:
            logger.error(f"删除对话向量数据失败: {e}")
            return False

    # 保留原有方法，用于知识库功能
    def add_documents(self, file_path, collection_name="document_store"):
        """添加文档到向量存储（用于后续知识库功能）"""
        # 保留原有实现...
        pass
