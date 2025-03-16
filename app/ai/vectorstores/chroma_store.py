from typing import List, Dict, Any, Optional

import chromadb
import numpy as np

from app.ai.embeddings.text_embedder import TextEmbedder
from app.core.config import settings
from app.core.logger import app_logger as logger


class ChromaVectorStore:
    """增强版向量存储，使用Chroma服务并采用余弦相似度进行语义搜索"""

    def __init__(self):
        self.embedder = TextEmbedder.get_instance()

        # 使用配置的Chroma服务URL
        self.chroma_url = settings.CHROMA_URL
        logger.info(f"连接到Chroma服务: {self.chroma_url}")

        # 解析URL
        if "://" not in self.chroma_url:
            raise ValueError(f"无效的Chroma URL格式: {self.chroma_url}")

        url_parts = self.chroma_url.split("://")
        if len(url_parts) != 2:
            raise ValueError(f"无效的Chroma URL格式: {self.chroma_url}")

        protocol, address = url_parts

        if ":" not in address:
            # 使用默认端口
            host = address
            port = 8000  # 默认端口
            logger.info(f"未指定端口，使用默认端口: {port}")
        else:
            # 解析主机和端口
            try:
                host_port = address.split(":")
                host = host_port[0]
                port = int(host_port[1])
            except (IndexError, ValueError) as e:
                raise ValueError(f"无法解析主机和端口: {address}, 错误: {str(e)}")

        try:
            # 连接到Chroma服务
            logger.info(f"尝试连接到Chroma服务: host={host}, port={port}")
            self.client = chromadb.HttpClient(host=host, port=port)
            logger.info("成功连接到Chroma服务")

            # 获取或创建集合
            self.message_collection = self._get_or_create_collection("message_vectors")
            self.conversation_collection = self._get_or_create_collection("conversation_vectors")
        except Exception as e:
            logger.error(f"连接Chroma服务失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _normalize_vector(self, vector: List[float]) -> List[float]:
        """将向量归一化为单位长度，确保余弦相似度计算准确"""
        try:
            # 转为numpy数组
            np_vector = np.array(vector, dtype=np.float32)
            # 计算向量的L2范数
            norm = np.linalg.norm(np_vector)
            # 归一化
            if norm > 0:
                normalized = np_vector / norm
                return normalized.tolist()
            return vector
        except Exception as e:
            logger.error(f"向量归一化失败: {e}")
            return vector

    def _get_or_create_collection(self, collection_name: str):
        """获取或创建集合"""
        try:
            # 尝试获取现有集合
            return self.client.get_collection(name=collection_name)
        except Exception:
            # 如果不存在则创建
            logger.info(f"创建新集合: {collection_name}")
            return self.client.create_collection(name=collection_name)

    def add_message(self, message_id: str, conversation_id: str,
                    role: str, content: str) -> bool:
        """
        添加消息向量到存储

        参数:
            message_id: 消息唯一ID
            conversation_id: 所属会话ID
            role: 消息角色(user或assistant)
            content: 消息内容

        返回:
            bool: 添加成功返回True，失败返回False
        """
        if not message_id or not conversation_id or not content:
            logger.error("添加消息参数无效")
            return False

        try:
            # 计算嵌入向量
            embedding = self.embedder.embed_text(content)

            if not embedding or len(embedding) == 0:
                logger.error(f"消息 {message_id} 嵌入计算失败")
                return False

            # 对向量进行归一化，确保余弦相似度计算准确
            normalized_embedding = self._normalize_vector(embedding)

            # 准备元数据
            metadata = {
                "conversation_id": conversation_id,
                "role": role,
                "content_preview": content[:100] + "..." if len(content) > 100 else content
            }

            # 添加或更新记录
            self.message_collection.upsert(
                ids=[message_id],
                embeddings=[normalized_embedding],
                metadatas=[metadata],
                documents=[content]
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
        """
        搜索与查询相关的消息

        使用余弦相似度进行向量搜索，查找与输入查询语义相似的消息。
        余弦相似度计算两个向量的夹角余弦值，范围在-1到1之间，值越大表示越相似。
        在文本语义搜索中，余弦相似度通常是最佳选择，因为它关注语义方向而非绝对距离。

        参数:
            query (str): 搜索查询文本
            limit (int, 可选): 返回结果的最大数量，默认为5
            conversation_id (str, 可选): 如果提供，将搜索限制在指定会话内
            threshold (float, 可选): 最低相似度阈值，默认为0.5

        返回:
            List[Dict[str, Any]]: 包含相关消息的列表
        """
        if not query or query.strip() == "":
            logger.error("搜索查询不能为空")
            return []

        try:
            # 计算查询的嵌入向量
            query_embedding = self.embedder.embed_text(query)

            if not query_embedding or len(query_embedding) == 0:
                logger.error("查询嵌入计算失败")
                return []

            # 对查询向量进行归一化，确保余弦相似度计算准确
            normalized_query = self._normalize_vector(query_embedding)

            # 准备查询条件
            where = {}
            if conversation_id:
                where = {"conversation_id": conversation_id}

            # 调试信息
            logger.info(f"执行查询: query='{query}', where={where}, limit={limit}")

            # 执行查询，明确指定使用余弦距离度量
            results = self.message_collection.query(
                query_embeddings=[normalized_query],
                n_results=min(limit * 3, 20),
                where=where if where else None,
                include=["metadatas", "documents", "distances"]
            )

            # 处理结果
            filtered_results = []

            if not results or 'ids' not in results or not results['ids']:
                logger.warning("查询返回空结果")
                return []

            logger.info(f"查询结果结构: {list(results.keys())}")

            try:
                # 获取结果数据
                if isinstance(results['ids'], list):
                    ids_data = results['ids'][0] if results['ids'] and isinstance(results['ids'][0], list) else results[
                        'ids']
                    distances_data = results['distances'][0] if 'distances' in results and results[
                        'distances'] and isinstance(results['distances'][0], list) else results.get('distances', [])
                    metadatas_data = results['metadatas'][0] if 'metadatas' in results and results[
                        'metadatas'] and isinstance(results['metadatas'][0], list) else results.get('metadatas', [])
                    documents_data = results['documents'][0] if 'documents' in results and results[
                        'documents'] and isinstance(results['documents'][0], list) else results.get('documents', [])

                    # 记录原始距离值以便调试
                    if distances_data:
                        if isinstance(distances_data, list):
                            logger.info(f"原始距离值(列表): 类型={type(distances_data)}, 值={distances_data}")
                        else:
                            logger.info(f"原始距离值(单值): 类型={type(distances_data)}, 值={distances_data}")

                    # 处理每个结果
                    for i, doc_id in enumerate(ids_data):
                        # 安全获取距离值
                        distance = None
                        if isinstance(distances_data, list) and i < len(distances_data):
                            distance = distances_data[i]
                        elif not isinstance(distances_data, list):
                            distance = distances_data

                        # 如果距离值是列表，取第一个元素
                        if isinstance(distance, list) and distance:
                            distance = distance[0]

                        # 如果距离值无效，使用默认值
                        if not isinstance(distance, (int, float)):
                            logger.warning(f"结果 {i} 的距离值无效: {distance}, 使用默认值1.0")
                            distance = 1.0  # 默认中等相似度

                        # 计算余弦相似度
                        # 余弦距离 = 1 - 余弦相似度，范围[0,2]
                        # 因此余弦相似度 = 1 - 余弦距离/2，范围[0,1]
                        similarity = 1.0 - (distance / 2.0)

                        # 确保相似度在有效范围内
                        similarity = max(0.0, min(1.0, similarity))

                        logger.info(f"结果 {i + 1}: ID={doc_id}, 余弦距离={distance:.4f}, 相似度={similarity:.4f}")

                        # 应用相似度阈值过滤
                        if similarity >= threshold:
                            # 安全获取元数据和内容
                            metadata = {}
                            if isinstance(metadatas_data, list) and i < len(metadatas_data):
                                metadata = metadatas_data[i] or {}

                            content = ""
                            if isinstance(documents_data, list) and i < len(documents_data):
                                content = documents_data[i] or ""

                            filtered_results.append({
                                "id": doc_id,
                                "conversation_id": metadata.get("conversation_id", ""),
                                "role": metadata.get("role", ""),
                                "content": content,
                                "content_preview": metadata.get("content_preview", ""),
                                "similarity": similarity
                            })
            except Exception as e:
                logger.error(f"处理查询结果时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # 按相似度排序
            sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
            logger.info(f"过滤后的结果数量: {len(sorted_results)}")
            return sorted_results[:limit]
        except Exception as e:
            logger.error(f"搜索消息失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def search_conversations(self, query: str, limit: int = 5,
                             threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        搜索相关对话

        使用余弦相似度查找与输入查询语义相关的对话。结果基于对话摘要进行匹配。

        参数:
            query (str): 搜索查询文本
            limit (int, 可选): 返回结果的最大数量，默认为5
            threshold (float, 可选): 最低相似度阈值，默认为0.3

        返回:
            List[Dict[str, Any]]: 包含相关对话的列表，每个对话包含相似度分数等信息
        """
        if not query:
            logger.error("搜索查询不能为空")
            return []

        try:
            # 计算查询的嵌入向量
            query_embedding = self.embedder.embed_text(query)

            if not query_embedding or len(query_embedding) == 0:
                logger.error("查询嵌入计算失败")
                return []

            # 执行查询，使用余弦相似度
            results = self.conversation_collection.query(
                query_embeddings=[query_embedding],
                n_results=min(limit * 3, 20),  # 获取更多候选结果但设置上限
                include=["metadatas", "documents", "distances"]
            )

            # 打印关键调试信息
            if results:
                logger.info(f"对话查询结果结构: {list(results.keys())}")
                # 安全地检查结果数量
                if 'ids' in results:
                    logger.info(f"对话IDs类型: {type(results['ids'])}")
                    if isinstance(results['ids'], list):
                        logger.info(f"找到对话结果数量: {len(results['ids'])}")
                    else:
                        logger.info(f"对话ID不是列表类型: {results['ids']}")
                else:
                    logger.warning("对话结果中没有ids字段")
                    return []
            else:
                logger.warning("对话查询返回空结果")
                return []

            # 处理结果
            filtered_results = []

            try:
                if 'ids' in results:
                    # 获取所有数据
                    ids_list = results['ids']
                    distances_list = results.get('distances', [])
                    metadatas_list = results.get('metadatas', [])
                    documents_list = results.get('documents', [])

                    # 检查数据结构
                    if isinstance(ids_list, list):
                        # 正常列表结构
                        for i, doc_id in enumerate(ids_list):
                            # 获取距离信息（若有）
                            distance = 0
                            if i < len(distances_list):
                                dist_value = distances_list[i]

                                # 记录实际获取的距离值类型和内容
                                logger.info(f"对话原始距离值类型: {type(dist_value)}, 值: {dist_value}")

                                # 处理不同类型的距离值
                                if isinstance(dist_value, (int, float)):
                                    # 单个数值
                                    distance = dist_value
                                elif isinstance(dist_value, list) and dist_value:
                                    # 列表 - 取第一个值
                                    logger.info(f"对话距离是列表: {dist_value}")
                                    if isinstance(distance, list) and distance:
                                        if all(isinstance(d, (int, float)) for d in distance):
                                            distance = min(distance)  # 取最小距离值（最相似）
                                            logger.info(f"使用最小距离值: {distance:.4f}")
                                else:
                                    logger.warning(f"距离列表为空，可能无匹配项，使用默认最大距离")
                                    distance = 2.0  # 余弦距离最大值，表示完全不相似

                            # 确保距离是数值类型
                            if not isinstance(distance, (int, float)):
                                logger.warning(f"最终对话距离不是数值类型: {type(distance)}, 使用默认值0")
                                distance = 0

                            # 计算相似度
                            similarity = 1.0 - (distance / 2.0)  # 余弦距离转相似度

                            logger.info(f"对话结果 {i + 1}: ID={doc_id}, 相似度={similarity:.4f}")

                            if similarity >= threshold:
                                # 获取元数据和摘要
                                metadata = {}
                                if i < len(metadatas_list) and metadatas_list[i]:
                                    metadata = metadatas_list[i]

                                summary = ""
                                if i < len(documents_list) and documents_list[i]:
                                    summary = documents_list[i]

                                filtered_results.append({
                                    "id": doc_id,
                                    "title": metadata.get("title", ""),
                                    "model": metadata.get("model", ""),
                                    "summary": summary,
                                    "summary_preview": metadata.get("summary_preview", ""),
                                    "similarity": similarity
                                })
                    else:
                        logger.warning(f"意外的对话IDs结构: {type(ids_list)}")
            except Exception as e:
                logger.error(f"处理对话查询结果时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())

            # 按相似度排序
            sorted_results = sorted(filtered_results, key=lambda x: x["similarity"], reverse=True)
            logger.info(f"对话结果数量: {len(sorted_results)}")
            return sorted_results[:limit]
        except Exception as e:
            logger.error(f"搜索对话失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def get_related_context(self, query: str, conversation_id: Optional[str] = None,
                            limit: int = 3, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        获取与查询相关的上下文信息，用于增强提示

        查找与输入查询语义相关的消息，可用于构建更丰富的上下文。

        参数:
            query (str): 搜索查询文本
            conversation_id (str, 可选): 如提供，将限制在特定会话内搜索
            limit (int, 可选): 返回结果的最大数量，默认为3
            threshold (float, 可选): 最低相似度阈值，默认为0.5

        返回:
            List[Dict[str, Any]]: 相关上下文消息列表
        """
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

    def delete_conversation_data(self, conversation_id: str) -> bool:
        """
        删除特定对话的所有向量数据

        参数:
            conversation_id (str): 要删除的对话ID

        返回:
            bool: 删除成功返回True，失败返回False
        """
        try:
            # 删除对话摘要
            self.conversation_collection.delete(
                ids=[conversation_id]
            )

            # 删除对话中的所有消息
            self.message_collection.delete(
                where={"conversation_id": conversation_id}
            )

            logger.info(f"成功删除对话数据: {conversation_id}")
            return True
        except Exception as e:
            logger.error(f"删除对话向量数据失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
