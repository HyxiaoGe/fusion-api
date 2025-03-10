import logging
from typing import List

from langchain.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class TextEmbedder:
    """文本嵌入服务，用于将文本转换为向量表示"""

    _instance = None

    @classmethod
    def get_instance(cls):
        """单例模式获取文本嵌入器实例"""
        if cls._instance is None:
            cls._instance = TextEmbedder()
        return cls._instance

    def __init__(self):
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="shibing624/text2vec-base-chinese"
            )
            logger.info("文本嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"文本嵌入模型初始化失败: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """将单个文本转换为向量"""
        try:
            if not text or text.strip() == "":
                logger.warning("尝试嵌入空文本")
                return []

            vector = self.embeddings.embed_query(text)
            return vector
        except Exception as e:
            logger.error(f"文本嵌入失败: {e}")
            return []

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量将文本转换为向量"""
        try:
            if not texts:
                return []

            # 过滤空文本
            valid_texts = [text for text in texts if text and text.strip() != ""]
            if not valid_texts:
                return []

            vectors = self.embeddings.embed_documents(valid_texts)
            return vectors
        except Exception as e:
            logger.error(f"批量文本嵌入失败: {e}")
            return []
