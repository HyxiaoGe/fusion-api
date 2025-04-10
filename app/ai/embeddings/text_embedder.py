import logging
import threading
from typing import List

import torch
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)


class TextEmbedder:
    """文本嵌入服务，用于将文本转换为向量表示"""

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = TextEmbedder()
                    cls._initialized = True
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        if not settings.ENABLE_VECTOR_EMBEDDINGS:
            logger.info("向量嵌入功能已禁用")
            self.embeddings = None
            return

        try:

            # 检查GPU是否可用
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                logger.info(f"使用GPU进行文本嵌入: {torch.cuda.get_device_name(0)}")
            else:
                logger.info("未检测到可用GPU，使用CPU进行文本嵌入")

            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                model_kwargs={"device": device}
            )
            logger.info("问答优化的文本嵌入模型初始化成功")
        except Exception as e:
            logger.error(f"文本嵌入模型初始化失败: {e}")
            raise

    def embed_text(self, text: str) -> List[float]:
        """将单个文本转换为向量"""
        if self.embeddings is None:
            return []

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
