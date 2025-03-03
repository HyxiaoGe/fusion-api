from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
import os


class ChromaDocStore:
    def __init__(self, persist_directory="./chroma_db"):
        self.persist_directory = persist_directory
        self.embeddings = HuggingFaceEmbeddings(model_name="shibing624/text2vec-base-chinese")

        if os.path.exists(persist_directory):
            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)
        else:
            self.db = Chroma(persist_directory=persist_directory, embedding_function=self.embeddings)

    def add_documents(self, file_path, collection_name="default"):
        """添加文档到向量存储"""
        # 根据文件类型选择加载器
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)

        documents = loader.load()

        # 分割文档
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)

        # 添加到向量库
        self.db.add_documents(splits)
        self.db.persist()

        return len(splits)

    def search(self, query, k=5):
        """搜索相关文档"""
        docs = self.db.similarity_search(query, k=k)
        return docs