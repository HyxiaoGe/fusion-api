from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from app.ai.llm_manager import llm_manager
from app.ai.vectorstores.chroma_store import ChromaDocStore


class CustomAgent:
    def __init__(self, model_name="deepseek"):
        self.llm = llm_manager.get_model(model_name)
        self.doc_store = ChromaDocStore()

        # 定义工具
        self.tools = [
            Tool(
                name="知识库搜索",
                func=self.doc_store.search,
                description="用于搜索知识库中的相关信息，输入应该是一个问题或关键词"
            ),
            # 可以添加更多工具
        ]

        # 初始化Agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def run(self, query):
        """运行Agent推理"""
        return self.agent.run(query)