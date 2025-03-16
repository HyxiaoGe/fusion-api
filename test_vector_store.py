# test_vector_store.py
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.vectorstores.chroma_store import ChromaVectorStore
from app.core.config import settings
from dotenv import load_dotenv

# 加载开发环境变量
load_dotenv(".env.dev")


def test_vector_store():
    """测试向量存储功能"""
    try:
        print(f"使用Chroma URL: {settings.CHROMA_URL}")

        # 初始化向量存储
        vector_store = ChromaVectorStore()
        print("向量存储初始化成功")

        # 添加几条消息用于测试
        test_messages = [
            {
                "id": "test_msg_001",
                "conversation_id": "test_conv_001",
                "role": "user",
                "content": "我想了解如何使用向量数据库进行语义搜索"
            },
            {
                "id": "test_msg_002",
                "conversation_id": "test_conv_001",
                "role": "assistant",
                "content": "向量数据库是一种特殊的数据库，专门用于存储和检索向量数据，非常适合语义搜索场景。"
            },
            {
                "id": "test_msg_003",
                "conversation_id": "test_conv_002",
                "role": "user",
                "content": "AI桌面聊天应用支持哪些模型？"
            }
        ]

        # 添加测试消息
        for msg in test_messages:
            result = vector_store.add_message(
                message_id=msg["id"],
                conversation_id=msg["conversation_id"],
                role=msg["role"],
                content=msg["content"]
            )
            print(f"添加消息 {msg['id']} 结果: {result}")

        # 等待一小段时间，让服务器处理完数据
        print("等待数据处理...")
        time.sleep(1)

        # 测试各种搜索场景
        search_tests = [
            "向量数据库",
            "语义搜索",
            "AI模型",
            "测试消息"
        ]

        for query in search_tests:
            print(f"\n搜索查询: '{query}'")
            results = vector_store.search_messages(query)
            print(f"找到 {len(results)} 条相关消息")
            for i, result in enumerate(results):
                print(f"结果 {i + 1}: ID={result['id']}, 相似度={result['similarity']:.4f}")
                print(f"内容: {result['content']}")

        # 测试按会话ID过滤
        print("\n按会话ID过滤搜索:")
        results = vector_store.search_messages("向量", conversation_id="test_conv_001")
        print(f"在会话 test_conv_001 中找到 {len(results)} 条相关消息")
        for i, result in enumerate(results):
            print(f"结果 {i + 1}: ID={result['id']}, 相似度={result['similarity']:.4f}")
            print(f"内容: {result['content']}")

        return True
    except Exception as e:
        print(f"测试向量存储失败: {e}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    test_vector_store()