import os
import sys
import time
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.vectorstores.chroma_store import ChromaVectorStore
from app.core.config import settings
from dotenv import load_dotenv

# 加载开发环境变量
load_dotenv(".env.dev")


def test_vector_store():
    """测试向量存储功能 - 余弦相似度版本"""
    try:
        print(f"使用Chroma URL: {settings.CHROMA_URL}")
        print("测试余弦相似度向量搜索...")

        # 初始化向量存储
        vector_store = ChromaVectorStore()
        print("向量存储初始化成功")

        # 添加几条消息用于测试
        test_messages = [
            {
                "id": "cosine_test_001",
                "conversation_id": "cosine_conv_001",
                "role": "user",
                "content": "我想了解如何使用向量数据库进行语义搜索"
            },
            {
                "id": "cosine_test_002",
                "conversation_id": "cosine_conv_001",
                "role": "assistant",
                "content": "向量数据库是一种特殊的数据库，专门用于存储和检索向量数据，非常适合语义搜索场景。"
            },
            {
                "id": "cosine_test_003",
                "conversation_id": "cosine_conv_002",
                "role": "user",
                "content": "AI桌面聊天应用支持哪些模型？"
            },
            {
                "id": "cosine_test_004",
                "conversation_id": "cosine_conv_002",
                "role": "assistant",
                "content": "我们的AI桌面应用支持多种大型语言模型，包括OpenAI的GPT系列、Anthropic的Claude系列，以及开源模型如Llama 2等。"
            }
        ]

        # 清理之前的测试数据
        try:
            print("清理之前的测试数据...")
            # 尝试删除之前的测试会话数据
            vector_store.delete_conversation_data("cosine_conv_001")
            vector_store.delete_conversation_data("cosine_conv_002")
            print("清理完成")
        except Exception as e:
            print(f"清理数据时出错: {e}")
            # 继续测试，不终止

        # 添加测试消息
        for msg in test_messages:
            result = vector_store.add_message(
                message_id=msg["id"],
                conversation_id=msg["conversation_id"],
                role=msg["role"],
                content=msg["content"]
            )
            print(f"添加消息 {msg['id']} 结果: {result}")

        # 等待数据处理
        print("等待数据处理...")
        time.sleep(2)  # 增加等待时间确保数据已经写入

        # 逐步测试查询功能
        # 1. 先测试简单查询是否能返回结果
        print("\n基础查询测试:")
        simple_query = "向量数据库"
        print(f"查询: '{simple_query}'")
        results = vector_store.search_messages(simple_query, limit=3)
        if results:
            print(f"找到 {len(results)} 条相关消息:")
            for i, result in enumerate(results):
                print(f"  结果 {i + 1}: ID={result['id']}, 相似度={result['similarity']:.4f}")
                print(f"  内容: {result['content'][:100]}")
        else:
            print("没有找到结果，这可能表明存在问题")

        # 2. 测试不同阈值的影响
        print("\n阈值测试:")
        for threshold in [0.1, 0.3, 0.5, 0.7]:
            print(f"使用阈值 {threshold}:")
            results = vector_store.search_messages(simple_query, threshold=threshold)
            print(f"  找到 {len(results)} 条相关消息")

        # 3. 测试不同查询内容
        print("\n多样化查询测试:")
        search_tests = [
            "向量数据库",
            "语义搜索技术",
            "AI聊天模型",
            "测试查询"
        ]

        for query in search_tests:
            print(f"\n查询: '{query}'")
            results = vector_store.search_messages(query)
            print(f"找到 {len(results)} 条相关消息")
            for i, result in enumerate(results):
                if i < 2:  # 只显示前两条结果，避免输出过多
                    print(f"  结果 {i + 1}: ID={result['id']}, 相似度={result['similarity']:.4f}")
                    print(f"  内容: {result['content'][:100]}")

        # 4. 测试按会话ID过滤
        print("\n按会话ID过滤测试:")
        for conv_id in ["cosine_conv_001", "cosine_conv_002"]:
            print(f"在会话 {conv_id} 中搜索:")
            results = vector_store.search_messages("AI", conversation_id=conv_id)
            print(f"  找到 {len(results)} 条相关消息")
            for i, result in enumerate(results):
                print(f"  结果 {i + 1}: ID={result['id']}, 相似度={result['similarity']:.4f}")
                print(f"  内容预览: {result['content_preview']}")

        return True
    except Exception as e:
        print(f"测试向量存储失败: {e}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    test_result = test_vector_store()
    print(f"\n测试结果: {'成功' if test_result else '失败'}")