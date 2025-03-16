import chromadb


def test_chroma_connection():
    """测试Chroma服务连接"""
    try:
        # 测试心跳接口
        client = chromadb.HttpClient(host="localhost", port=8001)
        collections = client.list_collections()
        print("tenants: ", client.tenant)
        print("Collections: ", collections)


        collection = client.get_collection("message_vectors")

        # 查看集合中的数据
        data = collection.get()
        print("Data:", data)

        return True
    except Exception as e:
        print(f"连接Chroma服务失败: {e}")
        return False


if __name__ == "__main__":
    test_chroma_connection()