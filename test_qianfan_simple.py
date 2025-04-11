from langchain_community.chat_models import QianfanChatEndpoint
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()


def test_qianfan():
    # 创建模型实例 - 使用endpoint参数
    model = QianfanChatEndpoint(
        # 使用文档中支持的模型名称，不使用endpoint参数
        model="ERNIE-Bot-4",  
        # 如果要使用ERNIE-4.5等新模型，需要使用正确的endpoint
        # endpoint="https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions_pro",
        streaming=True,
        timeout=60,
        max_retries=2,
    )

    # 准备问题
    messages = [HumanMessage(content="你是谁呀？请简单介绍一下你自己")]

    print("=== 测试流式响应 ===")
    print("开始接收流式响应...")
    
    final_content = ""
    
    for chunk in model.stream(messages):
        # 获取最终答案
        content = chunk.content if hasattr(chunk, 'content') else chunk
        if content:
            final_content += content
            print(f"收到内容: {content}")

    print("\n=== 最终答案 ===")
    print(final_content)


if __name__ == '__main__':
    test_qianfan() 