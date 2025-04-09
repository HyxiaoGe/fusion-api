from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()


def test_qwq():
    # 创建模型实例
    from langchain_qwq import ChatQwQ

    llm = ChatQwQ(
        model="qwq-plus",
        max_tokens=3_000,
        timeout=None,
        max_retries=2,
        api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )

    # 准备问题
    messages = [HumanMessage(content="你是谁呀")]

    print("=== 测试流式响应 ===")
    print("开始接收流式响应...")
    
    reasoning_content = ""
    final_content = ""
    
    for chunk in llm.stream(messages):
        
        # 获取思考过程
        if hasattr(chunk, 'additional_kwargs') and 'reasoning_content' in chunk.additional_kwargs:
            reasoning_content += chunk.additional_kwargs['reasoning_content']
            print(f"收到思考过程: {reasoning_content}")
            
        # 获取最终答案
        content = chunk.content if hasattr(chunk, 'content') else chunk
        if content:
            final_content += content
            print(f"收到最终答案: {content}")

    # print("\n=== 思考过程 ===")
    # print(reasoning_content)
    
    # print("\n=== 最终答案 ===")
    # print(final_content)

if __name__ == '__main__':
    test_qwq() 