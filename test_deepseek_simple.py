from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage

def test_deepseek():
    # 创建模型实例
    model = ChatDeepSeek(
        model="deepseek-reasoner",
        temperature=0.7,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    # 准备问题
    messages = [HumanMessage(content="hello")]

    print("=== 测试流式响应 ===")
    print("开始接收流式响应...")
    
    reasoning_content = ""
    final_content = ""
    
    for chunk in model.stream(messages, reasoning_effort="medium"):
        
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
    test_deepseek() 