import anthropic

def chat_with_claude_stream(api_key, message, base_url=None):
    client = anthropic.Anthropic(
        api_key=api_key,
        base_url=base_url
    )
    
    try:
        print("开始请求流式响应...")
        # 先测试普通API是否能正常工作
        print("测试普通API...")
        normal_response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": message}]
        )
        print(f"普通API响应: {normal_response.content}")
        
        # 再测试流式API
        print("测试流式API...")
        with client.messages.stream(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": message}]
        ) as stream:
            print("开始接收流...")
            full_response = ""
            chunk_count = 0
            for chunk in stream:
                chunk_count += 1
                print(f"收到块 #{chunk_count}，类型: {chunk.type}")
                print(f"完整块内容: {chunk}")
                
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text'):
                    text = chunk.delta.text
                    print(f"文本内容: {text}")
                    full_response += text
            
            print(f"\n\n收到总计 {chunk_count} 个数据块")
            print(f"完整响应: {full_response}")
    except Exception as e:
        print(f"错误详情: {str(e)}")
        print(f"错误类型: {type(e)}")
        raise

if __name__ == "__main__":
    api_key = "sk-KASfcb4ef34d23dee87e9a3213acc97246adbf12c6b4QGNS"
    base_url = "https://api.gptsapi.net"
    chat_with_claude_stream(api_key, "你好", base_url)