from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import os
import time
import json

# 加载环境变量
load_dotenv()

def test_gemini():
    """测试Gemini模型连接与功能"""
    try:
        # 获取配置
        api_key = os.getenv("GOOGLE_API_KEY")
        worker_url = os.getenv("GEMINI_API_ENDPOINT", "https://api.seanfield.org/gemini")
        print(f"使用Worker URL: {worker_url}")
        
        # 设置环境变量
        os.environ["GOOGLE_API_BASE"] = worker_url
        
        # 创建模型实例
        model = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            max_retries=2,
            timeout=30,
            google_api_key=api_key,
        )
        
        print("=== 1. 基本响应测试 ===")
        messages = [HumanMessage(content="你好，请简单介绍一下你自己")]
        
        # 非流式调用测试
        try:
            print("发送请求...")
            start_time = time.time()
            response = model.invoke(messages)
            elapsed = time.time() - start_time
            print(f"请求耗时: {elapsed:.2f}秒")
            print(f"收到响应: {response.content}")
            
            # 检查响应元数据
            if hasattr(response, "response_metadata") and response.response_metadata:
                print("\n响应元数据:")
                print(json.dumps(response.response_metadata, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"基本响应测试失败: {e}")
        
        print("\n=== 2. 流式响应测试 ===")
        try:
            messages = [HumanMessage(content="写一首关于AI的短诗")]
            print("开始流式响应...")
            
            full_response = ""
            chunk_count = 0
            start_time = time.time()
            
            for chunk in model.stream(messages):
                chunk_count += 1
                content = chunk.content if hasattr(chunk, "content") else str(chunk)
                if content:
                    full_response += content
                    print(f"[块{chunk_count}] {content}")
            
            elapsed = time.time() - start_time
            print(f"\n流式响应完成，共 {chunk_count} 个数据块，耗时: {elapsed:.2f}秒")
            print(f"完整响应:\n{full_response}")
        except Exception as e:
            print(f"流式响应测试失败: {e}")
        
        print("\n=== 3. 系统指令测试 ===")
        try:
            # 模拟系统指令 (注意Gemini对系统指令的处理方式不同)
            messages = [
                SystemMessage(content="你是一个专业的技术顾问，回答要简洁专业"),
                HumanMessage(content="什么是向量数据库？")
            ]
            
            print("发送带系统指令的请求...")
            response = model.invoke(messages)
            print(f"收到响应: {response.content}")
        except Exception as e:
            print(f"系统指令测试失败: {e}")
            print("注意：某些Gemini模型可能不直接支持系统指令")
        
        print("\n=== 4. 多轮对话测试 ===")
        try:
            # 模拟多轮对话
            messages = [
                HumanMessage(content="什么是Python？"),
                {"role": "assistant", "content": "Python是一种高级编程语言，以其简洁、易读的语法和强大的生态系统而闻名。"},
                HumanMessage(content="它与JavaScript相比有什么优势？")
            ]
            
            print("发送多轮对话请求...")
            response = model.invoke(messages)
            print(f"收到响应: {response.content}")
        except Exception as e:
            print(f"多轮对话测试失败: {e}")
        
        print("\n=== 测试完成 ===")
        return True
    except Exception as e:
        print(f"测试过程中出现未捕获的异常: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_gemini()
    print(f"\n总体测试结果: {'成功' if success else '失败'}")