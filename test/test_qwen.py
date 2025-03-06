# test_qianwen.py
import asyncio
import logging
import os

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_community.chat_models.tongyi import ChatTongyi

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 加载环境变量
load_dotenv()


async def test_qianwen_direct():
    """直接使用LangChain的ChatTongyi测试，绕过应用架构"""
    api_key = os.getenv("QIANWEN_API_KEY")

    if not api_key:
        logger.error("未找到QIANWEN_API_KEY环境变量")
        return False

    logger.info("正在直接测试通义千问模型...")

    try:
        # 创建通义千问实例
        qianwen = ChatTongyi(
            model="qwen-max-0125",  # 可以根据实际可用模型调整
            api_key=api_key
        )

        # 测试简单问题
        messages = [HumanMessage(content="你好，请简单介绍一下你自己")]

        logger.info("正在发送请求...")
        response = qianwen.invoke(messages)

        if hasattr(response, 'content'):
            logger.info(f"模型响应: {response.content}")
        else:
            logger.info(f"模型响应: {response}")

        logger.info("直接调用通义千问成功!")
        return True
    except Exception as e:
        logger.error(f"直接调用通义千问失败: {e}")
        return False


async def test_qianwen_app():
    """测试应用中的通义千问集成"""
    try:
        logger.info("正在导入应用依赖...")

        # 导入应用中的LLM管理器
        from app.ai.llm_manager import llm_manager

        # 检查通义千问是否在可用模型列表中
        available_models = llm_manager.list_available_models()
        logger.info(f"可用模型: {available_models}")

        if 'qianwen' not in available_models:
            logger.error("通义千问模型未在可用模型列表中，请检查API密钥和初始化过程")
            return False

        # 获取通义千问模型
        llm = llm_manager.get_model('qianwen')

        # 准备测试消息
        messages = [HumanMessage(content="你好，请介绍一下通义千问大模型的特点")]

        # 调用模型
        logger.info("正在调用通义千问模型...")
        response = llm.invoke(messages)

        # 打印响应
        if hasattr(response, 'content'):
            logger.info(f"模型响应: {response.content}")
        else:
            logger.info(f"模型响应: {response}")

        logger.info("应用集成通义千问测试成功!")
        return True
    except Exception as e:
        logger.error(f"应用集成通义千问测试失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


async def main():
    # 首先测试直接调用
    direct_test_result = await test_qianwen_direct()

    if direct_test_result:
        # 如果直接调用成功，测试应用集成
        app_test_result = await test_qianwen_app()

        if app_test_result:
            logger.info("通义千问模型集成测试全部通过！")
        else:
            logger.error("通义千问在应用中集成测试失败，但直接调用成功，请检查应用代码")
    else:
        logger.error("通义千问直接调用测试失败，请检查API密钥和网络连接")


if __name__ == "__main__":
    asyncio.run(main())
