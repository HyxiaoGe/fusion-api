import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(name, log_file, level=logging.INFO):
    """设置日志记录器"""
    # 确保日志目录存在
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # 创建格式化器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 文件处理器
    file_handler = RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5)
    file_handler.setFormatter(formatter)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 获取记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 清空现有处理器，避免重复
    if logger.handlers:
        logger.handlers = []

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 确保传播设置正确
    logger.propagate = False

    return logger


# 配置根日志记录器
def setup_root_logger():
    root_logger = logging.getLogger()
    # 设置为 DEBUG 级别，确保捕获所有级别的日志
    root_logger.setLevel(logging.DEBUG)

    # 清空现有处理器
    if root_logger.handlers:
        root_logger.handlers = []

    # 添加控制台处理器
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    root_logger.addHandler(console)


# 设置根日志记录器
setup_root_logger()

# 主应用日志
app_logger = setup_logger('app', './logs/app.log')
# API日志
api_logger = setup_logger('api', './logs/api.log')
# LLM日志
llm_logger = setup_logger('llm', './logs/llm.log')

# 输出一条明显的日志，帮助确认配置生效
app_logger.info("=== 日志系统初始化完成 ===")