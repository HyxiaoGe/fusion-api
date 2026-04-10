"""
提供商相关常量定义
"""

# 提供商展示名称映射
MODEL_DISPLAY_NAMES = {
    "anthropic": "Anthropic",
    "deepseek": "DeepSeek",
    "google": "Google",
    "openai": "OpenAI",
    "qwen": "通义千问",
    "volcengine": "火山引擎",
    "xai": "xAI",
    "xiaomi": "小米 MiMo",
    "minimax": "MiniMax",
    "moonshot": "月之暗面",
}


def get_model_display_name(provider: str) -> str:
    return MODEL_DISPLAY_NAMES.get(provider, provider)
