"""
提供商相关常量定义

注意：provider 显示名称已迁移到 providers 表，
此模块仅保留向后兼容的辅助函数。
"""


def get_model_display_name(provider: str) -> str:
    """向后兼容：直接返回 provider id，实际显示名称从 DB 获取"""
    return provider
