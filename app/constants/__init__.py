"""
常量模块

集中管理所有应用常量
"""

from .chat import MessageRoles, FunctionNames, MessageTexts, FUNCTION_DESCRIPTIONS
from .events import EventTypes

__all__ = [
    'MessageRoles',
    'FunctionNames', 
    'MessageTexts',
    'FUNCTION_DESCRIPTIONS',
    'EventTypes'
] 