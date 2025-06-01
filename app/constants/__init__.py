"""
常量模块

集中管理所有应用常量
"""

from .chat import MessageRoles, MessageTypes, FunctionNames, MessageTexts, FUNCTION_DESCRIPTIONS, USER_FRIENDLY_FUNCTION_DESCRIPTIONS
from .events import EventTypes

__all__ = [
    'MessageRoles',
    'MessageTypes',
    'FunctionNames', 
    'MessageTexts',
    'FUNCTION_DESCRIPTIONS',
    'USER_FRIENDLY_FUNCTION_DESCRIPTIONS',
    'EventTypes'
] 