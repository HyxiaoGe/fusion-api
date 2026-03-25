"""
常量模块

集中管理所有应用常量
"""

from .chat import MessageRoles, MessageTypes
from .events import EventTypes, FinishReasons

__all__ = [
    'MessageRoles',
    'MessageTypes',
    'EventTypes',
    'FinishReasons',
]
