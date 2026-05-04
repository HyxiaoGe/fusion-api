"""把 litellm 抛出的异常分类成稳定的 ErrorKind，供 health service 决策。

逻辑已迁移到 app.ai.litellm_utils，此处保持向后兼容 re-export。
"""

# 向后兼容：service 层原有导入不需改动
from app.ai.litellm_utils import ErrorKind, categorize

__all__ = ["ErrorKind", "categorize"]
