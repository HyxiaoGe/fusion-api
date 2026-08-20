"""统一的 UTC 时间工具。"""

from datetime import datetime, timezone


def utc_now() -> datetime:
    """返回带 UTC 时区信息的当前时间。"""
    return datetime.now(timezone.utc)


def as_utc(value: datetime) -> datetime:
    """把数据库时间统一为 UTC aware；历史无时区值按其既有 UTC 语义解释。"""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
