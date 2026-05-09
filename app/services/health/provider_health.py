"""Provider 全局健康状态：仅由 system key 失败触发。"""

from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.db.models import Provider
from app.services.error_categorizer import ErrorKind

OFFLINE_THRESHOLD = 2
CHINA_TZ = timezone(timedelta(hours=8))


class ProviderHealthService:
    def __init__(self, db: Session):
        self.db = db

    def mark_failure(self, provider_id: str, error_kind: ErrorKind, message: str) -> bool:
        """触发条件：source=='system' 且 kind in {KEY_INVALID, QUOTA_EXCEEDED, TOS_BLOCKED}。
        返回 status 是否从 ok 变到 offline。"""
        provider = self.db.query(Provider).filter(Provider.id == provider_id).first()
        if not provider:
            return False

        kind_str = error_kind.value
        if provider.last_failure_kind != kind_str:
            provider.consecutive_failures = 1
        else:
            provider.consecutive_failures += 1

        provider.last_failure_kind = kind_str
        provider.last_failure_at = datetime.now(CHINA_TZ).replace(tzinfo=None)
        provider.offline_message = (message or "")[:500]

        changed = False
        if provider.status == "ok" and provider.consecutive_failures >= OFFLINE_THRESHOLD:
            provider.status = "offline"
            provider.offline_reason = kind_str
            changed = True
            logger.warning(
                f"Provider 下线 [system key]: {provider_id} reason={kind_str} consecutive={provider.consecutive_failures}"
            )

        self.db.commit()
        return changed

    def mark_success(self, provider_id: str) -> None:
        provider = self.db.query(Provider).filter(Provider.id == provider_id).first()
        if not provider:
            return
        if provider.consecutive_failures != 0:
            provider.consecutive_failures = 0
            self.db.commit()

    def manual_recover(self, provider_id: str, by_user_id: str) -> None:
        provider = self.db.query(Provider).filter(Provider.id == provider_id).first()
        if not provider:
            return
        provider.status = "ok"
        provider.offline_reason = None
        provider.offline_message = None
        provider.consecutive_failures = 0
        provider.last_failure_kind = None
        self.db.commit()
        logger.info(f"Provider 手动恢复: {provider_id} by user={by_user_id}")
