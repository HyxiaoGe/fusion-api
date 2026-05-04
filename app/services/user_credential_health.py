"""User credential 维度的健康状态：仅由用户自己 key 的失败触发。"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.core.logger import app_logger as logger
from app.db.models import UserCredential
from app.services.error_categorizer import ErrorKind

USER_CRED_DISABLE_THRESHOLD = 2
CHINA_TZ = timezone(timedelta(hours=8))


class UserCredentialHealthService:
    def __init__(self, db: Session):
        self.db = db

    def _get(self, user_id: str, provider_id: str) -> Optional[UserCredential]:
        return (
            self.db.query(UserCredential)
            .filter(UserCredential.user_id == user_id, UserCredential.provider_id == provider_id)
            .first()
        )

    def mark_failure(self, user_id: str, provider_id: str, error_kind: ErrorKind, message: str) -> None:
        cred = self._get(user_id, provider_id)
        if not cred:
            return

        kind_str = error_kind.value
        if cred.last_error_kind != kind_str:
            cred.consecutive_failures = 1
        else:
            cred.consecutive_failures += 1
        cred.last_error_kind = kind_str
        cred.last_error_message = (message or "")[:500]
        cred.last_failure_at = datetime.now(CHINA_TZ).replace(tzinfo=None)

        if cred.is_active and cred.consecutive_failures >= USER_CRED_DISABLE_THRESHOLD:
            cred.is_active = False
            logger.warning(f"User credential 停用 [user key]: user={user_id} provider={provider_id} reason={kind_str}")
        self.db.commit()

    def mark_success(self, user_id: str, provider_id: str) -> None:
        cred = self._get(user_id, provider_id)
        if not cred or cred.consecutive_failures == 0:
            return
        cred.consecutive_failures = 0
        self.db.commit()

    def reactivate(self, user_id: str, provider_id: str) -> None:
        cred = self._get(user_id, provider_id)
        if not cred:
            return
        cred.is_active = True
        cred.consecutive_failures = 0
        cred.last_error_kind = None
        cred.last_error_message = None
        cred.last_failure_at = None
        self.db.commit()
