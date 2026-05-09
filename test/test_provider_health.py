import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import Provider
from app.services.error_categorizer import ErrorKind
from app.services.health.provider_health import ProviderHealthService


def fresh_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(bind=engine)
    db.add(Provider(id="xai", name="xAI", litellm_prefix="openrouter/x-ai", auth_config={}))
    db.commit()
    return db


class ProviderHealthServiceTests(unittest.TestCase):
    def test_first_failure_no_offline(self):
        db = fresh_db()
        svc = ProviderHealthService(db)
        changed = svc.mark_failure("xai", ErrorKind.KEY_INVALID, "401")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.consecutive_failures, 1)
        self.assertEqual(provider.status, "ok")
        self.assertFalse(changed)

    def test_two_same_kind_failures_offline(self):
        db = fresh_db()
        svc = ProviderHealthService(db)
        svc.mark_failure("xai", ErrorKind.KEY_INVALID, "401")
        changed = svc.mark_failure("xai", ErrorKind.KEY_INVALID, "401 again")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.status, "offline")
        self.assertEqual(provider.offline_reason, "key_invalid")
        self.assertTrue(changed)

    def test_different_kind_resets_count(self):
        db = fresh_db()
        svc = ProviderHealthService(db)
        svc.mark_failure("xai", ErrorKind.KEY_INVALID, "401")
        svc.mark_failure("xai", ErrorKind.QUOTA_EXCEEDED, "402")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.consecutive_failures, 1)
        self.assertEqual(provider.last_failure_kind, "quota_exceeded")
        self.assertEqual(provider.status, "ok")

    def test_success_resets_count_keeps_status(self):
        db = fresh_db()
        svc = ProviderHealthService(db)
        svc.mark_failure("xai", ErrorKind.KEY_INVALID, "401")
        svc.mark_success("xai")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.consecutive_failures, 0)

    def test_manual_recover_clears_offline(self):
        db = fresh_db()
        svc = ProviderHealthService(db)
        svc.mark_failure("xai", ErrorKind.KEY_INVALID, "1")
        svc.mark_failure("xai", ErrorKind.KEY_INVALID, "2")
        svc.manual_recover("xai", by_user_id="admin1")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.status, "ok")
        self.assertEqual(provider.consecutive_failures, 0)
        self.assertIsNone(provider.offline_reason)


if __name__ == "__main__":
    unittest.main()
