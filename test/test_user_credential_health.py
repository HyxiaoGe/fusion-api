import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
from cryptography.fernet import Fernet

os.environ.setdefault("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import Provider, User
from app.db.repositories import UserCredentialRepository
from app.services.error_categorizer import ErrorKind
from app.services.user_credential_health import UserCredentialHealthService


def fresh_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(bind=engine)
    db.add(User(id="u1", username="u1", email="u1@example.com"))
    db.add(Provider(id="xai", name="xAI", litellm_prefix="openrouter/x-ai", auth_config={}))
    db.commit()
    UserCredentialRepository(db).upsert("u1", "xai", "sk-bad")
    return db


class UserCredentialHealthServiceTests(unittest.TestCase):
    def test_two_user_failures_disable_credential(self):
        db = fresh_db()
        svc = UserCredentialHealthService(db)
        svc.mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "401")
        svc.mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "401 again")
        cred = UserCredentialRepository(db).get("u1", "xai")
        self.assertFalse(cred.is_active)
        self.assertEqual(cred.last_error_kind, "key_invalid")

    def test_provider_status_unchanged_on_user_failure(self):
        db = fresh_db()
        UserCredentialHealthService(db).mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "1")
        UserCredentialHealthService(db).mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "2")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.status, "ok")

    def test_success_resets_count(self):
        db = fresh_db()
        svc = UserCredentialHealthService(db)
        svc.mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "1")
        svc.mark_success("u1", "xai")
        cred = UserCredentialRepository(db).get("u1", "xai")
        self.assertEqual(cred.consecutive_failures, 0)

    def test_reactivate_clears_state(self):
        db = fresh_db()
        svc = UserCredentialHealthService(db)
        svc.mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "1")
        svc.mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "2")
        svc.reactivate("u1", "xai")
        cred = UserCredentialRepository(db).get("u1", "xai")
        self.assertTrue(cred.is_active)
        self.assertEqual(cred.consecutive_failures, 0)
        self.assertIsNone(cred.last_error_kind)
