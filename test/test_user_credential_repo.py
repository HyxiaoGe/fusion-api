import os
import unittest

from cryptography.fernet import Fernet

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import Provider, User
from app.db.repositories import UserCredentialRepository


def fresh_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(bind=engine)
    db.add(User(id="u1", username="u1", email="u1@example.com"))
    db.add(Provider(id="xai", name="xAI", litellm_prefix="openrouter/x-ai", auth_config={}))
    db.commit()
    return db


class UserCredentialRepoCRUDTests(unittest.TestCase):
    def test_upsert_then_get_returns_decrypted_plaintext(self):
        db = fresh_db()
        repo = UserCredentialRepository(db)
        repo.upsert("u1", "xai", "sk-or-plaintext-123")

        cred = repo.get("u1", "xai")
        assert cred is not None
        self.assertNotEqual(cred.api_key, "sk-or-plaintext-123")
        self.assertEqual(repo.decrypt(cred.api_key), "sk-or-plaintext-123")

    def test_resolve_returns_user_when_active(self):
        db = fresh_db()
        repo = UserCredentialRepository(db)
        repo.upsert("u1", "xai", "sk-user-key")

        key, source = repo.resolve("u1", "xai")
        self.assertEqual(key, "sk-user-key")
        self.assertEqual(source, "user")

    def test_resolve_returns_none_system_when_no_credential(self):
        db = fresh_db()
        repo = UserCredentialRepository(db)
        key, source = repo.resolve("u1", "xai")
        self.assertIsNone(key)
        self.assertEqual(source, "system")

    def test_resolve_returns_none_system_when_inactive(self):
        db = fresh_db()
        repo = UserCredentialRepository(db)
        repo.upsert("u1", "xai", "sk-user-key", is_active=False)
        key, source = repo.resolve("u1", "xai")
        self.assertIsNone(key)
        self.assertEqual(source, "system")

    def test_to_masked_schema_only_shows_tail(self):
        db = fresh_db()
        repo = UserCredentialRepository(db)
        cred = repo.upsert("u1", "xai", "sk-or-1234567890abcd")
        masked = repo.to_masked_schema(cred)
        self.assertEqual(masked.provider_id, "xai")
        self.assertNotIn("1234567890", masked.api_key_masked)
        self.assertTrue(masked.api_key_masked.endswith("abcd"))

    def test_delete_removes_row(self):
        db = fresh_db()
        repo = UserCredentialRepository(db)
        repo.upsert("u1", "xai", "sk-user-key")
        self.assertTrue(repo.delete("u1", "xai"))
        self.assertIsNone(repo.get("u1", "xai"))
