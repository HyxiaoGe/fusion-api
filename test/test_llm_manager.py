import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
from cryptography.fernet import Fernet

os.environ.setdefault("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.ai.litellm_utils import ProviderOfflineError
from app.ai.llm_manager import LLMManager
from app.db.database import Base
from app.db.models import ModelSource, Provider, User
from app.db.repositories import UserCredentialRepository


def fresh_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(bind=engine)
    db.add(User(id="u1", username="u1", email="u1@example.com"))
    db.add(Provider(id="xai", name="xAI", litellm_prefix="openrouter/x-ai", auth_config={}, status="ok"))
    db.add(
        ModelSource(
            model_id="grok-4.20-beta",
            name="Grok 4.20",
            provider="xai",
            capabilities={},
            pricing={},
            model_configuration={},
            enabled=True,
        )
    )
    db.commit()
    return db


class ResolveModelTests(unittest.TestCase):
    def test_returns_user_key_in_extra_body(self):
        db = fresh_db()
        UserCredentialRepository(db).upsert("u1", "xai", "sk-or-user")

        litellm_model, provider_id, kwargs = LLMManager().resolve_model("grok-4.20-beta", db, user_id="u1")
        self.assertEqual(litellm_model, "openai/openrouter/x-ai/grok-4.20-beta")
        self.assertEqual(provider_id, "xai")
        self.assertIn("extra_body", kwargs)
        self.assertEqual(kwargs["extra_body"]["api_key"], "sk-or-user")
        self.assertEqual(kwargs["metadata"]["credential_source"], "user")

    def test_no_user_key_no_extra_body(self):
        db = fresh_db()
        _, _, kwargs = LLMManager().resolve_model("grok-4.20-beta", db, user_id="u1")
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(kwargs["metadata"]["credential_source"], "system")

    def test_offline_provider_raises(self):
        db = fresh_db()
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        provider.status = "offline"
        provider.offline_reason = "quota_exceeded"
        provider.offline_message = "no balance"
        db.commit()

        with self.assertRaises(ProviderOfflineError) as ctx:
            LLMManager().resolve_model("grok-4.20-beta", db, user_id="u1")
        self.assertEqual(ctx.exception.provider_id, "xai")
        self.assertEqual(ctx.exception.reason, "quota_exceeded")

    def test_no_user_id_skips_credential_lookup(self):
        # 兼容路径：legacy 调用方未传 user_id，等同 system
        db = fresh_db()
        UserCredentialRepository(db).upsert("u1", "xai", "sk-or-user")
        _, _, kwargs = LLMManager().resolve_model("grok-4.20-beta", db)
        self.assertNotIn("extra_body", kwargs)
        self.assertEqual(kwargs["metadata"]["credential_source"], "system")


if __name__ == "__main__":
    unittest.main()
