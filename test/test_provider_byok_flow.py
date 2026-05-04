"""BYOK 端到端集成测试 — 4 场景。

A：system key 路径 → provider 全局 offline
B：user key 路径 → 仅停用该 user_credential，provider.status 保持 ok
C：用户 A 坏 key 不影响用户 B
D：extra_body merge 不丢字段（user_key + thinking 共存）
"""

import os
import unittest

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
from cryptography.fernet import Fernet

os.environ.setdefault("CREDENTIAL_ENCRYPTION_KEY", Fernet.generate_key().decode())

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db.database import Base
from app.db.models import ModelSource, Provider, User, UserCredential
from app.db.repositories import UserCredentialRepository
from app.services.error_categorizer import ErrorKind
from app.services.provider_health import ProviderHealthService
from app.services.user_credential_health import UserCredentialHealthService


def fresh_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(bind=engine)
    db.add(User(id="u1", username="u1", email="u1@example.com"))
    db.add(User(id="u2", username="u2", email="u2@example.com"))
    db.add(Provider(id="xai", name="xAI", litellm_prefix="openrouter/x-ai", auth_config={}))
    db.add(
        ModelSource(
            model_id="grok-test",
            name="Grok",
            provider="xai",
            capabilities={},
            pricing={},
            model_configuration={},
            enabled=True,
        )
    )
    db.commit()
    return db


class ScenarioASystemKeyOfflineTests(unittest.TestCase):
    """场景 A：system key 路径 — 2 次 401 → provider 全局 offline，不写 user_credentials"""

    def test_system_key_failures_offline_provider(self):
        db = fresh_db()
        # 用户没配 user credential
        ProviderHealthService(db).mark_failure("xai", ErrorKind.KEY_INVALID, "401")
        ProviderHealthService(db).mark_failure("xai", ErrorKind.KEY_INVALID, "401 again")
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.status, "offline")
        self.assertEqual(provider.offline_reason, "key_invalid")

        # user_credentials 表零变化
        self.assertEqual(db.query(UserCredential).count(), 0)


class ScenarioBUserKeyOnlyDisablesCredentialTests(unittest.TestCase):
    """场景 B：user key 路径 — 2 次 401 → 仅停用该 user_credential，provider 保持 ok"""

    def test_user_key_failures_disable_only_user_cred(self):
        db = fresh_db()
        UserCredentialRepository(db).upsert("u1", "xai", "sk-bad")
        UserCredentialHealthService(db).mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "401")
        UserCredentialHealthService(db).mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "401 again")

        cred = UserCredentialRepository(db).get("u1", "xai")
        self.assertFalse(cred.is_active)
        self.assertEqual(cred.last_error_kind, "key_invalid")

        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.status, "ok")  # 关键防护

    def test_disabled_user_cred_falls_back_to_system(self):
        db = fresh_db()
        UserCredentialRepository(db).upsert("u1", "xai", "sk-bad", is_active=False)
        key, source = UserCredentialRepository(db).resolve("u1", "xai")
        self.assertIsNone(key)
        self.assertEqual(source, "system")


class ScenarioCMultiUserIsolationTests(unittest.TestCase):
    """场景 C：用户 A 坏 key 不影响用户 B"""

    def test_user_a_failure_does_not_affect_user_b(self):
        db = fresh_db()
        UserCredentialRepository(db).upsert("u1", "xai", "sk-a-bad")
        # u2 没配 user credential（走 system）

        UserCredentialHealthService(db).mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "1")
        UserCredentialHealthService(db).mark_failure("u1", "xai", ErrorKind.KEY_INVALID, "2")

        # u2 的 resolve 不受影响
        key, source = UserCredentialRepository(db).resolve("u2", "xai")
        self.assertIsNone(key)
        self.assertEqual(source, "system")
        # u1 的 cred 是停用状态
        cred_a = UserCredentialRepository(db).get("u1", "xai")
        self.assertFalse(cred_a.is_active)
        # provider 全局仍然 ok
        provider = db.query(Provider).filter(Provider.id == "xai").one()
        self.assertEqual(provider.status, "ok")


class ScenarioDExtraBodyMergeTests(unittest.TestCase):
    """场景 D：extra_body merge 不丢字段（user_key + thinking 共存）"""

    def test_user_key_and_thinking_coexist(self):
        from app.ai.litellm_utils import merge_extra_body

        # 模拟 LLMManager 已经塞了 user key
        kwargs = {"extra_body": {"api_key": "sk-user"}}
        # volcengine 后续加 thinking
        merge_extra_body(kwargs, {"thinking": {"type": "disabled"}})

        self.assertEqual(kwargs["extra_body"]["api_key"], "sk-user")
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "disabled"})

    def test_merge_does_not_drop_existing_when_extra_overrides_one_key(self):
        from app.ai.litellm_utils import merge_extra_body

        kwargs = {"extra_body": {"api_key": "sk-user", "other": "value"}}
        merge_extra_body(kwargs, {"thinking": {"type": "disabled"}, "other": "new"})

        self.assertEqual(kwargs["extra_body"]["api_key"], "sk-user")
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["extra_body"]["other"], "new")  # 同 key 后值覆盖前值


if __name__ == "__main__":
    unittest.main()
