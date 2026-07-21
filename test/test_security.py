import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from auth_service_client import AuthenticatedUser
from sqlalchemy.exc import IntegrityError

from app.core import security


class SecurityTests(unittest.TestCase):
    @staticmethod
    def _integrity_error(constraint_name: str, message: str = "duplicate key") -> IntegrityError:
        original = RuntimeError(message)
        original.diag = SimpleNamespace(constraint_name=constraint_name)
        return IntegrityError("INSERT", {}, original)

    def test_sync_user_updates_nickname_and_avatar_from_auth_service(self):
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="old@example.com",
            username="old-user",
            nickname=None,
            avatar=None,
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = existing_user
        user_repo.get_by_email.return_value = None

        social_repo = MagicMock()
        social_repo.get_by_provider.side_effect = [None, None]

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={
                    "email": "new@example.com",
                    "name": "Sean",
                    "avatar_url": "https://example.com/avatar.png",
                },
            ),
        ):
            user = security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="old@example.com"),
                "token-123",
            )

        self.assertIs(user, existing_user)
        self.assertEqual(existing_user.email, "new@example.com")
        self.assertEqual(existing_user.nickname, "Sean")
        self.assertEqual(existing_user.avatar, "https://example.com/avatar.png")
        db.commit.assert_called()
        social_repo.create.assert_called_once()

    def test_sync_user_updates_existing_social_account_user_profile(self):
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="18889592303@163.com",
            username="18889592303",
            nickname=None,
            avatar=None,
            is_superuser=False,
        )
        user_repo = MagicMock()
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = SimpleNamespace(user=existing_user)

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={
                    "email": "18889592303@163.com",
                    "name": "Xiao",
                    "avatar_url": "https://avatars.githubusercontent.com/u/72925253?v=4",
                },
            ),
        ):
            user = security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="18889592303@163.com"),
                "token-123",
            )

        self.assertIs(user, existing_user)
        self.assertEqual(existing_user.nickname, "Xiao")
        self.assertEqual(existing_user.avatar, "https://avatars.githubusercontent.com/u/72925253?v=4")
        social_repo.create.assert_not_called()
        db.commit.assert_called()

    def test_sync_user_preserves_existing_avatar_when_userinfo_fetch_fails(self):
        # 慢隧道/抖动致 userinfo 拉取失败（except 分支 userinfo={} → avatar=None）时，
        # 绝不能把既有头像/昵称抹空——否则 /api/auth/me 返回 avatar:null、前端头像回退单字母。
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="Sean",
            avatar="https://lh3.googleusercontent.com/a/keep-me=s96-c",
            is_superuser=False,
        )
        user_repo = MagicMock()
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = SimpleNamespace(user=existing_user)

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                side_effect=httpx.HTTPError("boom"),
            ),
        ):
            user = security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertIs(user, existing_user)
        self.assertEqual(existing_user.nickname, "Sean")
        self.assertEqual(
            existing_user.avatar,
            "https://lh3.googleusercontent.com/a/keep-me=s96-c",
        )
        # 无字段变化 → 不应 commit（避免无谓写库，更别写空值）
        db.commit.assert_not_called()

    def test_sync_user_preserves_existing_avatar_when_userinfo_omits_fields(self):
        # userinfo 返回 200 但缺 name/avatar_url（avatar=None）时，同样不得覆盖既有头像。
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="Sean",
            avatar="https://lh3.googleusercontent.com/a/keep-me=s96-c",
            is_superuser=False,
        )
        user_repo = MagicMock()
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = SimpleNamespace(user=existing_user)

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c"},
            ),
        ):
            security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertEqual(existing_user.nickname, "Sean")
        self.assertEqual(
            existing_user.avatar,
            "https://lh3.googleusercontent.com/a/keep-me=s96-c",
        )
        db.commit.assert_not_called()

    def test_sync_user_maps_admin_scope_to_is_superuser(self):
        # 'admin' scope（来自 AuthenticatedUser.scopes）→ 本地 users.is_superuser=True
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        user_repo = MagicMock()
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = SimpleNamespace(user=existing_user)

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c", "name": "N", "avatar_url": "A"},
            ),
        ):
            security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c", scopes=["admin"]),
                "token-123",
            )

        self.assertTrue(existing_user.is_superuser)
        db.commit.assert_called()

    def test_sync_user_recovers_social_account_unique_race_and_returns_winner_user(self):
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        winner_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = existing_user
        social_repo = MagicMock()
        social_repo.get_by_provider.side_effect = [None, SimpleNamespace(user=winner_user)]
        db.commit.side_effect = [self._integrity_error("uix_provider_user_id")]

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c", "name": "N", "avatar_url": "A"},
            ),
        ):
            user = security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertIs(user, winner_user)
        db.rollback.assert_called_once_with()
        self.assertEqual(social_repo.get_by_provider.call_count, 2)
        social_repo.create.assert_called_once_with(
            {
                "user_id": "user-1",
                "provider": security.AUTH_PROVIDER,
                "provider_user_id": "user-1",
            }
        )

    def test_sync_user_does_not_swallow_unrelated_integrity_error(self):
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = existing_user
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = None
        unrelated = self._integrity_error("some_other_unique_constraint")
        db.commit.side_effect = unrelated

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c", "name": "N", "avatar_url": "A"},
            ),
            self.assertRaises(IntegrityError) as raised,
        ):
            security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertIs(raised.exception, unrelated)
        db.rollback.assert_called_once_with()
        self.assertEqual(social_repo.get_by_provider.call_count, 1)

    def test_sync_user_applies_profile_and_admin_scope_to_social_race_winner(self):
        db = MagicMock()
        current_user = SimpleNamespace(
            id="user-1",
            email="new@example.com",
            username="user-1",
            nickname="New",
            avatar="https://example.com/new.png",
            is_superuser=True,
        )
        winner_user = SimpleNamespace(
            id="user-1",
            email="old@example.com",
            username="user-1",
            nickname="Old",
            avatar="https://example.com/old.png",
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = current_user
        social_repo = MagicMock()
        social_repo.get_by_provider.side_effect = [None, SimpleNamespace(user=winner_user)]
        db.commit.side_effect = [self._integrity_error("uix_provider_user_id"), None]

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={
                    "email": "new@example.com",
                    "name": "New",
                    "avatar_url": "https://example.com/new.png",
                },
            ),
        ):
            user = security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="new@example.com", scopes=["admin"]),
                "token-123",
            )

        self.assertIs(user, winner_user)
        self.assertEqual(winner_user.email, "new@example.com")
        self.assertEqual(winner_user.nickname, "New")
        self.assertEqual(winner_user.avatar, "https://example.com/new.png")
        self.assertTrue(winner_user.is_superuser)
        self.assertEqual(db.commit.call_count, 2)
        db.rollback.assert_called_once_with()

    def test_sync_user_reraises_social_unique_race_when_winner_cannot_be_reloaded(self):
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = existing_user
        social_repo = MagicMock()
        social_repo.get_by_provider.side_effect = [None, None]
        conflict = self._integrity_error("uix_provider_user_id")
        db.commit.side_effect = conflict

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c", "name": "N", "avatar_url": "A"},
            ),
            self.assertRaises(IntegrityError) as raised,
        ):
            security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertIs(raised.exception, conflict)
        db.rollback.assert_called_once_with()
        self.assertEqual(social_repo.get_by_provider.call_count, 2)

    def test_sync_user_recovers_concurrent_same_subject_user_insert_then_creates_link(self):
        db = MagicMock()
        pending_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        winner_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.side_effect = [None, winner_user]
        user_repo.get_by_email.return_value = None
        user_repo.create.return_value = pending_user
        social_repo = MagicMock()
        social_repo.get_by_provider.side_effect = [None, None]
        db.commit.side_effect = [self._integrity_error("users_pkey"), None]

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c", "name": "N", "avatar_url": "A"},
            ),
        ):
            user = security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertIs(user, winner_user)
        db.rollback.assert_called_once_with()
        social_repo.create.assert_called_once_with(
            {
                "user_id": "user-1",
                "provider": security.AUTH_PROVIDER,
                "provider_user_id": "user-1",
            }
        )

    def test_sync_user_does_not_swallow_unrelated_user_insert_integrity_error(self):
        db = MagicMock()
        pending_user = SimpleNamespace(
            id="user-1",
            email="a@b.c",
            username="user-1",
            nickname="N",
            avatar="A",
            is_superuser=False,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = None
        user_repo.get_by_email.return_value = None
        user_repo.create.return_value = pending_user
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = None
        unrelated = self._integrity_error("users_check_unrelated")
        db.commit.side_effect = unrelated

        with (
            patch("app.core.security.UserRepository", return_value=user_repo),
            patch("app.core.security.SocialAccountRepository", return_value=social_repo),
            patch(
                "app.core.security._fetch_auth_service_userinfo",
                return_value={"email": "a@b.c", "name": "N", "avatar_url": "A"},
            ),
            self.assertRaises(IntegrityError) as raised,
        ):
            security._sync_user_from_claims(
                db,
                AuthenticatedUser(sub="user-1", email="a@b.c"),
                "token-123",
            )

        self.assertIs(raised.exception, unrelated)
        db.rollback.assert_called_once_with()
        self.assertEqual(user_repo.get.call_count, 1)
