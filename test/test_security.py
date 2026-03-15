import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.core import security


class SecurityTests(unittest.TestCase):
    def test_sync_user_updates_nickname_and_avatar_from_auth_service(self):
        db = MagicMock()
        existing_user = SimpleNamespace(
            id="user-1",
            email="old@example.com",
            username="old-user",
            nickname=None,
            avatar=None,
        )
        user_repo = MagicMock()
        user_repo.get.return_value = existing_user
        user_repo.get_by_email.return_value = None

        social_repo = MagicMock()
        social_repo.get_by_provider.side_effect = [None, None]

        with patch("app.core.security.UserRepository", return_value=user_repo), patch(
            "app.core.security.SocialAccountRepository", return_value=social_repo
        ), patch(
            "app.core.security._fetch_auth_service_userinfo",
            return_value={
                "email": "new@example.com",
                "name": "Sean",
                "avatar_url": "https://example.com/avatar.png",
            },
        ):
            user = security._sync_user_from_claims(
                db,
                {"sub": "user-1", "email": "old@example.com"},
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
        )
        user_repo = MagicMock()
        social_repo = MagicMock()
        social_repo.get_by_provider.return_value = SimpleNamespace(user=existing_user)

        with patch("app.core.security.UserRepository", return_value=user_repo), patch(
            "app.core.security.SocialAccountRepository", return_value=social_repo
        ), patch(
            "app.core.security._fetch_auth_service_userinfo",
            return_value={
                "email": "18889592303@163.com",
                "name": "Xiao",
                "avatar_url": "https://avatars.githubusercontent.com/u/72925253?v=4",
            },
        ):
            user = security._sync_user_from_claims(
                db,
                {"sub": "user-1", "email": "18889592303@163.com"},
                "token-123",
            )

        self.assertIs(user, existing_user)
        self.assertEqual(existing_user.nickname, "Xiao")
        self.assertEqual(existing_user.avatar, "https://avatars.githubusercontent.com/u/72925253?v=4")
        social_repo.create.assert_not_called()
        db.commit.assert_called()
