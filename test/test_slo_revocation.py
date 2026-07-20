"""跨应用单点登出（SLO）：fusion-api 在校验访问令牌后检查共享 Redis 的吊销标记。

访问令牌是无状态 RS256 JWT，签名离线校验 —— 用户在别处（如 audio）退出登录后，
auth-service 已销毁会话并吊销刷新令牌，但 fusion 手里这张访问令牌的签名仍然有效，会一直
被接受直到过期。auth-service 退出时向**共享 Redis** 写入 ``revoked_user:{sub}`` = 登出时刻，
fusion-api 在 ``jwt_validator.verify`` 成功之后增加一次检查：``iat < 标记`` 即 401，使
「一处退出 = 处处退出」在下一次接口调用即生效（约定见 auth-service AUTH_CONTRACT.md）。

该检查处于每请求鉴权热路径：Redis 故障必须失败开放（放行）而非 500 拖垮全站。

不依赖真实 Redis / auth-service：fake redis + fake 校验器。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from auth import AuthenticatedUser
from fastapi import HTTPException

from app.api import files
from app.core import revocation, security


class _FakeRedis:
    """最小同步 redis 替身：可预置标记值，或令 get 抛错以模拟 Redis 故障。"""

    def __init__(self, store=None, raise_exc=None):
        self._store = store or {}
        self._raise = raise_exc
        self.calls = 0

    def get(self, key):
        self.calls += 1
        if self._raise is not None:
            raise self._raise
        return self._store.get(key)


class IsUserAccessRevokedTests(unittest.TestCase):
    def test_token_issued_before_logout_is_revoked(self):
        with patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})):
            self.assertTrue(revocation.is_user_access_revoked("u1", 1000))

    def test_token_issued_after_logout_survives(self):
        # 退出后重新登录拿到的新令牌 iat > 登出时刻 —— 仍然有效。
        with patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})):
            self.assertFalse(revocation.is_user_access_revoked("u1", 3000))

    def test_no_marker_is_not_revoked(self):
        with patch.object(revocation, "get_redis", return_value=_FakeRedis({})):
            self.assertFalse(revocation.is_user_access_revoked("u2", 1000))

    def test_fractional_marker_over_revokes_same_second(self):
        # 与 auth-service/audio-web 一致的过度吊销语义：标记为小数墙钟秒、iat 为整数秒，严格 <
        # 保证登出前所有令牌（含同秒内更早签发者）被吊销；重新登录因需多次往返，新令牌 iat 落入
        # 下一秒得以存活。
        fake = _FakeRedis({"revoked_user:u1": "2000.5"})
        with patch.object(revocation, "get_redis", return_value=fake):
            self.assertTrue(revocation.is_user_access_revoked("u1", 2000))
            self.assertFalse(revocation.is_user_access_revoked("u1", 2001))

    def test_missing_iat_is_not_revoked(self):
        with patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})):
            self.assertFalse(revocation.is_user_access_revoked("u1", None))

    def test_empty_sub_is_not_revoked(self):
        with patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:": "2000.0"})):
            self.assertFalse(revocation.is_user_access_revoked("", 1000))

    def test_redis_outage_fails_open(self):
        # 热路径上 Redis 故障必须放行（视为未吊销）而非抛错；断言确有触达 Redis 且吞掉异常。
        fake = _FakeRedis({}, raise_exc=ConnectionError("redis down"))
        with patch.object(revocation, "get_redis", return_value=fake):
            self.assertFalse(revocation.is_user_access_revoked("u1", 1000))
        self.assertGreater(fake.calls, 0)


class IsSessionAccessRevokedTests(unittest.TestCase):
    def test_revoked_sid_marker_revokes_session(self):
        fake = _FakeRedis({"revoked_sid:sid-1": "1"})
        with patch.object(revocation, "get_redis", return_value=fake):
            self.assertTrue(revocation.is_session_access_revoked("sid-1"))

    def test_missing_sid_keeps_legacy_token_compatible(self):
        with patch.object(revocation, "get_redis", return_value=_FakeRedis({})) as redis_mock:
            self.assertFalse(revocation.is_session_access_revoked(None))
        redis_mock.assert_not_called()

    def test_session_revocation_redis_outage_fails_open(self):
        fake = _FakeRedis({}, raise_exc=ConnectionError("redis down"))
        with patch.object(revocation, "get_redis", return_value=fake):
            self.assertFalse(revocation.is_session_access_revoked("sid-1"))
        self.assertGreater(fake.calls, 0)


class GetCurrentUserRevocationTests(unittest.TestCase):
    """get_current_user 在签名校验通过后，对已吊销令牌返回 401，并且不再触达用户同步逻辑。"""

    def _auth_user(self, sub="u1", iat=1000):
        return AuthenticatedUser(sub=sub, email="a@b.c", raw_payload={"sub": sub, "iat": iat, "type": "access"})

    def test_revoked_sid_raises_401_before_user_sync(self):
        sentinel_db = object()
        auth_user = AuthenticatedUser(
            sub="u1",
            email="a@b.c",
            raw_payload={"sub": "u1", "iat": 3000, "sid": "sid-1", "type": "access"},
        )
        with (
            patch.object(security.jwt_validator, "verify", return_value=auth_user),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_sid:sid-1": "1"})),
            patch.object(security, "_sync_user_from_claims") as sync_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                security.get_current_user(db=sentinel_db, token="tok")
        self.assertEqual(ctx.exception.status_code, 401)
        sync_mock.assert_not_called()

    def test_malformed_sid_raises_401_without_user_sync(self):
        auth_user = AuthenticatedUser(
            sub="u1",
            email="a@b.c",
            raw_payload={"sub": "u1", "iat": 3000, "sid": 123, "type": "access"},
        )
        with (
            patch.object(security.jwt_validator, "verify", return_value=auth_user),
            patch.object(security, "_sync_user_from_claims") as sync_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                security.get_current_user(db=object(), token="tok")
        self.assertEqual(ctx.exception.status_code, 401)
        sync_mock.assert_not_called()

    def test_revoked_token_raises_401_before_user_sync(self):
        sentinel_db = object()
        with (
            patch.object(security.jwt_validator, "verify", return_value=self._auth_user(sub="u1", iat=1000)),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})),
            patch.object(security, "_sync_user_from_claims") as sync_mock,
        ):
            with self.assertRaises(HTTPException) as ctx:
                security.get_current_user(db=sentinel_db, token="tok")
        self.assertEqual(ctx.exception.status_code, 401)
        sync_mock.assert_not_called()  # 吊销令牌不应触发 userinfo 拉取 / DB 写

    def test_valid_token_passes_through_to_user_sync(self):
        sentinel_db = object()
        sentinel_user = SimpleNamespace(id="u1")
        with (
            patch.object(security.jwt_validator, "verify", return_value=self._auth_user(sub="u1", iat=3000)),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})),
            patch.object(security, "_sync_user_from_claims", return_value=sentinel_user) as sync_mock,
        ):
            result = security.get_current_user(db=sentinel_db, token="tok")
        self.assertIs(result, sentinel_user)
        sync_mock.assert_called_once()

    def test_redis_outage_does_not_block_auth(self):
        # 失败开放：Redis 故障时有效令牌仍放行（不因 SLO 检查 500）。
        sentinel_user = SimpleNamespace(id="u1")
        with (
            patch.object(security.jwt_validator, "verify", return_value=self._auth_user(sub="u1", iat=1000)),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({}, raise_exc=ConnectionError("down"))),
            patch.object(security, "_sync_user_from_claims", return_value=sentinel_user),
        ):
            result = security.get_current_user(db=object(), token="tok")
        self.assertIs(result, sentinel_user)


class ResolveUserFromBearerRevocationTests(unittest.TestCase):
    """文件内容下载走 Bearer 时的次要校验路径同样必须拒绝已吊销令牌，否则登出后仍能下载文件。"""

    def _request(self):
        return SimpleNamespace(headers={"authorization": "Bearer tok"})

    def test_revoked_bearer_resolves_to_none(self):
        repo = MagicMock()
        repo.get.return_value = SimpleNamespace(id="u1")  # 若不查吊销则会返回该用户
        with (
            patch.object(
                security.jwt_validator,
                "verify",
                return_value=AuthenticatedUser(sub="u1", email="a@b.c", raw_payload={"iat": 1000}),
            ),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})),
            patch("app.api.files.UserRepository", return_value=repo),
        ):
            self.assertIsNone(files._resolve_user_from_bearer(self._request(), db=object()))

    def test_revoked_sid_bearer_resolves_to_none(self):
        repo = MagicMock()
        repo.get.return_value = SimpleNamespace(id="u1")
        with (
            patch.object(
                security.jwt_validator,
                "verify",
                return_value=AuthenticatedUser(
                    sub="u1",
                    email="a@b.c",
                    raw_payload={"iat": 3000, "sid": "sid-1"},
                ),
            ),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_sid:sid-1": "1"})),
            patch("app.api.files.UserRepository", return_value=repo),
        ):
            self.assertIsNone(files._resolve_user_from_bearer(self._request(), db=object()))
        repo.get.assert_not_called()

    def test_valid_bearer_resolves_user(self):
        user = SimpleNamespace(id="u1")
        repo = MagicMock()
        repo.get.return_value = user
        with (
            patch.object(
                security.jwt_validator,
                "verify",
                return_value=AuthenticatedUser(sub="u1", email="a@b.c", raw_payload={"iat": 3000}),
            ),
            patch.object(revocation, "get_redis", return_value=_FakeRedis({"revoked_user:u1": "2000.0"})),
            patch("app.api.files.UserRepository", return_value=repo),
        ):
            self.assertIs(files._resolve_user_from_bearer(self._request(), db=object()), user)


if __name__ == "__main__":
    unittest.main()
