"""Parity oracle for the auth-client consolidation (P3.4).

These lock the OBSERVABLE behavior of fusion-api's stable auth boundary —
``get_current_user`` (route guard) and ``_resolve_user_from_bearer`` (files.py dual
auth) — independent of which JWT validator implementation sits behind them. They are
written to pass against the current hand-rolled ``AuthServiceJWTValidator`` AND must
stay green, unedited, after the swap to the shared ``auth-client`` ``JWTValidator``.
That green-across-the-swap property is the definition of parity.

A locally-generated RSA keypair signs the tokens; a matching fake JWKS (kid
'auth-key-1') is injected straight into the active validator's cache, so no network
or running Auth Service is needed. ``_jwks_cache`` / ``_cache_time`` are attribute
names shared by both the fork and the shared client, so the injection works for either.
"""

import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException
from jwt.algorithms import RSAAlgorithm

from app.api import files
from app.core import security

KID = "auth-key-1"


def _make_keypair():
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _make_jwks(keypair):
    jwk = json.loads(RSAAlgorithm.to_jwk(keypair.public_key()))
    jwk.update({"kid": KID, "alg": "RS256", "use": "sig"})
    return {"keys": [jwk]}


def _mint(keypair, **claims):
    # Match the live validator's configured issuer/audience so the token is accepted
    # by whatever the env wired up — and by both the fork and the shared client (both
    # expose .issuer / .audience). Keeps the oracle env-agnostic and swap-stable.
    v = security.jwt_validator
    now = int(time.time())
    payload = {"sub": "user-1", "iat": now, "exp": now + 3600, **claims}
    if v.issuer:
        payload["iss"] = v.issuer
    if v.audience:
        payload["aud"] = v.audience
    return jwt.encode(payload, keypair, algorithm="RS256", headers={"kid": KID})


class _InjectedJWKSMixin:
    @classmethod
    def setUpClass(cls):
        cls.keypair = _make_keypair()
        cls.jwks = _make_jwks(cls.keypair)

    def setUp(self):
        # Inject JWKS into the validator get_current_user / files.py actually use,
        # so verify() does no network. Works for fork and shared client alike.
        security.jwt_validator._jwks_cache = self.jwks
        security.jwt_validator._cache_time = time.time()


class GetCurrentUserParityTests(_InjectedJWKSMixin, unittest.TestCase):
    def test_valid_access_token_passes_and_flows_to_user_sync(self):
        token = _mint(self.keypair, type="access", scopes=["admin"])
        sentinel = SimpleNamespace(id="user-1", is_superuser=True)
        db = MagicMock()
        with patch("app.core.security._sync_user_from_claims", return_value=sentinel) as sync:
            user = security.get_current_user(db=db, token=token)
        self.assertIs(user, sentinel)
        sync.assert_called_once()
        # The raw token is forwarded to sync as the 3rd positional arg (for the
        # userinfo enrichment fetch) — a contract files/sync depend on.
        args = sync.call_args.args
        self.assertIs(args[0], db)
        self.assertEqual(args[2], token)

    def test_sid_claim_is_parsed_and_checked_without_replacing_subject(self):
        token = _mint(self.keypair, type="access", sid="sid-1")
        sentinel = SimpleNamespace(id="user-1", is_superuser=False)
        fake_redis = MagicMock()
        fake_redis.get.return_value = None
        with (
            patch("app.core.revocation.get_redis", return_value=fake_redis),
            patch("app.core.security._sync_user_from_claims", return_value=sentinel),
        ):
            user = security.get_current_user(db=MagicMock(), token=token)
        self.assertIs(user, sentinel)
        fake_redis.get.assert_any_call("revoked_sid:sid-1")
        fake_redis.get.assert_any_call("revoked_user:user-1")

    def test_refresh_type_token_is_rejected_401(self):
        # The critical guard: a non-access token must NOT authenticate a protected route.
        token = _mint(self.keypair, type="refresh")
        with self.assertRaises(HTTPException) as ctx:
            security.get_current_user(db=MagicMock(), token=token)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_token_missing_type_claim_is_rejected_401(self):
        token = _mint(self.keypair)  # no "type" claim at all
        with self.assertRaises(HTTPException) as ctx:
            security.get_current_user(db=MagicMock(), token=token)
        self.assertEqual(ctx.exception.status_code, 401)

    def test_tampered_signature_is_rejected_401(self):
        token = _mint(self.keypair, type="access") + "tamper"
        with self.assertRaises(HTTPException) as ctx:
            security.get_current_user(db=MagicMock(), token=token)
        self.assertEqual(ctx.exception.status_code, 401)


class ResolveUserFromBearerParityTests(_InjectedJWKSMixin, unittest.TestCase):
    @staticmethod
    def _request(token):
        return SimpleNamespace(headers={"authorization": f"Bearer {token}"})

    def test_returns_user_for_valid_bearer(self):
        token = _mint(self.keypair, type="access")
        user = SimpleNamespace(id="user-1")
        repo = MagicMock()
        repo.get.return_value = user
        with patch("app.api.files.UserRepository", return_value=repo):
            result = files._resolve_user_from_bearer(self._request(token), MagicMock())
        self.assertIs(result, user)
        repo.get.assert_called_once_with("user-1")

    def test_returns_none_on_bad_token(self):
        # Must swallow and return None (not raise) so the dual file-auth can fall
        # through to the HMAC file-token path.
        result = files._resolve_user_from_bearer(self._request("not-a-jwt"), MagicMock())
        self.assertIsNone(result)

    def test_returns_none_without_bearer_prefix(self):
        result = files._resolve_user_from_bearer(SimpleNamespace(headers={}), MagicMock())
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
