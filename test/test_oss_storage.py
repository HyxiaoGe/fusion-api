import sys
import types
import unittest
from unittest.mock import patch

from app.services.storage.oss_storage import OSSStorageBackend


class OSSStorageBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_upload_url_signs_put_with_content_type_header(self):
        calls = {}

        class FakeAuth:
            def __init__(self, access_key_id, access_key_secret):
                calls["auth"] = (access_key_id, access_key_secret)

        class FakeBucket:
            def __init__(self, auth, endpoint, bucket):
                calls["bucket"] = (auth, endpoint, bucket)

            def sign_url(self, method, key, expires, headers=None, slash_safe=False):
                calls["sign_url"] = {
                    "method": method,
                    "key": key,
                    "expires": expires,
                    "headers": headers,
                    "slash_safe": slash_safe,
                }
                return "https://oss.example.com/signed"

            def head_object(self, key):
                calls["head_object"] = key
                return types.SimpleNamespace(headers={"Content-Length": "123"})

        fake_oss2 = types.SimpleNamespace(
            Auth=FakeAuth,
            Bucket=FakeBucket,
            exceptions=types.SimpleNamespace(NoSuchKey=RuntimeError),
        )

        with patch.dict(sys.modules, {"oss2": fake_oss2}):
            backend = OSSStorageBackend(
                endpoint="oss-cn-shenzhen.aliyuncs.com",
                access_key_id="access-key-id",
                access_key_secret="access-key-secret",
                bucket="fusion-file",
                use_ssl=True,
            )

            result = await backend.get_upload_url(
                "conv-1/file-1/original/photo.png",
                content_type="image/png",
                expires=600,
            )
            size = await backend.get_size("conv-1/file-1/original/photo.png")

        self.assertEqual(calls["auth"], ("access-key-id", "access-key-secret"))
        self.assertEqual(calls["bucket"][1:], ("https://oss-cn-shenzhen.aliyuncs.com", "fusion-file"))
        self.assertEqual(
            calls["sign_url"],
            {
                "method": "PUT",
                "key": "conv-1/file-1/original/photo.png",
                "expires": 600,
                "headers": {"Content-Type": "image/png"},
                "slash_safe": True,
            },
        )
        self.assertEqual(calls["head_object"], "conv-1/file-1/original/photo.png")
        self.assertEqual(size, 123)
        self.assertEqual(
            result,
            {
                "url": "https://oss.example.com/signed",
                "method": "PUT",
                "headers": {"Content-Type": "image/png"},
                "expires_in": 600,
            },
        )


if __name__ == "__main__":
    unittest.main()
