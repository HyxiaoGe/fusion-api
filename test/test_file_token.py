import unittest
from unittest.mock import patch

from app.core import file_token


class FileTokenTests(unittest.TestCase):
    def setUp(self):
        if hasattr(file_token, "_TOKEN_CACHE"):
            file_token._TOKEN_CACHE.clear()

    def tearDown(self):
        if hasattr(file_token, "_TOKEN_CACHE"):
            file_token._TOKEN_CACHE.clear()

    def test_generate_file_token_reuses_cached_token_before_refresh_window(self):
        with patch("app.core.file_token.time.time", side_effect=[1000.0, 1005.0]):
            first = file_token.generate_file_token("file-1", expires=3600)
            second = file_token.generate_file_token("file-1", expires=3600)

        self.assertEqual(first, second)
        with patch("app.core.file_token.time.time", return_value=1005.0):
            self.assertEqual(file_token.verify_file_token(second), "file-1")

    def test_generate_file_token_refreshes_near_cached_token_expiry(self):
        with patch("app.core.file_token.time.time", side_effect=[1000.0, 4561.0]):
            first = file_token.generate_file_token("file-1", expires=3600)
            second = file_token.generate_file_token("file-1", expires=3600)

        self.assertNotEqual(first, second)
        with patch("app.core.file_token.time.time", return_value=4561.0):
            self.assertEqual(file_token.verify_file_token(second), "file-1")


if __name__ == "__main__":
    unittest.main()
