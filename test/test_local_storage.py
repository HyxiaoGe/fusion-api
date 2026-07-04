import tempfile
import unittest

from app.services.storage.local_storage import LocalStorageBackend


class LocalStorageBackendTests(unittest.IsolatedAsyncioTestCase):
    async def test_get_url_understands_current_file_storage_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(temp_dir)

            processed_url = await storage.get_url(
                "files/v1/users/user-1/conversations/conv-1/files/file-1/processed.jpg"
            )
            thumbnail_url = await storage.get_url(
                "files/v1/users/user-1/conversations/conv-1/files/file-1/thumbnail.jpg"
            )

        self.assertEqual(processed_url, "/api/files/file-1/content?variant=processed")
        self.assertEqual(thumbnail_url, "/api/files/file-1/content?variant=thumbnail")


if __name__ == "__main__":
    unittest.main()
