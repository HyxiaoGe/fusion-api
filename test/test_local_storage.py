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

    async def test_get_url_understands_legacy_file_storage_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(temp_dir)

            processed_url = await storage.get_url("conv-1/file-1/processed.jpg")
            thumbnail_url = await storage.get_url("conv-1/file-1/thumbnail.jpg")

        self.assertEqual(processed_url, "/api/files/file-1/content?variant=processed")
        self.assertEqual(thumbnail_url, "/api/files/file-1/content?variant=thumbnail")

    async def test_get_url_rejects_invalid_variants(self):
        invalid_keys = [
            "files/v1/users/user-1/conversations/conv-1/files/file-1/preview.jpg",
            "files/v1/users/user-1/conversations/conv-1/files/file-1/processed-copy.jpg",
            "files/v1/users/user-1/conversations/conv-1/files/file-1/thumbnail_backup.jpg",
            "conv-1/file-1/preview.jpg",
            "conv-1/file-1/processed-copy.jpg",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(temp_dir)

            for key in invalid_keys:
                with self.subTest(key=key):
                    self.assertEqual(await storage.get_url(key), f"/api/files/content/{key}")

    async def test_get_url_does_not_misclassify_malformed_storage_keys(self):
        malformed_keys = [
            "files/v1/users//conversations/conv-1/files/file-1/thumbnail.jpg",
            "files/v1/users/user-1/conversations//files/file-1/thumbnail.jpg",
            "files/v1/users/user-1/conversations/conv-1/files//thumbnail.jpg",
            "files/v1/users/./conversations/conv-1/files/file-1/thumbnail.jpg",
            "files/v1/users/user-1/conversations/../files/file-1/thumbnail.jpg",
            "files/v1/users/user-1/conversations/conv-1/files/./thumbnail.jpg",
            "/file-1/thumbnail.jpg",
            "conv-1//thumbnail.jpg",
            "./file-1/thumbnail.jpg",
            "../file-1/thumbnail.jpg",
            "conv-1/../thumbnail.jpg",
            "archive/files/file-1/thumbnail.jpg",
            "users/user-1/conversations/conv-1/files/file-1/thumbnail.jpg",
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageBackend(temp_dir)

            for key in malformed_keys:
                with self.subTest(key=key):
                    self.assertEqual(await storage.get_url(key), f"/api/files/content/{key}")


if __name__ == "__main__":
    unittest.main()
