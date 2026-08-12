import unittest
from unittest.mock import MagicMock

from botocore.exceptions import ClientError

from app.services.storage.minio_storage import MinIOStorageBackend


def client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": code}}, "HeadObject")


class MinIOStorageBackendTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = object.__new__(MinIOStorageBackend)
        self.backend.bucket = "fusion-files"
        self.backend.s3_client = MagicMock()

    async def test_exists_only_maps_not_found_to_false(self):
        self.backend.s3_client.head_object.side_effect = client_error("NoSuchKey")

        self.assertFalse(await self.backend.exists("missing"))

    async def test_exists_does_not_hide_dependency_or_permission_failures(self):
        self.backend.s3_client.head_object.side_effect = client_error("AccessDenied")

        with self.assertRaises(ClientError):
            await self.backend.exists("private")

    async def test_delete_does_not_turn_storage_outage_into_successful_cleanup(self):
        self.backend.s3_client.delete_object.side_effect = client_error("ServiceUnavailable")

        with self.assertRaises(ClientError):
            await self.backend.delete("document")


if __name__ == "__main__":
    unittest.main()
