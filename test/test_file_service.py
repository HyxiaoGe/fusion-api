import tempfile
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.core.config import settings
from app.services.file_service import FileService
from app.services.storage.local_storage import LocalStorageBackend

ORIGINAL_KEY = "files/v1/users/user-1/conversations/conv-1/files/file-1/original"
PROCESSED_KEY = "files/v1/users/user-1/conversations/conv-1/files/file-1/processed.jpg"
THUMBNAIL_KEY = "files/v1/users/user-1/conversations/conv-1/files/file-1/thumbnail.jpg"


class FileServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # FileService.__init__ 里 get_storage() 依赖 lifespan 初始化的全局 storage，
        # 单测里没走 lifespan，所以 patch 掉，给 service.storage 喂个 MagicMock
        self._storage_patcher = patch(
            "app.services.file_service.get_storage",
            return_value=MagicMock(),
        )
        self._storage_patcher.start()
        self.service = FileService(MagicMock())
        self._storage_patcher.stop()
        self._storage_for_backend_patcher = patch(
            "app.services.file_service.get_storage_for_backend",
            return_value=self.service.storage,
        )
        self._storage_for_backend_patcher.start()
        self.addCleanup(self._storage_for_backend_patcher.stop)
        self.service.file_repo = MagicMock()
        self.service.file_processor = MagicMock()
        self.service.file_processor.process_files = AsyncMock()
        self.service.storage.exists = AsyncMock(return_value=True)
        self.service.file_repo.get_stale_uploading_files.return_value = []

    async def test_parse_file_marks_processed_only_after_success(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            mimetype="text/plain",
            original_filename="note.txt",
        )
        self.service.file_processor.process_files.return_value = {"content": "整理后的内容"}

        await self.service._parse_file_with_llm("file-123", "/tmp/file-123_note.txt")

        self.service.file_repo.update_file.assert_called_once_with(
            file_id="file-123",
            updates={
                "status": "processed",
                "parsed_content": "整理后的内容",
                "processing_result": {
                    "status": "success",
                    "timestamp": self.service.file_repo.update_file.call_args.kwargs["updates"]["processing_result"][
                        "timestamp"
                    ],
                },
            },
        )

    def test_get_file_status_uses_shared_status_serializer(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-1",
            status="processed",
            processing_result={"status": "success"},
        )

        result = self.service.get_file_status("file-1", "user-1")

        self.assertEqual(
            result,
            {
                "id": "file-1",
                "status": "processed",
                "processing_result": {"status": "success"},
                "thumbnail_url": None,
            },
        )

    def test_get_file_status_returns_none_when_file_is_missing(self):
        self.service.file_repo.get_file_by_id.return_value = None

        result = self.service.get_file_status("missing-file", "user-1")

        self.assertIsNone(result)

    async def test_parse_file_marks_error_when_processor_returns_error(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            mimetype="text/plain",
            original_filename="note.txt",
        )
        self.service.file_processor.process_files.return_value = {
            "content": "处理文件时发生错误: 文件解析模型未配置",
            "error": "文件解析模型未配置",
        }

        await self.service._parse_file_with_llm("file-456", "/tmp/file-456_note.txt")

        self.service.file_repo.update_file.assert_called_once_with(
            file_id="file-456",
            updates={
                "status": "error",
                "processing_result": {
                    "status": "error",
                    "message": "文件解析模型未配置",
                },
            },
        )

    async def test_parse_file_marks_error_when_content_is_empty(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            mimetype="text/plain",
            original_filename="note.txt",
        )
        self.service.file_processor.process_files.return_value = {"content": "   "}

        await self.service._parse_file_with_llm("file-789", "/tmp/file-789_note.txt")

        self.service.file_repo.update_file.assert_called_once_with(
            file_id="file-789",
            updates={
                "status": "error",
                "processing_result": {
                    "status": "error",
                    "message": "无法解析文件内容",
                },
            },
        )

    async def test_parse_file_marks_error_when_processor_raises_unexpected_exception(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            mimetype="text/plain",
            original_filename="note.txt",
        )
        self.service.file_processor.process_files.side_effect = Exception("boom")

        await self.service._parse_file_with_llm("file-999", "/tmp/file-999_note.txt")

        self.service.file_repo.update_file.assert_called_once_with(
            file_id="file-999",
            updates={
                "status": "error",
                "processing_result": {
                    "status": "error",
                    "message": "boom",
                },
            },
        )

    async def test_get_conversation_files_includes_thumbnail_url_created_at_and_error_message(self):
        created_at = datetime(2026, 7, 3, 10, 11, 12)
        conversation_file = SimpleNamespace(
            file=SimpleNamespace(
                id="file-1",
                original_filename="photo.png",
                mimetype="image/png",
                size=12,
                status="error",
                thumbnail_key="conv-1/file-1/thumb.png",
                width=640,
                height=480,
                created_at=created_at,
                processing_result={"status": "error", "message": "解析失败"},
            )
        )
        self.service.file_repo.get_conversation_files.return_value = [conversation_file]
        self.service.storage.get_url = AsyncMock(return_value="/files/file-1/thumb.png")

        result = await self.service.get_conversation_files("conv-1")

        self.service.storage.get_url.assert_awaited_once_with(
            "conv-1/file-1/thumb.png",
            expires=settings.MINIO_PRESIGN_EXPIRES,
        )
        self.assertEqual(len(result), 1)
        summary = result[0]
        self.assertTrue(summary["thumbnail_url"].startswith("/files/file-1/thumb.png?token="))
        self.assertEqual(summary["created_at"], "2026-07-03T10:11:12")
        self.assertEqual(summary["error_message"], "解析失败")

    async def test_get_files_by_user_includes_summary_fields_without_thumbnail(self):
        file_record = SimpleNamespace(
            id="file-2",
            original_filename="report.pdf",
            mimetype="application/pdf",
            size=24,
            status="error",
            thumbnail_key=None,
            width=None,
            height=None,
            created_at="2026-07-03T11:12:13",
            processing_result={"status": "error", "error": "解析超时"},
        )
        self.service.file_repo.get_files_by_user_id.return_value = [file_record]
        self.service.storage.get_url = AsyncMock(return_value="/files/file-2/thumb.png")

        result = await self.service.get_files_by_user("user-1")

        self.service.storage.get_url.assert_not_awaited()
        self.assertEqual(
            result,
            [
                {
                    "id": "file-2",
                    "filename": "report.pdf",
                    "mimetype": "application/pdf",
                    "size": 24,
                    "status": "error",
                    "thumbnail_key": None,
                    "thumbnail_url": None,
                    "width": None,
                    "height": None,
                    "created_at": "2026-07-03T11:12:13",
                    "error_message": "解析超时",
                }
            ],
        )

    async def test_get_conversation_files_for_user_returns_none_when_conversation_missing(self):
        with patch("app.services.file_service.ConversationRepository") as repo_class:
            repo = MagicMock()
            repo.get_by_id.return_value = None
            repo_class.return_value = repo

            result = await self.service.get_conversation_files_for_user("conv-missing", "user-1")

        self.assertIsNone(result)
        repo.get_by_id.assert_called_once_with("conv-missing", "user-1")
        self.service.file_repo.get_conversation_files.assert_not_called()

    async def test_get_conversation_files_keeps_summary_when_thumbnail_url_fails(self):
        conversation_file = SimpleNamespace(
            file=SimpleNamespace(
                id="file-1",
                original_filename="photo.png",
                mimetype="image/png",
                size=12,
                status="processed",
                thumbnail_key="conv-1/file-1/thumb.png",
                width=640,
                height=480,
                created_at=None,
                processing_result=None,
            )
        )
        self.service.file_repo.get_conversation_files.return_value = [conversation_file]
        self.service.storage.get_url = AsyncMock(side_effect=RuntimeError("storage unavailable"))

        result = await self.service.get_conversation_files("conv-1")

        self.service.storage.get_url.assert_awaited_once_with(
            "conv-1/file-1/thumb.png",
            expires=settings.MINIO_PRESIGN_EXPIRES,
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "file-1")
        self.assertIsNone(result[0]["thumbnail_url"])

    async def test_get_conversation_files_omits_thumbnail_when_storage_object_missing(self):
        conversation_file = SimpleNamespace(
            file=SimpleNamespace(
                id="file-1",
                original_filename="photo.png",
                mimetype="image/png",
                size=12,
                status="processed",
                thumbnail_key="conv-1/file-1/thumbnail.jpg",
                width=640,
                height=480,
                created_at=None,
                processing_result=None,
            )
        )
        self.service.file_repo.get_conversation_files.return_value = [conversation_file]
        self.service.storage.exists = AsyncMock(return_value=False)
        self.service.storage.get_url = AsyncMock(return_value="/files/file-1/thumb.png")

        result = await self.service.get_conversation_files("conv-1")

        self.service.storage.exists.assert_awaited_once_with("conv-1/file-1/thumbnail.jpg")
        self.service.storage.get_url.assert_not_awaited()
        self.assertEqual(len(result), 1)
        self.assertIsNone(result[0]["thumbnail_url"])

    async def test_get_file_url_returns_none_when_storage_object_missing(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-1",
            thumbnail_key="conv-1/file-1/thumbnail.jpg",
            storage_key="conv-1/file-1/processed.jpg",
        )
        self.service.storage.exists = AsyncMock(return_value=False)
        self.service.storage.get_url = AsyncMock(return_value="/files/file-1/thumb.png")

        result = await self.service.get_file_url("file-1", "user-1", "thumbnail")

        self.service.storage.exists.assert_awaited_once_with("conv-1/file-1/thumbnail.jpg")
        self.service.storage.get_url.assert_not_awaited()
        self.assertIsNone(result)

    async def test_get_file_url_uses_file_storage_backend_for_historical_local_file(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-local",
            storage_backend="local",
            thumbnail_key="conv-1/file-local/thumbnail.jpg",
            storage_key="conv-1/file-local/processed.jpg",
        )
        local_storage = MagicMock()
        local_storage.exists = AsyncMock(return_value=True)
        local_storage.get_url = AsyncMock(return_value="/api/files/file-local/content?variant=thumbnail")

        with patch("app.services.file_service.get_storage_for_backend", return_value=local_storage) as get_backend:
            result = await self.service.get_file_url("file-local", "user-1", "thumbnail")

        get_backend.assert_called_once_with("local")
        local_storage.exists.assert_awaited_once_with("conv-1/file-local/thumbnail.jpg")
        local_storage.get_url.assert_awaited_once_with(
            "conv-1/file-local/thumbnail.jpg",
            expires=settings.MINIO_PRESIGN_EXPIRES,
        )
        self.assertTrue(result.startswith("/api/files/file-local/content?variant=thumbnail&token="))

    async def test_get_file_url_and_content_support_legacy_key_with_real_local_storage_backend(self):
        legacy_key = "conv-1/file-local/thumbnail.jpg"
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-local",
            storage_backend="local",
            thumbnail_key=legacy_key,
            storage_key="conv-1/file-local/processed.jpg",
            mimetype="image/jpeg",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            local_storage = LocalStorageBackend(temp_dir)
            await local_storage.upload(legacy_key, b"thumbnail", "image/jpeg")

            with patch("app.services.file_service.get_storage_for_backend", return_value=local_storage):
                result = await self.service.get_file_url("file-local", "user-1", "thumbnail")
                content = await self.service.get_file_content("file-local", "user-1", "thumbnail")

        self.assertTrue(result.startswith("/api/files/file-local/content?variant=thumbnail&token="))
        self.assertEqual(content, (b"thumbnail", "image/jpeg"))

    async def test_get_files_by_user_does_not_surface_success_message_as_error(self):
        file_record = SimpleNamespace(
            id="file-3",
            original_filename="report.pdf",
            mimetype="application/pdf",
            size=24,
            status="processed",
            thumbnail_key=None,
            width=None,
            height=None,
            created_at=None,
            processing_result={"status": "success", "message": "解析完成"},
        )
        self.service.file_repo.get_files_by_user_id.return_value = [file_record]

        result = await self.service.get_files_by_user("user-1")

        self.assertIsNone(result[0]["error_message"])

    async def test_get_conversation_files_uses_shared_summary_serializer(self):
        conversation_file = SimpleNamespace(
            file=SimpleNamespace(
                id="file-1",
                original_filename="note.txt",
                mimetype="text/plain",
                size=12,
                status="processed",
                thumbnail_key=None,
                width=None,
                height=None,
                created_at=None,
                processing_result=None,
            )
        )
        self.service.file_repo.get_conversation_files.return_value = [conversation_file]

        result = await self.service.get_conversation_files("conv-1")

        self.assertEqual(
            result,
            [
                {
                    "id": "file-1",
                    "filename": "note.txt",
                    "mimetype": "text/plain",
                    "size": 12,
                    "status": "processed",
                    "thumbnail_key": None,
                    "thumbnail_url": None,
                    "width": None,
                    "height": None,
                    "created_at": None,
                    "error_message": None,
                }
            ],
        )

    async def test_get_files_by_user_uses_shared_summary_serializer(self):
        file_record = SimpleNamespace(
            id="file-2",
            original_filename="report.pdf",
            mimetype="application/pdf",
            size=24,
            status="parsing",
            thumbnail_key=None,
            width=None,
            height=None,
            created_at=None,
            processing_result=None,
        )
        self.service.file_repo.get_files_by_user_id.return_value = [file_record]

        result = await self.service.get_files_by_user("user-1")

        self.assertEqual(
            result,
            [
                {
                    "id": "file-2",
                    "filename": "report.pdf",
                    "mimetype": "application/pdf",
                    "size": 24,
                    "status": "parsing",
                    "thumbnail_key": None,
                    "thumbnail_url": None,
                    "width": None,
                    "height": None,
                    "created_at": None,
                    "error_message": None,
                }
            ],
        )

    async def test_create_direct_upload_creates_uploading_record_and_signed_put_url(self):
        self.service.storage.get_upload_url = AsyncMock(
            return_value={
                "url": f"https://oss.example.com/{ORIGINAL_KEY}?signature=1",
                "method": "PUT",
                "headers": {"Content-Type": "image/png"},
                "expires_in": 600,
            }
        )
        self.service.file_repo.count_conversation_files.return_value = 0

        with (
            patch("app.services.file_service.uuid.uuid4", return_value="file-1"),
            patch("app.services.file_service.ConversationRepository") as repo_class,
        ):
            conv_repo = MagicMock()
            conv_repo.get_by_id.return_value = SimpleNamespace(id="conv-1")
            repo_class.return_value = conv_repo

            result = await self.service.create_direct_upload(
                user_id="user-1",
                conversation_id="conv-1",
                provider="qwen",
                model="qwen-vl-max",
                filename="photo.png",
                mimetype="image/png",
                size=123,
            )

        self.service.storage.get_upload_url.assert_awaited_once_with(
            ORIGINAL_KEY,
            content_type="image/png",
            expires=settings.MINIO_PRESIGN_EXPIRES,
        )
        self.assertNotIn("photo.png", self.service.storage.get_upload_url.await_args.args[0])
        self.service.file_repo.create_file.assert_called_once_with(
            {
                "id": "file-1",
                "user_id": "user-1",
                "filename": "file-1_photo.png",
                "original_filename": "photo.png",
                "mimetype": "image/png",
                "size": 123,
                "path": ORIGINAL_KEY,
                "status": "uploading",
                "processing_result": None,
                "storage_key": ORIGINAL_KEY,
                "thumbnail_key": None,
                "storage_backend": settings.STORAGE_BACKEND,
                "width": None,
                "height": None,
            }
        )
        self.service.file_repo.link_file_to_conversation.assert_called_once_with("conv-1", "file-1")
        self.assertEqual(
            result,
            {
                "file_id": "file-1",
                "upload_url": f"https://oss.example.com/{ORIGINAL_KEY}?signature=1",
                "method": "PUT",
                "headers": {"Content-Type": "image/png"},
                "expires_in": 600,
            },
        )

    async def test_create_direct_upload_cleans_stale_uploading_records_before_counting_quota(self):
        stale_file = SimpleNamespace(
            id="stale-file",
            user_id="user-1",
            storage_backend=settings.STORAGE_BACKEND,
            storage_key="files/v1/users/user-1/conversations/conv-1/files/stale-file/original",
            thumbnail_key=None,
            path="files/v1/users/user-1/conversations/conv-1/files/stale-file/original",
        )
        self.service.file_repo.get_stale_uploading_files.return_value = [stale_file]
        self.service.file_repo.count_conversation_files.return_value = 0
        self.service.storage.delete = AsyncMock(return_value=True)
        self.service.storage.get_upload_url = AsyncMock(
            return_value={
                "url": "https://oss.example.com/files/v1/users/user-1/conversations/conv-1/files/file-2/original?signature=1",
                "method": "PUT",
                "headers": {"Content-Type": "image/png"},
                "expires_in": 600,
            }
        )

        with (
            patch("app.services.file_service.uuid.uuid4", return_value="file-2"),
            patch("app.services.file_service.ConversationRepository") as repo_class,
            patch("app.services.file_service.get_storage_for_backend", return_value=self.service.storage),
        ):
            conv_repo = MagicMock()
            conv_repo.get_by_id.return_value = SimpleNamespace(id="conv-1")
            repo_class.return_value = conv_repo

            await self.service.create_direct_upload(
                user_id="user-1",
                conversation_id="conv-1",
                provider="qwen",
                model="qwen-vl-max",
                filename="photo.png",
                mimetype="image/png",
                size=123,
            )

        self.service.file_repo.get_stale_uploading_files.assert_called_once()
        self.service.storage.delete.assert_awaited_once_with(
            "files/v1/users/user-1/conversations/conv-1/files/stale-file/original"
        )
        self.service.file_repo.delete_file.assert_called_once_with("stale-file", "user-1")
        self.service.file_repo.count_conversation_files.assert_called_once_with("conv-1")

    async def test_complete_direct_upload_processes_image_from_uploaded_object(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-1",
            user_id="user-1",
            original_filename="photo.png",
            filename="file-1_photo.png",
            mimetype="image/png",
            size=123,
            path=ORIGINAL_KEY,
            storage_key=ORIGINAL_KEY,
            storage_backend=settings.STORAGE_BACKEND,
            status="uploading",
        )
        self.service.storage.exists = AsyncMock(return_value=True)
        self.service.storage.get_size = AsyncMock(return_value=321)
        self.service.storage.download = AsyncMock(return_value=b"original-image")
        self.service.storage.upload = AsyncMock()
        self.service.storage.get_url = AsyncMock(return_value=f"https://oss.example.com/{THUMBNAIL_KEY}")
        self.service.image_processor.process = AsyncMock(
            return_value={
                "processed": b"processed-image",
                "thumbnail": b"thumbnail-image",
                "mime_type": "image/jpeg",
                "width": 640,
                "height": 480,
            }
        )

        result = await self.service.complete_direct_upload("file-1", "user-1")

        self.service.storage.exists.assert_awaited_once_with(ORIGINAL_KEY)
        self.service.storage.get_size.assert_awaited_once_with(ORIGINAL_KEY)
        self.service.storage.download.assert_awaited_once_with(ORIGINAL_KEY)
        self.service.image_processor.process.assert_awaited_once_with(b"original-image", "image/png")
        self.service.file_repo.update_file.assert_called_once_with(
            file_id="file-1",
            updates={
                "status": "processed",
                "mimetype": "image/jpeg",
                "storage_key": PROCESSED_KEY,
                "thumbnail_key": THUMBNAIL_KEY,
                "width": 640,
                "height": 480,
                "size": 321,
                "processing_result": None,
            },
        )
        self.assertEqual(
            result,
            {
                "file_id": "file-1",
                "thumbnail_url": result["thumbnail_url"],
                "status": "processed",
                "filename": "photo.png",
                "mimetype": "image/jpeg",
                "size": 321,
            },
        )
        self.assertTrue(result["thumbnail_url"].startswith(f"https://oss.example.com/{THUMBNAIL_KEY}"))

    def test_conversation_id_from_key_requires_current_storage_schema(self):
        self.assertEqual(FileService._conversation_id_from_key(ORIGINAL_KEY), "conv-1")

        with self.assertRaises(ValueError):
            FileService._conversation_id_from_key("conv-1/file-1/original/photo.png")

    async def test_complete_direct_upload_rejects_legacy_key_before_storage_access(self):
        legacy_key = "conv-1/file-legacy/original.png"
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-legacy",
            user_id="user-1",
            path=legacy_key,
            storage_key=legacy_key,
            storage_backend="local",
            status="uploading",
        )
        self.service.storage.exists = AsyncMock(return_value=True)
        self.service.storage.get_size = AsyncMock(return_value=123)
        self.service.storage.download = AsyncMock(return_value=b"original-image")

        with self.assertRaisesRegex(ValueError, "文件对象 key 不符合当前存储结构"):
            await self.service.complete_direct_upload("file-legacy", "user-1")

        self.service.storage.exists.assert_not_awaited()
        self.service.storage.get_size.assert_not_awaited()
        self.service.storage.download.assert_not_awaited()

    async def test_delete_storage_objects_deletes_original_processed_and_thumbnail_objects(self):
        file_obj = SimpleNamespace(
            id="file-1",
            user_id="user-1",
            storage_backend=settings.STORAGE_BACKEND,
            path=ORIGINAL_KEY,
            storage_key=PROCESSED_KEY,
            thumbnail_key=THUMBNAIL_KEY,
        )
        self.service.storage.delete = AsyncMock(return_value=True)

        await self.service._delete_storage_objects(file_obj)

        self.service.storage.delete.assert_has_awaits(
            [
                call(PROCESSED_KEY),
                call(THUMBNAIL_KEY),
                call(ORIGINAL_KEY),
            ],
            any_order=True,
        )
        self.assertEqual(self.service.storage.delete.await_count, 3)

    async def test_complete_direct_upload_rejects_oversized_object_before_download(self):
        self.service.file_repo.get_file_by_id.return_value = SimpleNamespace(
            id="file-large",
            user_id="user-1",
            original_filename="large.png",
            filename="file-large_large.png",
            mimetype="image/png",
            size=123,
            path="files/v1/users/user-1/conversations/conv-1/files/file-large/original",
            storage_key="files/v1/users/user-1/conversations/conv-1/files/file-large/original",
            storage_backend=settings.STORAGE_BACKEND,
            status="uploading",
        )
        self.service.storage.exists = AsyncMock(return_value=True)
        self.service.storage.get_size = AsyncMock(return_value=settings.MAX_FILE_SIZE + 1)
        self.service.storage.download = AsyncMock()
        self.service.storage.delete = AsyncMock(return_value=True)

        with patch("app.services.file_service.get_storage_for_backend", return_value=self.service.storage):
            with self.assertRaises(ValueError):
                await self.service.complete_direct_upload("file-large", "user-1")

        self.service.storage.get_size.assert_awaited_once_with(
            "files/v1/users/user-1/conversations/conv-1/files/file-large/original"
        )
        self.service.storage.download.assert_not_awaited()
        self.service.storage.delete.assert_awaited_once_with(
            "files/v1/users/user-1/conversations/conv-1/files/file-large/original"
        )
        self.service.file_repo.delete_file.assert_called_once_with("file-large", "user-1")


if __name__ == "__main__":
    unittest.main()
