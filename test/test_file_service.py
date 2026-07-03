import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.config import settings
from app.services.file_service import FileService


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
        self.service.file_repo = MagicMock()
        self.service.file_processor = MagicMock()
        self.service.file_processor.process_files = AsyncMock()

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


if __name__ == "__main__":
    unittest.main()
