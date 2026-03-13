from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock

from app.services.file_service import FileService


class FileServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.service = FileService(MagicMock())
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
                    "timestamp": self.service.file_repo.update_file.call_args.kwargs["updates"]["processing_result"]["timestamp"],
                },
            },
        )

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

    def test_get_conversation_files_uses_shared_summary_serializer(self):
        conversation_file = SimpleNamespace(
            file=SimpleNamespace(
                id="file-1",
                original_filename="note.txt",
                mimetype="text/plain",
                size=12,
                status="processed",
            )
        )
        self.service.file_repo.get_conversation_files.return_value = [conversation_file]

        result = self.service.get_conversation_files("conv-1")

        self.assertEqual(
            result,
            [
                {
                    "id": "file-1",
                    "filename": "note.txt",
                    "mimetype": "text/plain",
                    "size": 12,
                    "status": "processed",
                }
            ],
        )

    def test_get_files_by_user_uses_shared_summary_serializer(self):
        file_record = SimpleNamespace(
            id="file-2",
            original_filename="report.pdf",
            mimetype="application/pdf",
            size=24,
            status="parsing",
        )
        self.service.file_repo.get_files_by_user_id.return_value = [file_record]

        result = self.service.get_files_by_user("user-1")

        self.assertEqual(
            result,
            [
                {
                    "id": "file-2",
                    "filename": "report.pdf",
                    "mimetype": "application/pdf",
                    "size": 24,
                    "status": "parsing",
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
