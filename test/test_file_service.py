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


if __name__ == "__main__":
    unittest.main()
