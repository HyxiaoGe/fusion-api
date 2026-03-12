from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.function_handlers.file_analysis import analyze_file_handler


class AnalyzeFileHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_reuses_existing_parsed_content(self):
        file_record = SimpleNamespace(
            path="/tmp/note.txt",
            mimetype="text/plain",
            original_filename="note.txt",
            parsed_content="已有摘要",
        )
        repo = MagicMock()
        repo.get_file_by_id.return_value = file_record

        with patch("app.services.function_handlers.file_analysis.FileRepository", return_value=repo), \
             patch("app.services.function_handlers.file_analysis.FileProcessor") as file_processor_cls:
            result = await analyze_file_handler(
                {"file_id": "file-123", "analysis_type": "summary"},
                {"db": object()},
            )

        self.assertEqual(result["summary"], "已有摘要")
        file_processor_cls.assert_called_once()
        file_processor_cls.return_value.process_files.assert_not_called()
        repo.update_file.assert_not_called()

    async def test_summary_returns_error_instead_of_persisting_failed_parse(self):
        file_record = SimpleNamespace(
            path="/tmp/note.txt",
            mimetype="text/plain",
            original_filename="note.txt",
            parsed_content=None,
        )
        repo = MagicMock()
        repo.get_file_by_id.return_value = file_record
        processor = MagicMock()
        processor.process_files = AsyncMock(return_value={"error": "文件解析模型未配置"})

        with patch("app.services.function_handlers.file_analysis.FileRepository", return_value=repo), \
             patch("app.services.function_handlers.file_analysis.FileProcessor", return_value=processor):
            result = await analyze_file_handler(
                {"file_id": "file-456", "analysis_type": "summary"},
                {"db": object()},
            )

        self.assertEqual(result, {"error": "文件分析失败: 文件解析模型未配置"})
        repo.update_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
