import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from app.processor.file_processor import FileProcessor


class FileProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_file_bypasses_remote_model_and_returns_extracted_content(self):
        processor = FileProcessor()
        processor._prepare_file_data = lambda path, mime_type: {
            "file_name": "note.txt",
            "mime_type": "text/plain",
            "content": "ignored",
            "extracted_text": "第一行\n第二行",
        }

        result = await processor.process_files(["/tmp/note.txt"], mime_types=["text/plain"])

        self.assertEqual(result["model"], "local-text-extraction")
        self.assertIn("文件 1: note.txt", result["content"])
        self.assertIn("第一行", result["content"])

    async def test_process_files_returns_error_payload_when_prepare_fails(self):
        processor = FileProcessor()

        def raise_error(path, mime_type):
            raise ValueError("bad file")

        processor._prepare_file_data = raise_error

        result = await processor.process_files(["/tmp/bad.txt"], mime_types=["text/plain"])

        self.assertEqual(result["error"], "bad file")
        self.assertIn("处理文件时发生错误", result["content"])

    def test_build_text_only_content_truncates_long_text(self):
        processor = FileProcessor()
        long_text = "a" * (processor.LOCAL_TEXT_PREVIEW_LIMIT + 20)

        content = processor._build_text_only_content(
            [
                {
                    "file_name": "note.txt",
                    "mime_type": "text/plain",
                    "extracted_text": long_text,
                }
            ]
        )

        self.assertIn("...(内容过长已截断)", content)
        self.assertIn("文件 1: note.txt", content)

    def test_build_prompt_truncates_extracted_text(self):
        processor = FileProcessor()
        long_text = "b" * (processor.PROMPT_TEXT_PREVIEW_LIMIT + 20)

        prompt = processor._build_prompt(
            "总结一下",
            [
                {
                    "file_name": "note.txt",
                    "mime_type": "text/plain",
                    "extracted_text": long_text,
                }
            ],
        )

        self.assertIn("...(内容过长已截断)", prompt)
        self.assertIn("总结一下", prompt)

    def test_run_optional_text_extractor_returns_none_on_missing_dependency(self):
        result = FileProcessor._run_optional_text_extractor(
            lambda: (_ for _ in ()).throw(ImportError("missing")),
            missing_dependency_message="missing dependency",
            failure_message="failure",
        )

        self.assertIsNone(result)

    def test_run_optional_text_extractor_returns_none_on_runtime_error(self):
        with patch("app.processor.file_processor.logger") as logger:
            result = FileProcessor._run_optional_text_extractor(
                lambda: (_ for _ in ()).throw(ValueError("bad content")),
                missing_dependency_message="missing dependency",
                failure_message="failure",
            )

        self.assertIsNone(result)
        logger.error.assert_called_once()

    async def test_call_model_attaches_file_processing_tags(self):
        processor = FileProcessor()
        processor.client = MagicMock()
        chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="图片描述"))])

        with patch("app.processor.file_processor.asyncio.to_thread", new=AsyncMock(return_value=[chunk])) as to_thread:
            result = await processor._call_model(
                "请分析图片",
                [{"mime_type": "image/png", "content": "base64-image"}],
            )

        self.assertEqual(result, "图片描述")
        self.assertEqual(
            to_thread.await_args.kwargs["extra_body"],
            {"metadata": {"tags": ["app:fusion", "phase:file_processing"]}},
        )


if __name__ == "__main__":
    unittest.main()
