import unittest

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


if __name__ == "__main__":
    unittest.main()
