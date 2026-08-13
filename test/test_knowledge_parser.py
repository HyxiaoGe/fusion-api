import io
import signal
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from app.services.knowledge.parser import KnowledgeDocumentParser, KnowledgeParseError, parse_document_isolated


class KnowledgeDocumentParserTests(unittest.TestCase):
    def setUp(self):
        self.parser = KnowledgeDocumentParser()

    def test_text_parser_normalizes_deterministically_without_truncation(self):
        content = ("Ａ  B\r\n" + "长文本" * 3000).encode()

        first = self.parser.parse(content, mimetype="text/plain", filename="note.txt")
        second = self.parser.parse(content, mimetype="text/plain", filename="note.txt")

        self.assertEqual(first, second)
        self.assertTrue(first[0].text.startswith("A B\n"))
        self.assertGreater(len(first[0].text), 6000)

    def test_non_utf8_text_is_rejected(self):
        with self.assertRaisesRegex(KnowledgeParseError, "UTF-8") as raised:
            self.parser.parse(b"\xff\xfe", mimetype="text/plain", filename="note.txt")

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_ENCODING_UNSUPPORTED")

    def test_nul_in_parsed_text_is_rejected_as_invalid_document(self):
        with self.assertRaisesRegex(KnowledgeParseError, "NUL") as raised:
            self.parser.parse(b"safe\x00unsafe", mimetype="text/plain", filename="note.txt")

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_INVALID")

    def test_scanned_pdf_is_rejected_without_ocr(self):
        import PyPDF2

        stream = io.BytesIO()
        writer = PyPDF2.PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(stream)

        with self.assertRaises(KnowledgeParseError) as raised:
            self.parser.parse(stream.getvalue(), mimetype="application/pdf", filename="scan.pdf")

        self.assertEqual(raised.exception.code, "KNOWLEDGE_SCANNED_DOCUMENT_UNSUPPORTED")

    def test_docx_zip_bomb_is_rejected_before_python_docx_parsing(self):
        archive = MagicMock()
        archive.__enter__.return_value.infolist.return_value = [
            SimpleNamespace(file_size=KnowledgeDocumentParser.MAX_DOCX_UNCOMPRESSED_BYTES + 1, flag_bits=0)
        ]
        with (
            patch("app.services.knowledge.parser.zipfile.ZipFile", return_value=archive),
            self.assertRaises(KnowledgeParseError) as raised,
        ):
            KnowledgeDocumentParser._validate_docx_archive(b"fake")

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_TOO_LARGE")

    def test_legacy_doc_is_not_supported(self):
        with self.assertRaises(KnowledgeParseError) as raised:
            self.parser.parse(b"legacy", mimetype="application/msword", filename="legacy.doc")

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_UNSUPPORTED")

    def test_mime_suffix_and_magic_mismatch_is_rejected(self):
        with self.assertRaises(KnowledgeParseError) as raised:
            self.parser.parse(b"%PDF-fake", mimetype="text/plain", filename="manual.pdf")

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_TYPE_MISMATCH")

    def test_isolated_parser_maps_child_crash_to_non_retryable_document_error(self):
        context = MagicMock()
        parent = MagicMock()
        child = MagicMock()
        process = MagicMock(exitcode=-signal.SIGKILL)
        parent.poll.return_value = True
        parent.recv.side_effect = EOFError
        process.is_alive.return_value = False
        context.Pipe.return_value = (parent, child)
        context.Process.return_value = process

        with (
            patch("app.services.knowledge.parser.multiprocessing.get_context", return_value=context),
            self.assertRaises(KnowledgeParseError) as raised,
        ):
            parse_document_isolated(
                b"%PDF-fake",
                mimetype="application/pdf",
                filename="manual.pdf",
                timeout_seconds=60,
            )

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_INVALID")
        process.join.assert_called()

    def test_isolated_parser_maps_cpu_limit_exit_to_parse_timeout(self):
        context = MagicMock()
        parent = MagicMock()
        child = MagicMock()
        process = MagicMock(exitcode=-signal.SIGXCPU)
        parent.poll.return_value = True
        parent.recv.side_effect = EOFError
        process.is_alive.return_value = False
        context.Pipe.return_value = (parent, child)
        context.Process.return_value = process

        with (
            patch("app.services.knowledge.parser.multiprocessing.get_context", return_value=context),
            self.assertRaises(KnowledgeParseError) as raised,
        ):
            parse_document_isolated(
                b"PK-fake",
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                filename="manual.docx",
                timeout_seconds=60,
            )

        self.assertEqual(raised.exception.code, "KNOWLEDGE_DOCUMENT_PARSE_TIMEOUT")


if __name__ == "__main__":
    unittest.main()
