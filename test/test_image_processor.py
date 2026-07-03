import unittest

from app.processor.image_processor import ImageProcessor

BAD_IMAGE_MESSAGE = "图片文件损坏或无法读取，请重新保存后再上传"


class ImageProcessorTests(unittest.IsolatedAsyncioTestCase):
    async def test_process_rejects_unreadable_image_with_value_error(self):
        processor = ImageProcessor()

        try:
            await processor.process(b"not-an-image", "image/png")
        except Exception as exc:
            self.assertIsInstance(exc, ValueError)
            self.assertEqual(str(exc), BAD_IMAGE_MESSAGE)
        else:
            self.fail("坏图片必须抛出 ValueError")


if __name__ == "__main__":
    unittest.main()
