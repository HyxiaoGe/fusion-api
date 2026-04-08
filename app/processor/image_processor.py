"""图片预处理器 — Pillow 处理管线

处理流程：
1. 格式检测与转换（HEIC/BMP/TIFF → JPEG，保留 PNG 透明和 GIF 动画）
2. EXIF 旋转修正
3. 长边缩放至 1568px（兼容所有 LLM 提供商）
4. 压缩编码（JPEG q=85，超 4MB 则降至 q=70）
5. 生成 400px 宽缩略图
"""

import io
from typing import Dict, Any

from PIL import Image, ImageOps

from app.core.logger import app_logger as logger

# 尝试导入 HEIC 支持
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORTED = True
except ImportError:
    HEIC_SUPPORTED = False
    logger.warning("pillow-heif 未安装，HEIC 格式图片将无法处理")


class ImageProcessor:
    """图片预处理管线"""

    # 处理后的长边最大值（兼容 Claude 1568px 建议）
    MAX_LONG_SIDE = 1568
    # 缩略图最大宽度
    THUMBNAIL_WIDTH = 400
    # JPEG 压缩质量
    JPEG_QUALITY = 85
    JPEG_QUALITY_FALLBACK = 70
    # 处理后最大文件大小（4MB，预留 1MB 给 Claude 5MB 限制）
    MAX_OUTPUT_SIZE = 4 * 1024 * 1024

    # 需要转换为 JPEG 的格式
    CONVERT_TO_JPEG = {"BMP", "TIFF", "ICO", "HEIF"}
    # 保持原格式的类型
    KEEP_FORMAT = {"PNG", "WEBP", "GIF", "JPEG"}

    async def process(self, data: bytes, mime_type: str) -> Dict[str, Any]:
        """
        处理图片，生成处理图和缩略图。

        Args:
            data: 原始图片二进制数据
            mime_type: 原始 MIME 类型

        Returns:
            {
                "processed": bytes,      # 处理后的图（发给 LLM）
                "thumbnail": bytes,      # 缩略图（前端预览）
                "mime_type": str,        # 处理后的 MIME 类型
                "width": int,            # 处理后宽度
                "height": int,           # 处理后高度
            }
        """
        img = Image.open(io.BytesIO(data))
        original_format = (img.format or "JPEG").upper()
        logger.info(f"图片预处理开始: 原始格式={original_format}, 尺寸={img.size}")

        # 1. 确定输出格式
        output_format = self._resolve_output_format(img, original_format)

        # 2. EXIF 旋转修正（手机拍照经常带旋转信息）
        img = ImageOps.exif_transpose(img) or img

        # 3. 处理透明通道：JPEG 不支持 alpha，需转为 RGB
        if output_format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[-1] if "A" in img.mode else None)
            img = background
        elif img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")

        # 4. 长边缩放
        img = self._resize_long_side(img, self.MAX_LONG_SIDE)

        # 5. 编码处理图
        processed_bytes = self._encode(img, output_format, self.JPEG_QUALITY)

        # 6. 若超 4MB，降低质量重新编码
        if len(processed_bytes) > self.MAX_OUTPUT_SIZE and output_format == "JPEG":
            logger.info(f"处理图超过 4MB ({len(processed_bytes)} bytes)，降低质量重新编码")
            processed_bytes = self._encode(img, output_format, self.JPEG_QUALITY_FALLBACK)

        # 7. 生成缩略图
        thumb = img.copy()
        thumb_height = int(img.height * (self.THUMBNAIL_WIDTH / img.width))
        thumb = thumb.resize((self.THUMBNAIL_WIDTH, thumb_height), Image.Resampling.LANCZOS)
        thumbnail_bytes = self._encode(thumb, output_format, 75)

        # MIME 类型映射
        mime_map = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
            "GIF": "image/gif",
        }
        result_mime = mime_map.get(output_format, "image/jpeg")

        logger.info(
            f"图片预处理完成: 输出格式={output_format}, 尺寸={img.size}, "
            f"处理图={len(processed_bytes)} bytes, 缩略图={len(thumbnail_bytes)} bytes"
        )

        return {
            "processed": processed_bytes,
            "thumbnail": thumbnail_bytes,
            "mime_type": result_mime,
            "width": img.width,
            "height": img.height,
        }

    def _resolve_output_format(self, img: Image.Image, original_format: str) -> str:
        """确定输出格式，保留 PNG 透明和 GIF 动画"""
        # GIF 动图保持原格式
        if original_format == "GIF":
            return "GIF"

        # 带透明通道的 PNG 保持原格式
        if original_format == "PNG" and img.mode in ("RGBA", "LA", "P"):
            # 检查是否真有透明像素
            if img.mode == "P" and "transparency" in img.info:
                return "PNG"
            if img.mode in ("RGBA", "LA"):
                return "PNG"

        # WebP 保持原格式
        if original_format == "WEBP":
            return "WEBP"

        # 需要转换的格式（BMP/TIFF/ICO/HEIF 等）→ JPEG
        if original_format in self.CONVERT_TO_JPEG:
            return "JPEG"

        # 默认输出 JPEG
        return "JPEG"

    @staticmethod
    def _resize_long_side(img: Image.Image, max_side: int) -> Image.Image:
        """等比缩放，使长边不超过 max_side"""
        w, h = img.size
        long_side = max(w, h)
        if long_side <= max_side:
            return img

        ratio = max_side / long_side
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _encode(img: Image.Image, fmt: str, quality: int = 85) -> bytes:
        """将图片编码为指定格式的字节流"""
        buf = io.BytesIO()

        if fmt == "JPEG":
            img.save(buf, format="JPEG", quality=quality, optimize=True)
        elif fmt == "PNG":
            img.save(buf, format="PNG", optimize=True)
        elif fmt == "WEBP":
            img.save(buf, format="WEBP", quality=quality)
        elif fmt == "GIF":
            img.save(buf, format="GIF")
        else:
            img.save(buf, format="JPEG", quality=quality, optimize=True)

        return buf.getvalue()
