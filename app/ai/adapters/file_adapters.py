import base64
import os
from typing import List, Dict, Any


class FileDialogAdapter:
    """文件对话适配器基类"""

    def prepare_file_for_request(self, file_paths: List[str], message: str) -> Dict[str, Any]:
        """准备包含文件的请求数据"""
        raise NotImplementedError()


class WenxinFileAdapter(FileDialogAdapter):
    """文心一言文件适配器"""

    def prepare_file_for_request(self, file_paths: List[str], message: str) -> Dict[str, Any]:
        """准备文心一言模型的文件请求"""
        files = []
        for path in file_paths:
            with open(path, "rb") as f:
                file_content = f.read()
                # 文心一言支持base64编码的文件
                b64_content = base64.b64encode(file_content).decode("utf-8")
                mime_type = self._get_mime_type(path)
                files.append({"file": b64_content, "mime_type": mime_type})

        return {
            "message": message,
            "files": files
        }

    def _get_mime_type(self, file_path: str) -> str:
        """根据文件扩展名获取MIME类型"""
        ext = os.path.splitext(file_path)[1].lower()
        mime_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".pdf": "application/pdf"
        }
        return mime_map.get(ext, "application/octet-stream")


class QwenFileAdapter(FileDialogAdapter):
    """通义千问文件适配器"""

    def prepare_file_for_request(self, file_paths: List[str], message: str) -> Dict[str, Any]:
        """准备通义千问模型的文件请求"""
        images = []
        for path in file_paths:
            with open(path, "rb") as f:
                image_content = f.read()
                # 通义千问支持base64编码的图片
                base64_image = base64.b64encode(image_content).decode("utf-8")
                images.append(base64_image)

        return {
            "message": message,
            "images": images
        }


class DefaultFileAdapter(FileDialogAdapter):
    """默认文件适配器(不支持文件)"""

    def prepare_file_for_request(self, file_paths: List[str], message: str) -> Dict[str, Any]:
        """准备不支持文件的模型请求"""
        # 仅返回消息，忽略文件
        return {"message": message}


def get_file_adapter(provider: str) -> FileDialogAdapter:
    """获取指定模型的文件适配器"""
    adapters = {
        "wenxin": WenxinFileAdapter(),
        "qwen": QwenFileAdapter(),
        "deepseek": DefaultFileAdapter()
    }
    return adapters.get(provider, DefaultFileAdapter())
