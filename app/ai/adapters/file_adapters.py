import base64
from typing import Any, Dict, List


class FileDialogAdapter:
    """文件对话适配器基类"""

    def prepare_file_for_request(self, file_paths: List[str], message: str) -> Dict[str, Any]:
        """准备包含文件的请求数据"""
        raise NotImplementedError()


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
                images.append({"url": f"data:image/jpeg;base64,{base64_image}"})

        return {"message": message, "images": images}


class DefaultFileAdapter(FileDialogAdapter):
    """默认文件适配器(不支持文件)"""

    def prepare_file_for_request(self, file_paths: List[str], message: str) -> Dict[str, Any]:
        """准备不支持文件的模型请求"""
        # 仅返回消息，忽略文件
        return {"message": message}


def get_file_adapter(provider: str) -> FileDialogAdapter:
    """获取指定模型的文件适配器"""
    adapters = {"qwen": QwenFileAdapter(), "deepseek": DefaultFileAdapter()}
    return adapters.get(provider, DefaultFileAdapter())
