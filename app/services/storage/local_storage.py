"""本地磁盘存储后端实现"""

import os

import aiofiles

from app.services.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    """本地文件系统存储后端"""

    def __init__(self, base_path: str, base_url_prefix: str = "/api/files"):
        """
        Args:
            base_path: 存储根目录（如 "./storage/files"）
            base_url_prefix: 文件访问 URL 前缀
        """
        self.base_path = os.path.abspath(base_path)
        self.base_url_prefix = base_url_prefix
        os.makedirs(self.base_path, exist_ok=True)

    def _full_path(self, key: str) -> str:
        """将存储键转为本地完整路径"""
        return os.path.join(self.base_path, key)

    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """上传文件到本地磁盘"""
        full_path = self._full_path(key)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        async with aiofiles.open(full_path, "wb") as f:
            await f.write(data)

        return key

    async def download(self, key: str) -> bytes:
        """从本地磁盘读取文件"""
        full_path = self._full_path(key)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"文件不存在: {key}")

        async with aiofiles.open(full_path, "rb") as f:
            return await f.read()

    async def get_size(self, key: str) -> int:
        """获取本地文件大小"""
        full_path = self._full_path(key)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"文件不存在: {key}")
        return os.path.getsize(full_path)

    async def get_url(self, key: str, expires: int = 3600) -> str:
        """返回本地文件的 API 访问路径（通过后端代理）"""
        # 本地模式通过 API 端点代理访问，不需要 presigned URL
        parts = key.split("/")
        for index in range(len(parts) - 3, -1, -1):
            if parts[index] != "files":
                continue
            file_id = parts[index + 1]
            variant_name = parts[index + 2]
            if not file_id:
                continue
            variant = "processed" if variant_name.startswith("processed") else "thumbnail"
            return f"{self.base_url_prefix}/{file_id}/content?variant={variant}"
        return f"{self.base_url_prefix}/content/{key}"

    async def delete(self, key: str) -> bool:
        """删除本地磁盘上的文件"""
        full_path = self._full_path(key)
        try:
            if os.path.exists(full_path):
                os.remove(full_path)
                return True
            return False
        except OSError:
            return False

    async def exists(self, key: str) -> bool:
        """判断本地文件是否存在"""
        return os.path.exists(self._full_path(key))
