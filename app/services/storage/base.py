"""存储后端抽象基类"""

from abc import ABC, abstractmethod


class StorageBackend(ABC):
    """文件存储后端抽象基类，定义统一的存储接口"""

    @abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """
        上传文件到存储后端。

        Args:
            key: 存储键（如 "conv_id/file_id/processed.jpg"）
            data: 文件二进制内容
            content_type: MIME 类型

        Returns:
            存储键（与输入 key 相同）
        """
        ...

    @abstractmethod
    async def download(self, key: str) -> bytes:
        """
        从存储后端下载文件内容。

        Args:
            key: 存储键

        Returns:
            文件二进制内容

        Raises:
            FileNotFoundError: 文件不存在
        """
        ...

    @abstractmethod
    async def get_url(self, key: str, expires: int = 3600) -> str:
        """
        获取文件的访问 URL。

        Args:
            key: 存储键
            expires: URL 有效期（秒），仅 MinIO 等需要签名的后端使用

        Returns:
            可访问的 URL 字符串
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        删除存储中的文件。

        Args:
            key: 存储键

        Returns:
            是否成功删除
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        判断文件是否存在。

        Args:
            key: 存储键

        Returns:
            文件是否存在
        """
        ...
