"""存储后端抽象基类"""

from abc import ABC, abstractmethod


class DefinitiveStorageUploadError(RuntimeError):
    """对象存储已明确拒绝 PUT，远端不会在稍后生成对象。"""


def is_definitive_upload_rejection(exc: Exception) -> bool:
    """只把有明确 4xx 响应的 PUT 失败视为未写入；超时和限流仍属不确定。"""
    status = getattr(exc, "status", None)
    response = getattr(exc, "response", None)
    if status is None and isinstance(response, dict):
        metadata = response.get("ResponseMetadata")
        if isinstance(metadata, dict):
            status = metadata.get("HTTPStatusCode")
    try:
        status_code = int(status)
    except (TypeError, ValueError):
        return False
    return 400 <= status_code < 500 and status_code not in {408, 425, 429}


class StorageBackend(ABC):
    """文件存储后端抽象基类，定义统一的存储接口"""

    @abstractmethod
    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """
        上传文件到存储后端。

        Args:
            key: 存储键（如 "files/v1/users/{user_id}/conversations/{conversation_id}/files/{file_id}/processed.jpg"）
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

    async def get_upload_url(self, key: str, content_type: str, expires: int = 3600) -> dict:
        """
        获取浏览器直传 URL。

        默认后端不支持直传；对象存储后端可覆盖该方法。
        """
        raise NotImplementedError("当前存储后端不支持直传上传")

    async def get_size(self, key: str) -> int:
        """
        获取文件大小。

        默认实现通过下载后计算长度；对象存储后端应覆盖为 HEAD 元数据查询。
        """
        return len(await self.download(key))

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
