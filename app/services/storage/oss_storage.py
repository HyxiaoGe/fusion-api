"""阿里云 OSS 对象存储后端实现"""

import asyncio
import io

from app.core.logger import app_logger as logger
from app.services.storage.base import StorageBackend


class OSSStorageBackend(StorageBackend):
    """阿里云 OSS 存储后端"""

    def __init__(
        self,
        endpoint: str,
        access_key_id: str,
        access_key_secret: str,
        bucket: str,
        use_ssl: bool = True,
    ):
        """
        Args:
            endpoint: OSS endpoint，如 oss-cn-shenzhen.aliyuncs.com
            access_key_id: RAM 用户 AccessKey ID
            access_key_secret: RAM 用户 AccessKey Secret
            bucket: Bucket 名称
            use_ssl: endpoint 未带协议时是否使用 HTTPS
        """
        if not endpoint:
            raise ValueError("OSS_ENDPOINT 未配置")
        if not access_key_id or not access_key_secret:
            raise ValueError("OSS_ACCESS_KEY_ID 或 OSS_ACCESS_KEY_SECRET 未配置")
        if not bucket:
            raise ValueError("OSS_BUCKET 未配置")

        import oss2

        protocol = "https" if use_ssl else "http"
        self.endpoint_url = endpoint if endpoint.startswith(("http://", "https://")) else f"{protocol}://{endpoint}"
        self.bucket_name = bucket
        self._oss2 = oss2
        self._bucket = oss2.Bucket(
            oss2.Auth(access_key_id, access_key_secret),
            self.endpoint_url,
            bucket,
        )
        logger.info("OSS 存储后端初始化: endpoint=%s, bucket=%s", self.endpoint_url, bucket)

    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """上传文件到 OSS"""
        headers = {"Content-Type": content_type}
        await asyncio.to_thread(
            self._bucket.put_object,
            key,
            io.BytesIO(data),
            headers=headers,
        )
        logger.debug("OSS 上传成功: %s (%s bytes)", key, len(data))
        return key

    async def download(self, key: str) -> bytes:
        """从 OSS 下载文件内容"""
        try:
            result = await asyncio.to_thread(self._bucket.get_object, key)
            return await asyncio.to_thread(result.read)
        except self._oss2.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(f"文件不存在: {key}") from exc

    async def get_size(self, key: str) -> int:
        """通过 HEAD 获取 OSS 对象大小"""
        try:
            result = await asyncio.to_thread(self._bucket.head_object, key)
            return int(result.headers.get("Content-Length", "0"))
        except self._oss2.exceptions.NoSuchKey as exc:
            raise FileNotFoundError(f"文件不存在: {key}") from exc

    async def get_url(self, key: str, expires: int = 3600) -> str:
        """生成 OSS GET 签名 URL"""
        return await asyncio.to_thread(
            self._bucket.sign_url,
            "GET",
            key,
            expires,
            slash_safe=True,
        )

    async def get_upload_url(self, key: str, content_type: str, expires: int = 3600) -> dict:
        """生成浏览器直传 OSS 的 PUT 签名 URL"""
        headers = {"Content-Type": content_type}
        url = await asyncio.to_thread(
            self._bucket.sign_url,
            "PUT",
            key,
            expires,
            headers=headers,
            slash_safe=True,
        )
        return {
            "url": url,
            "method": "PUT",
            "headers": headers,
            "expires_in": expires,
        }

    async def delete(self, key: str) -> bool:
        """删除 OSS 对象"""
        try:
            await asyncio.to_thread(self._bucket.delete_object, key)
            return True
        except self._oss2.exceptions.NoSuchKey:
            return False

    async def exists(self, key: str) -> bool:
        """判断 OSS 对象是否存在"""
        try:
            await asyncio.to_thread(self._bucket.head_object, key)
            return True
        except self._oss2.exceptions.NoSuchKey:
            return False
