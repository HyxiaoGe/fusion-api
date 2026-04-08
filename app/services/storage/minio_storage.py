"""MinIO/S3 兼容对象存储后端实现"""

import io
from functools import lru_cache

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError

from app.core.logger import app_logger as logger
from app.services.storage.base import StorageBackend


class MinIOStorageBackend(StorageBackend):
    """MinIO/S3 兼容存储后端"""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        use_ssl: bool = False,
    ):
        """
        Args:
            endpoint: MinIO 服务地址（如 "192.168.1.10:9000"）
            access_key: 访问密钥
            secret_key: 秘密密钥
            bucket: 存储桶名称
            use_ssl: 是否使用 HTTPS
        """
        self.bucket = bucket
        protocol = "https" if use_ssl else "http"
        self.endpoint_url = f"{protocol}://{endpoint}"

        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=BotoConfig(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )
        logger.info(f"MinIO 存储后端初始化: endpoint={self.endpoint_url}, bucket={bucket}")

    async def ensure_bucket(self) -> None:
        """确保存储桶存在，不存在则创建"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
            logger.info(f"MinIO bucket 已存在: {self.bucket}")
        except ClientError:
            self.s3_client.create_bucket(Bucket=self.bucket)
            logger.info(f"MinIO bucket 已创建: {self.bucket}")

    async def upload(self, key: str, data: bytes, content_type: str) -> str:
        """上传文件到 MinIO"""
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=io.BytesIO(data),
            ContentLength=len(data),
            ContentType=content_type,
        )
        logger.debug(f"MinIO 上传成功: {key} ({len(data)} bytes)")
        return key

    async def download(self, key: str) -> bytes:
        """从 MinIO 下载文件"""
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            data = response["Body"].read()
            return data
        except ClientError as e:
            if e.response["Error"]["Code"] == "NoSuchKey":
                raise FileNotFoundError(f"文件不存在: {key}")
            raise

    async def get_url(self, key: str, expires: int = 3600) -> str:
        """生成 MinIO presigned URL"""
        url = self.s3_client.generate_presigned_url(
            "get_object",
            Params={"Bucket": self.bucket, "Key": key},
            ExpiresIn=expires,
        )
        return url

    async def delete(self, key: str) -> bool:
        """删除 MinIO 中的文件"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False

    async def exists(self, key: str) -> bool:
        """判断 MinIO 中文件是否存在"""
        try:
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
            return True
        except ClientError:
            return False
