import logging

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError

from kern.config import Settings

logger = logging.getLogger(__name__)


class ObjectStore:
    """Minimal S3-compatible store. Keys are opaque; callers choose them."""

    def __init__(self, settings: Settings):
        self._bucket = settings.s3_bucket
        self._client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint,
            aws_access_key_id=settings.s3_access_key,
            aws_secret_access_key=settings.s3_secret_key,
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
            ),
        )

    def ensure_bucket(self) -> None:
        try:
            self._client.head_bucket(Bucket=self._bucket)
            return
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchBucket"}:
                raise
        self._client.create_bucket(Bucket=self._bucket)
        logger.info("created bucket %s", self._bucket)

    def put_if_absent(self, key: str, data: bytes) -> None:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code not in {"404", "NoSuchBucket", "NoSuchKey"}:
                raise
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data)

    def get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code")
            if code in {"404", "NoSuchKey"}:
                raise KeyError(key) from error
            raise
        return response["Body"].read()
