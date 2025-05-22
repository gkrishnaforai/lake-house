import boto3
from typing import Dict, Any
from botocore.exceptions import ClientError


class S3Service:
    def __init__(self, bucket: str, aws_region: str):
        self.bucket = bucket
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.exceptions = boto3.client('s3').exceptions

    async def upload_file(self, file_content: bytes, key: str) -> Dict[str, Any]:
        """Upload a file to S3."""
        try:
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=file_content
            )
            return {"status": "success", "key": key}
        except Exception as e:
            raise Exception(f"Failed to upload file: {str(e)}")

    async def get_file(self, key: str) -> bytes:
        """Get a file from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=self.bucket,
                Key=key
            )
            return response['Body'].read()
        except Exception as e:
            raise Exception(f"Failed to get file: {str(e)}")

    async def delete_file(self, key: str) -> Dict[str, Any]:
        """Delete a file from S3."""
        try:
            self.s3_client.delete_object(
                Bucket=self.bucket,
                Key=key
            )
            return {"status": "success", "key": key}
        except Exception as e:
            raise Exception(f"Failed to delete file: {str(e)}")

    def list_objects_v2(self, Bucket: str, Prefix: str) -> Dict[str, Any]:
        """List objects in an S3 bucket with a given prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=Bucket,
                Prefix=Prefix
            )
            return response
        except Exception as e:
            raise Exception(f"Failed to list objects: {str(e)}")

    def head_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """Get metadata for an object in S3."""
        try:
            response = self.s3_client.head_object(
                Bucket=Bucket,
                Key=Key
            )
            return response
        except Exception as e:
            raise Exception(f"Failed to get object metadata: {str(e)}")

    def get_object(self, Bucket: str, Key: str) -> Dict[str, Any]:
        """Get an object from S3."""
        try:
            response = self.s3_client.get_object(
                Bucket=Bucket,
                Key=Key
            )
            return response
        except Exception as e:
            raise Exception(f"Failed to get object: {str(e)}") 