import boto3
from typing import Dict, Any, Optional

class S3Service:
    def __init__(self, bucket: str, aws_region: str):
        self.bucket = bucket
        self.s3_client = boto3.client('s3', region_name=aws_region)

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