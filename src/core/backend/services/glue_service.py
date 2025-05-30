"""Glue service implementation."""

import logging
from typing import List, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class GlueService:
    """Service for managing AWS Glue operations."""

    def __init__(self):
        """Initialize the Glue service."""
        self.glue_client = boto3.client('glue')
        self.s3_client = boto3.client('s3')

    async def create_database(
        self,
        database_name: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new Glue database."""
        try:
            response = self.glue_client.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': description
                }
            )
            return {
                'status': 'success',
                'database_name': database_name,
                'message': 'Database created successfully'
            }
        except ClientError as e:
            logger.error(f"Error creating database: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def create_table(
        self,
        database_name: str,
        table_name: str,
        columns: List[Dict[str, str]],
        s3_location: str,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new Glue table."""
        try:
            response = self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput={
                    'Name': table_name,
                    'Description': description,
                    'StorageDescriptor': {
                        'Columns': columns,
                        'Location': s3_location,
                        'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                        'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                        'SerdeInfo': {
                            'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
                        }
                    },
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            return {
                'status': 'success',
                'table_name': table_name,
                'message': 'Table created successfully'
            }
        except ClientError as e:
            logger.error(f"Error creating table: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def start_crawler(
        self,
        crawler_name: str
    ) -> Dict[str, Any]:
        """Start a Glue crawler."""
        try:
            response = self.glue_client.start_crawler(
                Name=crawler_name
            )
            return {
                'status': 'success',
                'crawler_name': crawler_name,
                'message': 'Crawler started successfully'
            }
        except ClientError as e:
            logger.error(f"Error starting crawler: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def get_crawler_status(
        self,
        crawler_name: str
    ) -> Dict[str, Any]:
        """Get the status of a Glue crawler."""
        try:
            response = self.glue_client.get_crawler(
                Name=crawler_name
            )
            return {
                'status': 'success',
                'crawler_name': crawler_name,
                'state': response['Crawler']['State'],
                'last_updated': response['Crawler']['LastUpdated'].isoformat()
            }
        except ClientError as e:
            logger.error(f"Error getting crawler status: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            } 