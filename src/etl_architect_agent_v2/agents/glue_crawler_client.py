"""AWS Glue Crawler API Client."""

import boto3
from typing import Dict, Any, Optional
import logging
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)

class GlueCrawlerAPIClient:
    """Client for Glue Crawler API interactions."""
    
    def __init__(self, region_name: str = "us-east-1"):
        """Initialize the Glue Crawler API client.
        
        Args:
            region_name: AWS region name
        """
        self.glue_client = boto3.client('glue', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
    
    async def create_crawler(
        self,
        crawler_name: str,
        database_name: str,
        s3_path: str,
        role_arn: str
    ) -> Dict[str, Any]:
        """Create a new Glue crawler.
        
        Args:
            crawler_name: Name of the crawler
            database_name: Target Glue database name
            s3_path: S3 path to crawl
            role_arn: IAM role ARN for the crawler
            
        Returns:
            Status information
        """
        try:
            # Check if crawler exists
            try:
                self.glue_client.get_crawler(Name=crawler_name)
                logger.info(f"Crawler {crawler_name} already exists")
                return {"status": "exists"}
            except ClientError as e:
                if e.response['Error']['Code'] != 'EntityNotFoundException':
                    raise
            
            # Create crawler
            self.glue_client.create_crawler(
                Name=crawler_name,
                Role=role_arn,
                DatabaseName=database_name,
                Targets={'S3Targets': [{'Path': s3_path}]}
            )
            
            logger.info(f"Created crawler {crawler_name}")
            return {"status": "created"}
        except ClientError as e:
            logger.error(f"Error creating crawler: {str(e)}")
            raise
    
    async def start_crawler(self, crawler_name: str) -> Dict[str, Any]:
        """Start a Glue crawler.
        
        Args:
            crawler_name: Name of the crawler
            
        Returns:
            Status information
        """
        try:
            self.glue_client.start_crawler(Name=crawler_name)
            logger.info(f"Started crawler {crawler_name}")
            return {"status": "started"}
        except ClientError as e:
            if e.response['Error']['Code'] == 'CrawlerRunningException':
                return {"status": "already_running"}
            logger.error(f"Error starting crawler: {str(e)}")
            raise
    
    async def get_crawler_status(self, crawler_name: str) -> Dict[str, Any]:
        """Get crawler status.
        
        Args:
            crawler_name: Name of the crawler
            
        Returns:
            Status information
        """
        try:
            response = self.glue_client.get_crawler(Name=crawler_name)
            state = response['Crawler']['State']
            last_crawl = response['Crawler'].get('LastCrawl', {})
            
            return {
                "state": state,
                "last_crawl_status": last_crawl.get('Status'),
                "last_crawl_time": last_crawl.get('StartTime'),
                "last_crawl_duration": last_crawl.get('Duration')
            }
        except ClientError as e:
            logger.error(f"Error getting crawler status: {str(e)}")
            raise
    
    async def extract_schema(
        self,
        database_name: str,
        table_name: str
    ) -> Dict[str, Any]:
        """Extract schema from a Glue table.
        
        Args:
            database_name: Glue database name
            table_name: Glue table name
            
        Returns:
            Schema information
        """
        try:
            response = self.glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            table_info = response['Table']
            schema_info = {
                "database_name": database_name,
                "table_name": table_name,
                "storage_location": table_info.get(
                    'StorageDescriptor', {}).get('Location'),
                "columns": [
                    {
                        "name": col['Name'],
                        "type": col['Type'],
                        "comment": col.get('Comment', '')
                    }
                    for col in table_info.get(
                        'StorageDescriptor', {}).get('Columns', [])
                ],
                "partition_keys": [
                    {
                        "name": col['Name'],
                        "type": col['Type'],
                        "comment": col.get('Comment', '')
                    }
                    for col in table_info.get('PartitionKeys', [])
                ],
                "table_type": table_info.get('TableType'),
                "parameters": table_info.get('Parameters', {}),
                "last_updated": (
                    table_info.get('UpdateTime').isoformat()
                    if table_info.get('UpdateTime') else None
                )
            }
            
            logger.info(f"Extracted schema for {database_name}.{table_name}")
            return schema_info
        except ClientError as e:
            logger.error(f"Error extracting schema: {str(e)}")
            raise 