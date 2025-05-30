"""Service for managing saved SQL queries."""

import json
import uuid
import logging
from datetime import datetime
from typing import List, Optional
import boto3
from botocore.exceptions import ClientError

from ..models.catalog import SavedQuery

# Configure logging
logger = logging.getLogger(__name__)

class QueryService:
    def __init__(self, bucket: str, aws_region: str = 'us-east-1'):
        """Initialize the query service.
        
        Args:
            bucket: S3 bucket name
            aws_region: AWS region
        """
        self.bucket = bucket
        logger.info(f"Initializing QueryService with bucket: {bucket}, region: {aws_region}")
        self.s3_client = boto3.client('s3', region_name=aws_region)
        self.queries_prefix = 'queries/'

    def _get_queries_key(self, user_id: str) -> str:
        """Get the S3 key for user's queries file."""
        key = f"{self.queries_prefix}{user_id}/all_queries.json"
        logger.debug(f"Generated S3 key: {key}")
        return key

    async def _ensure_queries_file_exists(self, user_id: str) -> None:
        """Ensure the queries file exists in S3.
        
        Args:
            user_id: User ID
        """
        try:
            key = self._get_queries_key(user_id)
            logger.info(f"Ensuring queries file exists at: s3://{self.bucket}/{key}")
            
            try:
                # Try to get the file
                self.s3_client.head_object(Bucket=self.bucket, Key=key)
                logger.info(f"Queries file already exists at: s3://{self.bucket}/{key}")
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    logger.info(f"Queries file not found, creating new file at: s3://{self.bucket}/{key}")
                    # File doesn't exist, create it with empty queries list
                    self.s3_client.put_object(
                        Bucket=self.bucket,
                        Key=key,
                        Body=json.dumps({"queries": []}, indent=2),
                        ContentType='application/json'
                    )
                    logger.info(f"Successfully created new queries file at: s3://{self.bucket}/{key}")
                else:
                    logger.error(f"Error checking file existence: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"Error ensuring queries file exists: {str(e)}")
            raise Exception(f"Error ensuring queries file exists: {str(e)}")

    async def save_query(
        self,
        user_id: str,
        name: str,
        query: str,
        tables: List[str],
        description: Optional[str] = None,
        query_id: Optional[str] = None
    ) -> SavedQuery:
        """Save a new query or update an existing one.
        
        Args:
            user_id: User ID
            name: Query name
            query: SQL query
            tables: List of tables used in the query
            description: Optional query description
            query_id: Optional query ID for updates
            
        Returns:
            SavedQuery object
        """
        try:
            logger.info(f"Saving query for user {user_id}: {name}")
            
            # Ensure queries file exists
            await self._ensure_queries_file_exists(user_id)
            
            # Get existing queries
            queries = await self.get_all_queries(user_id)
            logger.debug(f"Retrieved {len(queries)} existing queries")
            
            # Create new query or update existing
            now = datetime.utcnow()
            if query_id and any(q.query_id == query_id for q in queries):
                logger.info(f"Updating existing query with ID: {query_id}")
                # Update existing query
                for q in queries:
                    if q.query_id == query_id:
                        q.name = name
                        q.query = query
                        q.tables = tables
                        q.description = description
                        q.updated_at = now
                        saved_query = q
                        break
            else:
                logger.info("Creating new query")
                # Create new query
                saved_query = SavedQuery(
                    query_id=str(uuid.uuid4()),
                    name=name,
                    description=description,
                    query=query,
                    tables=tables,
                    created_at=now,
                    updated_at=now,
                    created_by=user_id,
                    is_favorite=False,
                    execution_count=0
                )
                queries.append(saved_query)
            
            # Save to S3
            await self._save_queries_to_s3(user_id, queries)
            logger.info(f"Successfully saved query with ID: {saved_query.query_id}")
            return saved_query
            
        except Exception as e:
            logger.error(f"Error saving query: {str(e)}")
            raise Exception(f"Error saving query: {str(e)}")

    async def get_all_queries(self, user_id: str) -> List[SavedQuery]:
        """Get all saved queries for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            List of SavedQuery objects
        """
        try:
            # Ensure queries file exists
            await self._ensure_queries_file_exists(user_id)
            
            key = self._get_queries_key(user_id)
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=key
                )
                queries_data = json.loads(response['Body'].read().decode('utf-8'))
                return [SavedQuery(**q) for q in queries_data.get('queries', [])]
            except ClientError as e:
                if e.response['Error']['Code'] == 'NoSuchKey':
                    return []
                raise
        except Exception as e:
            raise Exception(f"Error getting queries: {str(e)}")

    async def get_query(self, user_id: str, query_id: str) -> Optional[SavedQuery]:
        """Get a specific query by ID.
        
        Args:
            user_id: User ID
            query_id: Query ID
            
        Returns:
            SavedQuery object or None if not found
        """
        queries = await self.get_all_queries(user_id)
        for query in queries:
            if query.query_id == query_id:
                return query
        return None

    async def delete_query(self, user_id: str, query_id: str) -> bool:
        """Delete a saved query.
        
        Args:
            user_id: User ID
            query_id: Query ID
            
        Returns:
            True if deleted, False if not found
        """
        try:
            queries = await self.get_all_queries(user_id)
            original_length = len(queries)
            queries = [q for q in queries if q.query_id != query_id]
            
            if len(queries) < original_length:
                await self._save_queries_to_s3(user_id, queries)
                return True
            return False
            
        except Exception as e:
            raise Exception(f"Error deleting query: {str(e)}")

    async def update_query_favorite(
        self,
        user_id: str,
        query_id: str,
        is_favorite: bool
    ) -> Optional[SavedQuery]:
        """Update query favorite status.
        
        Args:
            user_id: User ID
            query_id: Query ID
            is_favorite: New favorite status
            
        Returns:
            Updated SavedQuery object or None if not found
        """
        try:
            queries = await self.get_all_queries(user_id)
            for query in queries:
                if query.query_id == query_id:
                    query.is_favorite = is_favorite
                    query.updated_at = datetime.utcnow()
                    await self._save_queries_to_s3(user_id, queries)
                    return query
            return None
            
        except Exception as e:
            raise Exception(f"Error updating query favorite status: {str(e)}")

    async def update_query_execution(
        self,
        user_id: str,
        query_id: str
    ) -> Optional[SavedQuery]:
        """Update query execution stats.
        
        Args:
            user_id: User ID
            query_id: Query ID
            
        Returns:
            Updated SavedQuery object or None if not found
        """
        try:
            queries = await self.get_all_queries(user_id)
            for query in queries:
                if query.query_id == query_id:
                    query.execution_count += 1
                    query.last_run = datetime.utcnow()
                    await self._save_queries_to_s3(user_id, queries)
                    return query
            return None
            
        except Exception as e:
            raise Exception(f"Error updating query execution stats: {str(e)}")

    async def _save_queries_to_s3(
        self,
        user_id: str,
        queries: List[SavedQuery]
    ) -> None:
        """Save queries to S3.
        
        Args:
            user_id: User ID
            queries: List of SavedQuery objects
        """
        try:
            key = self._get_queries_key(user_id)
            logger.info(f"Saving {len(queries)} queries to S3: s3://{self.bucket}/{key}")
            
            queries_data = {"queries": [q.dict() for q in queries]}
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=json.dumps(queries_data, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Successfully saved queries to S3: s3://{self.bucket}/{key}")
        except Exception as e:
            logger.error(f"Error saving queries to S3: {str(e)}")
            raise Exception(f"Error saving queries to S3: {str(e)}") 