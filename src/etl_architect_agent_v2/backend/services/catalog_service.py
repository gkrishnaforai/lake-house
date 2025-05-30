"""Catalog service for managing data catalog and metadata."""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging
import asyncio
from io import BytesIO
import os
import re
import tempfile
import pandas as pd
import pyarrow
import avro
import duckdb
from pathlib import Path
import boto3
from pydantic import BaseModel
import hashlib

from core.llm.manager import LLMManager
from .glue_service import GlueService, GlueTableConfig
from .s3_service import S3Service
from .sql_generation_service import SQLGenerationService, SQLGenerationRequest
from ..models.transformation import (
    TransformationTemplate,
    TransformationConfig,
    TransformationResult
)
from .athena_service import AthenaService


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TableInfo(BaseModel):
    name: str
    description: Optional[str] = None
    schema: Optional[List[Dict[str, str]]] = None  # Changed to List[Dict[str, str]]
    rowCount: Optional[int] = None
    lastUpdated: Optional[datetime] = None
    createdBy: Optional[str] = None
    createdAt: Optional[datetime] = None
    etag: Optional[str] = None
    location: Optional[str] = None  # Made location optional

    def dict(self, *args, **kwargs):
        d = super().dict(*args, **kwargs)
        # Convert datetime objects to ISO format strings
        if d.get('lastUpdated'):
            d['lastUpdated'] = d['lastUpdated'].isoformat()
        if d.get('createdAt'):
            d['createdAt'] = d['createdAt'].isoformat()
        return d


class QualityCheckConfig(BaseModel):
    """Configuration for data quality checks."""
    enabled_metrics: List[str] = ["completeness", "uniqueness", "consistency"]
    thresholds: Dict[str, float] = {
        "completeness": 0.95,
        "uniqueness": 0.90,
        "consistency": 0.85
    }
    schedule: Optional[str] = None  # Cron expression for periodic checks
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None


class CatalogService:
    """Service for managing data catalog and metadata."""
    
    def __init__(
        self,
        bucket: str,
        aws_region: str = 'us-east-1',
        llm_manager: Optional[LLMManager] = None,
        start_background_sync: bool = True
    ):
        """Initialize the catalog service."""
        self.bucket = bucket
        self.aws_region = aws_region
        self.base_database_name = os.getenv(
            'GLUE_DATABASE_NAME', 
            'data_lakehouse'
        )
        
        # Initialize AWS clients first
        self.s3_client = boto3.client('s3')
        self.tables_prefix = 'tables/'
        
        # Initialize services
        self.s3_service = S3Service(bucket, aws_region)
        self.glue_service = GlueService(region_name=aws_region)
        self.athena_service = AthenaService(
            database="",  # Empty database name - will be set per query
            output_location=f's3://{bucket}/athena-results/',
            aws_region=aws_region
        )
        
        # Initialize LLM and SQL generation
        self.llm_manager = llm_manager or LLMManager()
        self.sql_generation_service = SQLGenerationService(
            llm_manager=self.llm_manager,
            glue_service=self.glue_service
        )

        # Initialize DuckDB with persistent storage in S3
        self.duckdb_path = Path("data/duckdb/catalog.db")
        self.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        self.duckdb_s3_key = "duckdb/catalog.db"
        self._sync_duckdb_from_s3()
        self.conn = duckdb.connect(str(self.duckdb_path))
        
        # Initialize DuckDB schema
        self._init_duckdb_schema()
        
        # Store flag for background sync
        self.start_background_sync = start_background_sync
        self._background_sync_task = None
        self._last_sync_time = {}  # Track last sync time per table

        self._ensure_tables_bucket()
        self._init_duckdb()

    def _sync_duckdb_from_s3(self):
        """Sync DuckDB database from S3."""
        try:
            # Try to get DuckDB from S3
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=self.duckdb_s3_key
                )
                with open(self.duckdb_path, 'wb') as f:
                    f.write(response['Body'].read())
                logger.info("Successfully synced DuckDB from S3")
            except self.s3_client.exceptions.NoSuchKey:
                logger.info("No existing DuckDB in S3, creating new")
                # Create empty DuckDB file
                conn = duckdb.connect(str(self.duckdb_path))
                conn.close()
                # Sync the new file back to S3
                self._sync_duckdb_to_s3()
        except Exception as e:
            logger.error(f"Error syncing DuckDB from S3: {str(e)}")

    def _sync_duckdb_to_s3(self):
        """Sync DuckDB database to S3."""
        try:
            with open(self.duckdb_path, 'rb') as f:
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=self.duckdb_s3_key,
                    Body=f.read()
                )
            logger.info("Successfully synced DuckDB to S3")
        except Exception as e:
            logger.error(f"Error syncing DuckDB to S3: {str(e)}")

    def _get_table_etag(self, table_name: str, user_id: str) -> str:
        """Get ETag for a table to detect changes."""
        try:
            # Get table metadata from Glue
            response = self.glue_service.get_table(
                DatabaseName=self.get_user_database_name(user_id),
                Name=table_name
            )
            
            # Create ETag from table metadata
            metadata = {
                'name': table_name,
                'lastUpdated': response['Table'].get('LastUpdatedTime', ''),
                'schema': response['Table'].get(
                    'StorageDescriptor', {}
                ).get('Columns', []),
                'location': response['Table'].get(
                    'StorageDescriptor', {}
                ).get('Location', '')
            }
            
            return hashlib.md5(
                json.dumps(metadata, sort_keys=True).encode()
            ).hexdigest()
        except Exception as e:
            logger.error(f"Error getting table ETag: {str(e)}")
            return ''

    async def sync_table_to_duckdb(self, table_name: str, user_id: str):
        """Synchronize a table from S3 to DuckDB cache."""
        try:
            # Get current ETag
            current_etag = self._get_table_etag(table_name, user_id)
            
            # Check if table needs sync
            cached_etag = self.conn.execute("""
                SELECT etag FROM table_cache WHERE table_name = ?
            """, (table_name,)).fetchone()
            
            if cached_etag and cached_etag[0] == current_etag:
                logger.info(f"Table {table_name} is up to date")
                return True
            
            # Get table schema and data from S3
            schema = await self.get_table_schema(table_name, user_id)
            data = await self.get_table_data(table_name, user_id)
            
            # Update schema cache
            self.conn.execute("""
                INSERT INTO table_cache (
                    table_name, s3_path, last_updated, schema, etag
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (table_name) DO UPDATE SET
                    s3_path = excluded.s3_path,
                    last_updated = excluded.last_updated,
                    schema = excluded.schema,
                    etag = excluded.etag
            """, (
                table_name,
                f"s3://{self.bucket}/data/{user_id}/{table_name}",
                datetime.now(),
                json.dumps(schema),
                current_etag
            ))
            
            # Update data cache
            self.conn.execute("""
                INSERT INTO table_data (
                    table_name, data, last_updated
                ) VALUES (?, ?, ?)
                ON CONFLICT (table_name) DO UPDATE SET
                    data = excluded.data,
                    last_updated = excluded.last_updated
            """, (
                table_name,
                json.dumps(data),
                datetime.now()
            ))
            
            # Sync DuckDB to S3
            self._sync_duckdb_to_s3()
            
            return True
        except Exception as e:
            logger.error(f"Error syncing table to DuckDB: {str(e)}")
            return False

    async def run_background_sync(self, sync_interval: int = 3600):
        """Start background task to periodically sync tables."""
        while True:
            try:
                logger.info("Starting periodic table sync to DuckDB")
                
                # Get all tables
                tables = await self.list_tables("test_user")  # TODO: Handle multiple users
                
                # Sync only changed tables
                for table in tables:
                    current_etag = self._get_table_etag(
                        table.name, "test_user"
                    )
                    cached_etag = self.conn.execute("""
                        SELECT etag FROM table_cache WHERE table_name = ?
                    """, (table.name,)).fetchone()
                    
                    if not cached_etag or cached_etag[0] != current_etag:
                        logger.info(
                            f"Syncing changed table: {table.name}"
                        )
                        await self.sync_table_to_duckdb(
                            table.name, "test_user"
                        )
                
                logger.info("Completed periodic table sync to DuckDB")
            except Exception as e:
                logger.error(f"Error in background sync: {str(e)}")
            
            # Wait for next sync
            await asyncio.sleep(sync_interval)

    async def start(self):
        """Start the service asynchronously."""
       # if self.start_background_sync:
       #     self._background_sync_task = asyncio.create_task(
       #         self.run_background_sync()
       #     )

    def _init_duckdb_schema(self):
        """Initialize DuckDB schema."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS table_cache (
                table_name VARCHAR PRIMARY KEY,
                s3_path VARCHAR,
                last_updated TIMESTAMP,
                schema JSON,
                etag VARCHAR,
                version VARCHAR  -- Add version field to track Glue table changes
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS table_data (
                table_name VARCHAR PRIMARY KEY,
                data JSON,
                last_updated TIMESTAMP
            )
        """)

    async def get_cached_table_data(self, table_name: str) -> Optional[Dict[str, Any]]:
        """Get table data from DuckDB cache."""
        try:
            result = self.conn.execute("""
                SELECT data, last_updated
                FROM table_data
                WHERE table_name = ?
            """, (table_name,)).fetchone()
            
            if result:
                return {
                    'data': json.loads(result[0]),
                    'last_updated': result[1]
                }
            return None
        except Exception as e:
            logger.error(f"Error getting cached table data: {str(e)}")
            return None

    async def execute_query(
        self,
        query: str,
        user_id: str = "test_user",
        output_location: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute a query using DuckDB if possible, fallback to Athena."""
        try:
            # Try to execute query in DuckDB first
            try:
                result = self.conn.execute(query).fetchall()
                return {
                    'status': 'success',
                    'results': result
                }
            except Exception as duckdb_error:
                logger.warning(f"DuckDB query failed, falling back to Athena: {str(duckdb_error)}")
                
                # Fallback to Athena
                database_name = self.get_user_database_name(user_id)
                return await self.athena_service.execute_query(
                    query=query,
                    database_name=database_name
                )
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def process_descriptive_query(
        self,
        query: str,
        table_name: Optional[str] = None,
        preserve_column_names: bool = True,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Process a descriptive query using SQLGeneratorAgent."""
        try:
            logger.info(f"Starting process_descriptive_query with query: {query}")
            logger.info(f"Table name: {table_name}, User ID: {user_id}")

            # Validate that a table is selected
            if not table_name:
                logger.error("No table selected for descriptive query")
                return {
                    'status': 'error',
                    'message': 'Please select a table before running the query'
                }

            # Get table schema
            if table_name:
                schema = await self.get_table_schema(table_name, user_id)
                logger.info(
                    f"Retrieved schema for table {table_name}: "
                    f"{json.dumps(schema, indent=2)}"
                )
            else:
                schema = {"tables": await self.list_tables(user_id)}
                logger.info(f"Retrieved list of tables: {json.dumps(schema, indent=2)}")

            # Generate SQL using SQLGenerationService
            logger.info("Generating SQL using SQLGenerationService")
            sql_request = SQLGenerationRequest(
                query=query,
                schema=schema,
                table_name=table_name,
                preserve_column_names=preserve_column_names,
                user_id=user_id
            )
            
            sql_response = await self.sql_generation_service.generate_sql(
                sql_request
            )
            logger.info(f"SQL generation response: {json.dumps(sql_response.dict(), indent=2)}")
            
            if sql_response.status == "error":
                logger.error(f"SQL generation failed: {sql_response.error}")
                return {
                    'status': 'error',
                    'message': sql_response.error
                }

            # Execute query - let execute_query handle database name
            logger.info("Executing query in Athena")
            query_result = await self.execute_query(sql_response.sql_query, user_id)
            logger.info(f"Query execution result: {json.dumps(query_result, indent=2)}")

            if query_result['status'] == 'success':
                if not query_result.get('results'):
                    # Check if table exists and has data
                    try:
                        logger.info(f"Checking if table {table_name} exists and has data")
                        table_info = await self.glue_service.get_table(user_id, table_name)
                        if not table_info:
                            logger.error(f"Table {table_name} does not exist")
                            return {
                                'status': 'error',
                                'message': f'Table {table_name} does not exist'
                            }
                        
                        # Check if table has data
                        count_query = f"SELECT COUNT(*) FROM {table_name}"
                        logger.info(f"Executing count query: {count_query}")
                        count_result = await self.execute_query(count_query, user_id)
                        if count_result['status'] == 'success':
                            count = (
                                count_result['results'][0][0] 
                                if count_result.get('results') 
                                else 0
                            )
                            logger.info(f"Table has {count} rows")
                            if count == 0:
                                return {
                                    'status': 'success',
                                    'query': sql_response.sql_query,
                                    'results': [],
                                    'message': 'Table exists but has no data'
                                }
                    except Exception as e:
                        logger.error(f"Error checking table data: {str(e)}")
                        return {
                            'status': 'error',
                            'message': f'Error checking table data: {str(e)}'
                        }

                return {
                    'status': 'success',
                    'query': sql_response.sql_query,
                    'results': query_result.get('results', []),
                    'metadata': {
                        'explanation': sql_response.explanation,
                        'confidence': sql_response.confidence,
                        'tables_used': sql_response.tables_used,
                        'columns_used': sql_response.columns_used,
                        'filters': sql_response.filters
                    }
                }
            else:
                logger.error(f"Query execution failed: {query_result.get('message', 'Query execution failed')}")
                return {
                    'status': 'error',
                    'message': query_result.get('message', 'Query execution failed')
                }

        except Exception as e:
            logger.error(f"Error processing descriptive query: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def update_catalog(
        self,
        file_info: Dict[str, Any],
        schema: str
    ) -> Dict[str, Any]:
        """
        Update catalog with new file information
        """
        try:
            # Prepare catalog update task
            update_task = f"""
            Update catalog with new file information:
            File: {file_info['file_name']}
            Schema: {schema}
            Track schema evolution and check data quality.
            """

            # Execute update using agent
            result = await self.agent_executor.ainvoke({
                "input": update_task,
                "chat_history": []
            })

            return {
                "status": "success",
                "details": result
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }

    def _update_schema(
        self,
        file_name: str,
        schema: Dict[str, Any],
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update schema in catalog
        """
        try:
            # Create schema entry
            schema_entry = {
                "file_name": file_name,
                "schema": schema,
                "version": version or datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
                "fields": schema.get("fields", [])
            }

            # Store in S3
            schema_key = f"metadata/schema/{file_name}.json"
            self.s3_client.put_object(
                Key=schema_key,
                Body=json.dumps(schema_entry)
            )

            return {
                "status": "success",
                "schema_path": f"s3://{self.bucket}/{schema_key}"
            }

        except Exception as e:
            raise Exception(f"Error updating schema: {str(e)}")

    async def _check_data_quality(
        self,
        file_name: str,
        data: pd.DataFrame,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """
        Check data quality metrics for a file.
        """
        try:
            # Calculate completeness
            completeness = 1.0 - data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
            
            # Calculate uniqueness
            uniqueness = 1.0 - data.duplicated().sum() / data.shape[0]
            
            # Calculate consistency
            consistency = 0.0
            valid_columns = 0
            
            for col in data.columns:
                if data[col].dtype == "object":
                    # For string columns, check pattern consistency
                    unique_patterns = data[col].str.len().nunique()
                    if unique_patterns <= 3:  # Allow some variation
                        consistency += 1
                elif data[col].dtype in ["int64", "float64"]:
                    # For numeric columns, check value distribution
                    if (data[col].std() / data[col].mean() < 0.5):
                        consistency += 1
                valid_columns += 1
            
            consistency = consistency / valid_columns if valid_columns > 0 else 0.0
            
            # Type validation
            type_validation = {}
            for col in data.columns:
                type_validation[col] = {
                    "expected_type": str(data[col].dtype),
                    "valid_count": data[col].count(),
                    "invalid_count": data.shape[0] - data[col].count()
                }
            
            # Range checks for numeric columns
            range_checks = {}
            for col in data.select_dtypes(include=['int64', 'float64']).columns:
                range_checks[col] = {
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std())
                }
            
            # Pattern validation for string columns
            pattern_checks = {}
            for col in data.select_dtypes(include=['object']).columns:
                # Check for common patterns (email, phone, etc.)
                pattern_checks[col] = {
                    "email_pattern": bool(
                        data[col].str.match(r'^[\w\.-]+@[\w\.-]+\.\w+$').any()
                    ),
                    "phone_pattern": bool(
                        data[col].str.match(r'^\+?[\d\s-]{10,}$').any()
                    ),
                    "date_pattern": bool(
                        data[col].str.match(r'^\d{4}-\d{2}-\d{2}$').any()
                    )
                }
            
            metrics = {
                "row_count": len(data),
                "column_count": len(data.columns),
                "completeness": completeness,
                "uniqueness": uniqueness,
                "consistency": consistency,
                "type_validation": type_validation,
                "range_checks": range_checks,
                "pattern_checks": pattern_checks,
                "column_types": data.dtypes.astype(str).to_dict(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store quality metrics in S3
            quality_key = f"metadata/quality/{user_id}/{file_name}/metrics.json"
            self.s3_client.put_object(
                Key=quality_key,
                Body=json.dumps(metrics)
            )
            
            return {
                "status": "success",
                "quality_metrics": metrics,
                "metadata": {
                    "file_name": file_name,
                    "checked_at": datetime.utcnow().isoformat(),
                    "metrics_location": f"s3://{self.bucket}/{quality_key}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _track_schema_evolution(
        self,
        file_name: str,
        new_schema: str
    ) -> Dict[str, Any]:
        """
        Track schema evolution
        """
        try:
            # Get current schema
            schema_key = f"metadata/schema/{file_name}.json"
            try:
                response = self.s3_service.get_object(
                    Key=schema_key
                )
                current_schema = json.loads(response['Body'].read())
            except self.s3_service.exceptions.NoSuchKey:
                current_schema = None

            # Parse new schema
            new_avro_schema = avro.schema.parse(new_schema)
            new_fields = {
                field.name: str(field.type)
                for field in new_avro_schema.fields
            }

            # Compare schemas
            evolution = {
                "file_name": file_name,
                "timestamp": datetime.utcnow().isoformat(),
                "changes": []
            }

            if current_schema:
                current_fields = {
                    field["name"]: field["type"]
                    for field in current_schema["fields"]
                }

                # Find added fields
                for field_name, field_type in new_fields.items():
                    if field_name not in current_fields:
                        evolution["changes"].append({
                            "type": "added",
                            "field": field_name,
                            "new_type": field_type
                        })

                # Find removed fields
                for field_name, field_type in current_fields.items():
                    if field_name not in new_fields:
                        evolution["changes"].append({
                            "type": "removed",
                            "field": field_name,
                            "old_type": field_type
                        })

                # Find type changes
                for field_name, new_type in new_fields.items():
                    if field_name in current_fields:
                        old_type = current_fields[field_name]
                        if old_type != new_type:
                            evolution["changes"].append({
                                "type": "modified",
                                "field": field_name,
                                "old_type": old_type,
                                "new_type": new_type
                            })

            # Store evolution history
            evolution_key = f"metadata/evolution/{file_name}.json"
            self.s3_client.put_object(
                Key=evolution_key,
                Body=json.dumps(evolution)
            )

            return {
                "status": "success",
                "evolution": evolution,
                "evolution_path": f"s3://{self.bucket}/{evolution_key}"
            }

        except Exception as e:
            raise Exception(f"Error tracking schema evolution: {str(e)}")

    async def get_quality_metrics(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, float]:
        """Get overall quality metrics for all tables.
        
        Args:
            user_id: The ID of the user whose tables to analyze
            
        Returns:
            Dict containing overall quality metrics
        """
        try:
            # Get all tables
            tables = await self.list_tables(user_id=user_id)
            
            if not tables:
                return {
                    "completeness": 0.0,
                    "uniqueness": 0.0,
                    "consistency": 0.0,
                    "timeliness": 0.0
                }
            
            # Get quality metrics for each table
            table_metrics = []
            for table in tables:
                try:
                    metrics = await self.get_table_quality(
                        table["name"],
                        user_id=user_id
                    )
                    table_metrics.append(metrics)
                except Exception as e:
                    logger.warning(f"Error getting metrics for table {table['name']}: {str(e)}")
                    continue
            
            if not table_metrics:
                return {
                    "completeness": 0.0,
                    "uniqueness": 0.0,
                    "consistency": 0.0,
                    "timeliness": 0.0
                }
            
            # Calculate overall metrics
            overall_metrics = {
                "completeness": sum(m["completeness"] for m in table_metrics) / len(table_metrics),
                "uniqueness": sum(m["uniqueness"] for m in table_metrics) / len(table_metrics),
                "consistency": sum(m["consistency"] for m in table_metrics) / len(table_metrics),
                "timeliness": sum(m["timeliness"] for m in table_metrics) / len(table_metrics)
            }
            
            return overall_metrics
            
        except Exception as e:
            logger.error(f"Error getting quality metrics: {str(e)}")
            raise Exception(f"Error getting quality metrics: {str(e)}")

    async def get_table_quality(
        self,
        table_name: str,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Get quality metrics for a specific table.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing quality metrics
        """
        try:
            # First try to get stored quality metrics
            quality_key = f"metadata/quality/{user_id}/{table_name}/metrics.json"
            logger.info(f"Attempting to get quality metrics from S3 key: {quality_key}")
            try:
                response = self.s3_client.get_object(
                    Bucket=self.bucket,
                    Key=quality_key
                )
                stored_metrics = json.loads(response['Body'].read())
                return stored_metrics
            except self.s3_client.exceptions.NoSuchKey:
                # If no stored metrics, calculate them
                logger.info(
                    f"No stored quality metrics found for {table_name}, "
                    "calculating on the fly..."
                )
            
            # Get table schema
            schema = await self.get_table_schema(table_name, user_id)
            
            # Get table data
            query = f"SELECT * FROM {table_name} LIMIT 1000"
            result = await self.execute_query(query, user_id)
            
            if result["status"] == "error":
                raise Exception(f"Error executing query: {result['message']}")
            
            # Convert results to DataFrame
            df = pd.DataFrame(result["results"])
            
            # Calculate completeness
            completeness = 1.0 - df.isnull().sum().sum() / (
                df.shape[0] * df.shape[1]
            )
            
            # Calculate accuracy (based on data type validation and value ranges)
            accuracy = 0.0
            valid_columns = 0
            
            for column in df.columns:
                col_type = schema.get(column, {}).get("type", "")
                if col_type:
                    valid_columns += 1
                    # Basic type validation
                    if (col_type.startswith("int") and 
                            df[column].dtype in ["int64", "int32"]):
                        accuracy += 1.0
                    elif (col_type.startswith("float") and 
                            df[column].dtype in ["float64", "float32"]):
                        accuracy += 1.0
                    elif col_type == "string" and df[column].dtype == "object":
                        accuracy += 1.0
                    elif col_type == "boolean" and df[column].dtype == "bool":
                        accuracy += 1.0
            
            accuracy = accuracy / valid_columns if valid_columns > 0 else 0.0
            
            # Calculate consistency (based on value patterns and constraints)
            consistency = 0.0
            if valid_columns > 0:
                # Check for consistent data patterns
                pattern_checks = 0
                for column in df.columns:
                    if df[column].dtype == "object":
                        # Check for consistent string patterns
                        unique_patterns = df[column].str.len().nunique()
                        if unique_patterns <= 3:  # Allow some variation
                            pattern_checks += 1
                    elif df[column].dtype in ["int64", "float64"]:
                        # Check for consistent numeric ranges
                        if (df[column].std() / df[column].mean() < 0.5):  
                            # Low coefficient of variation
                            pattern_checks += 1
                consistency = pattern_checks / valid_columns
            
            # Calculate timeliness (based on data freshness)
            timeliness = 1.0  # Default to 100% if we can't determine
            
            # Create metrics response
            metrics = {
                "completeness": {
                    "score": float(completeness),
                    "status": "success" if completeness >= 0.95 else "warning" if completeness >= 0.85 else "error",
                    "details": {
                        "total": int(df.shape[0] * df.shape[1]),
                        "valid": int((df.shape[0] * df.shape[1]) - df.isnull().sum().sum()),
                        "invalid": 0,
                        "missing": int(df.isnull().sum().sum())
                    }
                },
                "accuracy": {
                    "score": float(accuracy),
                    "status": "success" if accuracy >= 0.90 else "warning" if accuracy >= 0.80 else "error",
                    "details": {
                        "total": valid_columns,
                        "valid": int(accuracy * valid_columns),
                        "invalid": valid_columns - int(accuracy * valid_columns),
                        "missing": 0
                    }
                },
                "consistency": {
                    "score": float(consistency),
                    "status": "success" if consistency >= 0.85 else "warning" if consistency >= 0.75 else "error",
                    "details": {
                        "total": valid_columns,
                        "valid": int(consistency * valid_columns),
                        "invalid": valid_columns - int(consistency * valid_columns),
                        "missing": 0
                    }
                },
                "timeliness": {
                    "score": float(timeliness),
                    "status": "success",
                    "details": {
                        "total": 1,
                        "valid": 1,
                        "invalid": 0,
                        "missing": 0
                    }
                },
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Store metrics in S3 for future use
            try:
                self.s3_client.put_object(
                    Bucket=self.bucket,
                    Key=quality_key,
                    Body=json.dumps(metrics)
                )
                logger.info(f"Successfully stored quality metrics for {table_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to store quality metrics in S3: {str(e)}"
                )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting table quality metrics: {str(e)}")
            raise Exception(f"Error getting table quality metrics: {str(e)}")

    async def list_files(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> List[Dict[str, Any]]:
        """List all files in the catalog.
        
        Args:
            user_id: The ID of the user whose files to list
            
        Returns:
            List of file information dictionaries
        """
        try:
            # List all schema files in S3 for this user
            response = self.s3_service.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"metadata/schema/{user_id}/"  # Use user-specific prefix
            )

            files = []
            for obj in response.get('Contents', []):
                try:
                    # Get schema file content
                    schema_response = self.s3_service.get_object(
                        Bucket=self.bucket,
                        Key=obj['Key']
                    )
                    schema_data = json.loads(schema_response['Body'].read())
                    
                    # Create file info
                    file_info = {
                        "file_name": schema_data['file_name'],
                        "table_name": schema_data['table_name'],
                        "user_id": user_id,  # Add user_id to file info
                        "schema": schema_data['columns'],
                        "location": f"s3://{self.bucket}/data/{user_id}/{schema_data['table_name']}",  # Update path
                        "created_at": schema_data['created_at'],
                        "updated_at": schema_data['updated_at']
                    }
                    files.append(file_info)
                except Exception as e:
                    logger.error(f"Error processing schema file {obj['Key']}: {str(e)}")
                    continue
            
            return files
            
        except Exception as e:
            logger.error(f"Error listing files: {str(e)}")
            raise Exception(f"Error listing files: {str(e)}")

    async def get_catalog(
        self,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, Any]:
        """Get the complete catalog for a user.
        
        Args:
            user_id: The ID of the user whose catalog to get
            
        Returns:
            Dict containing:
            - tables: List of tables
            - files: List of files
            - quality_metrics: Overall quality metrics
        """
        try:
            # Get tables
            tables = await self.list_tables(user_id=user_id)
            
            # Get files
            files = await self.list_files(user_id=user_id)
            
            # Get quality metrics
            quality_metrics = await self.get_quality_metrics(user_id=user_id)
            
            return {
                "tables": tables,
                "files": files,
                "quality_metrics": quality_metrics,
                "user_id": user_id  # Add user_id to catalog info
            }
            
        except Exception as e:
            logger.error(f"Error getting catalog: {str(e)}")
            raise Exception(f"Error getting catalog: {str(e)}")

    async def list_tables(self, user_id: str) -> List[TableInfo]:
        """List all tables for a user."""
        try:
            # Get tables from Glue first
            database_name = self.get_user_database_name(user_id)
            tables_response = self.glue_service.glue_client.get_tables(
                DatabaseName=database_name
            )
            
            tables = []
            for table in tables_response.get('TableList', []):
                try:
                    # Transform schema columns into the expected format
                    schema_columns = []
                    for col in table.get('StorageDescriptor', {}).get('Columns', []):
                        schema_columns.append({
                            'name': col['Name'],
                            'type': col['Type']
                        })
                    
                    # Create base table info from Glue
                    table_info = TableInfo(
                        name=table['Name'],
                        description=table.get('Description'),
                        schema=schema_columns,
                        lastUpdated=table.get('LastUpdatedTime'),
                        createdBy=user_id,
                        createdAt=table.get('CreateTime'),
                        location=table.get('StorageDescriptor', {}).get('Location')
                    )
                    
                    # Try to get additional metadata from S3 if available
                    try:
                        metadata_key = (
                            f"{self.tables_prefix}{user_id}/{table['Name']}.json"
                        )
                        metadata_response = self.s3_client.get_object(
                            Bucket=self.bucket,
                            Key=metadata_key
                        )
                        metadata = json.loads(
                            metadata_response['Body'].read().decode('utf-8')
                        )
                        
                        # Update table info with metadata if available
                        if metadata.get('description'):
                            table_info.description = metadata['description']
                        if metadata.get('rowCount'):
                            table_info.rowCount = metadata['rowCount']
                        if metadata.get('etag'):
                            table_info.etag = metadata['etag']
                    except Exception as e:
                        logger.warning(
                            f"Could not get additional metadata for table "
                            f"{table['Name']}: {str(e)}"
                        )
                    
                    tables.append(table_info)
                except Exception as e:
                    logger.error(
                        f"Error processing table "
                        f"{table.get('Name', 'unknown')}: {str(e)}"
                    )
                    continue
            
            return sorted(
                tables,
                key=lambda x: x.lastUpdated or datetime.min,
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise

    def _standardize_column_name(self, column_name: str) -> str:
        """Convert column name to snake_case following database naming conventions.
        
        Rules:
        1. Convert to lowercase
        2. Replace spaces and special characters with underscores
        3. Remove leading/trailing underscores
        4. Replace multiple consecutive underscores with a single one
        5. Remove any non-alphanumeric characters except underscores
        """
        # Convert to lowercase
        name = column_name.lower()
        
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-z0-9]+', '_', name)
        
        # Remove leading/trailing underscores
        name = name.strip('_')
        
        # Replace multiple consecutive underscores with a single one
        name = re.sub(r'_+', '_', name)
        
        return name

    async def _create_consistent_schema(
        self,
        df: pd.DataFrame,
        table_name: str,
        user_id: str
    ) -> Dict[str, Any]:
        """Create a consistent schema for both Parquet and Glue tables.
        
        Args:
            df: Input DataFrame
            table_name: Name of the table
            user_id: User ID
            
        Returns:
            Dict containing the standardized schema
        """
        # Create standardized column names
        column_mappings = {
            col: self._standardize_column_name(col)
            for col in df.columns
        }
        
        # Create schema with standardized names
        schema = {
            'columns': [
                {
                    'name': column_mappings[col],
                    'type': self._pandas_to_athena_type(str(df[col].dtype)),
                    'description': f'Original column name: {col}'
                }
                for col in df.columns
            ]
        }
        
        # Rename DataFrame columns to match schema
        df = df.rename(columns=column_mappings)
        
        return {
            'schema': schema,
            'dataframe': df,
            'column_mappings': column_mappings
        }

    async def _process_dataframe_to_parquet_and_glue(
        self,
        df: pd.DataFrame,
        parquet_s3_key: str,
        table_name: str,
        user_id: str,
        file_format: str = "parquet"
    ) -> Dict[str, Any]:
        """Process DataFrame to Parquet and create/update Glue table with consistent schema."""
        try:
            # Create consistent schema
            schema_result = await self._create_consistent_schema(df, table_name, user_id)
            df = schema_result['dataframe']
            schema = schema_result['schema']
            column_mappings = schema_result['column_mappings']
            
            # Convert to Parquet
            parquet_buffer = BytesIO()
            df.to_parquet(parquet_buffer, index=False, compression='snappy')
            parquet_buffer.seek(0)
            
            # Upload to S3 using S3Service
            await self.s3_service.upload_file(parquet_buffer.getvalue(), parquet_s3_key)
            
            # Create/update Glue table using GlueService
            database_name = self.get_user_database_name(user_id)
            
            # Ensure the S3 location is properly formatted
            # Remove any leading/trailing slashes and ensure it ends with a slash
            folder_path = os.path.dirname(parquet_s3_key.strip('/'))
            if not folder_path.endswith('/'):
                folder_path += '/'
            
            # Create the full S3 location
            s3_location = f"s3://{self.bucket}/{folder_path}"
            
            # Create or update table
            table_result = await self.glue_service.create_or_update_table(
                GlueTableConfig(
                    database_name=database_name,
                    table_name=table_name,
                    schema=schema,
                    location=s3_location,
                    file_format=file_format,
                    description=f"Table created at {datetime.utcnow().isoformat()}"
                )
            )
            
            return {
                'status': 'success',
                'message': f'Successfully processed {file_format} file to Parquet and created/updated Glue table',
                'schema': schema,
                'column_mappings': column_mappings,
                's3_location': s3_location
            }
            
        except Exception as e:
            logger.error(f"Error processing DataFrame to Parquet and Glue: {str(e)}")
            raise

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename by converting to snake case and removing special characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename in snake case
        """
        # Remove file extension
        name, ext = os.path.splitext(filename)
        
        # Convert to snake case
        # Replace spaces and special characters with underscores
        name = re.sub(r'[^a-zA-Z0-9]+', '_', name)
        # Remove leading/trailing underscores
        name = name.strip('_')
        # Convert to lowercase
        name = name.lower()
        
        # Reattach extension
        return f"{name}{ext.lower()}"

    async def upload_file(
        self,
        file: Any,  # FastAPI UploadFile
        table_name: str,
        create_new: bool = False,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Upload a file and create/update a table."""
        try:
            # Process file content
            content = await file.read()
            file_name = file.filename
            file_format = file_name.split('.')[-1].lower()
            
            # Process file and create/update table
            result = await self._process_file_content_to_catalog(
                content,
                file_name,
                file_format,
                table_name,
                user_id
            )
            
            if result['status'] == 'success':
                # Sync the table to DuckDB cache
                await self.sync_glue_table_to_duckdb(table_name, user_id)
            
            return result
            
        except Exception as e:
            logger.error(f"Error uploading file: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def _process_file_content_to_catalog(
        self,
        content: bytes,
        file_name: str,
        file_format: str,
        table_name: str,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Process file content and add to catalog."""
        try:
            # Create a temporary file to store the content
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_format}") as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Read the file into a DataFrame
                if file_format == 'csv':
                    df = pd.read_csv(temp_file_path)
                elif file_format == 'json':
                    df = pd.read_json(temp_file_path)
                elif file_format == 'parquet':
                    df = pd.read_parquet(temp_file_path)
                elif file_format in ['xlsx', 'xls']:
                    df = pd.read_excel(temp_file_path)
                else:
                    raise ValueError(f"Unsupported file format: {file_format}")
                
                # Generate S3 keys with sanitized filenames
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                original_s3_key = f"originals/{user_id}/{file_name}"
                parquet_s3_key = f"data/{user_id}/{table_name}/{self._sanitize_filename(file_name).replace(f'.{file_format}', '')}_{timestamp}.parquet"
                
                # Upload original file
                await self.s3_service.upload_file(
                    file_content=content,
                    key=original_s3_key
                )
                
                # Process DataFrame to Parquet and create Glue table
                result = await self._process_dataframe_to_parquet_and_glue(
                    df=df,
                    parquet_s3_key=parquet_s3_key,
                    table_name=table_name,
                    user_id=user_id,
                    file_format=file_format
                )
                
                return {
                    "status": "success",
                    "message": "File uploaded and processed successfully",
                    "original_file": original_s3_key,
                    "parquet_file": parquet_s3_key,
                    "table_name": table_name
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing file content: {str(e)}")
            raise ValueError(f"Error processing file content: {str(e)}")

    async def process_s3_file(
        self,
        original_s3_uri: str,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, Any]:
        """
        Process an existing file from S3: convert to Parquet, catalog, create Glue table.
        original_s3_uri: The S3 URI of the original file (e.g., s3://bucket-name/path/to/original.xlsx)
        table_name: The target Glue table name.
        user_id: The ID of the user who owns the table.
        """
        logger.info(
            f"Starting processing of existing S3 file '{original_s3_uri}' for table '{table_name}' for user '{user_id}'"
        )
        try:
            if not original_s3_uri.startswith("s3://"):
                raise ValueError("original_s3_uri must be a valid S3 URI (e.g., s3://bucket/key)")

            # Parse S3 URI
            path_parts = original_s3_uri.replace("s3://", "").split("/", 1)
            if len(path_parts) < 2: # Ensure there's a key part
                 raise ValueError(f"Invalid S3 URI: {original_s3_uri}. Must include bucket and key.")
            
            source_bucket = path_parts[0]
            source_key = path_parts[1]
            
            # For now, assume the source bucket is the same as self.bucket.
            # Could be enhanced to handle cross-bucket if needed, with proper permissions.
            if source_bucket != self.bucket:
                logger.warning(
                    f"Source S3 bucket '{source_bucket}' is different from service bucket '{self.bucket}'. "
                    f"Ensure cross-bucket permissions are configured if this is intended. Proceeding with configured service bucket.")
                # Forcing to use configured bucket, this might need adjustment based on actual requirements
                # For simplicity in this step, we'll assume the file processing and output always uses self.bucket
                # If the intention is to read from a different bucket and write to self.bucket, that's fine.

            file_name = os.path.basename(source_key)
            file_format = file_name.split('.')[-1].lower()

            logger.info(f"Downloading original file from s3://{source_bucket}/{source_key}")
            response = self.s3_service.get_object(Bucket=source_bucket, Key=source_key)
            content = response['Body'].read()
            logger.info(f"Successfully downloaded {len(content)} bytes from s3://{source_bucket}/{source_key}")

            # Delegate to the common processing logic
            processing_result = await self._process_file_content_to_catalog(
                content=content,
                file_name=file_name, # This is the name of the file from the S3 key
                file_format=file_format,
                table_name=table_name,
                user_id=user_id  # Pass user_id to _process_file_content_to_catalog
            )
            
            # Enhance the file_info with the original S3 path from this processing flow
            if processing_result.get("status") == "success" and "file_info" in processing_result:
                processing_result["file_info"]["s3_path_original_source"] = original_s3_uri

            return processing_result

        except ValueError as ve:
            logger.error(
                f"ValueError during S3 file processing for URI {original_s3_uri}, table {table_name}: {str(ve)}"
            )
            raise
        except self.s3_service.exceptions.NoSuchKey:
            logger.error(f"Original S3 file not found at {original_s3_uri}")
            raise Exception(f"Original S3 file not found at {original_s3_uri}")
        except Exception as e:
            logger.error(
                f"Generic error processing S3 file {original_s3_uri} for table {table_name}: {str(e)}", exc_info=True
            )
            raise Exception(f"Error processing S3 file {original_s3_uri}: {str(e)}")

    async def get_schema(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> Dict[str, Any]:
        """Get schema for a table.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table schema
        """
        try:
            # Get table from Glue
            response = self.glue_service.get_table(
                DatabaseName=f"user_{user_id}",  # Use user-specific database
                Name=table_name
            )
            
            # Convert Glue schema to our format
            schema = {
                "table_name": table_name,
                "user_id": user_id,  # Add user_id to schema
                "columns": [
                    {
                        "name": col["Name"],
                        "type": col["Type"],
                        "description": col.get("Comment", "")
                    }
                    for col in response["Table"]["StorageDescriptor"]["Columns"]
                ],
                "location": response["Table"]["StorageDescriptor"]["Location"],
                "created_at": response["Table"].get("CreateTime", "").isoformat(),
                "updated_at": response["Table"].get("UpdateTime", "").isoformat()
            }
            
            return schema
            
        except Exception as e:
            logger.error(f"Error getting schema for table {table_name}: {str(e)}")
            raise Exception(f"Error getting schema for table {table_name}: {str(e)}")

    async def update_schema(
        self,
        table_name: str,
        new_schema: Dict[str, Any],
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Update schema for a file, including updating the Glue table."""
        logger.info(f"Updating schema for table {table_name} for user {user_id}")
        try:
            # Get current table info
            table_info = await self.glue_service.get_table(user_id, table_name)
            if not table_info:
                raise ValueError(f"Table {table_name} not found")

            # Get current data
            current_data = await self.get_table_data(table_name, user_id)
            if not current_data or "data" not in current_data:
                raise ValueError(f"No data found for table {table_name}")

            # Convert to DataFrame
            df = pd.DataFrame(current_data["data"])

            # Apply new schema transformations
            for field in new_schema.get("fields", []):
                col_name = field["name"]
                if col_name in df.columns:
                    # Apply type conversion if needed
                    if "type" in field:
                        try:
                            df[col_name] = df[col_name].astype(field["type"])
                        except Exception as e:
                            logger.warning(
                                f"Could not convert column {col_name} to type "
                                f"{field['type']}: {str(e)}"
                            )

            # Generate new S3 key with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            parquet_s3_key = f"data/{user_id}/{table_name}/{table_name}_{timestamp}.parquet"

            # Process DataFrame to Parquet and create Glue table
            result = await self._process_dataframe_to_parquet_and_glue(
                df=df,
                parquet_s3_key=parquet_s3_key,
                table_name=table_name,
                user_id=user_id
            )

            # Sync updated table to DuckDB
            await self.sync_glue_table_to_duckdb(table_name, user_id)

            return {
                "status": "success",
                "message": f"Schema updated successfully for table {table_name}",
                "table_info": result.get("glue_table_details", {}).get("table_info", {}),
                "new_schema": new_schema
            }
        except Exception as e:
            logger.error(f"Error updating schema for table {table_name}: {str(e)}", exc_info=True)
            raise Exception(f"Error updating schema: {str(e)}")

    def _pandas_to_athena_type(self, dtype_str: str) -> str:
        """Convert pandas dtype string to Athena/Glue compatible type string."""
        # Handle pandas extension dtypes (nullable types)
        # Examples: 'Int64', 'Float64', 'boolean' (pandas nullable types)
        dtype_str_lower = dtype_str.lower()
        if dtype_str_lower in ("int64", "int32", "int16", "int8", "int"):  # numpy dtypes
            return "bigint"
        if dtype_str_lower in ("float64", "float32", "float"):  # numpy dtypes
            return "double"
        if dtype_str_lower in ("bool", "boolean"):  # numpy and pandas
            return "boolean"
        if dtype_str_lower in ("string", "object", "category"):
            return "string"
        if dtype_str_lower.startswith("int") or dtype_str_lower == "int64":
            return "bigint"
        if dtype_str_lower.startswith("float"):
            return "double"
        if dtype_str_lower.startswith("datetime64") or "datetime" in dtype_str_lower:
            return "timestamp"
        # Pandas extension dtypes
        if dtype_str_lower == "int64" or dtype_str_lower == "int32":
            return "bigint"
        if dtype_str_lower == "float64" or dtype_str_lower == "float32":
            return "double"
        if dtype_str_lower == "boolean":
            return "boolean"
        # Pandas nullable extension dtypes
        if dtype_str_lower == "int64[pyarrow]" or dtype_str_lower == "int64[nullable]" or dtype_str_lower == "int64[python]":
            return "bigint"
        if dtype_str_lower == "float64[pyarrow]" or dtype_str_lower == "float64[nullable]":
            return "double"
        # Fallback for unhandled types
        logger.warning(f"Unhandled pandas type: {dtype_str}, defaulting to string for Athena.")
        return "string"

    def _pandas_to_pyarrow_type(self, dtype_str: str) -> pyarrow.DataType:
        """Convert pandas dtype string to PyArrow DataType."""
        # logger.info(f"[_pandas_to_pyarrow_type] Input pandas_type_str: '{dtype_str}'")
        if "bool" in dtype_str:
            pa_type = pyarrow.bool_()
        elif "int64" in dtype_str:
            pa_type = pyarrow.int64()
        elif "int32" in dtype_str:
            pa_type = pyarrow.int32()
        elif "int16" in dtype_str:
            pa_type = pyarrow.int16()
        elif "int8" in dtype_str:
            pa_type = pyarrow.int8()
        elif "float64" in dtype_str:
            pa_type = pyarrow.float64()
        elif "float32" in dtype_str:
            pa_type = pyarrow.float32()
        elif "datetime64" in dtype_str: # Pandas datetime
            pa_type = pyarrow.timestamp('ns') # nanosecond precision
        elif "object" in dtype_str: # Pandas string/mixed
            pa_type = pyarrow.string()
        elif "category" in dtype_str:
            pa_type = pyarrow.string() # Treat categories as strings
        else: # Fallback for unhandled types
            logger.warning(f"Unhandled pandas type for PyArrow: {dtype_str}, defaulting to string.")
            pa_type = pyarrow.string()
        # logger.info(f"[_pandas_to_pyarrow_type] Output PyArrow type: '{pa_type}' for input '{dtype_str}'")
        return pa_type 

    async def get_file(
        self,
        s3_path: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get file information from S3.
        
        Args:
            s3_path: S3 path to the file
            user_id: The ID of the user who owns the file
            
        Returns:
            Dict containing file information
        """
        try:
            # Parse S3 path
            if not s3_path.startswith("s3://"):
                raise ValueError("s3_path must be a valid S3 URI (e.g., s3://bucket/key)")
            
            path_parts = s3_path.replace("s3://", "").split("/", 1)
            if len(path_parts) < 2:
                raise ValueError(f"Invalid S3 URI: {s3_path}. Must include bucket and key.")
            
            bucket = path_parts[0]
            key = path_parts[1]
            
            # Get file metadata
            response = self.s3_service.head_object(Bucket=bucket, Key=key)
            
            # Get schema if available
            schema = None
            try:
                schema_key = (
                    f"metadata/schema/{user_id}/{os.path.basename(key)}.json"
                )  # Update path
                schema_response = self.s3_service.get_object(Bucket=bucket, Key=schema_key)
                schema = json.loads(schema_response['Body'].read())
            except self.s3_service.exceptions.NoSuchKey:
                pass
            
            return {
                "name": os.path.basename(key),
                "size": response['ContentLength'],
                "last_modified": response['LastModified'].isoformat(),
                "format": os.path.splitext(key)[1][1:].lower(),
                "location": s3_path,
                "user_id": user_id,  # Add user_id to file info
                "schema": schema
            }
            
        except Exception as e:
            logger.error(f"Error getting file info: {str(e)}")
            raise Exception(f"Error getting file info: {str(e)}")

    async def get_table_schema(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get schema for a table from Glue catalog.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table schema
        """
        try:
            # Get table from Glue
            database_name = f"user_{user_id}"  # Use user-specific database
            
            # First, check if the table exists
            try:
                tables = await self.glue_service.list_tables(database_name)
                table_names = [table['Name'] for table in tables]
                if table_name not in table_names:
                    raise Exception(
                        f"Table {table_name} not found in database {database_name}. "
                        f"Available tables: {', '.join(table_names)}"
                    )
            except Exception as e:
                logger.error(f"Error checking table existence: {str(e)}")
                raise Exception(f"Error checking table existence: {str(e)}")
            
            # Get table details
            response = await self.glue_service.get_table(
                database_name=database_name,
                table_name=table_name
            )
            
            # Extract schema
            schema = []
            for column in response['StorageDescriptor']['Columns']:
                schema.append({
                    "name": column['Name'],
                    "type": column['Type'],
                    "comment": column.get('Comment', '')
                })
            
            return {
                "table_name": table_name,
                "database": database_name,
                "schema": schema,
                "user_id": user_id  # Add user_id to schema info
            }
            
        except self.glue_service.glue_client.exceptions.EntityNotFoundException:
            raise Exception(f"Table {table_name} not found in database {database_name}")
        except Exception as e:
            logger.error(f"Error getting table schema: {str(e)}")
            raise Exception(f"Error getting table schema: {str(e)}")

    async def get_table_data(
        self,
        table_name: str,
        limit: int = 1000,
        user_id: str = "test_user"
    ) -> dict:
        """Get table data, trying DuckDB cache first."""
        try:
            # Try to get from DuckDB cache first
            cached_data = await self.get_cached_table_data(table_name)
            if cached_data:
                return {
                    'status': 'success',
                    'data': cached_data['data'],
                    'source': 'duckdb_cache',
                    'last_updated': cached_data['last_updated']
                }
            
            # If not in cache, get from S3 and cache it
            result = await self.athena_service.execute_query(
                f"SELECT * FROM {table_name} LIMIT {limit}",
                database=self.get_user_database_name(user_id)
            )
            
            if result['status'] == 'success':
                # Cache the data without triggering a full sync
                self.conn.execute("""
                    INSERT INTO table_data (
                        table_name, data, last_updated
                    ) VALUES (?, ?, ?)
                    ON CONFLICT (table_name) DO UPDATE SET
                        data = excluded.data,
                        last_updated = excluded.last_updated
                """, (
                    table_name,
                    json.dumps(result['results']),
                    datetime.now()
                ))
                return {
                    'status': 'success',
                    'data': result['results'],
                    'source': 's3',
                    'last_updated': datetime.now()
                }
            else:
                return result
                
        except Exception as e:
            logger.error(f"Error getting table data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def get_table_metadata(
        self,
        table_name: str,
        user_id: str = "test_user"  # Add user_id parameter with default
    ) -> dict:
        """Get metadata for a table from Glue catalog.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            Dict containing table metadata
        """
        try:
            # Get table from Glue
            database_name = f"user_{user_id}"  # Use user-specific database
            response = self.glue_service.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            
            # Extract metadata
            table = response['Table']
            metadata = {
                "name": table_name,
                "database": database_name,
                "location": table['StorageDescriptor']['Location'],
                "input_format": table['StorageDescriptor'].get('InputFormat', ''),
                "output_format": table['StorageDescriptor'].get('OutputFormat', ''),
                "compressed": table['StorageDescriptor'].get('Compressed', False),
                "number_of_buckets": table['StorageDescriptor'].get('NumberOfBuckets', -1),
                "ser_de_info": table['StorageDescriptor'].get('SerdeInfo', {}),
                "bucket_columns": table['StorageDescriptor'].get('BucketColumns', []),
                "sort_columns": table['StorageDescriptor'].get('SortColumns', []),
                "parameters": table.get('Parameters', {}),
                "table_type": table.get('TableType', ''),
                "created_at": table.get('CreateTime', '').isoformat(),
                "last_updated": table.get('UpdateTime', '').isoformat(),
                "user_id": user_id  # Add user_id to metadata
            }
            
            return metadata
            
        except self.glue_service.glue_client.exceptions.EntityNotFoundException:
            raise Exception(f"Table {table_name} not found in database {database_name}")
        except Exception as e:
            logger.error(f"Error getting table metadata: {str(e)}")
            raise Exception(f"Error getting table metadata: {str(e)}")

    def get_user_database_name(self, user_id: str) -> str:
        """Get the user-specific database name.
        
        Args:
            user_id: The user ID to get the database name for
            
        Returns:
            The user-specific database name
        """
        return f"user_{user_id}"

    async def ensure_user_database_exists(self, user_id: str) -> None:
        """Ensure that the user-specific database exists.
        
        Args:
            user_id: The user ID to ensure the database exists for
        """
        database_name = self.get_user_database_name(user_id)
        try:
            self.glue_service.get_database(Name=database_name)
        except self.glue_service.exceptions.EntityNotFoundException:
            self.glue_service.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': f'Database for user {user_id}'
                }
            )
            logger.info(f"Created database {database_name} for user {user_id}")

    async def get_available_transformations(self) -> List[Dict[str, Any]]:
        """Get list of available transformation types and their configurations."""
        return [
            {
                "type": "sentiment_analysis",
                "name": "Sentiment Analysis",
                "description": "Analyze sentiment of text columns",
                "input_types": ["string"],
                "output_type": "struct<sentiment:string,score:double>",
                "parameters": {
                    "aspects": "List of aspects to analyze",
                    "language": "Language of the text"
                }
            },
            {
                "type": "categorization",
                "name": "Text Categorization",
                "description": "Categorize text into predefined categories",
                "input_types": ["string"],
                "output_type": "struct<category:string,confidence:double>",
                "parameters": {
                    "categories": "List of possible categories",
                    "confidence_threshold": "Minimum confidence score"
                }
            },
            {
                "type": "entity_extraction",
                "name": "Entity Extraction",
                "description": "Extract named entities from text",
                "input_types": ["string"],
                "output_type": "array<struct<entity:string,type:string>>",
                "parameters": {
                    "entity_types": "Types of entities to extract"
                }
            },
            {
                "type": "custom_llm",
                "name": "Custom LLM Transformation",
                "description": "Apply custom LLM-based transformation",
                "input_types": ["any"],
                "output_type": "any",
                "parameters": {
                    "prompt_template": "LLM prompt template",
                    "output_format": "Expected output format"
                }
            }
        ]

    async def get_transformation_templates(self, user_id: str) -> List[TransformationTemplate]:
        """Get list of saved transformation templates."""
        try:
            templates_key = f"metadata/transformations/{user_id}/templates.json"
            response = self.s3_service.get_object(
                Key=templates_key
            )
            templates = json.loads(response['Body'].read())
            return [TransformationTemplate(**t) for t in templates]
        except self.s3_service.exceptions.NoSuchKey:
            return []

    async def save_transformation_template(
        self,
        template: TransformationTemplate,
        user_id: str
    ) -> Dict[str, Any]:
        """Save a new transformation template."""
        templates = await self.get_transformation_templates(user_id)
        templates.append(template)
        
        templates_key = f"metadata/transformations/{user_id}/templates.json"
        self.s3_client.put_object(
            Key=templates_key,
            Body=json.dumps([t.dict() for t in templates])
        )
        return {"status": "success", "message": "Template saved successfully"}

    async def apply_transformation(
        self,
        table_name: str,
        config: TransformationConfig,
        user_id: str
    ) -> TransformationResult:
        """Apply a transformation to a table."""
        try:
            # 1. Get table schema and data
            schema = await self.get_table_schema(table_name, user_id)
            query = f"SELECT * FROM {table_name} LIMIT 1000"  # Get sample for preview
            query_result = await self.execute_query(query, user_id)
            
            if query_result["status"] != "success":
                return TransformationResult(
                    status="error",
                    message="Failed to fetch table data",
                    new_columns=[],
                    errors=[query_result.get("message", "Unknown error")]
                )

            # 2. Generate new column name if not provided
            if not config.new_column_name:
                config.new_column_name = self._generate_column_name(
                    config.source_columns[0],
                    config.transformation_type
                )

            # 3. Apply transformation based on type
            if config.transformation_type == "sentiment_analysis":
                result = await self._apply_sentiment_analysis(
                    query_result["results"],
                    config
                )
            elif config.transformation_type == "categorization":
                result = await self._apply_categorization(
                    query_result["results"],
                    config
                )
            elif config.transformation_type == "custom_llm":
                result = await self._apply_custom_llm(
                    query_result["results"],
                    config
                )
            else:
                return TransformationResult(
                    status="error",
                    message=f"Unsupported transformation type: {config.transformation_type}",
                    new_columns=[],
                    errors=[f"Unsupported transformation type: {config.transformation_type}"]
                )

            # 4. Update table schema
            await self.update_table_schema(
                table_name=table_name,
                schema={"columns": result.new_columns},
                user_id=user_id
            )

            # 5. Update Parquet file with transformed data
            parquet_location = (await self.get_file(table_name, user_id))["location"]
            transformed_df = pd.DataFrame(result.preview_data)
            transformed_df.to_parquet(parquet_location)

            return result

        except Exception as e:
            return TransformationResult(
                status="error",
                message=str(e),
                new_columns=[],
                errors=[str(e)]
            )

    def _generate_column_name(self, source_column: str, transformation_type: str) -> str:
        """Generate a new column name based on source column and transformation type."""
        prefix_map = {
            "sentiment_analysis": "sentiment",
            "categorization": "category",
            "entity_extraction": "entities",
            "custom_llm": "transformed"
        }
        prefix = prefix_map.get(transformation_type, "transformed")
        return f"{prefix}_{source_column}"

    async def _apply_sentiment_analysis(
        self,
        data: List[List[Any]],
        config: TransformationConfig
    ) -> TransformationResult:
        """Apply sentiment analysis transformation."""
        try:
            # Convert data to list of dicts
            columns = data[0]
            records = []
            for row in data[1:]:
                records.append(dict(zip(columns, row)))

            # Prepare metadata for sentiment analysis
            metadata = {
                "analysis_type": "sentiment_analysis",
                "aspects": config.parameters.get("aspects", []),
                "language": config.parameters.get("language", "en")
            }

            # Apply transformation using agent
            result = await self.transformation_agent.apply_transformation(
                data=records,
                transformation_type="sentiment",
                metadata=metadata,
                config=config
            )

            # Convert result to TransformationResult
            new_columns = [{
                "name": config.new_column_name,
                "type": "struct<sentiment:string,score:double,aspect_sentiments:map<string,struct<sentiment:string,score:double>>>"
            }]

            # Convert transformed data to list format
            preview_data = []
            for item in result.transformed_data:
                preview_data.append({
                    **item["original"],
                    config.new_column_name: item["transformed"]
                })

            return TransformationResult(
                status="success",
                message="Sentiment analysis completed successfully",
                new_columns=new_columns,
                preview_data=preview_data,
                errors=result.errors
            )

        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}", exc_info=True)
            return TransformationResult(
                status="error",
                message=str(e),
                new_columns=[],
                errors=[str(e)]
            )

    async def _apply_categorization(
        self,
        data: List[List[Any]],
        config: TransformationConfig
    ) -> TransformationResult:
        """Apply categorization transformation."""
        try:
            # Convert data to list of dicts
            columns = data[0]
            records = []
            for row in data[1:]:
                records.append(dict(zip(columns, row)))

            # Prepare metadata for categorization
            metadata = {
                "analysis_type": "categorization",
                "categories": config.parameters.get("categories", []),
                "confidence_threshold": config.parameters.get("confidence_threshold", 0.7)
            }

            # Apply transformation using agent
            result = await self.transformation_agent.apply_transformation(
                data=records,
                transformation_type="categorization",
                metadata=metadata,
                config=config
            )

            # Convert result to TransformationResult
            new_columns = [{
                "name": config.new_column_name,
                "type": "struct<category:string,confidence:double,details:struct<is_ai_company:boolean,reasoning:string,ai_technologies:array<string>,ai_focus_areas:array<string>>>"
            }]

            # Convert transformed data to list format
            preview_data = []
            for item in result.transformed_data:
                preview_data.append({
                    **item["original"],
                    config.new_column_name: item["transformed"]
                })

            return TransformationResult(
                status="success",
                message="Categorization completed successfully",
                new_columns=new_columns,
                preview_data=preview_data,
                errors=result.errors
            )

        except Exception as e:
            logger.error(f"Error in categorization: {str(e)}", exc_info=True)
            return TransformationResult(
                status="error",
                message=str(e),
                new_columns=[],
                errors=[str(e)]
            )

    async def _apply_custom_llm(
        self,
        data: List[List[Any]],
        config: TransformationConfig
    ) -> TransformationResult:
        """Apply custom LLM transformation."""
        try:
            # Convert data to list of dicts
            columns = data[0]
            records = []
            for row in data[1:]:
                records.append(dict(zip(columns, row)))

            # Prepare metadata for custom LLM transformation
            metadata = {
                "analysis_type": "custom_llm",
                "prompt_template": config.parameters.get("prompt_template", ""),
                "output_format": config.parameters.get("output_format", "json")
            }

            # Apply transformation using agent
            result = await self.transformation_agent.apply_transformation(
                data=records,
                transformation_type="custom_llm",
                metadata=metadata,
                config=config
            )

            # Convert result to TransformationResult
            new_columns = [{
                "name": config.new_column_name,
                "type": config.data_type or "string"  # Use provided data type or default to string
            }]

            # Convert transformed data to list format
            preview_data = []
            for item in result.transformed_data:
                preview_data.append({
                    **item["original"],
                    config.new_column_name: item["transformed"]
                })

            return TransformationResult(
                status="success",
                message="Custom LLM transformation completed successfully",
                new_columns=new_columns,
                preview_data=preview_data,
                errors=result.errors
            )

        except Exception as e:
            logger.error(f"Error in custom LLM transformation: {str(e)}", exc_info=True)
            return TransformationResult(
                status="error",
                message=str(e),
                new_columns=[],
                errors=[str(e)]
            )

    async def update_table_schema(
        self,
        table_name: str,
        schema: Dict[str, Any],
        user_id: str
    ) -> None:
        """Update schema for a table."""
        try:
            # Update schema in Glue
            await self.glue_service.update_table(
                database_name=f"user_{user_id}",
                table_name=table_name,
                table_input={
                    'Name': table_name,
                    'TableType': 'EXTERNAL_TABLE',
                    'Parameters': {
                        'classification': 'parquet',
                        'typeOfData': 'file',
                        'user_id': user_id
                    },
                    'StorageDescriptor': {
                        'Columns': schema['columns'],
                        'Location': schema['location'],
                        'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                        'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                        'SerdeInfo': {
                            'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
                            'Parameters': {
                                'serialization.format': '1'
                            }
                        }
                    }
                }
            )
        except Exception as e:
            logger.error(f"Error updating table schema: {str(e)}")
            raise Exception(f"Error updating table schema: {str(e)}")

    async def get_table_files(
        self,
        table_name: str,
        user_id: str = "test_user"
    ) -> List[Dict[str, Any]]:
        """Get files associated with a table.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            
        Returns:
            List of file information dictionaries
        """
        try:
            # Get table metadata to get the S3 location
            table_info = await self.glue_service.get_table(
                database_name=f"user_{user_id}",
                table_name=table_name
            )
            
            if not table_info:
                raise Exception(f"Table {table_name} not found")
            
            # Get the S3 location from the table's storage descriptor
            s3_location = table_info.get('StorageDescriptor', {}).get('Location')
            if not s3_location:
                raise ValueError(f"No S3 location found for table {table_name}")
            
            logger.info(f"Found S3 location for table {table_name}: {s3_location}")
            
            # Parse the S3 location to get bucket and prefix
            if not s3_location.startswith('s3://'):
                raise ValueError(f"Invalid S3 location: {s3_location}")
            
            path_parts = s3_location.replace('s3://', '').split('/', 1)
            if len(path_parts) != 2:
                raise ValueError(f"Invalid S3 location format: {s3_location}")
            
            bucket = path_parts[0]
            prefix = path_parts[1]
            
            logger.info(f"Listing objects in bucket {bucket} with prefix {prefix}")
            
            # List objects in the S3 location
            response = self.s3_service.list_objects_v2(
                Bucket=bucket,
                Prefix=prefix
            )
            
            # Process the files
            files = []
            for obj in response.get('Contents', []):
                try:
                    # Get file metadata
                    file_info = {
                        "name": os.path.basename(obj['Key']),
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat(),
                        "format": os.path.splitext(obj['Key'])[1][1:].lower(),
                        "location": f"s3://{bucket}/{obj['Key']}",
                        "user_id": user_id
                    }
                    
                    # Try to get schema if available
                    try:
                        schema_key = f"metadata/schema/{user_id}/{os.path.basename(obj['Key'])}.json"
                        schema_response = self.s3_service.get_object(Bucket=bucket, Key=schema_key)
                        file_info['schema'] = json.loads(schema_response['Body'].read())
                    except self.s3_service.exceptions.NoSuchKey:
                        file_info['schema'] = None
                    
                    files.append(file_info)
                except Exception as e:
                    logger.error(f"Error processing file {obj['Key']}: {str(e)}")
                    continue
            
            logger.info(f"Found {len(files)} files for table {table_name}")
            return files
            
        except Exception as e:
            logger.error(f"Error getting table files: {str(e)}")
            raise Exception(f"Error getting table files: {str(e)}")

    async def get_table_preview(
        self,
        table_name: str,
        user_id: str = "test_user",
        rows: int = 5
    ) -> Dict[str, Any]:
        """Get a preview of table contents.
        
        Args:
            table_name: Name of the table
            user_id: The ID of the user who owns the table
            rows: Number of rows to return in preview
            
        Returns:
            Dictionary containing preview data
        """
        try:
            # Get table schema
            schema = await self.get_table_schema(table_name, user_id)
            if not schema:
                raise Exception(f"Table {table_name} not found")

            # Execute query to get preview data
            query = f"SELECT * FROM user_{user_id}.{table_name} LIMIT {rows}"
            query_result = await self.execute_query(query, user_id)
            
            if query_result["status"] != "success":
                raise Exception(f"Failed to fetch preview data: {query_result.get('message', 'Unknown error')}")

            # Get total row count
            count_query = f"SELECT COUNT(*) FROM user_{user_id}.{table_name}"
            count_result = await self.execute_query(count_query, user_id)
            total_rows = count_result["results"][0][0] if count_result["status"] == "success" else 0

            return {
                "columns": [col["name"] for col in schema["columns"]],
                "data": query_result["results"],
                "total_rows": total_rows
            }

        except Exception as e:
            logger.error(f"Error getting table preview: {str(e)}")
            raise Exception(f"Error getting table preview: {str(e)}")

    async def sync_all_tables_to_duckdb(self, user_id: str = "test_user"):
        """Synchronize all tables from S3 to DuckDB cache."""
        try:
            # Get all tables
            tables = await self.list_tables(user_id)
            
            # Sync each table sequentially since DuckDB operations are not async
            for table in tables:
                try:
                    # Get table schema and data from S3
                    schema = await self.get_table_schema(table.name, user_id)
                    data = await self.get_table_data(table.name, user_id)
                    
                    # Update schema cache
                    self.conn.execute("""
                        INSERT INTO table_cache (table_name, s3_path, last_updated, schema)
                        VALUES (?, ?, ?, ?)
                        ON CONFLICT (table_name) DO UPDATE SET
                            s3_path = excluded.s3_path,
                            last_updated = excluded.last_updated,
                            schema = excluded.schema
                    """, (
                        table.name,
                        f"s3://{self.bucket}/data/{user_id}/{table.name}",
                        datetime.now(),
                        json.dumps(schema)
                    ))
                    
                    # Update data cache
                    self.conn.execute("""
                        INSERT INTO table_data (table_name, data, last_updated)
                        VALUES (?, ?, ?)
                        ON CONFLICT (table_name) DO UPDATE SET
                            data = excluded.data,
                            last_updated = excluded.last_updated
                    """, (
                        table.name,
                        json.dumps(data.get('data', [])),
                        datetime.now()
                    ))
                except Exception as e:
                    logger.error(f"Error syncing table {table.name} to DuckDB: {str(e)}")
                    continue
            
            return {
                'status': 'success',
                'message': f'Synced {len(tables)} tables to DuckDB'
            }
        except Exception as e:
            logger.error(f"Error syncing all tables to DuckDB: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

    async def run_background_sync(self, sync_interval: int = 3600):
        """Start background task to periodically sync tables."""
        while True:
            try:
                logger.info("Starting periodic table DDL sync to DuckDB")
                
                # Get all tables from Glue
                tables = await self.list_tables("test_user")  # TODO: Handle multiple users
                
                # Sync only tables with DDL changes
                for table in tables:
                    try:
                        # Get current table DDL from Glue
                        table_info = await self.glue_service.get_table(
                            self.get_user_database_name("test_user"),
                            table.name
                        )
                        
                        if not table_info:
                            logger.warning(f"Table {table.name} not found in Glue, skipping")
                            continue
                        
                        # Generate current DDL version
                        current_ddl_version = self._generate_table_version(table_info)
                        
                        # Check if DDL has changed
                        cached_version = self.conn.execute(
                            "SELECT version FROM table_cache WHERE table_name = ?",
                            (table.name,)
                        ).fetchone()
                        
                        if not cached_version or cached_version[0] != current_ddl_version:
                            logger.info(f"DDL changed for table {table.name}, syncing...")
                            await self.sync_glue_table_to_duckdb(table.name, "test_user")
                        else:
                            logger.info(f"Table {table.name} DDL unchanged, skipping sync")
                            
                    except Exception as e:
                        logger.error(f"Error checking DDL for table {table.name}: {str(e)}")
                        continue
                
                logger.info("Completed periodic table DDL sync to DuckDB")
            except Exception as e:
                logger.error(f"Error in background sync: {str(e)}")
            
            # Wait for next sync
            await asyncio.sleep(sync_interval)

    def _ensure_tables_bucket(self):
        """Ensure the tables bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except Exception as e:
            logger.error(f"Error checking bucket: {str(e)}")
            self.s3_client.create_bucket(Bucket=self.bucket)

    def _init_duckdb(self):
        """Initialize DuckDB connection and tables."""
        try:
            conn = duckdb.connect(str(self.duckdb_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tables (
                    name VARCHAR,
                    description VARCHAR,
                    schema JSON,
                    row_count INTEGER,
                    last_updated TIMESTAMP,
                    created_by VARCHAR,
                    created_at TIMESTAMP
                )
            """)
            conn.close()
        except Exception as e:
            logger.error(f"Error initializing DuckDB: {str(e)}")

    def _sync_table_to_duckdb(self, table: TableInfo):
        """Sync a table's metadata to DuckDB."""
        try:
            conn = duckdb.connect(str(self.duckdb_path))
            
            # Convert datetime objects to ISO format strings for JSON serialization
            table_dict = table.dict()
            
            conn.execute("""
                INSERT OR REPLACE INTO tables (
                    name, description, schema, row_count, 
                    last_updated, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                table.name,
                table.description,
                json.dumps(table_dict.get('schema', [])),
                table.rowCount,
                table.lastUpdated.isoformat() if table.lastUpdated else None,
                table.createdBy,
                table.createdAt.isoformat() if table.createdAt else None
            ))
            
            conn.close()
        except Exception as e:
            logger.error(f"Error syncing table to DuckDB: {str(e)}")

    async def sync_glue_table_to_duckdb(self, table_name: str, user_id: str) -> bool:
        """Synchronize a single Glue table to DuckDB cache only if DDL has changed.
        
        Args:
            table_name: Name of the table to sync
            user_id: User ID who owns the table
            
        Returns:
            bool: True if sync was successful, False otherwise
        """
        try:
            # Get table from Glue
            database_name = self.get_user_database_name(user_id)
            table_info = await self.glue_service.get_table(
                database_name, table_name
            )
            
            if not table_info:
                logger.warning(
                    f"Table {table_name} not found in Glue, removing from DuckDB if exists"
                )
                self._remove_table_from_duckdb(table_name)
                return False
            
            # Generate version hash from table DDL
            ddl_version = self._generate_table_version(table_info)
            
            # Check if version has changed
            cached_version = self.conn.execute(
                "SELECT version FROM table_cache WHERE table_name = ?",
                (table_name,)
            ).fetchone()
            
            if cached_version and cached_version[0] == ddl_version:
                logger.info(f"Table {table_name} DDL unchanged, skipping sync")
                return True
            
            # Get table data from S3
            data = await self.get_table_data(table_name, user_id)
            
            # Update schema cache with new version
            self.conn.execute("""
                INSERT INTO table_cache (
                    table_name, s3_path, last_updated, schema, version
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (table_name) DO UPDATE SET
                    s3_path = excluded.s3_path,
                    last_updated = excluded.last_updated,
                    schema = excluded.schema,
                    version = excluded.version
            """, (
                table_name,
                table_info['StorageDescriptor']['Location'],
                datetime.now(),
                json.dumps(table_info['StorageDescriptor']['Columns']),
                ddl_version
            ))
            
            # Update data cache
            self.conn.execute("""
                INSERT INTO table_data (
                    table_name, data, last_updated
                ) VALUES (?, ?, ?)
                ON CONFLICT (table_name) DO UPDATE SET
                    data = excluded.data,
                    last_updated = excluded.last_updated
            """, (
                table_name,
                json.dumps(data.get('data', [])),
                datetime.now()
            ))
            
            return True
            
        except Exception as e:
            logger.error(
                f"Error syncing table {table_name} to DuckDB: {str(e)}"
            )
            return False

    def _generate_table_version(self, table_info: Dict[str, Any]) -> str:
        """Generate a version hash from table DDL.
        
        This version will change only when the table's DDL changes (schema, location, etc).
        """
        # Create a string representation of the DDL-relevant parts
        ddl_parts = {
            'columns': table_info['StorageDescriptor']['Columns'],
            'location': table_info['StorageDescriptor']['Location'],
            'input_format': (
                table_info['StorageDescriptor'].get('InputFormat')
            ),
            'output_format': (
                table_info['StorageDescriptor'].get('OutputFormat')
            ),
            'serde_info': table_info['StorageDescriptor'].get('SerdeInfo'),
            'parameters': table_info.get('Parameters', {})
        }
        
        # Convert to JSON string and hash it
        ddl_str = json.dumps(ddl_parts, sort_keys=True)
        return hashlib.md5(ddl_str.encode()).hexdigest()

    def _remove_table_from_duckdb(self, table_name: str) -> None:
        """Remove a table from DuckDB cache."""
        try:
            self.conn.execute("DELETE FROM table_cache WHERE table_name = ?", (table_name,))
            self.conn.execute("DELETE FROM table_data WHERE table_name = ?", (table_name,))
        except Exception as e:
            logger.error(f"Error removing table {table_name} from DuckDB: {str(e)}")

    async def validate_schema(
        self,
        schema: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Validate data against schema constraints.
        
        Args:
            schema: Schema definition
            data: DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        try:
            validation_results = {
                "status": "success",
                "validations": {},
                "errors": []
            }
            
            # Validate column existence
            schema_columns = {col["name"] for col in schema.get("columns", [])}
            data_columns = set(data.columns)
            
            if schema_columns != data_columns:
                validation_results["errors"].append({
                    "type": "column_mismatch",
                    "message": (
                        f"Schema columns {schema_columns} don't match "
                        f"data columns {data_columns}"
                    ),
                    "severity": "error"
                })
            
            # Validate data types
            for col in schema.get("columns", []):
                col_name = col["name"]
                if col_name in data.columns:
                    expected_type = col.get("type", "")
                    actual_type = str(data[col_name].dtype)
                    
                    if expected_type and expected_type != actual_type:
                        validation_results["errors"].append({
                            "type": "type_mismatch",
                            "column": col_name,
                            "expected": expected_type,
                            "actual": actual_type,
                            "severity": "error"
                        })
                    
                    # Validate constraints
                    if (
                        col.get("nullable") is False 
                        and data[col_name].isnull().any()
                    ):
                        validation_results["errors"].append({
                            "type": "null_violation",
                            "column": col_name,
                            "message": (
                                f"Column {col_name} contains null values "
                                "but is marked as non-nullable"
                            ),
                            "severity": "error"
                        })
            
            # Update validation status
            if validation_results["errors"]:
                validation_results["status"] = "error"
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating schema: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def configure_quality_checks(
        self,
        table_name: str,
        config: QualityCheckConfig,
        user_id: str = "test_user"
    ) -> Dict[str, Any]:
        """Configure quality checks for a table."""
        try:
            # Store configuration in S3
            config_key = f"metadata/quality/{user_id}/{table_name}/config.json"
            
            # Convert config to dict and add timestamp
            config_dict = config.dict()
            config_dict["last_updated"] = datetime.now().isoformat()
            
            # Use s3_client directly instead of S3Service
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=config_key,
                Body=json.dumps(config_dict)
            )
            
            return {
                "status": "success",
                "message": "Quality check configuration saved successfully",
                "config": config_dict
            }
            
        except Exception as e:
            logger.error(f"Error configuring quality checks: {str(e)}")
            raise Exception(f"Error configuring quality checks: {str(e)}")

    async def run_quality_checks(
        self,
        table_name: str,
        user_id: str = "test_user",
        force: bool = False
    ) -> Dict[str, Any]:
        try:
            # Get quality check configuration
            config_key = (
                f"metadata/quality/{user_id}/{table_name}/config.json"
            )
            try:
                response = self.s3_service.get_object(
                    Bucket=self.bucket,
                    Key=config_key
                )
                config = json.loads(response['Body'].read())
            except self.s3_service.exceptions.NoSuchKey:
                # Use default configuration if none exists
                config = {
                    "enabled_metrics": [
                        "completeness",
                        "uniqueness",
                        "consistency"
                    ],
                    "thresholds": {
                        "completeness": 0.95,
                        "uniqueness": 0.90,
                        "consistency": 0.85
                    }
                }
            
            # Get table data
            query = f"SELECT * FROM {table_name} LIMIT 1000"
            result = await self.execute_query(query, user_id)
            
            if result["status"] == "error":
                raise Exception(f"Error executing query: {result['message']}")
            
            # Convert results to DataFrame
            df = pd.DataFrame(result["results"])
            
            # Calculate metrics based on configuration
            metrics = {}
            column_metrics = {}
            
            # Calculate metrics for each column
            for column in df.columns:
                column_metrics[column] = {
                    "quality_metrics": {
                        "completeness": float(
                            1.0 - df[column].isnull().sum() / len(df)
                        ),
                        "uniqueness": float(
                            1.0 - df[column].duplicated().sum() / len(df)
                        ),
                        "validity": 1.0  # Default validity score
                    }
                }
            
            # Calculate overall metrics
            if "completeness" in config["enabled_metrics"]:
                null_cells = int(df.isnull().sum().sum())
                total_cells = int(df.shape[0] * df.shape[1])
                completeness = float(1.0 - null_cells / total_cells)
                metrics["completeness"] = {
                    "score": completeness,
                    "status": (
                        "success" 
                        if completeness >= config["thresholds"]["completeness"]
                        else "warning"
                    ),
                    "details": {
                        "total_cells": total_cells,
                        "null_cells": null_cells,
                        "threshold": float(
                            config["thresholds"]["completeness"]
                        )
                    }
                }
            
            if "uniqueness" in config["enabled_metrics"]:
                duplicate_rows = int(df.duplicated().sum())
                total_rows = int(df.shape[0])
                uniqueness = float(1.0 - duplicate_rows / total_rows)
                metrics["uniqueness"] = {
                    "score": uniqueness,
                    "status": (
                        "success" 
                        if uniqueness >= config["thresholds"]["uniqueness"]
                        else "warning"
                    ),
                    "details": {
                        "total_rows": total_rows,
                        "duplicate_rows": duplicate_rows,
                        "threshold": float(
                            config["thresholds"]["uniqueness"]
                        )
                    }
                }
            
            if "consistency" in config["enabled_metrics"]:
                pattern_checks = 0
                for column in df.columns:
                    if df[column].dtype == "object":
                        unique_patterns = int(
                            df[column].str.len().nunique()
                        )
                        if unique_patterns <= 3:  # Allow some variation
                            pattern_checks += 1
                    elif df[column].dtype in ["int64", "float64"]:
                        std = float(df[column].std())
                        mean = float(df[column].mean())
                        if mean != 0 and (std / mean < 0.5):
                            pattern_checks += 1
                
                consistency = float(pattern_checks / len(df.columns))
                metrics["consistency"] = {
                    "score": consistency,
                    "status": (
                        "success" 
                        if consistency >= config["thresholds"]["consistency"]
                        else "warning"
                    ),
                    "details": {
                        "total_columns": int(len(df.columns)),
                        "consistent_columns": pattern_checks,
                        "threshold": float(
                            config["thresholds"]["consistency"]
                        )
                    }
                }
            
            # Store metrics
            metrics_key = (
                f"metadata/quality/{user_id}/{table_name}/metrics.json"
            )
            metrics_data = {
                "metrics": metrics,
                "column_metrics": column_metrics,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "enabled_metrics": config["enabled_metrics"],
                    "thresholds": {
                        k: float(v) 
                        for k, v in config["thresholds"].items()
                    }
                }
            }
            
            self.s3_client.put_object(
                Bucket=self.bucket,
                Key=metrics_key,
                Body=json.dumps(metrics_data)
            )
            
            return {
                "status": "success",
                "metrics": metrics,
                "column_metrics": column_metrics,
                "metadata": {
                    "table_name": table_name,
                    "checked_at": datetime.now().isoformat(),
                    "metrics_location": f"s3://{self.bucket}/{metrics_key}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error running quality checks: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }