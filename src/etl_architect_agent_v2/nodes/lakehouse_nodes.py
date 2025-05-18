"""Nodes for lakehouse automation workflow."""

from typing import Dict, Any, List
import logging
import boto3
from botocore.exceptions import ClientError
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime

from ..core.state import AgentState
from ..core.schema_registry import SchemaRegistry, SchemaFormat

logger = logging.getLogger(__name__)


class LakehouseNodes:
    """Nodes for lakehouse automation workflow."""
    
    def __init__(
        self,
        region_name: str = "us-east-1",
        schema_registry: SchemaRegistry = None
    ):
        """Initialize lakehouse nodes.
        
        Args:
            region_name: AWS region name
            schema_registry: Schema registry instance
        """
        self.s3_client = boto3.client('s3', region_name=region_name)
        self.glue_client = boto3.client('glue', region_name=region_name)
        self.schema_registry = schema_registry
    
    async def discover_files(
        self,
        state: AgentState,
        bucket_name: str,
        prefix: str = "",
        file_types: List[str] = None
    ) -> AgentState:
        """Discover new files in S3 bucket.
        
        Args:
            state: Current agent state
            bucket_name: S3 bucket name
            prefix: S3 prefix to scan
            file_types: List of file extensions to include
            
        Returns:
            Updated agent state
        """
        if file_types is None:
            file_types = ['.csv', '.json']
        
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            new_files = []
            for obj in response.get('Contents', []):
                key = obj['Key']
                if any(key.endswith(ext) for ext in file_types):
                    new_files.append({
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat()
                    })
            
            state["discovered_files"] = new_files
            state["bucket_name"] = bucket_name
            logger.info(f"Discovered {len(new_files)} new files")
            return state
        except ClientError as e:
            logger.error(f"Error discovering files: {str(e)}")
            state["error"] = str(e)
            return state
    
    async def infer_schema(
        self,
        state: AgentState,
        sample_size: int = 1000
    ) -> AgentState:
        """Infer schema from sample data.
        
        Args:
            state: Current agent state
            sample_size: Number of rows to sample
            
        Returns:
            Updated agent state
        """
        try:
            schemas = {}
            for file_info in state["discovered_files"]:
                key = file_info['key']
                bucket = state["bucket_name"]
                
                # Download sample data
                response = self.s3_client.get_object(
                    Bucket=bucket,
                    Key=key,
                    Range=f'bytes=0-{min(1024*1024, file_info["size"])}'
                )
                data = response['Body'].read().decode('utf-8')
                
                # Infer schema based on file type
                if key.endswith('.csv'):
                    df = pd.read_csv(pd.StringIO(data), nrows=sample_size)
                    schema = {
                        'columns': [
                            {
                                'name': col,
                                'type': str(df[col].dtype),
                                'nullable': True
                            }
                            for col in df.columns
                        ]
                    }
                elif key.endswith('.json'):
                    df = pd.read_json(pd.StringIO(data), lines=True, nrows=sample_size)
                    schema = {
                        'columns': [
                            {
                                'name': col,
                                'type': str(df[col].dtype),
                                'nullable': True
                            }
                            for col in df.columns
                        ]
                    }
                else:
                    continue
                
                schemas[key] = schema
            
            state["inferred_schemas"] = schemas
            logger.info(f"Inferred schemas for {len(schemas)} files")
            return state
        except Exception as e:
            logger.error(f"Error inferring schemas: {str(e)}")
            state["error"] = str(e)
            return state
    
    async def create_iceberg_tables(
        self,
        state: AgentState,
        database_name: str
    ) -> AgentState:
        """Create or update Iceberg tables.
        
        Args:
            state: Current agent state
            database_name: Glue database name
            
        Returns:
            Updated agent state
        """
        try:
            created_tables = []
            for key, schema in state["inferred_schemas"].items():
                table_name = key.split('/')[-1].split('.')[0]
                
                # Create table if not exists
                try:
                    self.glue_client.get_table(
                        DatabaseName=database_name,
                        Name=table_name
                    )
                    logger.info(f"Table {table_name} already exists")
                except ClientError as e:
                    if e.response['Error']['Code'] == 'EntityNotFoundException':
                        # Create new table
                        self.glue_client.create_table(
                            DatabaseName=database_name,
                            TableInput={
                                'Name': table_name,
                                'TableType': 'EXTERNAL_TABLE',
                                'StorageDescriptor': {
                                    'Columns': [
                                        {
                                            'Name': col['name'],
                                            'Type': col['type'],
                                            'Comment': ''
                                        }
                                        for col in schema['columns']
                                    ],
                                    'Location': f"s3://{state['bucket_name']}/{key}",
                                    'InputFormat': 'org.apache.iceberg.mr.hive.HiveIcebergInputFormat',
                                    'OutputFormat': 'org.apache.iceberg.mr.hive.HiveIcebergOutputFormat',
                                    'SerdeInfo': {
                                        'SerializationLibrary': 'org.apache.iceberg.mr.hive.HiveIcebergSerDe'
                                    }
                                },
                                'Parameters': {
                                    'table_type': 'ICEBERG',
                                    'format': 'iceberg/parquet'
                                }
                            }
                        )
                        logger.info(f"Created table {table_name}")
                    else:
                        raise
                
                created_tables.append(table_name)
            
            state["created_tables"] = created_tables
            return state
        except Exception as e:
            logger.error(f"Error creating tables: {str(e)}")
            state["error"] = str(e)
            return state
    
    async def load_data(
        self,
        state: AgentState,
        database_name: str
    ) -> AgentState:
        """Load data into Iceberg tables.
        
        Args:
            state: Current agent state
            database_name: Glue database name
            
        Returns:
            Updated agent state
        """
        try:
            loaded_tables = []
            for key, schema in state["inferred_schemas"].items():
                table_name = key.split('/')[-1].split('.')[0]
                
                # Load data using Athena
                query = f"""
                INSERT INTO {database_name}.{table_name}
                SELECT * FROM {database_name}.{table_name}_temp
                """
                
                # Execute query using Athena
                # Note: Athena client implementation needed
                
                loaded_tables.append(table_name)
            
            state["loaded_tables"] = loaded_tables
            return state
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            state["error"] = str(e)
            return state
    
    async def generate_report(
        self,
        state: AgentState,
        database_name: str
    ) -> AgentState:
        """Generate summary report.
        
        Args:
            state: Current agent state
            database_name: Glue database name
            
        Returns:
            Updated agent state
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "tables_processed": len(state["created_tables"]),
                "tables": []
            }
            
            for table_name in state["created_tables"]:
                # Get table statistics
                response = self.glue_client.get_table(
                    DatabaseName=database_name,
                    Name=table_name
                )
                
                table_info = response['Table']
                report["tables"].append({
                    "name": table_name,
                    "row_count": 0,  # Need to implement row count
                    "columns": len(table_info['StorageDescriptor']['Columns']),
                    "location": table_info['StorageDescriptor']['Location']
                })
            
            state["report"] = report
            return state
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            state["error"] = str(e)
            return state 