"""Service for managing AWS Glue operations."""

import logging
import boto3
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlueTableConfig(BaseModel):
    """Configuration for Glue table operations."""
    database_name: str
    table_name: str
    schema: Dict[str, Any]
    location: str
    file_format: str = "parquet"
    partition_keys: Optional[List[Dict[str, str]]] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class GlueService:
    """Service for managing AWS Glue operations."""

    # File format configurations
    FORMAT_CONFIGS = {
        "parquet": {
            "input_format": (
                "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat"
            ),
            "output_format": (
                "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat"
            ),
            "serialization_library": (
                "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            ),
            "serde_parameters": {
                "serialization.format": "1"
            },
            "table_properties": {
                "parquet.compression": "SNAPPY",
                "compressionType": "none",
                "classification": "parquet",
                "typeOfData": "file"
            }
        },
        "csv": {
            "input_format": "org.apache.hadoop.mapred.TextInputFormat",
            "output_format": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "serialization_library": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
            "serde_parameters": {
                "field.delim": ",",
                "serialization.format": ","
            },
            "table_properties": {
                "classification": "csv",
                "typeOfData": "file"
            }
        },
        "json": {
            "input_format": "org.apache.hadoop.mapred.TextInputFormat",
            "output_format": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "serialization_library": "org.openx.data.jsonserde.JsonSerDe",
            "serde_parameters": {
                "serialization.format": "1"
            },
            "table_properties": {
                "classification": "json",
                "typeOfData": "file"
            }
        },
        "avro": {
            "input_format": "org.apache.hadoop.hive.ql.io.avro.AvroContainerInputFormat",
            "output_format": "org.apache.hadoop.hive.ql.io.avro.AvroContainerOutputFormat",
            "serialization_library": "org.apache.hadoop.hive.serde2.avro.AvroSerDe",
            "serde_parameters": {
                "serialization.format": "1"
            },
            "table_properties": {
                "classification": "avro",
                "typeOfData": "file"
            }
        }
    }

    def __init__(self, region_name: str = "us-east-1"):
        """Initialize Glue service with AWS region."""
        self.region_name = region_name
        self.glue_client = boto3.client('glue', region_name=region_name)

    async def create_database(self, database_name: str) -> Dict[str, Any]:
        """Create a Glue database if it doesn't exist."""
        try:
            self.glue_client.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': f'Database created at {datetime.utcnow().isoformat()}'
                }
            )
            return {
                "status": "success",
                "message": f"Database {database_name} created"
            }
        except self.glue_client.exceptions.AlreadyExistsException:
            return {
                "status": "exists",
                "message": f"Database {database_name} already exists"
            }
        except Exception as e:
            logger.error(f"Error creating database: {str(e)}")
            raise Exception(f"Error creating database: {str(e)}")

    async def create_or_update_table(self, config: GlueTableConfig) -> Dict[str, Any]:
        """Create or update a Glue table with the given configuration."""
        try:
            # Ensure database exists
            await self.create_database(config.database_name)
            
            # Get format configuration
            format_config = self.FORMAT_CONFIGS.get(config.file_format, self.FORMAT_CONFIGS["parquet"])
            
            # Ensure location ends with a trailing slash for folder paths
            location = config.location.rstrip('/') + '/'
            
            # Prepare table input
            table_input = {
                'Name': config.table_name,
                'Description': config.description or f'Table created at {datetime.utcnow().isoformat()}',
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': {
                    'classification': config.file_format,
                    **format_config['table_properties']
                },
                'StorageDescriptor': {
                    'Columns': [
                        {
                            'Name': col.get('Name', col.get('name')),
                            'Type': col.get('Type', col.get('type')),
                            'Comment': col.get('Comment', col.get('comment', ''))
                        }
                        for col in config.schema.get('columns', [])
                    ],
                    'Location': location,
                    'InputFormat': format_config['input_format'],
                    'OutputFormat': format_config['output_format'],
                    'Compressed': False,
                    'NumberOfBuckets': -1,
                    'SerdeInfo': {
                        'SerializationLibrary': format_config['serialization_library'],
                        'Parameters': format_config['serde_parameters']
                    },
                    'BucketColumns': [],
                    'SortColumns': [],
                    'StoredAsSubDirectories': True  # Enable subdirectories for folder-based storage
                }
            }

            # Add partition keys if provided
            if config.partition_keys:
                table_input['PartitionKeys'] = [
                    {
                        'Name': key['name'],
                        'Type': key['type'],
                        'Comment': key.get('comment', '')
                    }
                    for key in config.partition_keys
                ]

            # Add metadata if provided
            if config.metadata:
                table_input['Parameters'].update(config.metadata)

            try:
                # Try to get existing table
                existing_table = await self.get_table(config.database_name, config.table_name)
                # Update existing table
                self.glue_client.update_table(
                    DatabaseName=config.database_name,
                    TableInput=table_input
                )
                return {
                    'status': 'success',
                    'message': f'Table {config.table_name} updated successfully',
                    'table_info': existing_table
                }
            except Exception:
                # Create new table if it doesn't exist
                self.glue_client.create_table(
                    DatabaseName=config.database_name,
                    TableInput=table_input
                )
                return {
                    'status': 'success',
                    'message': f'Table {config.table_name} created successfully',
                    'table_info': await self.get_table(config.database_name, config.table_name)
                }

        except Exception as e:
            logger.error(f"Error creating/updating table: {str(e)}")
            raise Exception(f"Error creating/updating table: {str(e)}")

    async def get_table(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """Get table information from Glue."""
        try:
            response = self.glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            return response['Table']
        except Exception as e:
            logger.error(f"Error getting table: {str(e)}")
            raise Exception(f"Error getting table: {str(e)}")

    async def list_tables(self, database_name: str) -> List[Dict[str, Any]]:
        """List all tables in a database."""
        try:
            response = self.glue_client.get_tables(
                DatabaseName=database_name
            )
            return response['TableList']
        except Exception as e:
            logger.error(f"Error listing tables: {str(e)}")
            raise Exception(f"Error listing tables: {str(e)}")

    async def delete_table(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """Delete a table from Glue."""
        try:
            self.glue_client.delete_table(
                DatabaseName=database_name,
                Name=table_name
            )
            return {
                "status": "success",
                "message": f"Table {table_name} deleted"
            }
        except Exception as e:
            logger.error(f"Error deleting table: {str(e)}")
            raise Exception(f"Error deleting table: {str(e)}")

    def _convert_to_glue_type(self, pandas_type: str) -> str:
        """Convert pandas data type to Glue data type."""
        type_mapping = {
            'int': 'bigint',
            'float': 'double',
            'datetime': 'timestamp',
            'bool': 'boolean',
            'object': 'string'
        }
        return type_mapping.get(pandas_type.lower(), 'string') 