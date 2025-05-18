import boto3
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class GlueService:
    def __init__(self, region_name: str = "us-east-1"):
        self.region_name = region_name
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')
        
        self.glue_client = boto3.client(
            'glue',
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token
        )
        self.s3_client = boto3.client(
            's3',
            region_name=region_name,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token
        )
        
    async def create_database(self, database_name: str) -> Dict[str, Any]:
        """Create a Glue database if it doesn't exist."""
        try:
            self.glue_client.create_database(
                DatabaseInput={
                    'Name': database_name,
                    'Description': f'Database created by ETL Architect Agent at {datetime.utcnow().isoformat()}'
                }
            )
            return {"status": "success", "message": f"Database {database_name} created"}
        except self.glue_client.exceptions.AlreadyExistsException:
            return {"status": "exists", "message": f"Database {database_name} already exists"}
        except Exception as e:
            raise Exception(f"Error creating database: {str(e)}")

    async def create_table(
        self,
        database_name: str,
        table_name: str,
        schema: Dict[str, Any],
        location: str,
        file_format: str,
        partition_keys: Optional[List[Dict[str, str]]] = None,
        original_data_location: Optional[str] = None,
        parquet_data_location: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a new table in the Glue catalog.
        
        Args:
            database_name: Name of the database
            table_name: Name of the table
            schema: Table schema definition
            location: S3 location where the data is stored
            file_format: Format of the data file
            partition_keys: Optional list of partition keys
            original_data_location: Optional S3 path to the original/raw data
            parquet_data_location: Optional S3 path to the parquet version
            metadata: Optional additional metadata to store
        """
        try:
            # Convert schema to Glue format
            columns = []
            for field in schema.get('fields', []):
                columns.append({
                    'Name': field['name'],
                    'Type': self._convert_to_glue_type(field['type']),
                    'Comment': field.get('description', '')
                })

            # Prepare partition keys if provided
            partition_keys_list = []
            if partition_keys:
                for key in partition_keys:
                    partition_keys_list.append({
                        'Name': key['name'],
                        'Type': self._convert_to_glue_type(key['type'])
                    })

            # Enhanced table parameters
            table_parameters = {
                'classification': file_format,
                'typeOfData': 'file',
                'has_encrypted_data': 'false',
                'compressionType': 'none',
                'data_location': location,
                'original_data_location': original_data_location or location,
                'parquet_data_location': parquet_data_location or location,
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat()
            }

            # Add any additional metadata
            if metadata:
                table_parameters.update(metadata)

            # Create table input
            table_input = {
                'Name': table_name,
                'Description': f'Table created from {table_name}',
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': table_parameters,
                'StorageDescriptor': {
                    'Columns': columns,
                    'Location': location,
                    'InputFormat': self._get_input_format(file_format),
                    'OutputFormat': self._get_output_format(file_format),
                    'SerdeInfo': {
                        'SerializationLibrary': self._get_serialization_library(file_format),
                        'Parameters': self._get_serde_parameters(file_format)
                    }
                }
            }

            if partition_keys_list:
                table_input['PartitionKeys'] = partition_keys_list

            # Create the table
            self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )

            return {
                "status": "success",
                "table_info": {
                    "database_name": database_name,
                    "table_name": table_name,
                    "location": location,
                    "original_data_location": original_data_location,
                    "parquet_data_location": parquet_data_location,
                    "parameters": table_parameters
                }
            }

        except Exception as e:
            logger.error(f"Error creating table: {str(e)}")
            raise Exception(f"Failed to create table: {str(e)}")

    async def update_table(
        self,
        database_name: str,
        table_name: str,
        schema: Dict[str, Any],
        location: str,
        file_format: str
    ) -> Dict[str, Any]:
        """Update an existing table in Glue Catalog."""
        try:
            # Get current table
            current_table = self.glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )

            # Convert schema to Glue format
            columns = []
            for field in schema.get("fields", []):
                columns.append({
                    'Name': field["name"],
                    'Type': self._convert_to_glue_type(field["type"])
                })

            # Update table input
            table_input = current_table['Table']
            table_input['StorageDescriptor']['Columns'] = columns
            table_input['StorageDescriptor']['Location'] = location
            table_input['Parameters']['classification'] = file_format

            # Update the table
            self.glue_client.update_table(
                DatabaseName=database_name,
                TableInput=table_input
            )

            return {
                "status": "success",
                "message": f"Table {table_name} updated in database {database_name}",
                "table_info": {
                    "database": database_name,
                    "table": table_name,
                    "location": location,
                    "format": file_format,
                    "columns": columns
                }
            }

        except Exception as e:
            raise Exception(f"Error updating table: {str(e)}")

    async def create_parquet_table(
        self,
        database_name: str,
        table_name: str,
        schema: Dict[str, Any],
        location: str,
        partition_keys: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Create a Parquet table in Glue Catalog directly with schema."""
        try:
            # Convert schema to Glue format
            columns = []
            for field in schema.get("fields", []):
                columns.append({
                    'Name': field["name"],
                    'Type': self._convert_to_glue_type(field["type"]),
                    'Comment': field.get("description", "")
                })

            # Create table input
            table_input = {
                'Name': table_name,
                'Description': f'Parquet table created by ETL Architect Agent at {datetime.utcnow().isoformat()}',
                'TableType': 'EXTERNAL_TABLE',
                'Parameters': {
                    'classification': 'parquet',
                    'typeOfData': 'file',
                    'compressionType': 'none',
                    'parquet.compression': 'SNAPPY'
                },
                'StorageDescriptor': {
                    'Columns': columns,
                    'Location': location,
                    'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                    'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                    'SerdeInfo': {
                        'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
                        'Parameters': {}
                    }
                }
            }

            if partition_keys:
                table_input['PartitionKeys'] = partition_keys

            # Create the table
            self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )

            return {
                "status": "success",
                "message": f"table {table_name} created in database {database_name}",
                "table_info": {
                    "database": database_name,
                    "table": table_name,
                    "location": location,
                    "format": "parquet",
                    "columns": columns
                }
            }

        except Exception as e:
            raise Exception(f"Error creating Parquet table: {str(e)}")

    def _convert_to_glue_type(self, pandas_type: str) -> str:
        """Convert pandas dtype to Glue data type."""
        type_mapping = {
            'int64': 'bigint',
            'float64': 'double',
            'bool': 'boolean',
            'datetime64[ns]': 'timestamp',
            'object': 'string'
        }
        return type_mapping.get(str(pandas_type), 'string')

    def _get_input_format(self, file_format: str) -> str:
        """Get Glue input format for file type."""
        format_mapping = {
            'csv': 'org.apache.hadoop.mapred.TextInputFormat',
            'json': 'org.apache.hadoop.mapred.TextInputFormat',
            'parquet': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat'
        }
        return format_mapping.get(file_format.lower(), 'org.apache.hadoop.mapred.TextInputFormat')

    def _get_output_format(self, file_format: str) -> str:
        """Get Glue output format for file type."""
        format_mapping = {
            'csv': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
            'json': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
            'parquet': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
        }
        return format_mapping.get(file_format.lower(), 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat')

    def _get_serialization_library(self, file_format: str) -> str:
        """Get Glue serialization library for file type."""
        library_mapping = {
            'csv': 'org.apache.hadoop.hive.serde2.OpenCSVSerde',
            'json': 'org.openx.data.jsonserde.JsonSerDe',
            'parquet': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
        }
        return library_mapping.get(file_format.lower(), 'org.apache.hadoop.hive.serde2.OpenCSVSerde')

    def _get_serde_parameters(self, file_format: str) -> Dict[str, str]:
        """Get Glue SerDe parameters for file type."""
        if file_format.lower() == 'csv':
            return {
                'separatorChar': ',',
                'quoteChar': '"',
                'escapeChar': '\\'
            }
        return {}

    async def create_user_table(
        self,
        user_id: str,
        table_name: str,
        schema: Dict[str, Any],
        location: str,
        file_format: str = "parquet",
        partition_keys: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Create a table in a user-specific database in Glue Catalog.
        
        Args:
            user_id: Unique identifier for the user
            table_name: Name of the table to create
            schema: Table schema definition
            location: S3 location where the data is stored
            file_format: Format of the data file (default: parquet)
            partition_keys: Optional list of partition keys
            
        Returns:
            Dict containing the status and table information
        """
        try:
            # Create user-specific database if it doesn't exist
            database_name = f"user_{user_id}"
            try:
                self.glue_client.create_database(
                    DatabaseInput={
                        'Name': database_name,
                        'Description': f'Database for user {user_id}'
                    }
                )
                logger.info(f"Created database {database_name} for user {user_id}")
            except self.glue_client.exceptions.AlreadyExistsException:
                logger.info(f"Database {database_name} already exists")

            # Convert schema to Glue format
            columns = []
            for col in schema["columns"]:
                col_type = col["type"]
                # Map Python types to Glue types
                if "int" in col_type: glue_type = "bigint"
                elif "float" in col_type: glue_type = "double"
                elif "datetime" in col_type: glue_type = "timestamp"
                else: glue_type = "string"
                
                columns.append({
                    "Name": col["name"],
                    "Type": glue_type,
                    "Comment": col.get("description", "")
                })

            # Configure storage descriptor based on file format
            if file_format.lower() == "parquet":
                storage_descriptor = {
                    "Columns": columns,
                    "Location": location,
                    "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
                    "SerdeInfo": {
                        "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe",
                        "Parameters": {"serialization.format": "1"}
                    }
                }
            else:  # Default to CSV
                storage_descriptor = {
                    "Columns": columns,
                    "Location": location,
                    "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
                    "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
                    "SerdeInfo": {
                        "SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                        "Parameters": {"field.delim": ","}
                    }
                }

            # Create table
            table_input = {
                "Name": table_name,
                "TableType": "EXTERNAL_TABLE",
                "StorageDescriptor": storage_descriptor,
                "Parameters": {
                    "classification": file_format,
                    "typeOfData": "file",
                    "user_id": user_id
                }
            }

            if partition_keys:
                table_input["PartitionKeys"] = [
                    {"Name": key["name"], "Type": key["type"]}
                    for key in partition_keys
                ]

            self.glue_client.create_table(
                DatabaseName=database_name,
                TableInput=table_input
            )

            return {
                "status": "success",
                "message": f"Table {table_name} created in database {database_name}",
                "table_info": {
                    "database": database_name,
                    "table": table_name,
                    "location": location,
                    "format": file_format,
                    "columns": columns
                }
            }

        except Exception as e:
            logger.error(f"Error creating user table: {str(e)}", exc_info=True)
            raise Exception(f"Error creating user table: {str(e)}") 