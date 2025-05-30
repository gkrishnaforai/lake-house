from typing import Dict, Any, List, Optional, ClassVar
import boto3
import pandas as pd
from .base_logging_tool import BaseLoggingTool
from pydantic import BaseModel, Field, PrivateAttr
import logging
import time
import os

logger = logging.getLogger(__name__)


class FileInfo(BaseModel):
    file_name: str
    file_type: str
    s3_path: str
    size: int
    table_schema: Dict[str, str] = Field(
        description="Schema mapping column names to types"
    )


class IcebergTableInfo(BaseModel):
    table_name: str
    database_name: str
    s3_location: str
    table_schema: Dict[str, str] = Field(
        description="Schema mapping column names to types"
    )
    partition_keys: Optional[List[str]] = None
    properties: Optional[Dict[str, str]] = None


class FileToIcebergAgent:
    def __init__(self, s3_client=None, glue_client=None, athena_client=None):
        self.s3 = s3_client or boto3.client('s3')
        self.glue = glue_client or boto3.client('glue')
        self.athena = athena_client or boto3.client('athena')

    def upload_file_to_s3(self, file_info: FileInfo, table_info: IcebergTableInfo):
        """Upload file to S3, handling XLSX conversion if needed."""
        bucket = table_info.s3_location.split('/')[2]
        key = '/'.join(table_info.s3_location.split('/')[3:])
        
        # Handle XLSX files
        if file_info.file_type.lower() == 'xlsx':
            # Convert XLSX to CSV
            df = pd.read_excel(file_info.s3_path)  # Use local file path
            temp_csv = '/tmp/temp_file.csv'
            df.to_csv(temp_csv, index=False)
            
            # Update file info
            file_info.file_name = file_info.file_name.replace('.xlsx', '.csv')
            file_info.file_type = 'csv'
            file_info.s3_path = file_info.s3_path.replace('.xlsx', '.csv')
            
            # Upload CSV
            self.s3.upload_file(temp_csv, bucket, key)
            
            # Cleanup
            os.remove(temp_csv)
        else:
            # Upload file directly
            self.s3.upload_file(file_info.s3_path, bucket, key)

    def ensure_database_exists(self, db_name: str):
        """Ensure the database exists in Glue."""
        try:
            self.glue.get_database(Name=db_name)
            logger.info(f"Database {db_name} exists")
        except self.glue.exceptions.EntityNotFoundException:
            logger.info(f"Creating database {db_name}")
            self.glue.create_database(
                DatabaseInput={
                    'Name': db_name,
                    'Description': f'Database for {db_name}'
                }
            )
            # Wait for database to be available
            time.sleep(2)

    def create_glue_table(self, table_info: IcebergTableInfo):
        """Create the Glue table with Iceberg configuration."""
        columns = [{'Name': k, 'Type': v} for k, v in table_info.table_schema.items()]
        table_input = {
            'Name': table_info.table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'table_type': 'ICEBERG',
                'classification': 'iceberg',
                'format': 'iceberg',
                'EXTERNAL': 'TRUE',
                'write.metadata.location': f"{table_info.s3_location}/metadata",
                'iceberg.table.format-version': '2',
                **(table_info.properties or {})
            },
            'StorageDescriptor': {
                'Location': table_info.s3_location,
                'Columns': columns
            },
            'PartitionKeys': (
                [{'Name': k, 'Type': 'string'} for k in table_info.partition_keys]
                if table_info.partition_keys else []
            )
        }
        try:
            self.glue.create_table(
                DatabaseName=table_info.database_name,
                TableInput=table_input
            )
            logger.info(f"Created Glue table {table_info.table_name}")
        except self.glue.exceptions.AlreadyExistsException:
            logger.info(f"Table {table_info.table_name} already exists")

    def run_athena_query(self, query: str, db: str, output_s3: str, workgroup: str = 'primary'):
        """Execute an Athena query and wait for completion."""
        execution = self.athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': db},
            ResultConfiguration={'OutputLocation': output_s3},
            WorkGroup=workgroup
        )
        query_id = execution['QueryExecutionId']
        
        while True:
            result = self.athena.get_query_execution(QueryExecutionId=query_id)
            state = result['QueryExecution']['Status']['State']
            logger.info(f"Query state: {state}")
            
            if state in ('SUCCEEDED', 'FAILED', 'CANCELLED'):
                break
            time.sleep(1)
            
        if state == 'FAILED':
            reason = result['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
            raise RuntimeError(f"Athena query failed: {reason}")

    def create_iceberg_table_in_athena(self, table_info: IcebergTableInfo):
        """Create the Iceberg table in Athena."""
        # Map Python types to Athena SQL types
        type_mapping = {
            'string': 'string',
            'bigint': 'bigint',
            'double': 'double',
            'boolean': 'boolean',
            'timestamp': 'timestamp',
            'int': 'integer',
            'float': 'float'
        }
        
        # Format column definitions
        columns = []
        for name, type_ in table_info.table_schema.items():
            sql_type = type_mapping.get(type_.lower(), 'string')
            columns.append(f"`{name}` {sql_type}")
        
        query = f"""
        CREATE TABLE IF NOT EXISTS {table_info.database_name}.{table_info.table_name} (
            {', '.join(columns)}
        )
        LOCATION '{table_info.s3_location}'
        TBLPROPERTIES ('table_type'='ICEBERG')
        """
        
        output_s3 = f"s3://{table_info.s3_location.split('/')[2]}/athena-results/"
        self.run_athena_query(query, table_info.database_name, output_s3)

    def load_data_into_table(self, file_info: FileInfo, table_info: IcebergTableInfo):
        """Load data from the uploaded file into the Iceberg table."""
        query = f"""
        INSERT INTO {table_info.database_name}.{table_info.table_name}
        SELECT *
        FROM "{table_info.s3_location.split('/')[2]}".{table_info.database_name}."{file_info.file_name}"
        """
        
        output_s3 = f"s3://{table_info.s3_location.split('/')[2]}/athena-results/"
        self.run_athena_query(query, table_info.database_name, output_s3)

    def execute(self, file_info: FileInfo, table_info: IcebergTableInfo):
        """Execute the complete workflow."""
        try:
            self.upload_file_to_s3(file_info, table_info)
            self.ensure_database_exists(table_info.database_name)
            self.create_glue_table(table_info)
            self.create_iceberg_table_in_athena(table_info)
            self.load_data_into_table(file_info, table_info)
            
            return {
                "status": "success",
                "message": f"File loaded and Iceberg table {table_info.table_name} created with data",
                "table_info": table_info.model_dump()
            }
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "table_info": None
            }


class FileLoadingTool(BaseLoggingTool):
    """Tool for loading files into S3 and creating Iceberg tables."""
    name: ClassVar[str] = "file_loader"
    description: ClassVar[str] = "Load files into S3 and create Iceberg tables"
    _agent: Any = PrivateAttr()

    def __init__(self, s3_client=None, glue_client=None, athena_client=None, **kwargs):
        super().__init__(**kwargs)
        self._agent = FileToIcebergAgent(s3_client, glue_client, athena_client)

    def _run(self, file_info: FileInfo, table_info: IcebergTableInfo) -> Dict[str, Any]:
        """Execute the file loading workflow."""
        return self._agent.execute(file_info, table_info)


class SchemaInferenceTool(BaseLoggingTool):
    """Tool for inferring schema from a file."""
    name: ClassVar[str] = "schema_inference"
    description: ClassVar[str] = "Infer schema from file"

    def _run(self, file_path: str) -> Dict[str, str]:
        """Infer schema from a file."""
        try:
            # Validate file path
            if not file_path:
                raise ValueError("No file path provided")
            
            # Check file extension
            file_ext = file_path.lower().split('.')[-1]
            if file_ext not in ['csv', 'parquet', 'json', 'xlsx']:
                raise ValueError(
                    f"Unsupported file type: {file_ext}. "
                    "Only CSV, Parquet, JSON, and XLSX files are supported."
                )
            
            # Read file based on extension
            try:
                if file_ext == 'csv':
                    df = pd.read_csv(file_path)
                elif file_ext == 'parquet':
                    df = pd.read_parquet(file_path)
                elif file_ext == 'json':
                    df = pd.read_json(file_path)
                else:  # xlsx
                    df = pd.read_excel(file_path)
            except Exception as e:
                raise ValueError(f"Failed to read file {file_path}: {str(e)}")
            
            # Convert pandas dtypes to SQL types
            type_mapping = {
                'int64': 'bigint',
                'float64': 'double',
                'bool': 'boolean',
                'datetime64[ns]': 'timestamp',
                'object': 'string'
            }
            
            # Infer schema
            schema = {}
            for col, dtype in df.dtypes.items():
                sql_type = type_mapping.get(str(dtype), 'string')
                schema[col] = sql_type
            
            if not schema:
                raise ValueError("No columns found in file")
            
            return schema
            
        except Exception as e:
            logger.error(f"Error inferring schema from {file_path}: {str(e)}")
            raise ValueError(f"Error inferring schema: {str(e)}")


class TableValidationTool(BaseLoggingTool):
    """Tool for validating Iceberg table creation in Glue."""
    name: ClassVar[str] = "table_validation"
    description: ClassVar[str] = "Validate Iceberg table creation"
    _glue_client: Any = PrivateAttr()

    def __init__(self, glue_client=None, **kwargs):
        super().__init__(**kwargs)
        self._glue_client = glue_client or boto3.client('glue')

    def _run(self, database_name: str, table_name: str) -> Dict[str, Any]:
        """Validate that an Iceberg table was created correctly."""
        try:
            response = self._glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            table = response['Table']
            # Validate table type
            if table.get('TableType') != 'ICEBERG':
                return {
                    "status": "error",
                    "message": "Table is not an Iceberg table"
                }
            # Validate location
            if not table['StorageDescriptor']['Location'].startswith('s3://'):
                return {
                    "status": "error",
                    "message": "Invalid S3 location"
                }
            return {
                "status": "success",
                "message": "Table validated successfully",
                "table_info": {
                    "name": table['Name'],
                    "database": table['DatabaseName'],
                    "location": table['StorageDescriptor']['Location'],
                    "schema": {
                        col['Name']: col['Type']
                        for col in table['StorageDescriptor']['Columns']
                    }
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            } 