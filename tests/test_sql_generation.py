"""Test SQL generation functionality."""

import pytest
import pandas as pd
import os
from datetime import datetime
import logging
from io import BytesIO

from src.etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from src.etl_architect_agent_v2.agents.catalog_agent import CatalogAgent
from src.etl_architect_agent_v2.agents.schema_extractor.sql_generator_agent import (
    SQLGeneratorAgent
)
from src.etl_architect_agent_v2.core.llm.manager import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def aws_credentials():
    """Ensure AWS credentials are set."""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_SESSION_TOKEN',
        'AWS_REGION',
        'AWS_S3_BUCKET'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing AWS credentials: {', '.join(missing)}")

@pytest.fixture
def catalog_service(aws_credentials):
    """Create Catalog Service instance."""
    bucket = os.getenv('AWS_S3_BUCKET')
    if not bucket:
        pytest.skip("Missing AWS_S3_BUCKET environment variable")
    logger.info(f"Creating CatalogService with bucket: {bucket}")
    service = CatalogService(
        bucket=bucket,
        aws_region=os.getenv('AWS_REGION', 'us-east-1')
    )
    return service

@pytest.fixture
def catalog_agent(catalog_service):
    """Create Catalog Agent instance."""
    logger.info("Creating CatalogAgent")
    return CatalogAgent(catalog_service)

@pytest.fixture
def sql_generator_agent():
    """Create SQL Generator Agent instance."""
    logger.info("Creating SQLGeneratorAgent")
    llm_manager = LLMManager()
    return SQLGeneratorAgent(llm_manager)

@pytest.fixture
def test_table_data():
    """Create sample table data for testing."""
    return {
        'employee_id': [1, 2, 3, 4],
        'name': ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"],
        'department': ["Engineering", "Sales", "Engineering", "Marketing"],
        'salary': [90000, 80000, 110000, 70000],
        'role': [
            "Engineer II", "Sales Rep", "Senior Engineer", "Marketing Specialist"
        ],
        'start_date': ['2022-05-01', '2021-11-15', '2020-03-10', '2023-01-20']
    }

@pytest.mark.asyncio
async def test_schema_transformation(catalog_service, test_table_data):
    """Test schema transformation functionality."""
    # Create test table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_schema_{timestamp}"
    user_id = "test_user"
    
    try:
        # Create database if it doesn't exist
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
        except catalog_service.glue.exceptions.AlreadyExistsException:
            pass
        
        # Create test data
        df = pd.DataFrame(test_table_data)
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = f"data/{user_id}/{table_name}/test_data.parquet"
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Create Glue table
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'employee_id', 'Type': 'bigint'},
                    {'Name': 'name', 'Type': 'string'},
                    {'Name': 'department', 'Type': 'string'},
                    {'Name': 'salary', 'Type': 'bigint'},
                    {'Name': 'role', 'Type': 'string'},
                    {'Name': 'start_date', 'Type': 'string'}
                ],
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.'
                        'ParquetHiveSerDe'
                    )
                }
            },
            'PartitionKeys': []
        }
        
        catalog_service.glue.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        
        # Test schema transformation
        schema = await catalog_service.get_table_schema(table_name, user_id)
        
        # Verify schema
        assert isinstance(schema, dict)
        assert "schema" in schema
        assert len(schema["schema"]) == 6  # We have 6 columns
        column_names = {col["name"] for col in schema["schema"]}
        assert column_names == {
            "employee_id", "name", "department", "salary", "role", "start_date"
        }
        
    finally:
        # Cleanup
        try:
            catalog_service.glue.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            catalog_service.s3.delete_object(
                Bucket=catalog_service.bucket,
                Key=s3_key
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")

@pytest.mark.asyncio
async def test_sql_generation(catalog_service, test_table_data):
    """Test SQL generation functionality."""
    # Create test table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_sql_{timestamp}"
    user_id = "test_user"
    
    try:
        # Create database if it doesn't exist
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
        except catalog_service.glue.exceptions.AlreadyExistsException:
            pass
        
        # Create test data
        df = pd.DataFrame(test_table_data)
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = f"data/{user_id}/{table_name}/test_data.parquet"
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Create Glue table
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'employee_id', 'Type': 'bigint'},
                    {'Name': 'name', 'Type': 'string'},
                    {'Name': 'department', 'Type': 'string'},
                    {'Name': 'salary', 'Type': 'bigint'},
                    {'Name': 'role', 'Type': 'string'},
                    {'Name': 'start_date', 'Type': 'string'}
                ],
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.'
                        'ParquetHiveSerDe'
                    )
                }
            },
            'PartitionKeys': []
        }
        
        catalog_service.glue.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        
        # Test SQL generation
        query = "Show me all employees in the Engineering department"
        result = await catalog_service.process_descriptive_query(
            query=query,
            table_name=table_name,
            user_id=user_id
        )
        
        # Verify SQL generation
        assert result["status"] == "success"
        assert "sql_query" in result
        assert "SELECT" in result["sql_query"].upper()
        assert "FROM" in result["sql_query"].upper()
        assert "WHERE" in result["sql_query"].upper()
        assert "department" in result["sql_query"].lower()
        assert "engineering" in result["sql_query"].lower()
        
        # Verify additional metadata
        assert "explanation" in result
        assert "confidence" in result
        assert "tables_used" in result
        assert "columns_used" in result
        assert "filters" in result
        
    finally:
        # Cleanup
        try:
            catalog_service.glue.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            catalog_service.s3.delete_object(
                Bucket=catalog_service.bucket,
                Key=s3_key
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")

@pytest.mark.asyncio
async def test_complex_sql_generation(catalog_service, test_table_data):
    """Test complex SQL generation functionality."""
    # Create test table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_complex_sql_{timestamp}"
    user_id = "test_user"
    
    try:
        # Create database if it doesn't exist
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
        except catalog_service.glue.exceptions.AlreadyExistsException:
            pass
        
        # Create test data
        df = pd.DataFrame(test_table_data)
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = f"data/{user_id}/{table_name}/test_data.parquet"
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Create Glue table
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'employee_id', 'Type': 'bigint'},
                    {'Name': 'name', 'Type': 'string'},
                    {'Name': 'department', 'Type': 'string'},
                    {'Name': 'salary', 'Type': 'bigint'},
                    {'Name': 'role', 'Type': 'string'},
                    {'Name': 'start_date', 'Type': 'string'}
                ],
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.'
                        'ParquetHiveSerDe'
                    )
                }
            },
            'PartitionKeys': []
        }
        
        catalog_service.glue.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        
        # Test complex SQL generation
        query = (
            "What is the average salary by department, and show me only "
            "departments with average salary above 80000?"
        )
        result = await catalog_service.process_descriptive_query(
            query=query,
            table_name=table_name,
            user_id=user_id
        )
        
        # Verify SQL generation
        assert result["status"] == "success"
        assert "sql_query" in result
        assert "SELECT" in result["sql_query"].upper()
        assert "FROM" in result["sql_query"].upper()
        assert "GROUP BY" in result["sql_query"].upper()
        assert "HAVING" in result["sql_query"].upper()
        assert "AVG" in result["sql_query"].upper()
        assert "salary" in result["sql_query"].lower()
        assert "department" in result["sql_query"].lower()
        
        # Verify additional metadata
        assert "explanation" in result
        assert "confidence" in result
        assert "tables_used" in result
        assert "columns_used" in result
        assert "filters" in result
        
    finally:
        # Cleanup
        try:
            catalog_service.glue.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            catalog_service.s3.delete_object(
                Bucket=catalog_service.bucket,
                Key=s3_key
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}") 