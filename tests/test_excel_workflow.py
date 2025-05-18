"""Test Excel workflow functionality."""

import pytest
import pandas as pd
import boto3
import os
from datetime import datetime
import logging
from io import BytesIO
import json

from src.etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from src.etl_architect_agent_v2.agents.catalog_agent import CatalogAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def test_excel_file(tmp_path):
    """Create a sample Excel file for testing."""
    # Create sample data
    data = {
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'age': [30, 25, 35],
        'department': ['IT', 'HR', 'Finance'],
        'salary': [75000, 65000, 85000],
        'join_date': ['2023-01-15', '2023-02-01', '2023-03-10']
    }
    df = pd.DataFrame(data)
    
    # Save to Excel
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    
    return str(file_path)


@pytest.fixture
def aws_credentials():
    """Ensure AWS credentials are set."""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'AWS_S3_BUCKET'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing AWS credentials: {', '.join(missing)}")
    
    # Verify AWS credentials work
    try:
        s3 = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        s3.list_buckets()
        logger.info("AWS credentials verified successfully")
    except Exception as e:
        pytest.skip(f"AWS credentials verification failed: {str(e)}")


@pytest.fixture
def catalog_service(aws_credentials):
    """Create Catalog Service instance."""
    bucket = os.getenv('AWS_S3_BUCKET')
    if not bucket:
        pytest.skip("Missing AWS_S3_BUCKET environment variable")
    
    logger.info(f"Creating CatalogService with bucket: {bucket}")
    return CatalogService(
        bucket=bucket,
        aws_region=os.getenv('AWS_REGION', 'us-east-1')
    )


@pytest.fixture
def catalog_agent(catalog_service):
    """Create Catalog Agent instance."""
    logger.info("Creating CatalogAgent")
    return CatalogAgent(catalog_service)


@pytest.mark.asyncio
async def test_excel_workflow(test_excel_file, catalog_service, catalog_agent):
    """Test Excel file upload and processing workflow."""
    # Generate unique table name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_{timestamp}"
    logger.info(f"Starting test with table name: {table_name}")
    
    try:
        # Read Excel file into DataFrame
        df = pd.read_excel(test_excel_file)
        logger.info("Successfully read Excel file into DataFrame")
        
        # Process file using catalog agent
        logger.info("Processing file with catalog agent...")
        result = await catalog_agent.process_file_upload(
            df=df,
            file_name="test_data.xlsx",
            table_name=table_name,
            create_new=True
        )
        
        logger.info(f"Processing result: {result}")
        
        # Verify results
        assert result["status"] == "success"
        assert "schema" in result
        assert "quality_metrics" in result
        assert "s3_path" in result
        
        # Verify schema
        schema = result["schema"]
        assert len(schema["columns"]) == 4  # id, name, age, department
        
        # Verify column names
        column_names = {col["name"] for col in schema["columns"]}
        assert column_names == {"id", "name", "age", "department"}
        
        # Verify S3 files exist
        s3 = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        bucket = os.getenv('AWS_S3_BUCKET')
        
        # Check original file
        try:
            s3.head_object(
                Bucket=bucket,
                Key=f"data/{table_name}/test_data.xlsx"
            )
            logger.info("Original Excel file verified in S3")
        except Exception as e:
            logger.error(f"Original file not found in S3: {str(e)}")
            raise
        
        # Check parquet file
        try:
            s3.head_object(
                Bucket=bucket,
                Key=f"data/{table_name}/test_data.parquet"
            )
            logger.info("Parquet file verified in S3")
        except Exception as e:
            logger.error(f"Parquet file not found in S3: {str(e)}")
            raise
        
        # Verify Glue table exists
        glue = boto3.client(
            'glue',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        try:
            table = glue.get_table(
                DatabaseName=catalog_service.database_name,
                Name=table_name
            )
            logger.info(f"Glue table verified: {table['Table']['Name']}")
        except Exception as e:
            logger.error(f"Glue table not found: {str(e)}")
            raise
        
    finally:
        # Cleanup
        logger.info("Starting cleanup...")
        try:
            # Delete S3 files
            s3 = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            bucket = os.getenv('AWS_S3_BUCKET')
            
            # Delete original file
            try:
                s3.delete_object(
                    Bucket=bucket,
                    Key=f"data/{table_name}/test_data.xlsx"
                )
                logger.info("Deleted original Excel file from S3")
            except Exception as e:
                logger.error(f"Error deleting original file: {str(e)}")
            
            # Delete parquet file
            try:
                s3.delete_object(
                    Bucket=bucket,
                    Key=f"data/{table_name}/test_data.parquet"
                )
                logger.info("Deleted parquet file from S3")
            except Exception as e:
                logger.error(f"Error deleting parquet file: {str(e)}")
            
            # Delete Glue table
            glue = boto3.client(
                'glue',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            try:
                glue.delete_table(
                    DatabaseName=catalog_service.database_name,
                    Name=table_name
                )
                logger.info("Deleted Glue table")
            except glue.exceptions.EntityNotFoundException:
                logger.info(f"Table {table_name} not found, skipping deletion")
                
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")


@pytest.mark.asyncio
async def test_error_handling(catalog_service, catalog_agent):
    """Test error handling with invalid file."""
    # Generate unique table name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_invalid_{timestamp}"
    logger.info(f"Testing error handling with table name: {table_name}")
    
    try:
        # Test with invalid DataFrame
        df = pd.DataFrame()  # Empty DataFrame
        
        with pytest.raises(Exception) as exc_info:
            await catalog_agent.process_file_upload(
                df=df,
                file_name="invalid.txt",
                table_name=table_name,
                create_new=True
            )
        
        assert "Error processing file upload" in str(exc_info.value)
        
    finally:
        # Cleanup
        logger.info("Starting cleanup for error handling test...")
        try:
            # Delete S3 files if they were created
            s3 = boto3.client(
                's3',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            bucket = os.getenv('AWS_S3_BUCKET')
            
            # Delete original file
            try:
                s3.delete_object(
                    Bucket=bucket,
                    Key=f"data/{table_name}/invalid.txt"
                )
                logger.info("Deleted invalid file from S3")
            except Exception as e:
                logger.error(f"Error deleting invalid file: {str(e)}")
            
            # Delete parquet file if it exists
            try:
                s3.delete_object(
                    Bucket=bucket,
                    Key=f"data/{table_name}/invalid.parquet"
                )
                logger.info("Deleted invalid parquet file from S3")
            except Exception as e:
                logger.error(f"Error deleting invalid parquet file: {str(e)}")
            
            # Delete Glue table if it exists
            glue = boto3.client(
                'glue',
                region_name=os.getenv('AWS_REGION', 'us-east-1')
            )
            try:
                glue.delete_table(
                    DatabaseName=catalog_service.database_name,
                    Name=table_name
                )
                logger.info("Deleted invalid Glue table")
            except glue.exceptions.EntityNotFoundException:
                logger.info(f"Table {table_name} not found, skipping deletion")
                
        except Exception as e:
            logger.error(f"Cleanup error in error handling test: {str(e)}")


@pytest.mark.asyncio
async def test_file_upload(test_excel_file, catalog_service, catalog_agent):
    """Test basic file upload functionality."""
    # Generate unique table name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_{timestamp}"
    logger.info(f"Starting test with table name: {table_name}")
    
    # Read Excel file into DataFrame
    df = pd.read_excel(test_excel_file)
    logger.info("Successfully read Excel file into DataFrame")
    
    # Upload file to S3
    s3_key = f"data/{table_name}/test_data.xlsx"
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    
    s3 = boto3.client(
        's3',
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    bucket = os.getenv('AWS_S3_BUCKET')
    
    # Upload file
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=buffer.getvalue()
    )
    logger.info(f"Uploaded file to S3: {s3_key}")
    
    # Verify file exists
    try:
        s3.head_object(
            Bucket=bucket,
            Key=s3_key
        )
        logger.info("File verified in S3")
    except Exception as e:
        logger.error(f"File not found in S3: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_file_upload_with_parquet_and_schema(test_excel_file, catalog_service, catalog_agent):
    """Test file upload with Parquet conversion and schema generation."""
    # Generate unique table name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_{timestamp}"
    logger.info(f"Starting test with table name: {table_name}")
    
    # Read Excel file into DataFrame
    df = pd.read_excel(test_excel_file)
    logger.info("Successfully read Excel file into DataFrame")
    
    # Upload Excel file to S3
    excel_s3_key = f"data/{table_name}/test_data.xlsx"
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    
    s3 = boto3.client(
        's3',
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    bucket = os.getenv('AWS_S3_BUCKET')
    
    # Upload Excel file
    s3.put_object(
        Bucket=bucket,
        Key=excel_s3_key,
        Body=buffer.getvalue()
    )
    logger.info(f"Uploaded Excel file to S3: {excel_s3_key}")
    
    # Convert to Parquet and upload
    parquet_s3_key = f"data/{table_name}/test_data.parquet"
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    
    s3.put_object(
        Bucket=bucket,
        Key=parquet_s3_key,
        Body=parquet_buffer.getvalue()
    )
    logger.info(f"Uploaded Parquet file to S3: {parquet_s3_key}")
    
    # Generate and upload schema
    schema = {
        "file_name": "test_data.xlsx",
        "table_name": table_name,
        "columns": [
            {
                "name": "id",
                "type": "int64",
                "description": "Unique identifier",
                "nullable": False
            },
            {
                "name": "name",
                "type": "object",
                "description": "Person's name",
                "nullable": False
            },
            {
                "name": "age",
                "type": "int64",
                "description": "Person's age",
                "nullable": False
            },
            {
                "name": "department",
                "type": "object",
                "description": "Department name",
                "nullable": False
            }
        ],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    schema_s3_key = f"metadata/schema/{table_name}.json"
    s3.put_object(
        Bucket=bucket,
        Key=schema_s3_key,
        Body=json.dumps(schema, indent=2)
    )
    logger.info(f"Uploaded schema to S3: {schema_s3_key}")
    
    # Verify all files exist
    try:
        # Check Excel file
        s3.head_object(
            Bucket=bucket,
            Key=excel_s3_key
        )
        logger.info("Excel file verified in S3")
        
        # Check Parquet file
        s3.head_object(
            Bucket=bucket,
            Key=parquet_s3_key
        )
        logger.info("Parquet file verified in S3")
        
        # Check schema file
        s3.head_object(
            Bucket=bucket,
            Key=schema_s3_key
        )
        logger.info("Schema file verified in S3")
        
    except Exception as e:
        logger.error(f"File verification failed: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_create_glue_catalog(test_excel_file, catalog_service, catalog_agent):
    """Test creating Glue catalog table from Parquet and schema."""
    # Generate unique table name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_{timestamp}"
    logger.info(f"Starting test with table name: {table_name}")
    
    # Read Excel file into DataFrame
    df = pd.read_excel(test_excel_file)
    logger.info("Successfully read Excel file into DataFrame")
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    
    # Upload Excel file to S3
    excel_s3_key = f"data/{table_name}/test_data.xlsx"
    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    
    s3 = boto3.client(
        's3',
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    bucket = os.getenv('AWS_S3_BUCKET')
    
    # Upload Excel file
    s3.put_object(
        Bucket=bucket,
        Key=excel_s3_key,
        Body=buffer.getvalue()
    )
    logger.info(f"Uploaded Excel file to S3: {excel_s3_key}")
    
    # Convert to Parquet and upload
    parquet_s3_key = f"data/{table_name}/test_data.parquet"
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    
    s3.put_object(
        Bucket=bucket,
        Key=parquet_s3_key,
        Body=parquet_buffer.getvalue()
    )
    logger.info(f"Uploaded Parquet file to S3: {parquet_s3_key}")
    
    # Generate schema from DataFrame
    schema = {
        "file_name": "test_data.xlsx",
        "table_name": table_name,
        "columns": [
            {
                "name": col,
                "type": str(df[col].dtype),
                "description": f"Column {col} from {table_name}",
                "nullable": bool(df[col].isnull().any())  # Convert numpy.bool_ to Python bool
            }
            for col in df.columns
        ],
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }
    
    schema_s3_key = f"metadata/schema/{table_name}.json"
    s3.put_object(
        Bucket=bucket,
        Key=schema_s3_key,
        Body=json.dumps(schema, indent=2)
    )
    logger.info(f"Uploaded schema to S3: {schema_s3_key}")
    
    # Create Glue catalog table
    glue = boto3.client(
        'glue',
        region_name=os.getenv('AWS_REGION', 'us-east-1')
    )
    
    # Create database if it doesn't exist
    try:
        glue.get_database(Name=catalog_service.database_name)
    except glue.exceptions.EntityNotFoundException:
        glue.create_database(
            DatabaseInput={
                'Name': catalog_service.database_name,
                'Description': 'Test database for catalog service'
            }
        )
        logger.info(f"Created Glue database: {catalog_service.database_name}")
    
    # Create table
    table_input = {
        'Name': table_name,
        'Description': f'Test table created from {table_name}',
        'TableType': 'EXTERNAL_TABLE',
        'Parameters': {
            'classification': 'parquet',
            'typeOfData': 'file'
        },
        'StorageDescriptor': {
            'Columns': [
                {
                    'Name': col['name'],
                    'Type': col['type'],
                    'Comment': col['description']
                }
                for col in schema['columns']
            ],
            'Location': f's3://{bucket}/{parquet_s3_key}',
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
    
    glue.create_table(
        DatabaseName=catalog_service.database_name,
        TableInput=table_input
    )
    logger.info(f"Created Glue table: {table_name}")
    
    # Verify table exists
    try:
        table = glue.get_table(
            DatabaseName=catalog_service.database_name,
            Name=table_name
        )
        logger.info(f"Verified Glue table: {table['Table']['Name']}")
        
        # Verify table schema
        columns = {
            col['Name']: col['Type'] 
            for col in table['Table']['StorageDescriptor']['Columns']
        }
        
        # Verify all columns from DataFrame are in Glue table
        for col in df.columns:
            assert col in columns, f"Column {col} missing from Glue table"
            assert columns[col] == str(df[col].dtype), \
                f"Column {col} type mismatch: {columns[col]} != {df[col].dtype}"
        
        logger.info("Verified table schema matches DataFrame schema")
        
    except Exception as e:
        logger.error(f"Error verifying Glue table: {str(e)}")
        raise 