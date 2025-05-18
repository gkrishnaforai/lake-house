"""Test Catalog Service with optimized Glue table creation."""

import pytest
import pandas as pd
import boto3
import os
from datetime import datetime
import logging
from fastapi import UploadFile
from io import BytesIO

from src.etl_architect_agent_v2.backend.services.catalog_service import CatalogService
from src.etl_architect_agent_v2.backend.services.glue_service import GlueService

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
        'department': ['IT', 'HR', 'Finance']
    }
    df = pd.DataFrame(data)
    
    # Save to Excel
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    
    return str(file_path)

@pytest.fixture
def aws_credentials():
    """Ensure AWS credentials are set."""
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'AWS_S3_BUCKET']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing AWS credentials: {', '.join(missing)}")
    
    # Verify AWS credentials work
    try:
        s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
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

@pytest.mark.asyncio
async def test_excel_workflow(test_excel_file, catalog_service):
    """Test complete workflow with Excel file using optimized catalog service."""
    # Generate unique table name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_{timestamp}"
    logger.info(f"Starting test with table name: {table_name}")
    
    try:
        # Create UploadFile object from the Excel file
        with open(test_excel_file, 'rb') as f:
            content = f.read()
            file = UploadFile(
                filename="test_data.xlsx",
                file=BytesIO(content)
            )
        
        logger.info("Uploading file and creating table...")
        # Upload file and create table
        result = await catalog_service.upload_file(
            file=file,
            table_name=table_name,
            create_new=True
        )
        
        logger.info(f"Upload result: {result}")
        
        # Verify results
        assert result["status"] == "success"
        assert "schema" in result
        assert "glue_table" in result
        
        # Verify schema
        schema = result["schema"]
        assert len(schema["fields"]) == 4
        
        # Verify column names
        column_names = {field["name"] for field in schema["fields"]}
        assert column_names == {"id", "name", "age", "department"}
        
        # Verify Glue table
        glue_table = result["glue_table"]
        assert glue_table["database"] == catalog_service.database_name
        assert glue_table["table"] == table_name
        assert glue_table["format"] == "parquet"
        
        # Verify S3 files exist
        s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
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
        glue = boto3.client('glue', region_name=os.getenv('AWS_REGION', 'us-east-1'))
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
            s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION', 'us-east-1'))
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
            glue = boto3.client('glue', region_name=os.getenv('AWS_REGION', 'us-east-1'))
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
async def test_error_handling(catalog_service):
    """Test error handling with invalid file."""
    # Test with invalid file
    file = UploadFile(
        filename="invalid.txt",
        file=BytesIO(b"invalid content")
    )
    
    with pytest.raises(Exception) as exc_info:
        await catalog_service.upload_file(
            file=file,
            table_name="invalid_table",
            create_new=True
        )
    
    assert "Unsupported file format" in str(exc_info.value) 