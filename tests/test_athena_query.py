"""Test Athena SQL query functionality."""

import pytest
import pandas as pd
import boto3
import os
from datetime import datetime
import logging
from io import BytesIO
import asyncio
from fastapi.testclient import TestClient
import time
import traceback

# Assuming your FastAPI app instance is named 'app' in main.py
from src.etl_architect_agent_v2.api.main import app as fastapi_app

from src.etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from src.etl_architect_agent_v2.agents.catalog_agent import CatalogAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def pandas_to_athena_type(dtype_str):
    """Convert pandas dtype string to Athena/Glue compatible type string."""
    if dtype_str == 'int64':
        return 'bigint'
    elif dtype_str == 'float64':
        return 'double'
    elif dtype_str == 'object':
        return 'string'
    elif dtype_str == 'bool':
        return 'boolean'
    elif dtype_str.startswith('datetime64'):
        return 'timestamp'
    else:
        return 'string'


@pytest.fixture(scope="module")  # Scope to module if app setup is expensive
def client():
    """Provides a FastAPI TestClient instance."""
    with TestClient(fastapi_app) as c:
        yield c


@pytest.fixture
def test_excel_file(tmp_path):
    """Create a sample Excel file for testing."""
    data = {
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'age': [30, 25, 35],
        'department': ['IT', 'HR', 'Finance'],
        'salary': [75000, 65000, 85000],
        'join_date': ['2023-01-15', '2023-02-01', '2023-03-10']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)


@pytest.fixture
def test_excel_file_variant_content(tmp_path):
    """Create a sample Excel file with different content for testing."""
    data = {
        'id': [101, 102, 103, 104],
        'name': ["Alice", "Bob", "Charlie", "David"],
        'age': [28, 32, 45, 38],
        'department': ["Engineering", "Sales", "Engineering", "Marketing"],
        'salary': [90000, 80000, 110000, 70000],
        'role': ["Engineer II", "Sales Rep", "Senior Engineer", "Marketing Specialist"],
        'start_date': ['2022-05-01', '2021-11-15', '2020-03-10', '2023-01-20']
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "test_data_variant.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)


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
    logger.info(
        "Creating CatalogService with bucket: %s", bucket
    )
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


@pytest.mark.asyncio
async def test_descriptive_query_manual_setup(
    client, test_excel_file, catalog_service, catalog_agent
):
    """Test querying data using descriptive language with manual S3/Glue setup."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_manual_{timestamp}"
    user_id = "test_user"
    
    print("\n=== Starting test_descriptive_query_manual_setup ===")
    print(f"Table name: {table_name}")
    print(f"User ID: {user_id}")
    
    try:
        # Create database if it doesn't exist
        print("\n=== Creating database if it doesn't exist ===")
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
            print(f"Created database: user_{user_id}")
        except catalog_service.glue.exceptions.AlreadyExistsException:
            print(f"Database user_{user_id} already exists")
        
        # Upload test file to S3
        print("\n=== Uploading test file to S3 ===")
        # Create test data
        data = {
            'employee_id': [1, 2, 3, 4],
            'name': ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"],
            'department': ["Engineering", "Sales", "Engineering", "Marketing"],
            'salary': [90000, 80000, 110000, 70000],
            'role': [
                "Engineer II", "Sales Rep", "Senior Engineer", 
                "Marketing Specialist"
            ],
            'start_date': [
                '2022-05-01', '2021-11-15', '2020-03-10', 
                '2023-01-20'
            ]
        }
        df = pd.DataFrame(data)
        
        print("\n=== Creating test data ===")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = f"data/{user_id}/{table_name}/test_data.parquet"
        print(f"\n=== Uploading to S3: {s3_key} ===")
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Create Glue table
        print("\n=== Creating Glue table ===")
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
                    'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                    )
                }
            }
        }
        
        catalog_service.glue.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        print(f"Created table: {table_name}")
        
        # Wait for table to be available
        print("\n=== Waiting for table to be available ===")
        await asyncio.sleep(10)
        
        # Test descriptive query
        print("\n=== Testing descriptive query ===")
        descriptive_query = (
            "Show me all employees in the Engineering department"
        )
        print(f"Query: {descriptive_query}")
        
        # Get table schema for SQL generation
        schema = await catalog_service.get_table_schema(table_name, user_id)
        print("Table schema:", schema)
        
        # Process descriptive query using catalog service
        result = await catalog_service.process_descriptive_query(
            query=descriptive_query,
            table_name=table_name,
            user_id=user_id
        )
        
        print("Query result:", result)
        
        # Verify SQL generation output
        assert result["status"] == "success"
        assert "sql_query" in result
        assert "explanation" in result
        assert "confidence" in result
        assert "tables_used" in result
        assert "columns_used" in result
        assert "filters" in result
        
        # Verify SQL query content
        sql_query = result["sql_query"]
        assert "SELECT" in sql_query.upper()
        assert "FROM" in sql_query.upper()
        assert "WHERE" in sql_query.upper()
        assert "department" in sql_query.lower()
        assert "engineering" in sql_query.lower()
        
        # Verify explanation
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0
        
        # Verify confidence score
        assert 0 <= result["confidence"] <= 1
        
        # Verify tables and columns used
        assert isinstance(result["tables_used"], list)
        assert table_name in result["tables_used"]
        assert isinstance(result["columns_used"], list)
        assert "department" in result["columns_used"]
        
        # Verify filters
        assert isinstance(result["filters"], dict)
        assert "department" in result["filters"]
        assert result["filters"]["department"].lower() == "engineering"
        
    except Exception as e:
        print("\n=== Test failed with exception ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\n=== Cleaning up test resources ===")
        try:
            catalog_service.glue.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            print(f"Deleted table: {table_name}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


@pytest.mark.asyncio
async def test_descriptive_query_backend_setup_variant_data(
    client, test_excel_file_variant_content, catalog_agent, catalog_service,
    aws_credentials
):
    """Test querying data using descriptive language with backend setup."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_backend_{timestamp}"
    user_id = "test_user"
    
    print("\n=== Starting test_descriptive_query_backend_setup_variant_data ===")
    print(f"Table name: {table_name}")
    print(f"User ID: {user_id}")
    
    try:
        # Create database if it doesn't exist
        print("\n=== Creating database if it doesn't exist ===")
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
            print(f"Created database: user_{user_id}")
        except catalog_service.glue.exceptions.AlreadyExistsException:
            print(f"Database user_{user_id} already exists")
        
        # Upload file through API
        print("\n=== Uploading file through API ===")
        with open(test_excel_file_variant_content, 'rb') as f:
            response = client.post(
                f"/api/catalog/tables/{table_name}/files",
                files={"file": ("test_data_variant.xlsx", f)},
                params={"user_id": user_id}
            )
        
        print(f"Upload response status code: {response.status_code}")
        print(f"Upload response content: {response.json()}")
        
        assert response.status_code == 200
        upload_result = response.json()
        assert upload_result["status"] == "success"
        
        # Wait for processing
        print("\n=== Waiting for processing ===")
        await asyncio.sleep(10)
        
        # Test descriptive query
        print("\n=== Testing descriptive query ===")
        descriptive_query = (
            "Show me all employees in the Engineering department"
        )
        print(f"Query: {descriptive_query}")
        
        # Get table schema for SQL generation
        schema = await catalog_service.get_table_schema(table_name, user_id)
        print("Table schema:", schema)
        
        # Process descriptive query using catalog service
        result = await catalog_service.process_descriptive_query(
            query=descriptive_query,
            table_name=table_name,
            user_id=user_id
        )
        
        print("Query result:", result)
        
        # Verify SQL generation output
        assert result["status"] == "success"
        assert "sql_query" in result
        assert "explanation" in result
        assert "confidence" in result
        assert "tables_used" in result
        assert "columns_used" in result
        assert "filters" in result
        
        # Verify SQL query content
        sql_query = result["sql_query"]
        assert "SELECT" in sql_query.upper()
        assert "FROM" in sql_query.upper()
        assert "WHERE" in sql_query.upper()
        assert "department" in sql_query.lower()
        assert "engineering" in sql_query.lower()
        
        # Verify explanation
        assert isinstance(result["explanation"], str)
        assert len(result["explanation"]) > 0
        
        # Verify confidence score
        assert 0 <= result["confidence"] <= 1
        
        # Verify tables and columns used
        assert isinstance(result["tables_used"], list)
        assert table_name in result["tables_used"]
        assert isinstance(result["columns_used"], list)
        assert "department" in result["columns_used"]
        
        # Verify filters
        assert isinstance(result["filters"], dict)
        assert "department" in result["filters"]
        assert result["filters"]["department"].lower() == "engineering"
        
    except Exception as e:
        print("\n=== Test failed with exception ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\n=== Cleaning up test resources ===")
        try:
            catalog_service.glue.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            print(f"Deleted table: {table_name}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


@pytest.mark.asyncio
async def test_manual_s3_then_backend_processing(
    client, test_excel_file_variant_content, catalog_service, aws_credentials
):
    """Test manual S3 upload followed by backend processing."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_manual_s3_{timestamp}"
    user_id = "test_user"
    
    print(f"\n=== Starting test_manual_s3_then_backend_processing ===")
    print(f"Table name: {table_name}")
    print(f"User ID: {user_id}")
    
    try:
        # Create database if it doesn't exist
        print("\n=== Creating database if it doesn't exist ===")
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
            print(f"Created database: user_{user_id}")
        except catalog_service.glue.exceptions.AlreadyExistsException:
            print(f"Database user_{user_id} already exists")
        
        # Upload file to S3 manually
        print("\n=== Uploading file to S3 manually ===")
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        bucket = os.getenv('AWS_S3_BUCKET')
        
        with open(test_excel_file_variant_content, 'rb') as f:
            s3_client.put_object(
                Bucket=bucket,
                Key=f"originals/{user_id}/{table_name}/test_data_variant.xlsx",
                Body=f.read()
            )
        
        # Process through backend
        print("\n=== Processing through backend ===")
        response = client.post(
            f"/api/catalog/tables/{table_name}/process",
            json={"user_id": user_id}
        )
        
        print(f"Process response status code: {response.status_code}")
        print(f"Process response content: {response.json()}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        
        # Wait for processing
        print("\n=== Waiting for processing ===")
        await asyncio.sleep(10)
        
        # Test descriptive query
        print("\n=== Testing descriptive query ===")
        descriptive_query = "Show me all employees in the Engineering department"
        print(f"Query: {descriptive_query}")
        
        response = client.post(
            "/api/catalog/descriptive_query",
            json={
                "query": descriptive_query,
                "table_name": table_name,
                "user_id": user_id
            }
        )
        
        print(f"Query response status code: {response.status_code}")
        print(f"Query response content: {response.json()}")
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "query" in result
        assert "results" in result
        
    except Exception as e:
        print(f"\n=== Test failed with exception ===")
        print(f"Exception type: {type(e)}")
        print(f"Exception message: {str(e)}")
        raise
    finally:
        # Cleanup
        print("\n=== Cleaning up test resources ===")
        try:
            catalog_service.glue.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            print(f"Deleted table: {table_name}")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")


@pytest.mark.asyncio
async def test_backend_upload_and_artifact_creation(
    client, test_excel_file_variant_content, catalog_service, aws_credentials
):
    """Test backend file upload and artifact creation."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_backend_upload_{timestamp}"
    user_id = "test_user"
    
    # Upload file through API
    with open(test_excel_file_variant_content, 'rb') as f:
        response = client.post(
            f"/api/catalog/tables/{table_name}/files",
            files={"file": ("test_data_variant.xlsx", f)},
            params={"user_id": user_id}
        )
    
    assert response.status_code == 200
    upload_result = response.json()
    assert upload_result["status"] == "success"
    
    # Wait for processing
    await asyncio.sleep(10)
    
    try:
        # Verify artifacts were created
        s3_client = boto3.client(
            's3',
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )
        bucket = os.getenv('AWS_S3_BUCKET')
        
        # Check for parquet file
        parquet_key = f"data/{user_id}/{table_name}/test_data_variant.parquet"
        try:
            s3_client.head_object(Bucket=bucket, Key=parquet_key)
            logger.info(f"Found parquet file at {parquet_key}")
        except Exception as e:
            logger.error(f"Parquet file not found: {str(e)}")
            pytest.fail("Parquet file not created")
        
        # Check for schema file
        schema_key = f"metadata/schema/{user_id}/{table_name}.json"
        try:
            s3_client.head_object(Bucket=bucket, Key=schema_key)
            logger.info(f"Found schema file at {schema_key}")
        except Exception as e:
            logger.error(f"Schema file not found: {str(e)}")
            pytest.fail("Schema file not created")
        
    except Exception as e:
        logger.error(f"Error in artifact verification: {str(e)}")
        pytest.fail(f"Artifact verification failed: {str(e)}")


@pytest.mark.asyncio
async def test_sql_query(client, test_excel_file, catalog_service, catalog_agent):
    """Test direct SQL query execution."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_sql_{timestamp}"
    user_id = "test_user"
    
    # Upload file through API
    with open(test_excel_file, 'rb') as f:
        response = client.post(
            f"/api/catalog/tables/{table_name}/files",
            files={"file": ("test_data.xlsx", f)},
            params={"user_id": user_id}
        )
    
    assert response.status_code == 200
    upload_result = response.json()
    assert upload_result["status"] == "success"
    
    # Wait for processing
    await asyncio.sleep(10)
    
    try:
        # Execute SQL query
        sql_query = f"SELECT * FROM {table_name} WHERE department = 'IT'"
        response = client.post(
            "/api/catalog/query",
            json={
                "query": sql_query,
                "user_id": user_id
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "data" in result
        assert len(result["data"]) > 0
        
        # Verify results
        it_employees = [
            row for row in result["data"]
            if row["department"] == "IT"
        ]
        assert len(it_employees) > 0
        
    except Exception as e:
        logger.error(f"Error in SQL query test: {str(e)}")
        pytest.fail(f"SQL query test failed: {str(e)}")


def test_descriptive_query_backend_descriptive_api(
    client, test_excel_file_variant_content, catalog_service, aws_credentials
):
    """Test descriptive query through API endpoint."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_descriptive_api_{timestamp}"
    user_id = "test_user"
    
    # Upload file through API
    with open(test_excel_file_variant_content, 'rb') as f:
        response = client.post(
            f"/api/catalog/tables/{table_name}/files",
            files={"file": ("test_data_variant.xlsx", f)},
            params={"user_id": user_id}
        )
    
    assert response.status_code == 200
    upload_result = response.json()
    assert upload_result["status"] == "success"
    
    # Wait for processing
    time.sleep(10)
    
    try:
        # Execute descriptive query through API
        response = client.post(
            "/api/catalog/query",
            json={
                "query": (
                    "Show me all employees in the Engineering department"
                ),
                "user_id": user_id
            }
        )
        
        assert response.status_code == 200
        result = response.json()
        assert result["status"] == "success"
        assert "data" in result
        assert len(result["data"]) > 0
        
        # Verify results
        eng_employees = [
            row for row in result["data"]
            if row["department"] == "Engineering"
        ]
        assert len(eng_employees) > 0
        
    except Exception as e:
        logger.error(f"Error in descriptive API test: {str(e)}")
        raise


def test_list_files_originals_with_user(
    client, test_excel_file_variant_content, aws_credentials
):
    """Test listing files in originals directory with user ID."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_list_files_{timestamp}"
    user_id = "test_user"
    
    # Upload file through API
    with open(test_excel_file_variant_content, 'rb') as f:
        response = client.post(
            f"/api/catalog/tables/{table_name}/files",
            files={"file": ("test_data_variant.xlsx", f)},
            params={"user_id": user_id}
        )
    
    assert response.status_code == 200
    upload_result = response.json()
    assert upload_result["status"] == "success"
    
    # Wait for processing
    time.sleep(10)
    
    try:
        # List files through API
        response = client.get(
            "/api/files",
            params={"prefix": "originals/", "user_id": user_id}
        )
        
        assert response.status_code == 200
        files = response.json()
        assert isinstance(files, list)
        assert len(files) > 0
        
        # Verify user's files are present
        user_files = [
            f for f in files
            if f["s3_path"].startswith(f"originals/{user_id}/")
        ]
        assert len(user_files) > 0
        
    except Exception as e:
        logger.error(f"Error in list files test: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_descriptive_query_with_organization_data(
    client, catalog_service, aws_credentials
):
    """Test descriptive query with organization data."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_org_table_{timestamp}"
    user_id = "test_user"
    
    print("\n=== Starting test_descriptive_query_with_organization_data ===")
    print(f"Table name: {table_name}")
    print(f"User ID: {user_id}")
    
    try:
        # Create database if it doesn't exist
        print("\n=== Creating database if it doesn't exist ===")
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
            print(f"Created database: user_{user_id}")
        except catalog_service.glue.exceptions.AlreadyExistsException:
            print(f"Database user_{user_id} already exists")
        
        # Create test data with organization information
        print("\n=== Creating test data with organization information ===")
        data = {
            'organization_id': [1, 2, 3, 4],
            'organization_name': [
                "Tech Corp", "Data Systems", "Cloud Solutions", "AI Innovations"
            ],
            'industry': ["Technology", "Data", "Cloud", "AI"],
            'employee_count': [500, 300, 200, 150],
            'founded_year': [2010, 2015, 2018, 2020],
            'headquarters': [
                "San Francisco", "New York", "Seattle", "Boston"
            ]
        }
        df = pd.DataFrame(data)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = f"data/{user_id}/{table_name}/org_data.parquet"
        print(f"\n=== Uploading to S3: {s3_key} ===")
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Create Glue table
        print("\n=== Creating Glue table ===")
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'organization_id', 'Type': 'bigint'},
                    {'Name': 'organization_name', 'Type': 'string'},
                    {'Name': 'industry', 'Type': 'string'},
                    {'Name': 'employee_count', 'Type': 'bigint'},
                    {'Name': 'founded_year', 'Type': 'bigint'},
                    {'Name': 'headquarters', 'Type': 'string'}
                ],
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                    )
                }
            }
        }
        
        catalog_service.glue.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        print(f"Created table: {table_name}")
        
        # Wait for table to be available
        print("\n=== Waiting for table to be available ===")
        await asyncio.sleep(10)
        
        # Test descriptive query
        print("\n=== Testing descriptive query ===")
        descriptive_query = "Show me all organizations in the Technology industry"
        print(f"Query: {descriptive_query}")
        
        # Get table schema for SQL generation
        schema = await catalog_service.get_table_schema(table_name, user_id)
        print("Table schema:", schema)
        
        # Process descriptive query using catalog service
        result = await catalog_service.process_descriptive_query(
            query=descriptive_query,
            table_name=table_name,
            user_id=user_id
        )
        
        print("Query result:", result)
        
        # Verify the result
        assert result["status"] == "success", f"Query failed: {result.get('error')}"
        assert "results" in result, "No results in response"
        assert len(result["results"]) > 0, "No results returned"
        
        # Verify SQL query contains correct table and column references
        assert table_name in result["sql_query"], "Table name not in SQL query"
        assert "organization_name" in result["sql_query"], "organization_name column not in SQL query"
        assert "industry" in result["sql_query"], "industry column not in SQL query"
        
        # Clean up
        print("\n=== Cleaning up ===")
        catalog_service.glue.delete_table(
            DatabaseName=f"user_{user_id}",
            Name=table_name
        )
        print(f"Deleted table: {table_name}")
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        raise


@pytest.mark.asyncio
async def test_descriptive_query_sample_columns(
    client, catalog_service, aws_credentials
):
    """Test descriptive query to show 3 columns in a sample of 10 rows."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_sample_table_{timestamp}"
    user_id = "test_user"
    
    print("\n=== Starting test_descriptive_query_sample_columns ===")
    print(f"Table name: {table_name}")
    print(f"User ID: {user_id}")
    
    try:
        # Create database if it doesn't exist
        print("\n=== Creating database if it doesn't exist ===")
        try:
            catalog_service.glue.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
            print(f"Created database: user_{user_id}")
        except catalog_service.glue.exceptions.AlreadyExistsException:
            print(f"Database user_{user_id} already exists")
        
        # Create test data
        print("\n=== Creating test data ===")
        data = {
            'employee_id': list(range(1, 21)),  # 20 rows of data
            'name': [f"Employee {i}" for i in range(1, 21)],
            'department': ["Engineering", "Sales", "Marketing", "IT"] * 5,
            'salary': [80000 + i * 1000 for i in range(20)],
            'role': ["Engineer", "Sales Rep", "Marketing Specialist", "Developer"] * 5,
            'start_date': [f"2023-{i:02d}-01" for i in range(1, 21)]
        }
        df = pd.DataFrame(data)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = f"data/{user_id}/{table_name}/sample_data.parquet"
        print(f"\n=== Uploading to S3: {s3_key} ===")
        try:
            catalog_service.s3.put_object(
                Bucket=catalog_service.bucket,
                Key=s3_key,
                Body=parquet_buffer.getvalue()
            )
            print(f"Successfully uploaded to S3: s3://{catalog_service.bucket}/{s3_key}")
        except Exception as e:
            print(f"Error uploading to S3: {str(e)}")
            raise
        
        # Create Glue table
        print("\n=== Creating Glue table ===")
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
                    'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe'
                    )
                }
            }
        }
        
        try:
            catalog_service.glue.create_table(
                DatabaseName=f"user_{user_id}",
                TableInput=table_input
            )
            print(f"Created table: {table_name}")
        except Exception as e:
            print(f"Error creating Glue table: {str(e)}")
            raise
        
        # Wait for table to be available and verify data
        print("\n=== Waiting for table to be available ===")
        await asyncio.sleep(20)  # Increased wait time
        
        # Verify table exists and has data
        try:
            table = catalog_service.glue.get_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            print(f"Table verification successful: {table['Table']['Name']}")
            print(f"Table location: {table['Table']['StorageDescriptor']['Location']}")
            
            # Verify data in S3
            try:
                s3_response = catalog_service.s3.get_object(
                    Bucket=catalog_service.bucket,
                    Key=s3_key
                )
                print(f"S3 object size: {s3_response['ContentLength']} bytes")
            except Exception as e:
                print(f"Error verifying S3 data: {str(e)}")
                raise
        except Exception as e:
            print(f"Error verifying table: {str(e)}")
            raise
        
        # Test descriptive query
        print("\n=== Testing descriptive query ===")
        descriptive_query = "Show me 3 columns in a sample of 10 rows"
        print(f"Query: {descriptive_query}")
        
        # Get table schema for SQL generation
        schema = await catalog_service.get_table_schema(table_name, user_id)
        print("Table schema:", schema)
        
        # Process descriptive query using catalog service
        result = await catalog_service.process_descriptive_query(
            query=descriptive_query,
            table_name=table_name,
            user_id=user_id
        )
        
        print("Query result:", result)
        
        # Verify the result
        assert result["status"] == "success", f"Query failed: {result.get('error')}"
        assert "results" in result, "No results in response"
        assert len(result["results"]) > 0, "No results returned"
        
        # Verify SQL query contains correct table and limit
        assert table_name in result["sql_query"], "Table name not in SQL query"
        assert "LIMIT 10" in result["sql_query"].upper(), "LIMIT 10 not in SQL query"
        
    except Exception as e:
        print(f"\n=== Test failed with error ===")
        print(f"Error type: {type(e)}")
        print(f"Error message: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        raise
    # finally:
    #     # Cleanup
    #     print("\n=== Cleaning up test resources ===")
    #     try:
    #         catalog_service.glue.delete_table(
    #             DatabaseName=f"user_{user_id}",
    #             Name=table_name
    #         )
    #         print(f"Deleted table: {table_name}")
    #         
    #         # Delete S3 objects
    #         catalog_service.s3.delete_object(
    #             Bucket=catalog_service.bucket,
    #             Key=s3_key
    #         )
    #         print(f"Deleted S3 object: {s3_key}")
    #     except Exception as e:
    #         print(f"Error during cleanup: {str(e)}") 