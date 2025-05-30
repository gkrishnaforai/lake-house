import pytest
import pandas as pd
import os
from fastapi.testclient import TestClient
from etl_architect_agent_v2.api.main import app
import boto3
from moto import mock_s3, mock_glue, mock_athena
import json
import uuid

# Create test client
client = TestClient(app)

@pytest.fixture
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
    os.environ['AWS_SECURITY_TOKEN'] = 'testing'
    os.environ['AWS_SESSION_TOKEN'] = 'testing'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

@pytest.fixture
def s3(aws_credentials):
    with mock_s3():
        yield boto3.client('s3', region_name='us-east-1')

@pytest.fixture
def glue(aws_credentials):
    with mock_glue():
        yield boto3.client('glue', region_name='us-east-1')

@pytest.fixture
def athena(aws_credentials):
    with mock_athena():
        yield boto3.client('athena', region_name='us-east-1')

@pytest.fixture
def test_bucket(s3):
    """Create test S3 bucket."""
    bucket_name = 'test-bucket'
    s3.create_bucket(Bucket=bucket_name)
    return bucket_name

@pytest.fixture
def test_database(glue):
    """Create test Glue database."""
    database_name = 'test_database'
    glue.create_database(
        DatabaseInput={
            'Name': database_name,
            'Description': 'Test database'
        }
    )
    return database_name

def create_test_excel():
    """Create a test Excel file with sample data."""
    df = pd.DataFrame({
        'id': range(1, 6),
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'age': [25, 30, 35, 28, 32],
        'department': ['IT', 'HR', 'Finance', 'Marketing', 'Sales']
    })
    
    # Create test directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    # Save to Excel
    file_path = 'test_data/test_data.xlsx'
    df.to_excel(file_path, index=False)
    return file_path

@pytest.mark.asyncio
async def test_end_to_end_flow(s3, glue, athena, test_bucket, test_database):
    """Test the complete ETL flow from file upload to query execution."""
    
    # 1. Create test Excel file
    test_file_path = create_test_excel()
    
    # 2. Upload file
    with open(test_file_path, 'rb') as f:
        response = client.post(
            "/api/upload",
            files={"file": ("test_data.xlsx", f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")},
            headers={"X-User-Id": "test_user"}
        )
    assert response.status_code == 200
    upload_response = response.json()
    assert upload_response["status"] == "success"
    
    # 3. Verify S3 upload
    s3_response = s3.list_objects_v2(Bucket=test_bucket, Prefix="raw/")
    assert len(s3_response.get('Contents', [])) > 0
    
    # 4. Verify Glue table creation
    tables = glue.get_tables(DatabaseName=test_database)
    assert len(tables.get('TableList', [])) > 0
    
    # 5. Test descriptive query
    query_response = client.post(
        "/api/catalog/descriptive_query",
        json={
            "query": "Show me all employees in the IT department",
            "table_name": upload_response["table_name"]
        }
    )
    assert query_response.status_code == 200
    query_result = query_response.json()
    assert query_result["status"] == "success"
    assert len(query_result.get("results", [])) > 0
    
    # 6. Test agent chat for ETL project
    chat_response = client.post(
        "/api/chat",
        json={
            "message": "Create an ETL project to process employee data",
            "schema": {
                "id": "integer",
                "name": "string",
                "age": "integer",
                "department": "string"
            },
            "sample_data": [
                {"id": 1, "name": "John", "age": 25, "department": "IT"}
            ]
        }
    )
    assert chat_response.status_code == 200
    chat_result = chat_response.json()
    assert "response" in chat_result
    
    # Cleanup
    os.remove(test_file_path)
    os.rmdir('test_data') 