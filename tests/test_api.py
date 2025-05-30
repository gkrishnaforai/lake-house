import os
import pytest
from fastapi.testclient import TestClient
import pandas as pd
from datetime import datetime
import tempfile
import json

from etl_architect_agent_v2.api.main import app

# Create test client
client = TestClient(app)

# Test data
TEST_USER_ID = "test_user"
TEST_TABLE_NAME = "test_table"
TEST_FILE_NAME = "test_data.xlsx"

def create_test_excel():
    """Create a test Excel file with sample data."""
    df = pd.DataFrame({
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'age': [30, 25, 35],
        'created_at': [datetime.now() for _ in range(3)]
    })
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as temp_file:
        df.to_excel(temp_file.name, index=False)
        return temp_file.name

@pytest.fixture(scope="session")
def test_file():
    """Create and return a test Excel file."""
    file_path = create_test_excel()
    yield file_path
    os.unlink(file_path)

def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_upload_file(test_file):
    """Test file upload endpoint."""
    with open(test_file, "rb") as f:
        files = {"file": (TEST_FILE_NAME, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        headers = {"X-User-Id": TEST_USER_ID}
        response = client.post(
            f"/api/catalog/tables/{TEST_TABLE_NAME}/files",
            files=files,
            headers=headers
        )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["file_name"] == TEST_FILE_NAME
    assert "s3_path" in data

def test_list_files():
    """Test listing files endpoint."""
    headers = {"X-User-Id": TEST_USER_ID}
    response = client.get("/api/files", headers=headers)
    assert response.status_code == 200
    files = response.json()
    assert isinstance(files, list)
    if files:
        assert all(isinstance(f, dict) for f in files)
        assert all("file_name" in f for f in files)

def test_get_file_details(test_file):
    """Test getting file details."""
    # First upload a file
    with open(test_file, "rb") as f:
        files = {"file": (TEST_FILE_NAME, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        headers = {"X-User-Id": TEST_USER_ID}
        upload_response = client.post(
            f"/api/catalog/tables/{TEST_TABLE_NAME}/files",
            files=files,
            headers=headers
        )
    
    assert upload_response.status_code == 200
    s3_path = upload_response.json()["s3_path"]
    
    # Then get its details
    response = client.get(f"/api/files/{s3_path}/details")
    assert response.status_code == 200
    details = response.json()
    assert "file_name" in details
    assert "size" in details
    assert "last_modified" in details

def test_get_file_preview(test_file):
    """Test getting file preview."""
    # First upload a file
    with open(test_file, "rb") as f:
        files = {"file": (TEST_FILE_NAME, f, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
        headers = {"X-User-Id": TEST_USER_ID}
        upload_response = client.post(
            f"/api/catalog/tables/{TEST_TABLE_NAME}/files",
            files=files,
            headers=headers
        )
    
    assert upload_response.status_code == 200
    s3_path = upload_response.json()["s3_path"]
    
    # Then get its preview
    response = client.get(f"/api/files/{s3_path}/preview")
    assert response.status_code == 200
    preview = response.json()
    assert isinstance(preview, list)
    assert len(preview) > 0
    assert all(isinstance(row, dict) for row in preview)

def test_list_catalog_files():
    """Test listing catalog files."""
    response = client.get("/api/catalog/files")
    assert response.status_code == 200
    files = response.json()
    assert isinstance(files, list)
    if files:
        assert all(isinstance(f, dict) for f in files)
        assert all("file_name" in f for f in files)

def test_get_table_files():
    """Test getting files for a specific table."""
    response = client.get(f"/api/catalog/tables/{TEST_TABLE_NAME}/files")
    assert response.status_code == 200
    files = response.json()
    assert isinstance(files, list)
    if files:
        assert all(isinstance(f, dict) for f in files)
        assert all("file_name" in f for f in files)

def test_get_database_catalog():
    """Test getting the database catalog."""
    response = client.get("/api/catalog")
    assert response.status_code == 200
    catalog = response.json()
    assert "database_name" in catalog
    assert "tables" in catalog
    assert isinstance(catalog["tables"], list)

def test_descriptive_query():
    """Test executing a descriptive query."""
    query_data = {
        "query": "Show me all data",
        "table_name": TEST_TABLE_NAME
    }
    response = client.post("/api/catalog/descriptive_query", json=query_data)
    assert response.status_code == 200
    result = response.json()
    assert "status" in result
    assert "results" in result

def test_get_audit_trail():
    """Test getting audit trail for a table."""
    response = client.get(f"/api/catalog/audit/{TEST_TABLE_NAME}")
    assert response.status_code in [200, 404]  # 404 is acceptable if no audit trail exists yet
    if response.status_code == 200:
        audit = response.json()
        assert "table_name" in audit
        assert "audit_metadata_found" in audit

def test_get_data_quality():
    """Test getting data quality metrics for a table."""
    response = client.get(f"/api/catalog/quality/{TEST_TABLE_NAME}")
    assert response.status_code in [200, 404]  # 404 is acceptable if no quality metrics exist yet
    if response.status_code == 200:
        quality = response.json()
        assert isinstance(quality, dict) 