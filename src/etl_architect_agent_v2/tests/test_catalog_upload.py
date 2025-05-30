import pandas as pd
import pytest
from datetime import datetime
from etl_architect_agent_v2.agents.catalog_agent import CatalogAgent
from etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService,
)
import httpx
from src.etl_architect_agent_v2.api.main import app

@pytest.fixture
def dummy_excel_file(tmp_path):
    """Create a dummy Excel file for testing."""
    # Create sample data
    data = {
        'id': [1, 2, 3],
        'name': ['John', 'Jane', 'Bob'],
        'age': [30, 25, 35],
        'date': [datetime.now(), datetime.now(), datetime.now()]
    }
    df = pd.DataFrame(data)
    
    # Save to Excel
    file_path = tmp_path / "test_data.xlsx"
    df.to_excel(file_path, index=False)
    return str(file_path)

@pytest.fixture
def catalog_service():
    """Create a catalog service instance."""
    return CatalogService(
        bucket="test-bucket",
        aws_region="us-east-1"
    )

@pytest.fixture
def catalog_agent(catalog_service):
    """Create a catalog agent instance."""
    return CatalogAgent(catalog_service)

@pytest.mark.asyncio
async def test_file_upload_flow(dummy_excel_file, catalog_agent):
    """Test the complete file upload flow."""
    try:
        # Read the Excel file
        df = pd.read_excel(dummy_excel_file)
        
        # Test schema extraction
        schema = await catalog_agent._extract_schema(
            file_name="test_data.xlsx",
            file_type=df.dtypes.to_dict(),
            data=df.to_dict(),
            table_name="test_table"
        )
        assert schema is not None
        assert "columns" in schema
        assert len(schema["columns"]) == 4  # id, name, age, date
        
        # Test data quality check
        quality_metrics = await catalog_agent._check_data_quality(
            file_name="test_data.xlsx",
            file_type=df.dtypes.to_dict(),
            data=df.to_dict(),
            user_id="test_user"
        )
        assert quality_metrics is not None
        assert "completeness" in quality_metrics
        assert "uniqueness" in quality_metrics
        
        # Test schema validation
        if catalog_agent._needs_schema_validation(schema):
            validated_schema = await catalog_agent._validate_schema_with_llm(schema)
            assert validated_schema is not None
        
        # Test schema update
        version = datetime.utcnow().isoformat()
        schema_result = catalog_agent._update_schema(
            file_name="test_data.xlsx",
            schema=schema,
            version=version
        )
        assert schema_result is not None
        assert schema_result["file_name"] == "test_data.xlsx"
        assert schema_result["schema"] == schema
        assert schema_result["version"] == version
        
        # Test complete file upload process
        result = await catalog_agent.process_file_upload(
            df=df,
            file_name="test_data.xlsx",
            table_name="test_table",
            create_new=True
        )
        
        assert result["status"] == "success"
        assert "schema" in result
        assert "quality_metrics" in result
        assert "schema_result" in result
        assert "agent_result" in result
        
        print("Test completed successfully!")
        print(f"Schema: {result['schema']}")
        print(f"Quality Metrics: {result['quality_metrics']}")
        
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")

@pytest.mark.asyncio
async def test_list_files_for_test_user():
    """Test listing all files for test_user from S3 user folder via API."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/api/files", headers={"X-User-Id": "test_user"})
        assert response.status_code == 200
        files = response.json()
        s3_keys = []
        for file in files:
            s3_key = file.get('location') or file.get('s3_path')
            assert s3_key is not None
            assert s3_key.startswith("originals/test_user/")
            s3_keys.append(s3_key)
        print("S3 keys listed for test_user (raw files only):")
        for key in s3_keys:
            print(key)
    print("test_list_files_for_test_user: PASSED")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_file_upload_flow()) 