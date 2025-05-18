import pytest
import pandas as pd
import os
from pathlib import Path
from datetime import datetime
from src.core.workflow.etl_agent_flow import ETLAgentFlow
from src.core.workflow.etl_state import ETLState, ETLStep
from src.core.llm.manager import LLMManager
from src.core.state_management.state_manager import StateManager
from src.etl_architect_agent_v2.agents.catalog_agent import CatalogAgent
from src.etl_architect_agent_v2.backend.services.catalog_service import CatalogService

@pytest.fixture
def test_excel_file(tmp_path):
    """Create a test Excel file with sample data."""
    # Create sample data
    data = {
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'age': [30, 25, 35, 28, 32],
        'salary': [50000, 60000, 55000, 65000, 70000],
        'department': ['IT', 'HR', 'Finance', 'Marketing', 'IT'],
        'join_date': [
            datetime(2020, 1, 1),
            datetime(2020, 2, 1),
            datetime(2020, 3, 1),
            datetime(2020, 4, 1),
            datetime(2020, 5, 1)
        ]
    }
    
    # Create DataFrame
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

@pytest.fixture
def llm_manager():
    """Create an LLM manager instance."""
    return LLMManager()

@pytest.fixture
def state_manager():
    """Create a state manager instance."""
    return StateManager()

@pytest.fixture
def etl_agent_flow(llm_manager, state_manager):
    """Create an ETL agent flow instance."""
    return ETLAgentFlow(llm_manager, state_manager)

async def test_excel_workflow(
    test_excel_file,
    catalog_service,
    catalog_agent,
    etl_agent_flow
):
    """Test the complete workflow for processing an Excel file."""
    
    # 1. Initialize state
    workflow_id = f"test_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    initial_state = ETLState(
        workflow_id=workflow_id,
        current_step=ETLStep.ANALYSIS,
        metadata={
            "file_path": test_excel_file,
            "file_type": "excel",
            "table_name": "employees"
        }
    )
    
    # 2. Process file upload
    file_result = await catalog_agent.process_file_upload(
        file_path=test_excel_file,
        table_name="employees"
    )
    
    assert file_result["status"] == "success"
    assert "schema" in file_result
    assert "quality_metrics" in file_result
    
    # 3. Verify schema extraction
    schema = file_result["schema"]
    assert "fields" in schema
    assert len(schema["fields"]) > 0
    
    # 4. Verify quality metrics
    quality_metrics = file_result["quality_metrics"]
    assert "completeness" in quality_metrics
    assert "accuracy" in quality_metrics
    
    # 5. Query the data
    query = "SELECT * FROM employees WHERE department = 'IT'"
    query_result = await catalog_service.execute_query(query)
    
    assert query_result is not None
    assert len(query_result) > 0
    
    # 6. Verify schema evolution
    evolution = await catalog_service.get_schema_evolution("employees")
    assert evolution is not None
    assert "changes" in evolution
    
    # 7. Verify audit log
    audit_log = await catalog_service.get_table_audit("employees")
    assert audit_log is not None
    assert len(audit_log) > 0
    
    # 8. Clean up
    # await catalog_service.delete_file(test_excel_file)
    
    # 9. Verify file deletion
    #with pytest.raises(Exception):
    #    await catalog_service.get_file(test_excel_file)

async def test_error_handling(
    test_excel_file,
    catalog_service,
    catalog_agent
):
    """Test error handling in the workflow."""
    
    # 1. Test invalid file
    with pytest.raises(Exception):
        await catalog_agent.process_file_upload(
            file_path="invalid_file.xlsx",
            table_name="employees"
        )
    
    # 2. Test invalid table name
    with pytest.raises(Exception):
        await catalog_agent.process_file_upload(
            file_path=test_excel_file,
            table_name=""
        )
    
    # 3. Test invalid query
    with pytest.raises(Exception):
        await catalog_service.execute_query("INVALID SQL QUERY")

async def test_retry_mechanism(
    test_excel_file,
    catalog_service,
    catalog_agent
):
    """Test retry mechanism in the workflow."""
    
    # 1. Simulate temporary failure
    catalog_service.s3_client = None  # Simulate S3 failure
    
    # 2. Attempt file upload
    with pytest.raises(Exception):
        await catalog_agent.process_file_upload(
            file_path=test_excel_file,
            table_name="employees"
        )
    
    # 3. Restore S3 client
    catalog_service.s3_client = boto3.client('s3')
    
    # 4. Retry file upload
    result = await catalog_agent.process_file_upload(
        file_path=test_excel_file,
        table_name="employees"
    )
    
    assert result["status"] == "success"

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 