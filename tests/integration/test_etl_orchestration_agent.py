import pytest
from datetime import datetime
from src.core.agents.etl_orchestration_agent import ETLOrchestrationAgent
from src.core.agents.base_agent import AgentEvent
from src.core.aws.aws_service_manager import AWSServiceManager

@pytest.fixture
async def aws_manager():
    """Create an AWS service manager instance for testing."""
    manager = AWSServiceManager(region_name="us-east-1")
    await manager.initialize()
    return manager

@pytest.fixture
async def etl_agent(aws_manager):
    """Create an ETL orchestration agent instance for testing."""
    agent = ETLOrchestrationAgent(
        agent_id="test-etl-agent",
        aws_manager=aws_manager,
        llm_model="gpt-4"
    )
    await agent.initialize()
    return agent

@pytest.mark.asyncio
async def test_etl_workflow_execution(etl_agent, aws_manager):
    """Test end-to-end ETL workflow execution."""
    # Create test data lake
    data_lake = await aws_manager.create_data_lake(
        name="test-data-lake",
        region="us-east-1"
    )
    
    # Create test ETL workflow request
    workflow_request = AgentEvent(
        event_type="etl_workflow_request",
        payload={
            "database": data_lake["database"],
            "table": "test_table",
            "transform_query": """
                CREATE TABLE IF NOT EXISTS transformed_data AS
                SELECT * FROM source_data
                WHERE date >= '2024-01-01'
            """,
            "load_query": """
                INSERT INTO target_table
                SELECT * FROM transformed_data
            """,
            "output_location": data_lake["output_location"]
        },
        source_agent="test-client",
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process the workflow request
    response = await etl_agent.process_event(workflow_request)
    
    # Verify response
    assert response is not None
    assert response.event_type == "etl_workflow_response"
    assert response.payload["status"] == "success"
    assert "workflow_result" in response.payload
    assert "metadata" in response.payload
    
    # Verify workflow steps
    workflow_steps = response.payload["metadata"]["workflow_steps"]
    assert "workflow_plan" in workflow_steps
    assert "schema_validation" in workflow_steps
    assert "transformation_result" in workflow_steps
    assert "quality_check" in workflow_steps
    assert "load_result" in workflow_steps

@pytest.mark.asyncio
async def test_schema_validation(etl_agent, aws_manager):
    """Test schema validation in ETL workflow."""
    # Create test data lake
    data_lake = await aws_manager.create_data_lake(
        name="test-schema-validation",
        region="us-east-1"
    )
    
    # Create test table with schema
    await aws_manager.execute_query(
        query="""
            CREATE TABLE IF NOT EXISTS test_table (
                id INT,
                name STRING,
                date DATE
            )
        """,
        database=data_lake["database"],
        output_location=data_lake["output_location"]
    )
    
    # Create test ETL workflow request
    workflow_request = AgentEvent(
        event_type="etl_workflow_request",
        payload={
            "database": data_lake["database"],
            "table": "test_table",
            "transform_query": "SELECT * FROM test_table",
            "load_query": "INSERT INTO target_table SELECT * FROM test_table",
            "output_location": data_lake["output_location"]
        },
        source_agent="test-client",
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process the workflow request
    response = await etl_agent.process_event(workflow_request)
    
    # Verify schema validation
    assert response is not None
    assert response.event_type == "etl_workflow_response"
    assert "schema_validation" in response.payload["workflow_result"]
    
    schema = response.payload["workflow_result"]["schema_validation"]
    assert "columns" in schema
    assert len(schema["columns"]) == 3
    assert any(col["name"] == "id" for col in schema["columns"])
    assert any(col["name"] == "name" for col in schema["columns"])
    assert any(col["name"] == "date" for col in schema["columns"])

@pytest.mark.asyncio
async def test_data_quality_check(etl_agent, aws_manager):
    """Test data quality checks in ETL workflow."""
    # Create test data lake
    data_lake = await aws_manager.create_data_lake(
        name="test-quality-check",
        region="us-east-1"
    )
    
    # Create and populate test table
    await aws_manager.execute_query(
        query="""
            CREATE TABLE IF NOT EXISTS test_table (
                id INT,
                name STRING,
                date DATE
            )
        """,
        database=data_lake["database"],
        output_location=data_lake["output_location"]
    )
    
    await aws_manager.execute_query(
        query="""
            INSERT INTO test_table VALUES
            (1, 'test1', DATE '2024-01-01'),
            (2, 'test2', DATE '2024-01-02'),
            (3, NULL, DATE '2024-01-03')
        """,
        database=data_lake["database"],
        output_location=data_lake["output_location"]
    )
    
    # Create test ETL workflow request
    workflow_request = AgentEvent(
        event_type="etl_workflow_request",
        payload={
            "database": data_lake["database"],
            "table": "test_table",
            "transform_query": "SELECT * FROM test_table",
            "load_query": "INSERT INTO target_table SELECT * FROM test_table",
            "output_location": data_lake["output_location"]
        },
        source_agent="test-client",
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process the workflow request
    response = await etl_agent.process_event(workflow_request)
    
    # Verify quality check results
    assert response is not None
    assert response.event_type == "etl_workflow_response"
    assert "quality_check" in response.payload["workflow_result"]
    
    quality = response.payload["workflow_result"]["quality_check"]
    assert "completeness" in quality
    assert "accuracy" in quality
    assert "consistency" in quality
    
    # Verify completeness (should be less than 100% due to NULL value)
    assert quality["completeness"] < 1.0

@pytest.mark.asyncio
async def test_error_handling(etl_agent):
    """Test error handling in ETL workflow."""
    # Create test ETL workflow request with invalid parameters
    workflow_request = AgentEvent(
        event_type="etl_workflow_request",
        payload={
            "database": "non_existent_db",
            "table": "non_existent_table",
            "transform_query": "SELECT * FROM non_existent_table",
            "load_query": (
                "INSERT INTO target_table SELECT * FROM non_existent_table"
            ),
            "output_location": "s3://non-existent-bucket/"
        },
        source_agent="test-client",
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process the workflow request
    response = await etl_agent.process_event(workflow_request)
    
    # Verify error handling
    assert response is None  # Should return None on error
    state = await etl_agent.get_state()
    assert "error" in state
    assert "error_timestamp" in state 