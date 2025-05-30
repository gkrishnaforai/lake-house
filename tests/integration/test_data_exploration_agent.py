import pytest
from datetime import datetime
from src.core.agents.data_exploration_agent import DataExplorationAgent
from src.core.aws.aws_service_manager import AWSServiceManager
from src.core.agents.base_agent import AgentEvent


@pytest.fixture
async def aws_manager():
    """Create an AWS service manager for testing."""
    return AWSServiceManager(
        region_name="us-east-1",
        profile_name="test"
    )


@pytest.fixture
async def data_exploration_agent(aws_manager):
    """Create a data exploration agent for testing."""
    agent = DataExplorationAgent(
        agent_id="test_agent",
        aws_manager=aws_manager
    )
    await agent.initialize()
    yield agent
    await agent.cleanup()


@pytest.mark.asyncio
async def test_create_data_lake(aws_manager):
    """Test creating a data lake."""
    # Create test data lake
    result = await aws_manager.create_data_lake(
        bucket_name="test-data-lake-bucket",
        database_name="test_database",
        table_name="test_table",
        schema=[
            {"Name": "id", "Type": "string"},
            {"Name": "name", "Type": "string"},
            {"Name": "value", "Type": "double"}
        ]
    )
    
    assert result["status"] == "success"
    assert result["bucket_name"] == "test-data-lake-bucket"
    assert result["database_name"] == "test_database"
    assert result["table_name"] == "test_table"


@pytest.mark.asyncio
async def test_execute_query(data_exploration_agent):
    """Test executing a query."""
    # Create test query event
    event = AgentEvent(
        event_type="query_request",
        source_agent="test_client",
        target_agent=None,
        payload={
            "query": "SELECT * FROM test_database.test_table LIMIT 10",
            "database": "test_database",
            "output_location": "s3://test-data-lake-bucket/query-results"
        },
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process query event
    response = await data_exploration_agent.process_event(event)
    
    assert response is not None
    assert response.event_type == "query_response"
    assert response.target_agent == "test_client"
    assert response.payload["status"] == "success"


@pytest.mark.asyncio
async def test_get_schema(data_exploration_agent):
    """Test getting table schema."""
    # Create test schema event
    event = AgentEvent(
        event_type="schema_request",
        source_agent="test_client",
        target_agent=None,
        payload={
            "database": "test_database",
            "table": "test_table"
        },
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process schema event
    response = await data_exploration_agent.process_event(event)
    
    assert response is not None
    assert response.event_type == "schema_response"
    assert response.target_agent == "test_client"
    assert response.payload["status"] == "success"
    assert len(response.payload["schema"]) > 0


@pytest.mark.asyncio
async def test_check_data_quality(data_exploration_agent):
    """Test checking data quality."""
    # Create test quality event
    event = AgentEvent(
        event_type="data_quality_request",
        source_agent="test_client",
        target_agent=None,
        payload={
            "database": "test_database",
            "table": "test_table",
            "metrics": ["completeness", "accuracy"]
        },
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process quality event
    response = await data_exploration_agent.process_event(event)
    
    assert response is not None
    assert response.event_type == "quality_response"
    assert response.target_agent == "test_client"
    assert response.payload["status"] == "success"
    assert "metrics" in response.payload


@pytest.mark.asyncio
async def test_agent_state_management(data_exploration_agent):
    """Test agent state management."""
    # Check initial state
    state = await data_exploration_agent.get_state()
    assert state.status == "ready"
    assert state.agent_id == "test_agent"
    
    # Update state
    await data_exploration_agent.update_state({
        "status": "processing",
        "metadata": {"current_task": "test_task"}
    })
    
    # Check updated state
    state = await data_exploration_agent.get_state()
    assert state.status == "processing"
    assert state.metadata["current_task"] == "test_task"


@pytest.mark.asyncio
async def test_error_handling(data_exploration_agent):
    """Test error handling."""
    # Create invalid query event
    event = AgentEvent(
        event_type="query_request",
        source_agent="test_client",
        target_agent=None,
        payload={
            "query": "INVALID QUERY",
            "database": "test_database",
            "output_location": "s3://test-data-lake-bucket/query-results"
        },
        timestamp=datetime.utcnow().isoformat()
    )
    
    # Process invalid event and check error state
    await data_exploration_agent.process_event(event)
    state = await data_exploration_agent.get_state()
    assert state.status == "error"
    assert "error" in state.metadata 