"""Test SQL agent functionality."""

import pytest
from typing import Dict, Any
from src.core.sql.sql_state import (
    SQLGenerationState,
    SQLGenerationStep,
    SQLRequirements,
    SQLGenerationOutput
)
from src.core.sql.sql_agent import SQLGenerationAgent
from src.core.state_management.state_manager import StateManager
from src.core.llm.manager import LLMManager


@pytest.fixture
def sample_schema() -> Dict[str, Any]:
    """Create a sample schema for testing."""
    return {
        "table_name": "employees",
        "columns": [
            {
                "name": "id",
                "type": "int",
                "description": "Employee ID"
            },
            {
                "name": "name",
                "type": "string",
                "description": "Employee name"
            },
            {
                "name": "department",
                "type": "string",
                "description": "Department name"
            },
            {
                "name": "salary",
                "type": "decimal",
                "description": "Annual salary"
            },
            {
                "name": "hire_date",
                "type": "date",
                "description": "Date of hire"
            }
        ]
    }


@pytest.fixture
def state_manager():
    """Create state manager instance."""
    return StateManager()


@pytest.fixture
def llm_manager():
    """Create LLM manager instance."""
    return LLMManager()


@pytest.fixture
def sql_agent(state_manager, llm_manager):
    """Create SQL agent instance."""
    return SQLGenerationAgent(
        llm_manager=llm_manager,
        state_manager=state_manager
    )


@pytest.mark.asyncio
async def test_basic_query_generation(sql_agent, sample_schema):
    """Test basic SQL query generation."""
    # Create requirements
    requirements = SQLRequirements(
        query="Show me all employees in the IT department",
        schema=sample_schema
    )
    
    # Generate SQL
    result = await sql_agent.run("test_workflow", requirements.query)
    
    # Verify result
    assert isinstance(result, SQLGenerationState)
    assert result.metadata.get("query_result")
    assert "sql" in result.metadata["query_result"]
    assert "SELECT" in result.metadata["query_result"]["sql"].upper()
    assert "FROM" in result.metadata["query_result"]["sql"].upper()
    assert "WHERE" in result.metadata["query_result"]["sql"].upper()
    assert "department" in result.metadata["query_result"]["sql"].lower()
    assert result.confidence > 0
    assert result.tables_used == ["employees"]
    assert "department" in result.columns_used


@pytest.mark.asyncio
async def test_query_with_aggregation(sql_agent, sample_schema):
    """Test SQL generation with aggregation functions."""
    requirements = SQLRequirements(
        query="What is the average salary by department?",
        schema=sample_schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "AVG" in result.sql_query.upper()
    assert "GROUP BY" in result.sql_query.upper()
    assert "department" in result.sql_query.lower()
    assert "salary" in result.sql_query.lower()


@pytest.mark.asyncio
async def test_query_with_date_filter(sql_agent, sample_schema):
    """Test SQL generation with date filters."""
    requirements = SQLRequirements(
        query="Show employees hired after 2020",
        schema=sample_schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "WHERE" in result.sql_query.upper()
    assert "hire_date" in result.sql_query.lower()
    assert "2020" in result.sql_query


@pytest.mark.asyncio
async def test_query_with_multiple_conditions(sql_agent, sample_schema):
    """Test SQL generation with multiple conditions."""
    requirements = SQLRequirements(
        query="Find IT department employees with salary above 100000",
        schema=sample_schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "WHERE" in result.sql_query.upper()
    assert "AND" in result.sql_query.upper()
    assert "department" in result.sql_query.lower()
    assert "salary" in result.sql_query.lower()
    assert "100000" in result.sql_query


@pytest.mark.asyncio
async def test_query_with_ordering(sql_agent, sample_schema):
    """Test SQL generation with ordering."""
    requirements = SQLRequirements(
        query="List employees by salary in descending order",
        schema=sample_schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "ORDER BY" in result.sql_query.upper()
    assert "DESC" in result.sql_query.upper()
    assert "salary" in result.sql_query.lower()


@pytest.mark.asyncio
async def test_query_with_limit(sql_agent, sample_schema):
    """Test SQL generation with limit clause."""
    requirements = SQLRequirements(
        query="Show top 5 highest paid employees",
        schema=sample_schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "ORDER BY" in result.sql_query.upper()
    assert "DESC" in result.sql_query.upper()
    assert "LIMIT" in result.sql_query.upper()
    assert "5" in result.sql_query


@pytest.mark.asyncio
async def test_query_with_joins(sql_agent):
    """Test SQL generation with table joins."""
    schema = {
        "tables": [
            {
                "name": "employees",
                "columns": [
                    {"name": "id", "type": "int"},
                    {"name": "name", "type": "string"},
                    {"name": "department_id", "type": "int"}
                ]
            },
            {
                "name": "departments",
                "columns": [
                    {"name": "id", "type": "int"},
                    {"name": "name", "type": "string"}
                ]
            }
        ]
    }
    
    requirements = SQLRequirements(
        query="Show employee names and their department names",
        schema=schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "JOIN" in result.sql_query.upper()
    assert "employees" in result.sql_query.lower()
    assert "departments" in result.sql_query.lower()
    assert "department_id" in result.sql_query.lower()


@pytest.mark.asyncio
async def test_error_handling(sql_agent, sample_schema):
    """Test error handling in SQL generation."""
    # Test with invalid query
    requirements = SQLRequirements(
        query="Invalid query that doesn't make sense",
        schema=sample_schema
    )
    
    with pytest.raises(Exception):
        await sql_agent.generate_sql(requirements)
    
    # Test with missing schema
    requirements = SQLRequirements(
        query="Show all employees",
        schema={}
    )
    
    with pytest.raises(Exception):
        await sql_agent.generate_sql(requirements)


@pytest.mark.asyncio
async def test_state_management(sql_agent, sample_schema):
    """Test state management during SQL generation."""
    workflow_id = "test_workflow"
    
    # Create initial state
    state = SQLGenerationState(workflow_id=workflow_id)
    assert state.current_step == SQLGenerationStep.REQUIREMENTS_ANALYSIS
    
    # Generate SQL
    requirements = SQLRequirements(
        query="Show all employees",
        schema=sample_schema
    )
    
    result = await sql_agent.generate_sql(requirements)
    
    # Verify state progression
    assert isinstance(result, SQLGenerationOutput)
    assert result.sql_query
    assert not state.error  # No errors should be present 