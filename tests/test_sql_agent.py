"""Test SQL agent functionality."""

import pytest
from typing import Dict, Any
from src.core.sql.sql_state import (
    SQLGenerationState,
    SQLRequirements
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
    print("\nStarting test_basic_query_generation...")
    
    # Create requirements
    requirements = SQLRequirements(
        query="Show me all employees in the IT department",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    # Generate SQL
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    # Verify result
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "sql" in result.metadata["query_result"]
    assert "SELECT" in result.metadata["query_result"]["sql"].upper()
    assert "FROM" in result.metadata["query_result"]["sql"].upper()
    assert "WHERE" in result.metadata["query_result"]["sql"].upper()
    assert "department" in result.metadata["query_result"]["sql"].lower()
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_query_with_aggregation(sql_agent, sample_schema):
    """Test SQL generation with aggregation functions."""
    print("\nStarting test_query_with_aggregation...")
    
    requirements = SQLRequirements(
        query="What is the average salary by department?",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "AVG" in result.metadata["query_result"]["sql"].upper()
    assert "GROUP BY" in result.metadata["query_result"]["sql"].upper()
    assert "department" in result.metadata["query_result"]["sql"].lower()
    assert "salary" in result.metadata["query_result"]["sql"].lower()
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_query_with_date_filter(sql_agent, sample_schema):
    """Test SQL generation with date filters."""
    print("\nStarting test_query_with_date_filter...")
    
    requirements = SQLRequirements(
        query="Show employees hired after 2020",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "WHERE" in result.metadata["query_result"]["sql"].upper()
    assert "hire_date" in result.metadata["query_result"]["sql"].lower()
    assert "2020" in result.metadata["query_result"]["sql"]
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_query_with_multiple_conditions(sql_agent, sample_schema):
    """Test SQL generation with multiple conditions."""
    print("\nStarting test_query_with_multiple_conditions...")
    
    requirements = SQLRequirements(
        query="Find IT department employees with salary above 100000",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "WHERE" in result.metadata["query_result"]["sql"].upper()
    assert "AND" in result.metadata["query_result"]["sql"].upper()
    assert "department" in result.metadata["query_result"]["sql"].lower()
    assert "salary" in result.metadata["query_result"]["sql"].lower()
    assert "100000" in result.metadata["query_result"]["sql"]
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_query_with_ordering(sql_agent, sample_schema):
    """Test SQL generation with ordering."""
    print("\nStarting test_query_with_ordering...")
    
    requirements = SQLRequirements(
        query="List employees by salary in descending order",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "ORDER BY" in result.metadata["query_result"]["sql"].upper()
    assert "DESC" in result.metadata["query_result"]["sql"].upper()
    assert "salary" in result.metadata["query_result"]["sql"].lower()
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_query_with_limit(sql_agent, sample_schema):
    """Test SQL generation with limit clause."""
    print("\nStarting test_query_with_limit...")
    
    requirements = SQLRequirements(
        query="Show top 5 highest paid employees",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "ORDER BY" in result.metadata["query_result"]["sql"].upper()
    assert "DESC" in result.metadata["query_result"]["sql"].upper()
    assert "LIMIT" in result.metadata["query_result"]["sql"].upper()
    assert "5" in result.metadata["query_result"]["sql"]
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_query_with_joins(sql_agent):
    """Test SQL generation with table joins."""
    print("\nStarting test_query_with_joins...")
    
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
    print(f"Created schema: {schema}")
    
    requirements = SQLRequirements(
        query="Show employee names and their department names",
        schema=schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    assert result.metadata.get("query_result")
    assert "JOIN" in result.metadata["query_result"]["sql"].upper()
    assert "employees" in result.metadata["query_result"]["sql"].lower()
    assert "departments" in result.metadata["query_result"]["sql"].lower()
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_error_handling(sql_agent, sample_schema):
    """Test error handling in SQL generation."""
    print("\nStarting test_error_handling...")
    
    requirements = SQLRequirements(
        query="Invalid query with non-existent table",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "error"
    assert result.error is not None
    print("Test completed successfully!")


@pytest.mark.asyncio
async def test_state_management(sql_agent, sample_schema):
    """Test state management in SQL generation."""
    print("\nStarting test_state_management...")
    
    requirements = SQLRequirements(
        query="Show me all employees",
        schema=sample_schema
    )
    print(f"Created requirements: {requirements}")
    
    print("Calling sql_agent.run...")
    result = await sql_agent.run("test_workflow", requirements.query)
    print(f"Got result: {result}")
    
    assert isinstance(result, SQLGenerationState)
    assert result.status == "success"
    assert result.is_done
    print("Test completed successfully!") 