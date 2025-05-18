"""Test SQL agent functionality using LangChain prompts and chaining."""

import pytest
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from src.core.sql.sql_state import SQLGenerationOutput, SQLGenerationError


class SQLGenerationChain(BaseModel):
    """Chain for SQL generation using LangChain."""
    llm: ChatOpenAI
    prompt: ChatPromptTemplate
    output_parser: PydanticOutputParser

    def __init__(self, llm: ChatOpenAI):
        """Initialize the chain with LLM."""
        # Define the SQL generation prompt template
        system_prompt = (
            "You are a SQL expert. Your task is to generate SQL queries "
            "based on natural language descriptions. Follow these "
            "guidelines:\n"
            "1. Use appropriate JOINs if multiple tables are involved\n"
            "2. Include WHERE clauses for any filters\n"
            "3. Use appropriate aggregation functions if needed\n"
            "4. Include ORDER BY if sorting is needed\n"
            "5. Limit the results to 1000 rows\n"
            "6. Ensure the query is valid SQL syntax\n"
            "7. Return your response in the following JSON format:\n"
            "{{\n"
            '    "sql_query": "your SQL query here",\n'
            '    "explanation": "explanation of the query",\n'
            '    "confidence": confidence_score (0-1),\n'
            '    "tables_used": ["list", "of", "tables"],\n'
            '    "columns_used": ["list", "of", "columns"],\n'
            '    "filters": {{"filter_name": "filter_value"}}\n'
            "}}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", """Given the following database schema:
            {schema}

            Generate a SQL query for this request:
            {query}

            {format_instructions}""")
        ])

        # Set up the output parser
        output_parser = PydanticOutputParser(
            pydantic_object=SQLGenerationOutput
        )

        # Initialize the parent class with all required fields
        super().__init__(
            llm=llm,
            prompt=prompt,
            output_parser=output_parser
        )

    def _validate_input(self, query: str, schema: Dict[str, Any]) -> None:
        """Validate input parameters."""
        if not query or not query.strip():
            raise SQLGenerationError("Query cannot be empty")
        
        if not schema:
            raise SQLGenerationError("Schema cannot be empty")
        
        if "tables" in schema:
            if not schema["tables"]:
                raise SQLGenerationError(
                    "Schema must contain at least one table"
                )
        elif "table_name" not in schema:
            raise SQLGenerationError(
                "Schema must contain table information"
            )

    async def generate_sql(
        self, query: str, schema: Dict[str, Any]
    ) -> SQLGenerationOutput:
        """Generate SQL query using the chain."""
        try:
            # Validate input
            self._validate_input(query, schema)

            # Format the prompt with schema and query
            formatted_prompt = self.prompt.format_messages(
                schema=schema,
                query=query,
                format_instructions=(
                    self.output_parser.get_format_instructions()
                )
            )

            # Generate response using LLM
            response = await self.llm.ainvoke(formatted_prompt)

            # Parse the response
            result = self.output_parser.parse(response.content)
            return result

        except Exception as e:
            raise SQLGenerationError(
                f"Error generating SQL: {str(e)}"
            )


@pytest.fixture
def llm():
    """Create LLM instance."""
    return ChatOpenAI(temperature=0)


@pytest.fixture
def sql_chain(llm):
    """Create SQL generation chain."""
    return SQLGenerationChain(llm=llm)


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


@pytest.mark.asyncio
async def test_basic_query_generation(sql_chain, sample_schema):
    """Test basic SQL query generation."""
    query = "Show me all employees in the IT department"
    
    result = await sql_chain.generate_sql(query, sample_schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "SELECT" in result.sql_query.upper()
    assert "FROM" in result.sql_query.upper()
    assert "WHERE" in result.sql_query.upper()
    assert "department" in result.sql_query.lower()
    assert result.confidence > 0
    assert result.tables_used == ["employees"]
    assert "department" in result.columns_used


@pytest.mark.asyncio
async def test_query_with_aggregation(sql_chain, sample_schema):
    """Test SQL generation with aggregation functions."""
    query = "What is the average salary by department?"
    
    result = await sql_chain.generate_sql(query, sample_schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "AVG" in result.sql_query.upper()
    assert "GROUP BY" in result.sql_query.upper()
    assert "department" in result.sql_query.lower()
    assert "salary" in result.sql_query.lower()


@pytest.mark.asyncio
async def test_query_with_date_filter(sql_chain, sample_schema):
    """Test SQL generation with date filters."""
    query = "Show employees hired after 2020"
    
    result = await sql_chain.generate_sql(query, sample_schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "WHERE" in result.sql_query.upper()
    assert "hire_date" in result.sql_query.lower()
    assert "2020" in result.sql_query


@pytest.mark.asyncio
async def test_query_with_multiple_conditions(sql_chain, sample_schema):
    """Test SQL generation with multiple conditions."""
    query = "Find IT department employees with salary above 100000"
    
    result = await sql_chain.generate_sql(query, sample_schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "WHERE" in result.sql_query.upper()
    assert "AND" in result.sql_query.upper()
    assert "department" in result.sql_query.lower()
    assert "salary" in result.sql_query.lower()
    assert "100000" in result.sql_query


@pytest.mark.asyncio
async def test_query_with_ordering(sql_chain, sample_schema):
    """Test SQL generation with ordering."""
    query = "List employees by salary in descending order"
    
    result = await sql_chain.generate_sql(query, sample_schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "ORDER BY" in result.sql_query.upper()
    assert "DESC" in result.sql_query.upper()
    assert "salary" in result.sql_query.lower()


@pytest.mark.asyncio
async def test_query_with_limit(sql_chain, sample_schema):
    """Test SQL generation with limit clause."""
    query = "Show top 5 highest paid employees"
    
    result = await sql_chain.generate_sql(query, sample_schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "ORDER BY" in result.sql_query.upper()
    assert "DESC" in result.sql_query.upper()
    assert "LIMIT" in result.sql_query.upper()
    assert "5" in result.sql_query


@pytest.mark.asyncio
async def test_query_with_joins(sql_chain):
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
    
    query = "Show employee names and their department names"
    
    result = await sql_chain.generate_sql(query, schema)
    
    assert isinstance(result, SQLGenerationOutput)
    assert "JOIN" in result.sql_query.upper()
    assert "employees" in result.sql_query.lower()
    assert "departments" in result.sql_query.lower()
    assert "department_id" in result.sql_query.lower()


@pytest.mark.asyncio
async def test_error_handling(sql_chain, sample_schema):
    """Test error handling in SQL generation."""
    # Test with empty query
    with pytest.raises(SQLGenerationError) as exc_info:
        await sql_chain.generate_sql("", sample_schema)
    assert "Query cannot be empty" in str(exc_info.value)
    
    # Test with whitespace-only query
    with pytest.raises(SQLGenerationError) as exc_info:
        await sql_chain.generate_sql("   ", sample_schema)
    assert "Query cannot be empty" in str(exc_info.value)
    
    # Test with invalid schema
    invalid_schema = {"invalid": "schema"}
    with pytest.raises(SQLGenerationError) as exc_info:
        await sql_chain.generate_sql(
            "Show me all employees",
            invalid_schema
        )
    assert "Schema must contain table information" in str(exc_info.value)
    
    # Test with empty schema
    with pytest.raises(SQLGenerationError) as exc_info:
        await sql_chain.generate_sql("Show me all employees", {})
    assert "Schema cannot be empty" in str(exc_info.value) 