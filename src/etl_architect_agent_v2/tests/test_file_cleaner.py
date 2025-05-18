"""Tests for the intelligent file cleaning agent."""

import pytest
import pandas as pd
from pathlib import Path
import tempfile
from langchain_openai import ChatOpenAI
from etl_architect_agent_v2.agents.csv_cleaner import FileCleaningAgent, CleaningState


@pytest.fixture
def sample_csv():
    """Create a sample CSV file with various data issues."""
    data = {
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', None, 'Alice Brown'],
        'Age': ['25', '30', '35', '40', '45'],
        'Email': ['john@example.com', 'jane@example.com', 'bob@example.com', 
                 'alice@example.com', None],
        'Salary': ['$50,000', '$60,000', '$70,000', '$80,000', '$90,000'],
        'Date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-04-01', '2023-05-01']
    }
    df = pd.DataFrame(data)
    
    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        return Path(tmp.name)


@pytest.fixture
def llm():
    """Create a mock LLM for testing."""
    return ChatOpenAI(
        temperature=0,
        model="gpt-3.5-turbo"
    )


@pytest.fixture
def file_cleaner(llm):
    """Create an instance of the file cleaning agent."""
    return FileCleaningAgent(llm)


@pytest.mark.asyncio
async def test_file_cleaning_workflow(sample_csv, file_cleaner):
    """Test the complete file cleaning workflow."""
    # Clean the file
    cleaned_file = await file_cleaner.clean_file(sample_csv, "csv")
    
    # Read the cleaned file
    df_cleaned = pd.read_csv(cleaned_file)
    
    # Verify the cleaning results
    assert df_cleaned['Name'].isna().sum() == 0, (
        "Null values in Name column should be handled"
    )
    assert df_cleaned['Email'].isna().sum() == 0, (
        "Null values in Email column should be handled"
    )
    assert pd.to_numeric(df_cleaned['Age'], errors='coerce').notna().all(), (
        "Age should be numeric"
    )
    assert df_cleaned['Salary'].str.startswith('$').all(), (
        "Salary format should be preserved"
    )
    assert pd.to_datetime(df_cleaned['Date']).notna().all(), (
        "Dates should be valid"
    )
    
    # Clean up
    sample_csv.unlink()
    cleaned_file.unlink()


@pytest.mark.asyncio
async def test_state_management(sample_csv, file_cleaner):
    """Test the state management in the file cleaning workflow."""
    # Initialize state
    state = CleaningState(
        input_file=sample_csv,
        file_type="csv",
        current_data=None,
        cleaning_suggestions=[],
        applied_cleaning=[],
        conversation_history=[],
        error=None,
        output_file=None,
        schema_analysis=None,
        user_feedback=None
    )
    
    # Run through the workflow steps
    state = await file_cleaner._analyze_file(state)
    assert state["current_data"] is not None, "Data should be loaded"
    assert state["error"] is None, "No errors should occur during analysis"
    
    state = await file_cleaner._analyze_schema(state)
    assert state["schema_analysis"] is not None, (
        "Schema analysis should be generated"
    )
    
    state = await file_cleaner._suggest_cleaning(state)
    assert len(state["cleaning_suggestions"]) > 0, (
        "Cleaning suggestions should be generated"
    )
    
    state = await file_cleaner._apply_cleaning(state)
    assert len(state["applied_cleaning"]) > 0, (
        "Cleaning strategies should be applied"
    )
    
    state = await file_cleaner._validate_results(state)
    assert state["error"] is None, "No errors should occur during validation"
    
    # Clean up
    sample_csv.unlink()


@pytest.mark.asyncio
async def test_error_handling(file_cleaner):
    """Test error handling in the file cleaning workflow."""
    # Try to clean a non-existent file
    with pytest.raises(RuntimeError) as exc_info:
        await file_cleaner.clean_file(
            Path("non_existent.csv"),
            "csv"
        )
    assert "Error analyzing file" in str(exc_info.value)
    
    # Try to clean with an unsupported file type
    with pytest.raises(RuntimeError) as exc_info:
        await file_cleaner.clean_file(
            Path("test.pdf"),
            "pdf"
        )
    assert "PDF support coming soon" in str(exc_info.value) 