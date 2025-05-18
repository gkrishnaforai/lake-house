"""Test cases for the CSV cleaning agent."""

import pytest
import pandas as pd
import tempfile
import logging
from pathlib import Path
from etl_architect_agent_v2.agents.csv_cleaner import FileCleaningAgent
from etl_architect_agent_v2.core.llm_manager import LLMManager

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@pytest.fixture
def llm_manager():
    """Create an LLM manager instance for testing."""
    return LLMManager(
        model_name="gpt-3.5-turbo",
        temperature=0.0,
        max_tokens=1000
    )


@pytest.fixture
def cleaning_agent(llm_manager):
    """Create a file cleaning agent instance for testing."""
    return FileCleaningAgent(llm_manager)


@pytest.fixture
def sample_csv_data():
    """Create a sample CSV file with various data quality issues."""
    data = {
        "name": ["John", "Jane", None, "Bob", "Alice"],
        "age": ["25", "30", "unknown", "35", "40"],
        "email": [
            "john@example.com",
            "jane@example.com",
            "invalid_email",
            "bob@example.com",
            "alice@example.com"
        ],
        "date": [
            "2023-01-01",
            "2023-02-01",
            "invalid_date",
            "2023-03-01",
            "2023-04-01"
        ]
    }
    df = pd.DataFrame(data)
    
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        return Path(f.name)


@pytest.mark.asyncio
async def test_csv_cleaning_workflow(cleaning_agent, sample_csv_data):
    """Test the complete CSV cleaning workflow."""
    try:
        logger.debug("Starting CSV cleaning workflow test")
        logger.debug(f"Input file: {sample_csv_data}")
        
        # Clean the file
        logger.debug("Calling clean_file method")
        output_file = await cleaning_agent.clean_file(
            input_file=sample_csv_data,
            file_type="csv"
        )
        logger.debug(f"Output file: {output_file}")
        
        # Verify output file exists
        assert Path(output_file).exists()
        logger.debug("Output file exists")
        
        # Load cleaned data
        logger.debug("Loading cleaned data")
        cleaned_df = pd.read_csv(output_file)
        logger.debug(f"Cleaned data shape: {cleaned_df.shape}")
        logger.debug(f"Cleaned data columns: {cleaned_df.columns.tolist()}")
        logger.debug(f"Cleaned data dtypes: {cleaned_df.dtypes}")
        
        # Verify basic cleaning operations
        logger.debug("Verifying cleaning operations")
        logger.debug(f"Null values in name: {cleaned_df['name'].isna().sum()}")
        logger.debug(f"Age dtype: {cleaned_df['age'].dtype}")
        logger.debug(f"Email validation: {cleaned_df['email'].str.contains('@').all()}")
        logger.debug(f"Date validation: {pd.to_datetime(cleaned_df['date'], errors='coerce').notna().all()}")
        
        assert cleaned_df["name"].isna().sum() == 0  # Nulls handled
        assert cleaned_df["age"].dtype == "int64"  # Type conversion
        assert cleaned_df["email"].str.contains("@").all()  # Email validation
        assert pd.to_datetime(cleaned_df["date"], errors="coerce").notna().all()  # Date validation
        
        # Clean up
        logger.debug("Cleaning up test files")
        Path(output_file).unlink()
        sample_csv_data.unlink()
        logger.debug("Test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        logger.error("Test state at failure:")
        logger.error(f"Input file exists: {sample_csv_data.exists()}")
        if 'output_file' in locals():
            logger.error(f"Output file exists: {Path(output_file).exists()}")
        raise


@pytest.mark.asyncio
async def test_error_handling(cleaning_agent):
    """Test error handling for invalid inputs."""
    # Test with non-existent file
    with pytest.raises(RuntimeError):
        await cleaning_agent.clean_file(
            input_file="non_existent.csv",
            file_type="csv"
        )
    
    # Test with unsupported file type
    with pytest.raises(ValueError):
        await cleaning_agent.clean_file(
            input_file="test.txt",
            file_type="txt"
        )


@pytest.mark.asyncio
async def test_schema_analysis(cleaning_agent, sample_csv_data):
    """Test schema analysis functionality."""
    # Initialize state
    state = {
        "input_file": str(sample_csv_data),
        "file_type": "csv",
        "current_data": pd.read_csv(sample_csv_data),
        "cleaning_suggestions": [],
        "applied_cleaning": [],
        "conversation_history": [],
        "error": None,
        "output_file": None,
        "schema_analysis": None,
        "user_feedback": None
    }
    
    # Run schema analysis
    state = await cleaning_agent._analyze_schema(state)
    
    # Verify schema analysis results
    assert state["schema_analysis"] is not None
    assert "data_types" in state["schema_analysis"]
    assert "missing_patterns" in state["schema_analysis"]
    assert "quality_issues" in state["schema_analysis"]
    assert "relationships" in state["schema_analysis"]
    
    # Clean up
    sample_csv_data.unlink()


@pytest.mark.asyncio
async def test_cleaning_suggestions(cleaning_agent, sample_csv_data):
    """Test cleaning suggestions generation."""
    # Initialize state
    state = {
        "input_file": str(sample_csv_data),
        "file_type": "csv",
        "current_data": pd.read_csv(sample_csv_data),
        "cleaning_suggestions": [],
        "applied_cleaning": [],
        "conversation_history": [],
        "error": None,
        "output_file": None,
        "schema_analysis": {
            "data_types": {"age": "string", "date": "string"},
            "missing_patterns": {"name": "null"},
            "quality_issues": ["invalid dates", "invalid emails"],
            "relationships": []
        },
        "user_feedback": None
    }
    
    # Generate cleaning suggestions
    state = await cleaning_agent._suggest_cleaning(state)
    
    # Verify suggestions
    assert state["cleaning_suggestions"] is not None
    assert len(state["cleaning_suggestions"]) > 0
    
    for suggestion in state["cleaning_suggestions"]:
        assert "strategy" in suggestion
        assert "description" in suggestion
        assert "columns" in suggestion
        assert "parameters" in suggestion
        assert "confidence" in suggestion
    
    # Clean up
    sample_csv_data.unlink() 