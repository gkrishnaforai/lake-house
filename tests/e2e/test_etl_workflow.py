import os
import tempfile

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from aws_architect_agent.graph.state.state_manager import StateManager
from aws_architect_agent.graph.workflow.workflow_manager import WorkflowManager
from aws_architect_agent.models.base import SolutionType
from aws_architect_agent.utils.logging import setup_logging

# Set up logging
setup_logging(log_level="DEBUG")

# Set up OpenAI API key for testing
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError(
        "Please set OPENAI_API_KEY environment variable for testing"
    )


@pytest.fixture
def sample_csv_data():
    """Create sample CSV data for testing."""
    data = {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "department": ["HR", "Engineering", "Marketing"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_csv_file(sample_csv_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        sample_csv_data.to_csv(temp_file.name, index=False)
        yield temp_file.name
        os.unlink(temp_file.name)


@pytest.fixture(scope="function")
def aws_credentials():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


@pytest.fixture(scope="function")
def s3(aws_credentials):
    """Mock S3 client."""
    with mock_aws():
        s3_client = boto3.client("s3")
        # Create test bucket
        s3_client.create_bucket(Bucket="test-bucket")
        yield s3_client


@pytest.fixture
def state_manager():
    """Create a state manager instance."""
    return StateManager()


@pytest.fixture
def workflow_manager(state_manager):
    """Create a workflow manager instance."""
    return WorkflowManager(state_manager)


def test_etl_workflow(
    temp_csv_file,
    s3,
    workflow_manager,
    state_manager,
):
    """Test the complete ETL workflow."""
    # Step 1: Initialize the architecture
    state_manager.initialize_architecture(
        name="ETL Pipeline",
        description="ETL pipeline for CSV to S3 with Athena reporting",
        solution_type=SolutionType.ETL,
    )

    # Step 2: Process requirements gathering
    response = workflow_manager.process_message(
        "I need an ETL pipeline that:\n"
        "1. Reads CSV files from local storage\n"
        "2. Uploads them to S3\n"
        "3. Creates an Athena table\n"
        "4. Generates a report\n\n"
        "The CSV files contain employee data with columns: "
        "id, name, age, and department. We need to analyze "
        "employee distribution across departments and age groups."
    )
    assert "requirements_analyzed" in response.lower(), (
        f"Expected 'requirements_analyzed' in response, got: {response}"
    )

    # Step 3: Process architecture design
    response = workflow_manager.process_message(
        "Design the architecture for this ETL pipeline. "
        "Consider data security, scalability, and cost optimization. "
        "Include specific AWS services and their configurations."
    )
    assert "architecture_designed" in response.lower(), (
        f"Expected 'architecture_designed' in response, got: {response}"
    )

    # Step 4: Process architecture review
    response = workflow_manager.process_message(
        "Review the architecture against AWS best practices. "
        "Focus on security, reliability, performance efficiency, "
        "cost optimization, and operational excellence."
    )
    assert "architecture_reviewed" in response.lower(), (
        f"Expected 'architecture_reviewed' in response, got: {response}"
    )

    # Step 5: Process architecture refinement
    response = workflow_manager.process_message(
        "Refine the architecture based on the review feedback. "
        "Ensure all security best practices are implemented and "
        "the solution is cost-effective for the given scale."
    )
    assert "architecture_refined" in response.lower(), (
        f"Expected 'architecture_refined' in response, got: {response}"
    )

    # Step 6: Upload CSV to S3
    s3.upload_file(
        temp_csv_file,
        "test-etl-bucket",
        "input/sample_data.csv",
    )

    # Step 7: Create Athena table
    create_table_query = """
    CREATE EXTERNAL TABLE IF NOT EXISTS test_db.sample_data (
        id INT,
        name STRING,
        age INT,
        department STRING
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES (
        'separatorChar' = ',',
        'quoteChar' = '"',
        'escapeChar' = '\\\\'
    )
    STORED AS TEXTFILE
    LOCATION 's3://test-etl-bucket/input/'
    TBLPROPERTIES ('skip.header.line.count'='1')
    """

    athena_client = boto3.client("athena")
    query_execution = athena_client.start_query_execution(
        QueryString=create_table_query,
        QueryExecutionContext={"Database": "test_db"},
        ResultConfiguration={
            "OutputLocation": "s3://test-etl-bucket/output/"
        },
    )

    # Step 8: Generate report
    report_query = """
    SELECT department, COUNT(*) as employee_count, AVG(age) as avg_age
    FROM test_db.sample_data
    GROUP BY department
    ORDER BY employee_count DESC
    """

    query_execution = athena_client.start_query_execution(
        QueryString=report_query,
        QueryExecutionContext={"Database": "test_db"},
        ResultConfiguration={
            "OutputLocation": "s3://test-etl-bucket/output/"
        },
    )

    # Verify the workflow state
    current_state = state_manager.get_current_state().value
    assert current_state == "finalized", (
        f"Expected state 'finalized', got: {current_state}"
    )
    assert state_manager.get_current_architecture() is not None, (
        "Expected current architecture to be set"
    )
    assert len(state_manager.get_conversation_history()) > 0, (
        "Expected conversation history to be non-empty"
    )

    # Verify S3 upload
    s3_objects = s3.list_objects_v2(Bucket="test-etl-bucket")
    assert "input/sample_data.csv" in [
        obj["Key"] for obj in s3_objects.get("Contents", [])
    ], "Expected sample_data.csv to be in S3 bucket"

    # Verify Athena query execution
    assert query_execution["QueryExecutionId"] is not None, (
        "Expected QueryExecutionId to be set"
    ) 