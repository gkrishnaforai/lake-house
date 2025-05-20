"""Integration tests for the transformation service."""

import pytest
import pandas as pd

from etl_architect_agent_v2.backend.services.transformation_service import (
    TransformationService
)
from etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from etl_architect_agent_v2.backend.config import get_settings
from etl_architect_agent_v2.core.exceptions import ValidationError

# Test data constants
TEST_USER_ID = "test_user"
TEST_TABLE_NAME = "test_companies"
TEST_DATA = pd.DataFrame({
    'company_id': ['C1', 'C2', 'C3'],
    'name': [
        'TechAI Solutions',
        'Green Energy Corp',
        'DataFlow Systems'
    ],
    'description': [
        'Leading provider of AI-powered analytics and machine learning '
        'solutions.',
        'Renewable energy company focused on solar and wind power solutions.',
        'Enterprise software company providing data integration and ETL '
        'solutions.'
    ],
    'revenue': ['50M', '30M', '25M'],
    'employees': [250, 150, 120],
    'city': ['San Francisco', 'Austin', 'Boston'],
    'state': ['CA', 'TX', 'MA']
})

@pytest.fixture(scope="module")
def settings():
    """Get application settings."""
    return get_settings()

@pytest.fixture(scope="module")
def catalog_service(settings):
    """Create catalog service instance."""
    return CatalogService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

@pytest.fixture(scope="module")
def transformation_service(settings):
    """Create transformation service instance."""
    return TransformationService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )

@pytest.fixture(scope="function")
async def setup_test_table(catalog_service):
    """Set up test table and clean up after test."""
    # Create test table
    try:
        await catalog_service.get_table(TEST_TABLE_NAME, TEST_USER_ID)
        await catalog_service.delete_table(TEST_TABLE_NAME, TEST_USER_ID)
    except Exception:
        pass  # Table doesn't exist, which is fine

    # Upload test data
    csv_buffer = TEST_DATA.to_csv(index=False).encode()
    await catalog_service.upload_file(
        file=csv_buffer,
        table_name=TEST_TABLE_NAME,
        create_new=True,
        user_id=TEST_USER_ID
    )

    yield

    # Cleanup
    try:
        await catalog_service.delete_table(TEST_TABLE_NAME, TEST_USER_ID)
    except Exception:
        pass

@pytest.mark.asyncio
async def test_get_available_transformations(transformation_service):
    """Test getting available transformations."""
    transformations = await transformation_service.get_available_transformations()
    assert isinstance(transformations, list)
    assert len(transformations) > 0
    assert all(isinstance(t, dict) for t in transformations)
    assert all("id" in t and "type" in t for t in transformations)

@pytest.mark.asyncio
async def test_get_transformation_tool(transformation_service):
    """Test getting a specific transformation tool."""
    # Test existing tool
    tool = await transformation_service.get_transformation_tool(
        "categorization"
    )
    assert tool is not None
    assert tool["id"] == "categorization"
    assert "type" in tool
    assert "description" in tool

    # Test non-existent tool
    tool = await transformation_service.get_transformation_tool("non_existent")
    assert tool is None

@pytest.mark.asyncio
async def test_get_table_columns(transformation_service, setup_test_table):
    """Test getting table columns."""
    columns = await transformation_service.get_table_columns(
        TEST_TABLE_NAME,
        TEST_USER_ID
    )
    assert isinstance(columns, list)
    assert len(columns) > 0
    assert all(isinstance(col, dict) for col in columns)
    assert all("name" in col and "type" in col for col in columns)
    
    # Verify expected columns exist
    column_names = [col["name"] for col in columns]
    for expected_col in ["company_id", "name", "description", "revenue", "employees"]:
        assert expected_col in column_names

@pytest.mark.asyncio
async def test_apply_transformation_categorization(
    transformation_service,
    setup_test_table
):
    """Test applying categorization transformation."""
    # Apply transformation
    result = await transformation_service.apply_transformation(
        table_name=TEST_TABLE_NAME,
        tool_id="categorization",
        source_columns=["description"],
        user_id=TEST_USER_ID
    )

    # Verify result structure
    assert result["status"] == "success"
    assert "new_columns" in result
    assert "preview_data" in result

    # Verify new columns
    new_columns = result["new_columns"]
    assert any(col["name"] == "is_ai_company" for col in new_columns)
    assert any(col["name"] == "is_ai_company_confidence" for col in new_columns)

    # Verify preview data
    preview_data = result["preview_data"]
    assert isinstance(preview_data, list)
    assert len(preview_data) > 0
    assert all("is_ai_company" in row for row in preview_data)
    assert all("is_ai_company_confidence" in row for row in preview_data)

@pytest.mark.asyncio
async def test_apply_transformation_sentiment(
    transformation_service,
    setup_test_table
):
    """Test applying sentiment analysis transformation."""
    # Apply transformation
    result = await transformation_service.apply_transformation(
        table_name=TEST_TABLE_NAME,
        tool_id="sentiment",
        source_columns=["description"],
        user_id=TEST_USER_ID
    )

    # Verify result structure
    assert result["status"] == "success"
    assert "new_columns" in result
    assert "preview_data" in result

    # Verify new columns
    new_columns = result["new_columns"]
    assert any(col["name"] == "sentiment_score" for col in new_columns)
    assert any(col["name"] == "sentiment_label" for col in new_columns)

    # Verify preview data
    preview_data = result["preview_data"]
    assert isinstance(preview_data, list)
    assert len(preview_data) > 0
    assert all("sentiment_score" in row for row in preview_data)
    assert all("sentiment_label" in row for row in preview_data)

@pytest.mark.asyncio
async def test_apply_transformation_invalid_columns(
    transformation_service,
    setup_test_table
):
    """Test applying transformation with invalid columns."""
    with pytest.raises(ValidationError) as exc_info:
        await transformation_service.apply_transformation(
            table_name=TEST_TABLE_NAME,
            tool_id="categorization",
            source_columns=["non_existent_column"],
            user_id=TEST_USER_ID
        )
    assert "Invalid source columns" in str(exc_info.value)

@pytest.mark.asyncio
async def test_apply_transformation_invalid_tool(
    transformation_service,
    setup_test_table
):
    """Test applying transformation with invalid tool."""
    with pytest.raises(ValidationError) as exc_info:
        await transformation_service.apply_transformation(
            table_name=TEST_TABLE_NAME,
            tool_id="non_existent_tool",
            source_columns=["description"],
            user_id=TEST_USER_ID
        )
    assert "Transformation tool not found" in str(exc_info.value)

@pytest.mark.asyncio
async def test_transformation_template_operations(
    transformation_service,
    setup_test_table
):
    """Test transformation template operations."""
    # Create test template
    template = {
        "name": "test_template",
        "description": "Test template for categorization",
        "tool_id": "categorization",
        "source_columns": ["description"],
        "config": {
            "categories": ["AI", "Non-AI"],
            "confidence_threshold": 0.7
        }
    }

    # Save template
    save_result = await transformation_service.save_transformation_template(
        template=template,
        user_id=TEST_USER_ID
    )
    assert save_result["status"] == "success"

    # Get templates
    templates = await transformation_service.get_transformation_templates(
        TEST_USER_ID
    )
    assert len(templates) > 0
    assert any(t.name == "test_template" for t in templates)

    # Delete template
    delete_result = await transformation_service.delete_transformation_template(
        template_name="test_template",
        user_id=TEST_USER_ID
    )
    assert delete_result["status"] == "success"

    # Verify template is deleted
    templates = await transformation_service.get_transformation_templates(
        TEST_USER_ID
    )
    assert not any(t.name == "test_template" for t in templates) 