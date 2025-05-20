"""Test full cycle of data processing, transformation, and schema updates."""

import pytest
import pandas as pd
from fastapi import UploadFile
from etl_architect_agent_v2.agents.transformation.transformation_agent import (
    TransformationAgent,
    TransformationConfig,
    CategorizationConfig
)
from etl_architect_agent_v2.backend.services.catalog_service import CatalogService
from etl_architect_agent_v2.backend.config import get_settings
from etl_architect_agent_v2.core.llm.manager import LLMManager
import io
from io import BytesIO
from datetime import datetime


@pytest.fixture
def test_data():
    """Create test data for company descriptions."""
    return pd.DataFrame({
        'company_id': ['C1', 'C2', 'C3'],
        'name': [
            'TechAI Solutions',
            'Green Energy Corp',
            'DataFlow Systems'
        ],
        'description': [
            'Leading provider of AI-powered analytics and machine learning solutions for enterprise businesses. '
            'We specialize in natural language processing and computer vision technologies.',
            'Renewable energy company focused on solar and wind power solutions. '
            'We provide sustainable energy alternatives for residential and commercial properties.',
            'Enterprise software company providing data integration and ETL solutions. '
            'Our platform helps businesses streamline their data workflows.'
        ],
        'revenue': ['50M', '30M', '25M'],
        'employees': [250, 150, 120],
        'city': ['San Francisco', 'Austin', 'Boston'],
        'state': ['CA', 'TX', 'MA']
    })


@pytest.fixture
def catalog_service():
    """Create catalog service instance."""
    settings = get_settings()
    return CatalogService(
        bucket=settings.AWS_S3_BUCKET,
        aws_region=settings.AWS_REGION
    )


@pytest.fixture
def transformation_agent():
    """Create transformation agent instance."""
    llm_manager = LLMManager()
    return TransformationAgent(llm_manager)


# Add utility function to cast DataFrame columns to match Glue schema
async def cast_df_to_glue_schema(df, catalog_service, table_name, user_id):
    """Cast DataFrame columns to match the Glue table schema."""
    # Fetch Glue table schema
    schema_response = await catalog_service.get_table_schema(table_name, user_id)
    glue_schema = schema_response.get("schema", [])
    
    # Mapping Glue types to pandas dtypes
    glue_to_pandas = {
        'string': 'string',
        'int': 'Int64',
        'bigint': 'Int64',
        'double': 'float',
        'boolean': 'bool',
        # add more as needed
    }
    
    # Cast DataFrame columns to match Glue schema
    for col in glue_schema:
        col_name = col.get('name')
        glue_type = col.get('type', 'string')
        pandas_type = glue_to_pandas.get(glue_type, 'string')
        if col_name in df.columns:
            df[col_name] = df[col_name].astype(pandas_type)
    
    return df


@pytest.mark.asyncio
async def test_full_cycle_transformation(
    test_data,
    catalog_service,
    transformation_agent,
    tmp_path
):
    """Test full cycle of data processing, transformation, and schema updates."""
    # 1. Save test data as CSV
    csv_path = tmp_path / "companies.csv"
    test_data.to_csv(csv_path, index=False)
    
    # 2. Create UploadFile object
    with open(csv_path, "rb") as f:
        file_bytes = f.read()
    upload_file = UploadFile(
        filename="companies.csv",
        file=io.BytesIO(file_bytes)
    )
    
    # 3. Check if table exists and delete if it does
    table_name = "test_companies"
    try:
        await catalog_service.get_table(
            table_name,
            user_id="test_user"
        )
        await catalog_service.delete_table(
            table_name,
            user_id="test_user"
        )
    except Exception:
        pass  # Table doesn't exist, which is fine
    
    # 4. Upload CSV to S3 and process it
    upload_result = await catalog_service.upload_file(
        file=upload_file,
        table_name=table_name,
        create_new=True,
        user_id="test_user"
    )
    assert upload_result["status"] == "success"
    
    # 5. Verify table exists in catalog
    table_info = await catalog_service.get_table(
        table_name,
        user_id="test_user"
    )
    assert table_info["name"] == table_name
    parquet_location = table_info["location"]
    
    # 6. Apply transformation to add isAI column
    config = TransformationConfig(
        categorization=CategorizationConfig(
            categories=["AI", "Non-AI"],
            confidence_threshold=0.7
        )
    )
    
    transform_result = await transformation_agent.apply_transformation(
        data=test_data.to_dict("records"),
        transformation_type="categorization",
        config=config
    )
    assert transform_result.transformed_data, "No transformed data returned"
    print("Transformation result:", transform_result.transformed_data)
    
    # 7. Update existing Parquet file with transformed data
    transformed_df = pd.DataFrame([
        {**item["original"], **item["transformed"]}
        for item in transform_result.transformed_data
    ])
    print("Transformed DataFrame columns:", transformed_df.columns.tolist())
    print("Transformed DataFrame values:\n", transformed_df.values)
    
    # Explicitly cast employees to bigint
    transformed_df['employees'] = transformed_df['employees'].astype('int64')
    
    # Cast DataFrame to match Glue schema before saving to Parquet
    transformed_df = await cast_df_to_glue_schema(transformed_df, catalog_service, table_name, "test_user")
    
    # Generate new S3 key with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_s3_key = parquet_location.replace(f"s3://{catalog_service.bucket}/", "")
    new_s3_key = base_s3_key.replace('.parquet', f'_{timestamp}.parquet')
    
    # Convert to Parquet buffer
    parquet_buffer = BytesIO()
    transformed_df.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    
    # Upload to S3 with new key
    catalog_service.s3.put_object(
        Bucket=catalog_service.bucket,
        Key=new_s3_key,
        Body=parquet_buffer.getvalue()
    )
    print(f"Uploaded transformed data to s3://{catalog_service.bucket}/{new_s3_key}")
    
    # Prepare new schema for Glue update
    def pandas_dtype_to_glue_type(dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return 'bigint'
        elif pd.api.types.is_float_dtype(dtype):
            return 'double'
        elif pd.api.types.is_bool_dtype(dtype):
            return 'boolean'
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return 'timestamp'
        else:
            return 'string'

    updated_columns = [
        {
            'name': col,
            'type': pandas_dtype_to_glue_type(transformed_df[col].dtype),
            'description': f'Transformed column {col}'
        }
        for col in transformed_df.columns
    ]
    new_schema = {'fields': updated_columns}
    new_location = f"s3://{catalog_service.bucket}/data/test_user/{table_name}/"
    
    # Update Glue table using correct signature
    await catalog_service.glue_service.update_table(
        database_name=f"user_test_user",
        table_name=table_name,
        schema=new_schema,
        location=new_location,
        file_format="parquet"
    )
    
    # Run MSCK REPAIR TABLE to refresh partitions/metadata
    await catalog_service.execute_query(
        f"MSCK REPAIR TABLE {table_name}",
        user_id="test_user"
    )
    
    # 9. Verify schema update
    updated_schema = await catalog_service.get_table_schema(
        table_name,
        user_id="test_user"
    )
    print("Updated Glue table schema:", updated_schema)
    assert any(
        col["name"] == "is_ai_company"
        for col in updated_schema["schema"]
    )
    
    # 10. Query transformed data
    query_result = await catalog_service.execute_query(
        f"SELECT * FROM {table_name}",
        user_id="test_user"
    )
    assert query_result["status"] == "success"
    assert len(query_result["results"]) > 0
    print("Full query result:", query_result["results"])
    # Handle both list-of-lists and list-of-dicts
    first_row = query_result["results"][0]
    if isinstance(first_row, dict):
        assert "is_ai_company" in first_row
        assert "is_ai_company_ai_confidence" in first_row
        assert "is_ai_company_ai_reasoning" in first_row
    elif isinstance(first_row, list):
        assert "is_ai_company" in first_row, f"Header row: {first_row}"
        assert "is_ai_company_ai_confidence" in first_row, f"Header row: {first_row}"
        assert "is_ai_company_ai_reasoning" in first_row, f"Header row: {first_row}"
    else:
        raise AssertionError(f"Unexpected result row type: {type(first_row)}") 