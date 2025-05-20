"""Test SQL generation functionality."""

import pytest
import pandas as pd
import os
from datetime import datetime
import logging
from io import BytesIO
import boto3
import time
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from src.etl_architect_agent_v2.backend.services.catalog_service import (
    CatalogService
)
from src.etl_architect_agent_v2.agents.catalog_agent import CatalogAgent
from src.etl_architect_agent_v2.agents.schema_extractor.sql_generator_agent import (
    SQLGeneratorAgent
)
from src.etl_architect_agent_v2.core.llm.manager import (
    LLMManager
)
from src.etl_architect_agent_v2.backend.services.sql_generation_service import (
    SQLGenerationService,
    SQLGenerationRequest
)
from src.etl_architect_agent_v2.backend.services.glue_service import (
    GlueService,
    GlueTableConfig
)

class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

@pytest.fixture
def aws_credentials():
    """Ensure AWS credentials are set."""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_SESSION_TOKEN',
        'AWS_REGION',
        'AWS_S3_BUCKET'
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        pytest.skip(f"Missing AWS credentials: {', '.join(missing)}")

@pytest.fixture
def catalog_service(aws_credentials):
    """Create Catalog Service instance."""
    bucket = os.getenv('AWS_S3_BUCKET')
    if not bucket:
        pytest.skip("Missing AWS_S3_BUCKET environment variable")
    logger.info(f"Creating CatalogService with bucket: {bucket}")
    service = CatalogService(
        bucket=bucket,
        aws_region=os.getenv('AWS_REGION', 'us-east-1')
    )
    return service

@pytest.fixture
def catalog_agent(catalog_service):
    """Create Catalog Agent instance."""
    logger.info("Creating CatalogAgent")
    return CatalogAgent(catalog_service)

@pytest.fixture
def sql_generator_agent():
    """Create SQL Generator Agent instance."""
    logger.info("Creating SQLGeneratorAgent")
    llm_manager = LLMManager()
    return SQLGeneratorAgent(llm_manager)

@pytest.fixture
def test_table_data():
    """Create sample table data for testing."""
    return {
        'employee_id': [1, 2, 3, 4],
        'name': ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown"],
        'department': ["Engineering", "Sales", "Engineering", "Marketing"],
        'salary': [90000, 80000, 110000, 70000],
        'role': [
            "Engineer II", "Sales Rep", "Senior Engineer", "Marketing Specialist"
        ],
        'start_date': ['2022-05-01', '2021-11-15', '2020-03-10', '2023-01-20']
    }

@pytest.mark.asyncio
async def test_schema_transformation(catalog_service, test_table_data):
    """Test schema transformation functionality."""
    # Create test table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_schema_{timestamp}"
    user_id = "test_user"
    
    try:
        # Create database if it doesn't exist
        try:
            catalog_service.glue_service.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
        except catalog_service.glue_service.exceptions.AlreadyExistsException:
            pass
        
        # Create test data
        df = pd.DataFrame(test_table_data)
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = (
            f"data/{user_id}/{table_name}/test_data.parquet"
        )
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Generate schema using service's _pandas_to_athena_type method
        columns = []
        for col_name, dtype in df.dtypes.items():
            athena_type = catalog_service._pandas_to_athena_type(str(dtype))
            columns.append({
                'Name': col_name,
                'Type': athena_type
            })
        
        # Create Glue table
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': columns,
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.'
                        'ParquetHiveSerDe'
                    )
                }
            },
            'PartitionKeys': []
        }
        
        catalog_service.glue_service.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        
        # Test schema transformation
        schema = await catalog_service.get_table_schema(table_name, user_id)
        
        # Verify schema
        assert isinstance(schema, dict)
        assert "schema" in schema
        assert len(schema["schema"]) == 6  # We have 6 columns
        column_names = {col["name"] for col in schema["schema"]}
        assert column_names == {
            "employee_id", "name", "department", "salary", "role", "start_date"
        }
        
        # Verify column types match the service's type mapping
        for col in schema["schema"]:
            original_dtype = str(df[col["name"]].dtype)
            expected_type = catalog_service._pandas_to_athena_type(original_dtype)
            assert col["type"] == expected_type, (
                f"Type mismatch for column {col['name']}: "
                f"expected {expected_type}, got {col['type']}"
            )
        
    finally:
        # Cleanup
        try:
            catalog_service.glue_service.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            catalog_service.s3.delete_object(
                Bucket=catalog_service.bucket,
                Key=s3_key
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")

@pytest.mark.asyncio
async def test_sql_generation(catalog_service, test_table_data):
    """Test SQL generation functionality."""
    # Create test table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_sql_{timestamp}"
    user_id = "test_user"
    
    try:
        # Create database if it doesn't exist
        try:
            catalog_service.glue_service.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
        except catalog_service.glue_service.exceptions.AlreadyExistsException:
            pass
        
        # Create test data
        df = pd.DataFrame(test_table_data)
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = (
            f"data/{user_id}/{table_name}/test_data.parquet"
        )
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Generate schema using service's _pandas_to_athena_type method
        columns = []
        for col_name, dtype in df.dtypes.items():
            athena_type = catalog_service._pandas_to_athena_type(str(dtype))
            columns.append({
                'Name': col_name,
                'Type': athena_type
            })
        
        # Create Glue table
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': columns,
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.'
                        'ParquetHiveSerDe'
                    )
                }
            },
            'PartitionKeys': []
        }
        
        catalog_service.glue_service.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        
        # Test SQL generation
        query = "Show me all employees in the Engineering department"
        result = await catalog_service.process_descriptive_query(
            query=query,
            table_name=table_name,
            user_id=user_id
        )
        
        # Verify SQL generation
        assert result["status"] == "success"
        assert "query" in result
        assert "SELECT" in result["query"].upper()
        assert "FROM" in result["query"].upper()
        assert "WHERE" in result["query"].upper()
        assert "department" in result["query"].lower()
        assert "engineering" in result["query"].lower()
        
    finally:
        # Cleanup
        try:
            catalog_service.glue_service.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            catalog_service.s3.delete_object(
                Bucket=catalog_service.bucket,
                Key=s3_key
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")

@pytest.mark.asyncio
async def test_complex_sql_generation(catalog_service, test_table_data):
    """Test complex SQL generation functionality."""
    # Create test table
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_table_complex_sql_{timestamp}"
    user_id = "test_user"
    
    try:
        # Create database if it doesn't exist
        try:
            catalog_service.glue_service.create_database(
                DatabaseInput={
                    'Name': f"user_{user_id}",
                    'Description': f"Database for user {user_id}"
                }
            )
        except catalog_service.glue_service.exceptions.AlreadyExistsException:
            pass
        
        # Create test data
        df = pd.DataFrame(test_table_data)
        
        # Convert to Parquet
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        # Upload to S3
        s3_key = (
            f"data/{user_id}/{table_name}/test_data.parquet"
        )
        catalog_service.s3.put_object(
            Bucket=catalog_service.bucket,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Create Glue table
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file'
            },
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'employee_id', 'Type': 'bigint'},
                    {'Name': 'name', 'Type': 'string'},
                    {'Name': 'department', 'Type': 'string'},
                    {'Name': 'salary', 'Type': 'bigint'},
                    {'Name': 'role', 'Type': 'string'},
                    {'Name': 'start_date', 'Type': 'string'}
                ],
                'Location': f"s3://{catalog_service.bucket}/{s3_key}",
                'InputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.hadoop.hive.ql.io.parquet.'
                    'MapredParquetOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.hadoop.hive.ql.io.parquet.serde.'
                        'ParquetHiveSerDe'
                    )
                }
            },
            'PartitionKeys': []
        }
        
        catalog_service.glue_service.create_table(
            DatabaseName=f"user_{user_id}",
            TableInput=table_input
        )
        
        # Test complex SQL generation
        query = (
            "What is the average salary by department, and show me only "
            "departments with average salary above 80000?"
        )
        result = await catalog_service.process_descriptive_query(
            query=query,
            table_name=table_name,
            user_id=user_id
        )
        
        # Verify SQL generation
        assert result["status"] == "success"
        assert "query" in result
        assert "SELECT" in result["query"].upper()
        assert "FROM" in result["query"].upper()
        assert "GROUP BY" in result["query"].upper()
        assert "HAVING" in result["query"].upper()
        assert "AVG" in result["query"].upper()
        assert "salary" in result["query"].lower()
        assert "department" in result["query"].lower()
        
        # Verify additional metadata
        assert "explanation" in result
        assert "confidence" in result
        assert "tables_used" in result
        assert "columns_used" in result
        assert "filters" in result
        
    finally:
        # Cleanup
        try:
            catalog_service.glue_service.delete_table(
                DatabaseName=f"user_{user_id}",
                Name=table_name
            )
            catalog_service.s3.delete_object(
                Bucket=catalog_service.bucket,
                Key=s3_key
            )
        except Exception as e:
            logger.warning(f"Cleanup failed: {str(e)}")

@pytest.mark.asyncio
async def test_generate_sql_from_description():
    """Integration test: Generate SQL from description 'show top 2 rows'."""
    # Setup services
    llm_manager = LLMManager()
    glue_service = GlueService(region_name="us-east-1")
    user_id = "test_user"
    table_name = "abc123"  # Use existing table
    database_name = "user_test_user"

    try:
        # Initialize SQL generation service
        service = SQLGenerationService(
            llm_manager=llm_manager,
            glue_service=glue_service
        )

        # First verify AWS credentials and connection
        try:
            sts = boto3.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS Identity: {identity}")
        except Exception as e:
            logger.error(f"AWS Credentials/Connection Error: {str(e)}")
            raise

        # Verify database exists
        try:
            databases = glue_service.glue_client.get_databases()
            available_dbs = [db['Name'] for db in databases['DatabaseList']]
            logger.info(f"Available databases: {available_dbs}")
            
            if database_name not in available_dbs:
                logger.error(f"Database {database_name} not found in available databases")
                raise ValueError(f"Database {database_name} not found")
            logger.info(f"Successfully verified database {database_name} exists")
        except Exception as e:
            logger.error(f"Error checking databases: {str(e)}")
            raise

        # Verify table exists and get its schema
        try:
            tables = glue_service.glue_client.get_tables(DatabaseName=database_name)
            available_tables = [t['Name'] for t in tables['TableList']]
            logger.info(f"Available tables in {database_name}: {available_tables}")
            
            if table_name not in available_tables:
                logger.error(f"Table {table_name} not found in database {database_name}")
                raise ValueError(f"Table {table_name} not found")
            
            # Get table details
            table_response = glue_service.glue_client.get_table(
                DatabaseName=database_name,
                Name=table_name
            )
            table_info = table_response['Table']
            logger.info(f"Table details: {json.dumps(table_info, indent=2, cls=DateTimeEncoder)}")
            
            # Extract schema
            schema = {
                "columns": [
                    {
                        "name": col["Name"],
                        "type": col["Type"],
                        "comment": col.get("Comment", "")
                    }
                    for col in table_info["StorageDescriptor"]["Columns"]
                ]
            }
            logger.info(f"Extracted schema: {json.dumps(schema, indent=2)}")
            
        except Exception as e:
            logger.error(f"Error checking tables: {str(e)}")
            raise

        # Generate SQL
        try:
            sql_request = SQLGenerationRequest(
                query="show top 2 rows",
                schema=schema,
                table_name=table_name,
                preserve_column_names=True,
                user_id=user_id
            )
            
            logger.info("Generating SQL with request:")
            logger.info(json.dumps(sql_request.dict(), indent=2))
            
            response = await service.generate_sql(sql_request)
            logger.info(f"SQL Generation Response: {json.dumps(response.dict(), indent=2)}")
            
            assert response.status == "success", f"SQL generation failed: {response.error}"
            assert response.sql_query, "Generated SQL query is empty"
            
            # Execute the generated query
            query_result = await service.execute_query(response.sql_query, user_id)
            logger.info(f"Query Execution Result: {json.dumps(query_result, indent=2)}")
            
            assert query_result["status"] == "success", f"Query execution failed: {query_result.get('message')}"
            assert len(query_result.get("results", [])) > 0, "Query returned no results"
            
        except Exception as e:
            logger.error(f"Error in SQL generation/execution: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise 