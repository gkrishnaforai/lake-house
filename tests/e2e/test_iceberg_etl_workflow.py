"""End-to-end tests for ETL workflow with Iceberg tables."""

import pytest
import pandas as pd
import boto3
import os
from datetime import datetime
import logging
from io import BytesIO
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@pytest.fixture
def test_data() -> pd.DataFrame:
    """Create test data for ETL workflow."""
    return pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
        'department': ['Engineering', 'Sales', 'Engineering', 'Marketing', 'Sales'],
        'salary': [90000, 80000, 110000, 70000, 85000],
        'hire_date': ['2022-01-01', '2021-06-15', '2020-03-10', '2023-01-20', '2022-08-01']
    })

@pytest.fixture
def aws_credentials():
    """Set up AWS credentials for testing."""
    os.environ['AWS_ACCESS_KEY_ID'] = 'test'
    os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
    os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
    return {
        'aws_access_key_id': 'test',
        'aws_secret_access_key': 'test',
        'region_name': 'us-east-1'
    }

@pytest.fixture
def s3_client(aws_credentials):
    """Create S3 client for testing."""
    return boto3.client('s3', **aws_credentials)

@pytest.fixture
def glue_client(aws_credentials):
    """Create Glue client for testing."""
    return boto3.client('glue', **aws_credentials)

@pytest.fixture
def athena_client(aws_credentials):
    """Create Athena client for testing."""
    return boto3.client('athena', **aws_credentials)

@pytest.mark.asyncio
async def test_iceberg_etl_workflow(
    test_data: pd.DataFrame,
    s3_client,
    glue_client,
    athena_client
):
    """Test end-to-end ETL workflow with Iceberg tables."""
    # Generate unique identifiers
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    database_name = f"test_db_{timestamp}"
    table_name = f"employees_{timestamp}"
    bucket_name = "test-etl-bucket"
    
    try:
        # Step 1: Create test database
        logger.info(f"Creating test database: {database_name}")
        glue_client.create_database(
            DatabaseInput={
                'Name': database_name,
                'Description': 'Test database for ETL workflow'
            }
        )
        
        # Step 2: Upload test data to S3
        logger.info("Uploading test data to S3")
        parquet_buffer = BytesIO()
        test_data.to_parquet(parquet_buffer, index=False)
        parquet_buffer.seek(0)
        
        s3_key = f"raw/{table_name}/data.parquet"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=s3_key,
            Body=parquet_buffer.getvalue()
        )
        
        # Step 3: Create Iceberg table in Glue catalog
        logger.info("Creating Iceberg table in Glue catalog")
        table_input = {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'table_type': 'ICEBERG',
                'format': 'iceberg/parquet'
            },
            'StorageDescriptor': {
                'Columns': [
                    {'Name': 'id', 'Type': 'bigint'},
                    {'Name': 'name', 'Type': 'string'},
                    {'Name': 'department', 'Type': 'string'},
                    {'Name': 'salary', 'Type': 'bigint'},
                    {'Name': 'hire_date', 'Type': 'string'}
                ],
                'Location': f"s3://{bucket_name}/{s3_key}",
                'InputFormat': (
                    'org.apache.iceberg.mr.hive.HiveIcebergInputFormat'
                ),
                'OutputFormat': (
                    'org.apache.iceberg.mr.hive.HiveIcebergOutputFormat'
                ),
                'SerdeInfo': {
                    'SerializationLibrary': (
                        'org.apache.iceberg.mr.hive.HiveIcebergSerDe'
                    )
                }
            }
        }
        
        glue_client.create_table(
            DatabaseName=database_name,
            TableInput=table_input
        )
        
        # Step 4: Create transformed table using Athena
        logger.info("Creating transformed table using Athena")
        transform_query = f"""
        CREATE TABLE {database_name}.transformed_{table_name}
        WITH (
            format = 'ICEBERG',
            location = 's3://{bucket_name}/transformed/{table_name}/'
        ) AS
        SELECT 
            id,
            name,
            department,
            salary,
            CAST(hire_date AS DATE) as hire_date,
            CASE 
                WHEN salary >= 100000 THEN 'High'
                WHEN salary >= 80000 THEN 'Medium'
                ELSE 'Low'
            END as salary_category
        FROM {database_name}.{table_name}
        """
        
        query_execution = athena_client.start_query_execution(
            QueryString=transform_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={
                'OutputLocation': f's3://{bucket_name}/query-results/'
            }
        )
        
        # Wait for query to complete
        while True:
            query_status = athena_client.get_query_execution(
                QueryExecutionId=query_execution['QueryExecutionId']
            )['QueryExecution']['Status']['State']
            
            if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            
            await asyncio.sleep(1)
        
        assert query_status == 'SUCCEEDED', f"Query failed with status: {query_status}"
        
        # Step 5: Verify transformed data
        logger.info("Verifying transformed data")
        verify_query = f"""
        SELECT 
            department,
            salary_category,
            COUNT(*) as employee_count,
            AVG(salary) as avg_salary
        FROM {database_name}.transformed_{table_name}
        GROUP BY department, salary_category
        ORDER BY department, salary_category
        """
        
        query_execution = athena_client.start_query_execution(
            QueryString=verify_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={
                'OutputLocation': f's3://{bucket_name}/query-results/'
            }
        )
        
        # Wait for query to complete
        while True:
            query_status = athena_client.get_query_execution(
                QueryExecutionId=query_execution['QueryExecutionId']
            )['QueryExecution']['Status']['State']
            
            if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            
            await asyncio.sleep(1)
        
        assert query_status == 'SUCCEEDED', f"Query failed with status: {query_status}"
        
        # Get results
        results = athena_client.get_query_results(
            QueryExecutionId=query_execution['QueryExecutionId']
        )
        
        # Verify results
        assert len(results['ResultSet']['Rows']) > 1  # Header row + data rows
        
        # Step 6: Verify data quality
        logger.info("Verifying data quality")
        quality_query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT id) as unique_ids,
            COUNT(CASE WHEN salary > 0 THEN 1 END) as valid_salaries,
            COUNT(CASE WHEN hire_date IS NOT NULL THEN 1 END) as valid_dates
        FROM {database_name}.transformed_{table_name}
        """
        
        query_execution = athena_client.start_query_execution(
            QueryString=quality_query,
            QueryExecutionContext={'Database': database_name},
            ResultConfiguration={
                'OutputLocation': f's3://{bucket_name}/query-results/'
            }
        )
        
        # Wait for query to complete
        while True:
            query_status = athena_client.get_query_execution(
                QueryExecutionId=query_execution['QueryExecutionId']
            )['QueryExecution']['Status']['State']
            
            if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            
            await asyncio.sleep(1)
        
        assert query_status == 'SUCCEEDED', f"Query failed with status: {query_status}"
        
        # Get results
        results = athena_client.get_query_results(
            QueryExecutionId=query_execution['QueryExecutionId']
        )
        
        # Verify quality metrics
        quality_metrics = results['ResultSet']['Rows'][1]['Data']
        total_rows = int(quality_metrics[0]['VarCharValue'])
        unique_ids = int(quality_metrics[1]['VarCharValue'])
        valid_salaries = int(quality_metrics[2]['VarCharValue'])
        valid_dates = int(quality_metrics[3]['VarCharValue'])
        
        assert total_rows == len(test_data)
        assert unique_ids == len(test_data)
        assert valid_salaries == len(test_data)
        assert valid_dates == len(test_data)
        
    finally:
        # Cleanup
        logger.info("Cleaning up test resources")
        try:
            # Delete tables
            glue_client.delete_table(
                DatabaseName=database_name,
                Name=table_name
            )
            glue_client.delete_table(
                DatabaseName=database_name,
                Name=f"transformed_{table_name}"
            )
            
            # Delete database
            glue_client.delete_database(Name=database_name)
            
            # Delete S3 objects
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=s3_key
            )
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=f"transformed/{table_name}/"
            )
            
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            raise 