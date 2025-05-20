import pytest
import pandas as pd
from io import BytesIO
import asyncio
from datetime import datetime
import logging


logger = logging.getLogger(__name__)


@pytest.fixture
def test_data_fixture():
    """Create test data fixture with realistic business data."""
    data = {
        'customer_id': list(range(1, 21)),
        'customer_name': [f"Customer {i}" for i in range(1, 21)],
        'segment': ["Enterprise", "SMB", "Startup"] * 7,
        'annual_revenue': [1000000 + i * 50000 for i in range(20)],
        'industry': ["Technology", "Healthcare", "Finance", "Retail"] * 5,
        'region': ["North", "South", "East", "West"] * 5,
        'last_purchase_date': [f"2024-{i:02d}-01" for i in range(1, 21)]
    }
    return pd.DataFrame(data)


@pytest.fixture
def test_table_setup(catalog_service, test_data_fixture):
    """Setup test table with data in S3 and Glue."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    table_name = f"test_integration_table_{timestamp}"
    user_id = "test_user"
    
    # Convert to Parquet
    parquet_buffer = BytesIO()
    test_data_fixture.to_parquet(parquet_buffer, index=False)
    parquet_buffer.seek(0)
    
    # Upload to S3
    s3_key = f"data/{user_id}/{table_name}/test_data.parquet"
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
                {'Name': 'customer_id', 'Type': 'bigint'},
                {'Name': 'customer_name', 'Type': 'string'},
                {'Name': 'segment', 'Type': 'string'},
                {'Name': 'annual_revenue', 'Type': 'bigint'},
                {'Name': 'industry', 'Type': 'string'},
                {'Name': 'region', 'Type': 'string'},
                {'Name': 'last_purchase_date', 'Type': 'string'}
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
        }
    }
    
    catalog_service.glue.create_table(
        DatabaseName=f"user_{user_id}",
        TableInput=table_input
    )
    
    yield {
        'table_name': table_name,
        'user_id': user_id,
        's3_key': s3_key
    }
    
    # Cleanup
    try:
        catalog_service.glue.delete_table(
            DatabaseName=f"user_{user_id}",
            Name=table_name
        )
        catalog_service.s3.delete_object(
            Bucket=catalog_service.bucket,
            Key=s3_key
        )
    except Exception as e:
        logger.warning("Cleanup error: %s", str(e))


@pytest.mark.asyncio
async def test_descriptive_query_integration(
    catalog_service,
    test_table_setup,
    test_data_fixture
):
    """Test comprehensive descriptive query functionality."""
    table_name = test_table_setup['table_name']
    user_id = test_table_setup['user_id']
    
    # Wait for table to be available
    await asyncio.sleep(10)
    
    test_cases = [
        {
            'query': (
                "Show me all Enterprise customers in the Technology industry"
            ),
            'expected_conditions': {
                'segment': 'Enterprise',
                'industry': 'Technology'
            }
        },
        {
            'query': "What is the average revenue by industry?",
            'expected_aggregation': {
                'column': 'annual_revenue',
                'function': 'AVG',
                'group_by': 'industry'
            }
        },
        {
            'query': "Show me the top 5 customers by revenue",
            'expected_limit': 5,
            'expected_order': {
                'column': 'annual_revenue',
                'direction': 'DESC'
            }
        }
    ]
    
    for test_case in test_cases:
        try:
            # Process descriptive query
            result = await catalog_service.process_descriptive_query(
                query=test_case['query'],
                table_name=table_name,
                user_id=user_id
            )
            
            # Verify basic response structure
            assert result["status"] == "success", \
                f"Query failed: {result.get('error')}"
            assert "sql_query" in result, "SQL query missing from response"
            assert "results" in result, "Results missing from response"
            
            # Verify SQL query content
            sql_query = result["sql_query"].upper()
            
            # Check for expected conditions
            if 'expected_conditions' in test_case:
                for column, value in test_case['expected_conditions'].items():
                    assert column.upper() in sql_query, \
                        f"Column {column} not in SQL query"
                    assert value.upper() in sql_query, \
                        f"Value {value} not in SQL query"
            
            # Check for expected aggregation
            if 'expected_aggregation' in test_case:
                agg = test_case['expected_aggregation']
                agg_func = f"{agg['function']}({agg['column'].upper()})"
                assert agg_func in sql_query, \
                    f"Aggregation {agg['function']} not found in SQL query"
                group_by = f"GROUP BY {agg['group_by'].upper()}"
                assert group_by in sql_query, \
                    "GROUP BY clause not found in SQL query"
            
            # Check for expected limit and order
            if 'expected_limit' in test_case:
                limit_clause = f"LIMIT {test_case['expected_limit']}"
                assert limit_clause in sql_query, \
                    "LIMIT clause not found in SQL query"
            
            if 'expected_order' in test_case:
                order = test_case['expected_order']
                order_clause = (
                    f"ORDER BY {order['column'].upper()} "
                    f"{order['direction']}"
                )
                assert order_clause in sql_query, \
                    "ORDER BY clause not found in SQL query"
            
            # Verify results structure
            results = result["results"]
            assert isinstance(results, list), "Results should be a list"
            if results:
                assert isinstance(results[0], dict), \
                    "Each result should be a dictionary"
                
                # Verify column presence in results
                if 'expected_conditions' in test_case:
                    for column in test_case['expected_conditions'].keys():
                        assert column in results[0], \
                            f"Column {column} missing from results"
                
                # Verify aggregation results
                if 'expected_aggregation' in test_case:
                    agg = test_case['expected_aggregation']
                    assert agg['group_by'] in results[0], \
                        f"Group by column {agg['group_by']} missing from results"
                    avg_col = f"avg_{agg['column']}"
                    assert avg_col in results[0], \
                        "Aggregated column missing from results"
            
        except Exception as e:
            logger.error("Test case failed: %s", test_case['query'])
            logger.error("Error: %s", str(e))
            raise 