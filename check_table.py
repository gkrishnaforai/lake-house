import boto3
import os
import json
import sys
from datetime import datetime


def datetime_handler(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def check_table(database_name, table_name):
    # Initialize AWS clients
    glue = boto3.client('glue', region_name=os.environ.get('AWS_REGION'))
    athena = boto3.client('athena', region_name=os.environ.get('AWS_REGION'))
    
    try:
        # Check if table exists
        print(f"Checking if table {table_name} exists in database {database_name}...")
        response = glue.get_table(
            DatabaseName=database_name,
            Name=table_name
        )
        print("Table exists!")
        print("\nTable details:")
        print(json.dumps(response['Table'], indent=2, default=datetime_handler))
        
        # Try to query the table
        print("\nExecuting query...")
        query_response = athena.start_query_execution(
            QueryString=f'SELECT COUNT(*) FROM {database_name}.{table_name}',
            QueryExecutionContext={
                'Database': database_name
            },
            ResultConfiguration={
                'OutputLocation': 's3://lambda-code-q/athena-results/'
            }
        )
        
        query_execution_id = query_response['QueryExecutionId']
        print(f"Query execution ID: {query_execution_id}")
        
        # Wait for query to complete
        while True:
            query_status = athena.get_query_execution(
                QueryExecutionId=query_execution_id
            )['QueryExecution']['Status']['State']
            
            if query_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
                
            print("Waiting for query to complete...")
            import time
            time.sleep(1)
        
        if query_status == 'SUCCEEDED':
            results = athena.get_query_results(
                QueryExecutionId=query_execution_id
            )
            print("\nQuery results:")
            print(json.dumps(results, indent=2, default=datetime_handler))
        else:
            error = athena.get_query_execution(
                QueryExecutionId=query_execution_id
            )['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
            print(f"\nQuery failed: {error}")
            
    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python check_table.py <database_name> <table_name>")
        sys.exit(1)
    
    database_name = sys.argv[1]
    table_name = sys.argv[2]
    check_table(database_name, table_name) 