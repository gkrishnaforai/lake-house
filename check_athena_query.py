import boto3
import time
import sys
import json
from datetime import datetime


def get_athena_client():
    """Initialize and return Athena client."""
    return boto3.client('athena')


def execute_query(query: str, database: str, workgroup: str = 'primary') -> str:
    """Execute an Athena query and return the query execution ID."""
    athena = get_athena_client()
    
    try:
        response = athena.start_query_execution(
            QueryString=query,
            QueryExecutionContext={
                'Database': database
            },
            WorkGroup=workgroup
        )
        return response['QueryExecutionId']
    except Exception as e:
        print(f"Error executing query: {str(e)}")
        return None


def get_query_execution(query_execution_id: str) -> dict:
    """Get details about a specific query execution."""
    athena = get_athena_client()
    
    try:
        response = athena.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        return response['QueryExecution']
    except Exception as e:
        print(f"Error getting query execution: {str(e)}")
        return None


def get_workgroup_config(workgroup: str = 'primary') -> dict:
    """Get Athena workgroup configuration."""
    athena = get_athena_client()
    
    try:
        response = athena.get_work_group(
            WorkGroup=workgroup
        )
        return response['WorkGroup']
    except Exception as e:
        print(f"Error getting workgroup config: {str(e)}")
        return None


def main():
    if len(sys.argv) != 3:
        print("Usage: python check_athena_query.py <database> <table>")
        print("Example: python check_athena_query.py user_test_user abcnote4")
        sys.exit(1)
        
    database = sys.argv[1]
    table = sys.argv[2]
    
    # 1. Check workgroup configuration
    print("\nChecking Athena Workgroup Configuration:")
    print("-" * 50)
    workgroup_config = get_workgroup_config()
    if workgroup_config:
        print(f"Workgroup: {workgroup_config['Name']}")
        print(f"State: {workgroup_config['State']}")
        print("\nConfiguration:")
        config = workgroup_config.get('Configuration', {})
        print(f"  Result Location: {config.get('ResultConfiguration', {}).get('OutputLocation', 'Not set')}")
        print(f"  Encryption: {config.get('ResultConfiguration', {}).get('EncryptionConfiguration', 'Not set')}")
        print(f"  Enforce Workgroup Configuration: {config.get('EnforceWorkGroupConfiguration', False)}")
    
    # 2. Execute a test query
    print("\nExecuting Test Query:")
    print("-" * 50)
    query = f"SELECT * FROM {database}.{table}"
    print(f"Query: {query}")
    
    query_execution_id = execute_query(query, database)
    if not query_execution_id:
        print("Failed to execute query")
        return
    
    # 3. Wait for query completion and get results
    print("\nWaiting for query completion...")
    while True:
        execution = get_query_execution(query_execution_id)
        if not execution:
            print("Failed to get query execution details")
            return
            
        state = execution['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(1)
    
    # 4. Print query execution details
    print("\nQuery Execution Details:")
    print("-" * 50)
    print(f"Query Execution ID: {query_execution_id}")
    print(f"State: {state}")
    
    if state == 'SUCCEEDED':
        print("\nQuery Statistics:")
        stats = execution['Statistics']
        print(f"  Data Scanned: {stats.get('DataScannedInBytes', 0)} bytes")
        print(f"  Execution Time: {stats.get('TotalExecutionTimeInMillis', 0)} ms")
        print(f"  Engine Execution Time: {stats.get('EngineExecutionTimeInMillis', 0)} ms")
        print(f"  Queue Time: {stats.get('QueryQueueTimeInMillis', 0)} ms")
        
        # Get query results
        athena = get_athena_client()
        try:
            results = athena.get_query_results(
                QueryExecutionId=query_execution_id
            )
            print("\nQuery Results:")
            rows = results.get('ResultSet', {}).get('Rows', [])
            if len(rows) > 0:
                # Get headers from first row
                headers = [col.get('VarCharValue', '') for col in rows[0].get('Data', [])]
                print("\nHeaders:")
                print(", ".join(headers))
                
                # Print data rows
                print("\nData Rows:")
                for row in rows[1:]:  # Skip header row
                    values = [col.get('VarCharValue', '') for col in row.get('Data', [])]
                    print(", ".join(values))
            else:
                print("No data found")
        except Exception as e:
            print(f"Error getting query results: {str(e)}")
    else:
        print(f"\nQuery Failed: {execution['Status'].get('StateChangeReason', 'Unknown error')}")


if __name__ == "__main__":
    main() 