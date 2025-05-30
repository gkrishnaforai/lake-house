import boto3
import pandas as pd
from io import BytesIO
import sys
from typing import Dict, List, Tuple


def get_parquet_columns(s3_path: str) -> List[str]:
    """Get column names from Parquet file in S3."""
    try:
        # Parse S3 path
        path_without_prefix = s3_path[5:]  # Remove 's3://'
        bucket = path_without_prefix.split('/')[0]
        key = '/'.join(path_without_prefix.split('/')[1:])
        
        # Read Parquet file
        s3 = boto3.client('s3')
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        return list(df.columns)
    except Exception as e:
        print(f"Error reading Parquet file: {str(e)}")
        return []


def get_glue_columns(table_name: str, database_name: str) -> List[str]:
    """Get column names from Glue table."""
    try:
        glue = boto3.client('glue')
        response = glue.get_table(
            DatabaseName=database_name,
            Name=table_name
        )
        
        # Extract column names from Glue table schema
        columns = response['Table']['StorageDescriptor']['Columns']
        return [col['Name'] for col in columns]
    except Exception as e:
        print(f"Error reading Glue table: {str(e)}")
        return []


def compare_columns(parquet_cols: List[str], glue_cols: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """Compare column names between Parquet and Glue table."""
    parquet_set = set(parquet_cols)
    glue_set = set(glue_cols)
    
    # Find differences
    only_in_parquet = list(parquet_set - glue_set)
    only_in_glue = list(glue_set - parquet_set)
    common = list(parquet_set.intersection(glue_set))
    
    return only_in_parquet, only_in_glue, common


def main():
    if len(sys.argv) != 4:
        print("Usage: python compare_schemas.py <s3_path> <database_name> <table_name>")
        print("Example: python compare_schemas.py s3://my-bucket/path/to/file.parquet my_database my_table")
        sys.exit(1)
    
    s3_path = sys.argv[1]
    database_name = sys.argv[2]
    table_name = sys.argv[3]
    
    print("\nComparing schemas:")
    print(f"Parquet file: {s3_path}")
    print(f"Glue table: {database_name}.{table_name}")
    print("-" * 80)
    
    # Get columns from both sources
    parquet_cols = get_parquet_columns(s3_path)
    glue_cols = get_glue_columns(table_name, database_name)
    
    if not parquet_cols or not glue_cols:
        print("Error: Could not retrieve columns from one or both sources")
        sys.exit(1)
    
    # Compare columns
    only_in_parquet, only_in_glue, common = compare_columns(parquet_cols, glue_cols)
    
    # Print results
    print("\nColumns only in Parquet file:")
    for col in sorted(only_in_parquet):
        print(f"  - {col}")
    
    print("\nColumns only in Glue table:")
    for col in sorted(only_in_glue):
        print(f"  - {col}")
    
    print("\nCommon columns:")
    for col in sorted(common):
        print(f"  - {col}")
    
    # Print summary
    print("\nSummary:")
    print(f"Total columns in Parquet: {len(parquet_cols)}")
    print(f"Total columns in Glue: {len(glue_cols)}")
    print(f"Common columns: {len(common)}")
    print(f"Columns only in Parquet: {len(only_in_parquet)}")
    print(f"Columns only in Glue: {len(only_in_glue)}")


if __name__ == "__main__":
    main() 