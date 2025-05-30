import boto3
import pandas as pd
from io import BytesIO
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import fastavro
import json
from typing import Dict, Any, List, Optional


def get_glue_table_config(
    table_name: str,
    location: str,
    columns: List[Dict[str, str]],
    file_format: str = "parquet"
) -> Dict[str, Any]:
    """Generate Glue table configuration based on file format."""
    if file_format.lower() == "parquet":
        return {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'parquet',
                'typeOfData': 'file',
                'compressionType': 'snappy',
                'parquet.compression': 'SNAPPY'
            },
            'StorageDescriptor': {
                'Columns': columns,
                'Location': location,
                'InputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat',
                'OutputFormat': 'org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat',
                'SerdeInfo': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe',
                    'Parameters': {'serialization.format': '1'}
                }
            }
        }
    elif file_format.lower() == "avro":
        return {
            'Name': table_name,
            'TableType': 'EXTERNAL_TABLE',
            'Parameters': {
                'classification': 'avro',
                'typeOfData': 'file',
                'compressionType': 'snappy',
                'avro.schema.literal': json.dumps({
                    'type': 'record',
                    'name': table_name,
                    'fields': [
                        {'name': col['Name'], 'type': _convert_to_avro_type(col['Type'])}
                        for col in columns
                    ]
                })
            },
            'StorageDescriptor': {
                'Columns': columns,
                'Location': location,
                'InputFormat': 'org.apache.hadoop.hive.ql.io.avro.AvroContainerInputFormat',
                'OutputFormat': 'org.apache.hadoop.hive.ql.io.avro.AvroContainerOutputFormat',
                'SerdeInfo': {
                    'SerializationLibrary': 'org.apache.hadoop.hive.serde2.avro.AvroSerDe',
                    'Parameters': {
                        'avro.schema.literal': json.dumps({
                            'type': 'record',
                            'name': table_name,
                            'fields': [
                                {'name': col['Name'], 'type': _convert_to_avro_type(col['Type'])}
                                for col in columns
                            ]
                        })
                    }
                }
            }
        }
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


def _convert_to_avro_type(glue_type: str) -> Dict[str, Any]:
    """Convert Glue data type to Avro type."""
    type_mapping = {
        'string': 'string',
        'bigint': 'long',
        'double': 'double',
        'boolean': 'boolean',
        'timestamp': {'type': 'long', 'logicalType': 'timestamp-millis'},
        'date': {'type': 'int', 'logicalType': 'date'},
        'decimal': {'type': 'bytes', 'logicalType': 'decimal', 'precision': 38, 'scale': 18}
    }
    return type_mapping.get(glue_type.lower(), 'string')


def _convert_to_glue_type(pandas_type: str) -> str:
    """Convert pandas dtype to Glue data type."""
    type_mapping = {
        'int64': 'bigint',
        'float64': 'double',
        'bool': 'boolean',
        'datetime64[ns]': 'timestamp',
        'object': 'string'
    }
    return type_mapping.get(str(pandas_type).lower(), 'string')


def rewrite_file(
    s3_path: str,
    file_format: str = "parquet",
    create_glue_table: bool = False,
    database_name: Optional[str] = None,
    table_name: Optional[str] = None
) -> None:
    """Read a file from S3, rewrite it in the specified format, and upload back."""
    try:
        # Parse S3 path
        if not s3_path.startswith('s3://'):
            raise ValueError("Invalid S3 path. Must start with 's3://'")
            
        bucket = s3_path.split('/')[2]
        key = '/'.join(s3_path.split('/')[3:])
        
        # Initialize AWS clients
        s3 = boto3.client('s3')
        glue = boto3.client('glue')
        
        # Read existing file
        print(f"Reading file from {s3_path}")
        response = s3.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # Print DataFrame info
        print("\nDataFrame Info:")
        print(f"Shape: {df.shape}")
        print("\nColumns and Types:")
        for col, dtype in df.dtypes.items():
            print(f"{col}: {dtype}")
        
        # Convert to PyArrow Table with explicit schema
        table = pa.Table.from_pandas(df)
        
        # Generate new S3 key
        new_key = key.replace('.parquet', f'_{file_format}.{file_format}')
        
        if file_format.lower() == "parquet":
            # Write to Parquet with proper settings
            parquet_buffer = BytesIO()
            pq.write_table(
                table,
                parquet_buffer,
                compression='snappy',
                version='2.6',
                write_statistics=True
            )
            parquet_buffer.seek(0)
            
            # Upload to S3
            print(f"\nUploading rewritten Parquet file to s3://{bucket}/{new_key}")
            s3.put_object(
                Bucket=bucket,
                Key=new_key,
                Body=parquet_buffer.getvalue()
            )
            
        elif file_format.lower() == "avro":
            # Convert to Avro schema
            avro_schema = {
                'type': 'record',
                'name': table_name or 'record',
                'fields': [
                    {
                        'name': col,
                        'type': _convert_to_avro_type(_convert_to_glue_type(str(dtype)))
                    }
                    for col, dtype in df.dtypes.items()
                ]
            }
            
            # Write to Avro
            avro_buffer = BytesIO()
            records = df.to_dict('records')
            fastavro.writer(avro_buffer, avro_schema, records)
            avro_buffer.seek(0)
            
            # Upload to S3
            print(f"\nUploading Avro file to s3://{bucket}/{new_key}")
            s3.put_object(
                Bucket=bucket,
                Key=new_key,
                Body=avro_buffer.getvalue()
            )
        
        print(f"\nSuccessfully rewrote file to {file_format} format")
        print(f"Original file: s3://{bucket}/{key}")
        print(f"New file: s3://{bucket}/{new_key}")
        
        # Create/update Glue table if requested
        if create_glue_table and database_name and table_name:
            # Convert DataFrame columns to Glue format
            columns = [
                {
                    'Name': col,
                    'Type': _convert_to_glue_type(str(dtype))
                }
                for col, dtype in df.dtypes.items()
            ]
            
            # Generate table configuration
            table_input = get_glue_table_config(
                table_name=table_name,
                location=f"s3://{bucket}/{new_key}",
                columns=columns,
                file_format=file_format
            )
            
            try:
                # Try to create new table
                glue.create_table(
                    DatabaseName=database_name,
                    TableInput=table_input
                )
                print(f"\nCreated new Glue table: {database_name}.{table_name}")
            except glue.exceptions.AlreadyExistsException:
                # Update existing table
                glue.update_table(
                    DatabaseName=database_name,
                    TableInput=table_input
                )
                print(f"\nUpdated existing Glue table: {database_name}.{table_name}")
        
    except Exception as e:
        print(f"Error rewriting file: {str(e)}")
        sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage: python rewrite_parquet.py <s3_path> [--format parquet|avro] [--create-glue] [--database DATABASE] [--table TABLE]")
        print("Example: python rewrite_parquet.py s3://bucket/path/to/file.parquet --format avro --create-glue --database mydb --table mytable")
        sys.exit(1)
        
    s3_path = sys.argv[1]
    file_format = "parquet"
    create_glue = False
    database_name = None
    table_name = None
    
    # Parse command line arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--format":
            file_format = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--create-glue":
            create_glue = True
            i += 1
        elif sys.argv[i] == "--database":
            database_name = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--table":
            table_name = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    if create_glue and (not database_name or not table_name):
        print("Error: --database and --table are required when --create-glue is specified")
        sys.exit(1)
    
    rewrite_file(s3_path, file_format, create_glue, database_name, table_name)


if __name__ == "__main__":
    main() 