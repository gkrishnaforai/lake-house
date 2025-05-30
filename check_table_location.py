import boto3
import sys


def get_table_location(database_name: str, table_name: str) -> dict:
    """Get the table's location and configuration from Glue."""
    try:
        glue = boto3.client('glue')
        response = glue.get_table(
            DatabaseName=database_name,
            Name=table_name
        )
        
        table_info = response['Table']
        storage_descriptor = table_info['StorageDescriptor']
        
        return {
            'location': storage_descriptor.get('Location'),
            'input_format': storage_descriptor.get('InputFormat'),
            'output_format': storage_descriptor.get('OutputFormat'),
            'serde_info': storage_descriptor.get('SerdeInfo'),
            'compressed': storage_descriptor.get('Compressed', False),
            'parameters': table_info.get('Parameters', {})
        }
    except Exception as e:
        print(f"Error getting table location: {str(e)}")
        return {}


def main():
    if len(sys.argv) != 3:
        print("Usage: python check_table_location.py <database_name> <table_name>")
        print("Example: python check_table_location.py user_test_user abcnote4")
        sys.exit(1)
    
    database_name = sys.argv[1]
    table_name = sys.argv[2]
    
    print("\nChecking Glue table configuration:")
    print(f"Database: {database_name}")
    print(f"Table: {table_name}")
    print("-" * 80)
    
    # Get table location info
    table_config = get_table_location(database_name, table_name)
    
    if not table_config:
        print("Error: Could not retrieve table configuration")
        sys.exit(1)
    
    # Print detailed configuration
    print("\nTable Configuration:")
    print(f"Location: {table_config['location']}")
    print(f"Input Format: {table_config['input_format']}")
    print(f"Output Format: {table_config['output_format']}")
    print(f"Compressed: {table_config['compressed']}")
    
    print("\nSerDe Info:")
    serde_info = table_config['serde_info']
    if serde_info:
        print(f"  Serialization Library: {serde_info.get('SerializationLibrary')}")
        print(f"  Parameters: {serde_info.get('Parameters', {})}")
    
    print("\nTable Parameters:")
    for key, value in table_config['parameters'].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main() 