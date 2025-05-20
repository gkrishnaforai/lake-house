import boto3
import os
from datetime import datetime

def get_glue_client():
    return boto3.client('glue',
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        region_name=os.environ.get("AWS_REGION", "us-east-1")
    )

def update_table_location(database_name: str, table_name: str, new_location: str):
    """Update the location of a Glue table."""
    glue = get_glue_client()
    
    try:
        # Get current table
        response = glue.get_table(
            DatabaseName=database_name,
            Name=table_name
        )
        
        # Create table input with only required fields
        table_input = {
            'Name': table_name,
            'StorageDescriptor': response['Table']['StorageDescriptor'],
            'TableType': response['Table']['TableType'],
            'Parameters': response['Table'].get('Parameters', {})
        }
        
        # Update the location
        table_input['StorageDescriptor']['Location'] = new_location
        
        # Update the table
        glue.update_table(
            DatabaseName=database_name,
            TableInput=table_input
        )
        
        print(f"Successfully updated table location to: {new_location}")
        return True
        
    except Exception as e:
        print(f"Error updating table location: {str(e)}")
        return False

def main():
    database_name = "user_test_user"
    table_name = "aa2"
    new_location = "s3://lambda-code-q/data/test_user/aa2/"
    
    print(f"Updating table {database_name}.{table_name} location to: {new_location}")
    success = update_table_location(database_name, table_name, new_location)
    
    if success:
        print("Table location updated successfully!")
    else:
        print("Failed to update table location.")

if __name__ == "__main__":
    main() 