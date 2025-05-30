import boto3
from botocore.exceptions import ClientError


TABLE_NAME = 'etl_workflow_states'
REGION = 'us-east-1'


def create_table():
    dynamodb = boto3.client('dynamodb', region_name=REGION)
    try:
        # Check if table already exists
        existing_tables = dynamodb.list_tables()['TableNames']
        if TABLE_NAME in existing_tables:
            print(f"Table '{TABLE_NAME}' already exists in region {REGION}.")
            return
        # Create the table
        dynamodb.create_table(
            TableName=TABLE_NAME,
            KeySchema=[
                {'AttributeName': 'workflow_id', 'KeyType': 'HASH'}
            ],
            AttributeDefinitions=[
                {'AttributeName': 'workflow_id', 'AttributeType': 'S'}
            ],
            ProvisionedThroughput={
                'ReadCapacityUnits': 5,
                'WriteCapacityUnits': 5
            }
        )
        print(f"Creating table '{TABLE_NAME}'...")
        # Wait for the table to become active
        waiter = dynamodb.get_waiter('table_exists')
        waiter.wait(TableName=TABLE_NAME)
        print(f"Table '{TABLE_NAME}' created and is now ACTIVE.")
    except ClientError as e:
        print(f"Error creating table: {e.response['Error']['Message']}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")


def main():
    create_table()


if __name__ == "__main__":
    main() 