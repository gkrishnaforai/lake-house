"""AWS credentials setup script."""

import boto3
import os
import logging
from botocore.exceptions import ClientError


logger = logging.getLogger(__name__)


def setup_aws_credentials():
    """Set up and validate AWS credentials."""
    try:
        # Get AWS credentials from environment variables
        aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_session_token = os.getenv('AWS_SESSION_TOKEN')
        aws_region = os.getenv('AWS_REGION', 'us-east-1')
        bucket_name = os.getenv('AWS_S3_BUCKET')
        database_name = os.getenv('GLUE_DATABASE_NAME', 'data_lakehouse')

        # Log masked credentials for debugging
        masked_key = (
            aws_access_key[:4] + '*' * (len(aws_access_key) - 8) + 
            aws_access_key[-4:] if aws_access_key else None
        )
        logger.info(f"Using AWS Access Key: {masked_key}")
        logger.info(f"Using AWS Region: {aws_region}")
        logger.info(f"Using S3 Bucket: {bucket_name}")
        logger.info(f"Using Glue Database: {database_name}")

        # Initialize S3 client
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )

        # Test S3 connection and permissions
        logger.info("Testing S3 connection and permissions...")
        
        # Test bucket listing
        try:
            buckets = s3_client.list_buckets()
            logger.info(f"Successfully listed {len(buckets['Buckets'])} buckets")
        except ClientError as e:
            logger.error(f"Failed to list buckets: {str(e)}")
            raise

        # Test bucket access
        try:
            s3_client.head_bucket(Bucket=bucket_name)
            logger.info(f"Successfully accessed bucket: {bucket_name}")
        except ClientError as e:
            logger.error(f"Failed to access bucket {bucket_name}: {str(e)}")
            raise

        # Test bucket permissions
        try:
            # Test write permission
            test_key = "test_permissions.txt"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=test_key,
                Body="test"
            )
            logger.info(f"Successfully wrote to bucket {bucket_name}")
            
            # Test read permission
            s3_client.get_object(
                Bucket=bucket_name,
                Key=test_key
            )
            logger.info(f"Successfully read from bucket {bucket_name}")
            
            # Clean up test file
            s3_client.delete_object(
                Bucket=bucket_name,
                Key=test_key
            )
            logger.info(f"Successfully deleted test file from bucket {bucket_name}")
        except ClientError as e:
            logger.error(f"Failed to test bucket permissions: {str(e)}")
            raise

        logger.info("Successfully connected to AWS S3")

        # Initialize Glue client
        glue_client = boto3.client(
            'glue',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
            region_name=aws_region
        )

        # Test Glue connection
        logger.info("Testing Glue connection...")
        try:
            # Check if database exists
            try:
                glue_client.get_database(Name=database_name)
                logger.info(f"Database {database_name} already exists")
            except glue_client.exceptions.EntityNotFoundException:
                # Create database if it doesn't exist
                glue_client.create_database(
                    DatabaseInput={
                        'Name': database_name,
                        'Description': 'Data Lakehouse Database'
                    }
                )
                logger.info(f"Created database {database_name}")

            # Test Glue permissions
            try:
                # List tables in database
                tables = glue_client.get_tables(DatabaseName=database_name)
                logger.info(
                    f"Successfully listed {len(tables['TableList'])} tables in "
                    f"database {database_name}"
                )
            except ClientError as e:
                logger.error(
                    f"Failed to list tables in database {database_name}: {str(e)}"
                )
                raise

            logger.info("Successfully connected to AWS Glue")

        except Exception as e:
            logger.error(f"Error testing Glue connection: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error setting up AWS credentials: {str(e)}")
        raise


if __name__ == "__main__":
    setup_aws_credentials() 