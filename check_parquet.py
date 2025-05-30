import boto3
import pandas as pd
from io import BytesIO
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_parquet_from_s3(bucket: str, key: str) -> pd.DataFrame:
    """Read a Parquet file from S3."""
    try:
        # Initialize S3 client with credentials from environment
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
            region_name=os.environ.get("AWS_REGION", "us-east-1")
        )
        
        # Get the object from S3
        logger.info(f"Reading Parquet file from s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        
        # Read the Parquet file
        df = pd.read_parquet(BytesIO(response['Body'].read()))
        
        # Display the specific column
        logger.info("\nDescription Scalable Reason for each company:")
        logger.info("-" * 80)
        for idx, row in df.iterrows():
            logger.info(f"\nCompany: {row['organization_name']}")
            logger.info(f"Reason: {row['description_scalable_reason']}")
            logger.info("-" * 80)
        
        return df
        
    except s3.exceptions.NoSuchKey:
        logger.error(f"File not found: s3://{bucket}/{key}")
        raise
    except s3.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        error_message = e.response['Error']['Message']
        logger.error(f"AWS Error ({error_code}): {error_message}")
        if error_code == 'ExpiredToken':
            logger.error(
                "Your AWS credentials have expired. Please refresh your "
                "credentials and try again."
            )
        raise
    except Exception as e:
        logger.error(f"Error reading Parquet file: {str(e)}")
        raise

def main():
    # S3 path components
    bucket = "lambda-code-q"
    key = "data/test_user/vcnote11/vcnote11_20250520_170304.parquet"
    
    try:
        # Read the Parquet file
        df = read_parquet_from_s3(bucket, key)
        
        # Additional analysis
        logger.info("\nDataFrame Info:")
        print(df.info())
        
        logger.info("\nDataFrame Description:")
        print(df.describe())
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")

if __name__ == "__main__":
    main() 